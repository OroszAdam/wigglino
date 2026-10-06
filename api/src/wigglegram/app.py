import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .depth_model import DepthEstimator, pick_device
from .jobs import JobQueue, QueueFull
from .pipeline import BadImage, JobRequest, decode_image, make_handler
from .ratelimit import MemoryRateLimiter, RateLimiter
from .security import client_ip, verify_turnstile
from .storage import ID_RE, FileStore, new_id

log = logging.getLogger("wigglegram")

SWEEP_INTERVAL_S = 60
# Multipart overhead on top of the image itself.
UPLOAD_OVERHEAD = 64 * 1024
JSON_BODY_LIMIT = 16 * 1024


class BodyLimit:
    """Reject POSTs without Content-Length or above the limit, before anything is read.

    The server enforces Content-Length framing, so the body can't grow past the declared size.
    """

    def __init__(self, app, upload_limit: int):
        self.app = app
        self.upload_limit = upload_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            limit = self.upload_limit if scope["path"] == "/api/images" else JSON_BODY_LIMIT
            raw = dict(scope["headers"]).get(b"content-length")
            if raw is None or not raw.isdigit():
                return await JSONResponse({"detail": "Content-Length required."}, 411)(scope, receive, send)
            if int(raw) > limit:
                return await JSONResponse({"detail": "Upload is too large."}, 413)(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None, estimator: DepthEstimator | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    device = pick_device(settings.device)
    estimator = estimator or DepthEstimator(device, settings.depth_res)
    store = FileStore(settings.data_dir, settings.file_ttl_s)
    limiters: dict[str, RateLimiter] = {
        "upload": MemoryRateLimiter(settings.rate_upload),
        "depth": MemoryRateLimiter(settings.rate_depth),
        "render": MemoryRateLimiter(settings.rate_render),
    }
    queue = JobQueue(make_handler(store, estimator, device), settings.max_queue, settings.workers,
                     settings.file_ttl_s)

    async def sweep_loop():
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            try:
                await asyncio.to_thread(store.sweep)
                queue.sweep()
                for limiter in limiters.values():
                    limiter.prune()
            except Exception:
                log.exception("sweep failed")

    async def preload(key: str):
        try:
            await asyncio.to_thread(estimator.load, key)
        except Exception:
            log.exception("preloading %s failed", key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.turnstile_secret:
            log.warning("TURNSTILE_SECRET is not set: bot protection is OFF")
        log.info("device=%s depth_res=%s", device, settings.depth_res)
        queue.start()
        app.state.http = httpx.AsyncClient()
        tasks = [asyncio.create_task(sweep_loop())]
        tasks += [asyncio.create_task(preload(k)) for k in settings.preload_models]
        yield
        for t in tasks:
            t.cancel()
        await queue.stop()
        await app.state.http.aclose()

    app = FastAPI(title="wigglegram", lifespan=lifespan, docs_url=None, redoc_url=None)
    # Added first = inner, so its 413/411 responses still get CORS headers.
    app.add_middleware(BodyLimit, upload_limit=settings.max_upload_bytes + UPLOAD_OVERHEAD)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_origin_regex=settings.allowed_origin_regex,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    def check_rate(name: str, ip: str) -> None:
        retry = limiters[name].hit(ip)
        if retry is not None:
            raise HTTPException(429, "Too many requests, please wait a bit.",
                                headers={"Retry-After": str(int(retry) + 1)})

    def ip_of(request: Request) -> str:
        ip = client_ip(request, settings.client_ip_header, settings.trusted_proxy_hops)
        if log.isEnabledFor(logging.DEBUG):  # LOG_LEVEL=DEBUG to check TRUSTED_PROXY_HOPS on a new host
            log.debug("peer=%s %s=%r -> %s", request.client and request.client.host,
                      settings.client_ip_header, request.headers.get(settings.client_ip_header), ip)
        return ip

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "device": str(device), "models_loaded": estimator.loaded,
                "queue": queue.pending, "max_frames": settings.max_frames}

    @app.post("/api/images", status_code=201)
    async def upload_image(request: Request, file: UploadFile = File(...), turnstile: str = Form("")):
        ip = ip_of(request)
        check_rate("upload", ip)
        if settings.turnstile_secret and not await verify_turnstile(
                app.state.http, settings.turnstile_secret, turnstile, ip):
            raise HTTPException(403, "Bot check failed, please try again.")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "Upload is too large.")
        try:
            img = await asyncio.to_thread(decode_image, data, settings.max_input_pixels, settings.max_side)
        except BadImage as e:
            raise HTTPException(415, str(e)) from None
        image_id = new_id()
        await asyncio.to_thread(store.save_image, image_id, img)
        return {"image_id": image_id, "width": img.width, "height": img.height}

    @app.post("/api/jobs", status_code=202)
    async def create_job(req: JobRequest, request: Request):
        if req.params.frames > settings.max_frames:
            raise HTTPException(422, f"At most {settings.max_frames} frames.")
        if not store.image_path(req.image_id).is_file():
            raise HTTPException(404, "The image has expired. Please upload it again.")
        ip = ip_of(request)
        if not store.has_depth(req.image_id, req.model):
            check_rate("depth", ip)
        check_rate("render", ip)
        try:
            job = queue.submit(req)
        except QueueFull:
            raise HTTPException(503, "The service is busy, please try again shortly.",
                                headers={"Retry-After": "30"}) from None
        return {"job_id": job.id, "position": queue.position(job)}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = queue.jobs.get(job_id) if ID_RE.match(job_id) else None
        if job is None:
            raise HTTPException(404, "Unknown job.")
        result = None
        if job.result:
            result = {
                "url": f"/api/files/{job.result['file']}",
                "depth_url": f"/api/files/{job.result['depth_file']}",
                **{k: job.result[k] for k in ("pivot_disparity", "width", "height")},
            }
        return {"status": job.status, "stage": job.stage, "position": queue.position(job),
                "result": result, "error": job.error}

    @app.get("/api/files/{name}")
    async def get_file(name: str, download: bool = False):
        path = store.public_path(name)
        if path is None:
            raise HTTPException(404, "Not found (files expire after a while).")
        headers = {"Cache-Control": "public, max-age=3600, immutable"}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="wigglegram{path.suffix}"'
        return FileResponse(path, headers=headers)

    return app
