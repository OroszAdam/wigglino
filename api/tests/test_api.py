import io
import time

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from wigglegram.app import create_app
from wigglegram.config import Settings, parse_rate
from wigglegram.ratelimit import MemoryRateLimiter
from wigglegram.security import client_ip


class FakeEstimator:
    """Left-to-right gradient instead of a real model."""

    loaded: list[str] = []

    def __init__(self):
        self.calls = 0

    def load(self, key):
        pass

    def estimate(self, image, key):
        self.calls += 1
        W, H = image.size
        return torch.linspace(0, 1, W).expand(H, W).contiguous()


def _png(w=64, h=48, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    # Textured, because Pillow merges identical GIF frames.
    Image.effect_noise((w, h), 64).convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(**overrides):
        settings = Settings(data_dir=tmp_path, device="cpu", max_side=64, **overrides)
        est = FakeEstimator()
        client = TestClient(create_app(settings, estimator=est))
        client.__enter__()
        clients.append(client)
        return client, est

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def _upload(client, data=None, name="a.png"):
    return client.post("/api/images", files={"file": (name, data or _png(), "image/png")})


def _wait(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_full_flow_and_depth_cache(make_client):
    client, est = make_client()
    r = _upload(client, _png(128, 96))
    assert r.status_code == 201
    up = r.json()
    assert (up["width"], up["height"]) == (64, 48)  # downscaled to max_side

    r = client.post("/api/jobs", json={"image_id": up["image_id"], "model": "mono",
                                       "params": {"frames": 3, "format": "gif"}})
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    res = job["result"]
    gif = client.get(res["url"])
    assert gif.status_code == 200
    with Image.open(io.BytesIO(gif.content)) as img:
        assert img.n_frames == 4  # ping-pong of 3
        assert img.size == (64, 48)
    assert client.get(res["depth_url"]).status_code == 200
    dl = client.get(res["url"], params={"download": 1})
    assert "attachment" in dl.headers["content-disposition"]

    # Second render reuses the cached depth.
    r = client.post("/api/jobs", json={"image_id": up["image_id"], "model": "mono",
                                       "params": {"frames": 4, "axis": "circular", "order": "forward",
                                                  "format": "webp"}})
    assert _wait(client, r.json()["job_id"])["status"] == "done"
    assert est.calls == 1


def test_rejects_bad_uploads(make_client):
    client, _ = make_client(max_upload_bytes=2000)
    assert _upload(client, b"not an image").status_code == 415
    assert _upload(client, _png(8, 8, fmt="BMP")).status_code == 415
    big = _png(400, 400)
    assert len(big) > 2000
    assert _upload(client, big).status_code == 413


def test_rejects_bad_job_requests(make_client):
    client, _ = make_client(max_frames=8)
    image_id = _upload(client).json()["image_id"]
    bad = [
        {"image_id": "../../etc/passwd", "model": "mono"},
        {"image_id": image_id, "model": "giant"},
        {"image_id": image_id, "params": {"frames": 1}},
        {"image_id": image_id, "params": {"rotation_deg": 999}},
        {"image_id": image_id, "params": {"unknown": 1}},
    ]
    for body in bad:
        assert client.post("/api/jobs", json=body).status_code == 422, body
    assert client.post("/api/jobs", json={"image_id": image_id, "params": {"frames": 9}}).status_code == 422
    assert client.post("/api/jobs", json={"image_id": "A" * 22}).status_code == 404


def test_files_and_jobs_only_serve_known_names(make_client):
    client, _ = make_client()
    image_id = _upload(client).json()["image_id"]
    assert client.get(f"/api/files/{image_id}.png").status_code == 404  # uploads are never served
    assert client.get("/api/files/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_rate_limits_uploads_and_depth(make_client):
    client, _ = make_client(rate_upload=parse_rate("2/600"), rate_depth=parse_rate("1/600"))
    ids = [_upload(client).json()["image_id"] for _ in range(2)]
    r = _upload(client)
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert client.post("/api/jobs", json={"image_id": ids[0]}).status_code == 202
    # A second image needs a new depth map -> limited.
    assert client.post("/api/jobs", json={"image_id": ids[1]}).status_code == 429


def test_queue_full_returns_503(make_client):
    client, _ = make_client(max_queue=0)
    image_id = _upload(client).json()["image_id"]
    r = client.post("/api/jobs", json={"image_id": image_id})
    assert r.status_code == 503 and r.headers["retry-after"]


def test_turnstile_required_when_configured(make_client, monkeypatch):
    seen = {}

    async def fake_verify(http, secret, token, ip):
        seen.update(secret=secret, token=token)
        return token == "good"

    monkeypatch.setattr("wigglegram.app.verify_turnstile", fake_verify)
    client, _ = make_client(turnstile_secret="s3cret")
    assert _upload(client).status_code == 403
    r = client.post("/api/images", files={"file": ("a.png", _png(), "image/png")}, data={"turnstile": "good"})
    assert r.status_code == 201 and seen == {"secret": "s3cret", "token": "good"}


def test_post_without_content_length_is_rejected(make_client):
    client, _ = make_client()

    def chunks():
        yield b'{"image_id": "x"}'

    r = client.post("/api/jobs", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 411


def test_cors_allows_configured_origin_only(make_client):
    client, _ = make_client(allowed_origins=["https://wiggle.pages.dev"])
    ok = client.get("/api/health", headers={"Origin": "https://wiggle.pages.dev"})
    assert ok.headers.get("access-control-allow-origin") == "https://wiggle.pages.dev"
    bad = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def _request(headers: dict, peer="10.0.0.1") -> Request:
    return Request({"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
                    "client": (peer, 1234)})


def test_client_ip_trusts_only_proxy_hops():
    spoofed = {"x-forwarded-for": "6.6.6.6, 1.2.3.4"}
    assert client_ip(_request(spoofed), "x-forwarded-for", 0) == "10.0.0.1"
    assert client_ip(_request(spoofed), "x-forwarded-for", 1) == "1.2.3.4"
    assert client_ip(_request({}), "x-forwarded-for", 1) == "10.0.0.1"


def test_memory_rate_limiter_refills():
    now = [0.0]
    lim = MemoryRateLimiter([(2, 10.0)], clock=lambda: now[0])
    assert lim.hit("a") is None and lim.hit("a") is None
    wait = lim.hit("a")
    assert wait == pytest.approx(5.0)
    assert lim.hit("b") is None  # separate key
    now[0] = 5.0
    assert lim.hit("a") is None
    now[0] = 100.0
    lim.prune()
    assert lim._state == {}
