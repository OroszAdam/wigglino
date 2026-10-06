"""Request schema, upload decoding and the job handler (depth -> relief -> animation)."""

import io
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .core.depth import resize
from .core.encode import encode_animation
from .core.paths import ping_pong
from .core.pivot import pivot_disparity
from .core.relief import render_relief
from .depth_model import DepthEstimator
from .jobs import Job, JobError
from .storage import FileStore

ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}


class RenderParams(BaseModel):
    """Defaults match the tuned ComfyUI v4 relief workflow."""

    model_config = ConfigDict(extra="forbid")

    pivot_x: float = Field(0.5, ge=0, le=1)
    pivot_y: float = Field(0.5, ge=0, le=1)
    pivot_radius: int = Field(4, ge=0, le=64)
    frames: int = Field(8, ge=2, le=64)
    rotation_deg: float = Field(4.5, ge=0.1, le=30)
    depth_pct: float = Field(30, ge=1, le=200)
    axis: Literal["vertical", "horizontal", "circular"] = "vertical"
    tear: float = Field(0.89, ge=0, le=1)
    hole_fill: Literal["stretch", "push_pull", "none"] = "stretch"
    order: Literal["ping_pong", "forward"] = "ping_pong"
    fps: float = Field(24, ge=1, le=60)
    format: Literal["gif", "webp"] = "gif"


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(pattern=r"^[A-Za-z0-9_-]{22}$")
    model: Literal["mono", "metric"] = "metric"
    params: RenderParams = RenderParams()


class BadImage(Exception):
    pass


def decode_image(data: bytes, max_pixels: int, max_side: int) -> Image.Image:
    """Validate and normalise an upload: known format, bounded size, EXIF-rotated RGB."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise BadImage("Please upload a JPEG, PNG or WebP image.")
            if img.width * img.height > max_pixels:
                raise BadImage("That image has too many pixels.")
            out = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise BadImage("Could not read that image.") from None
    if max(out.size) > max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def make_handler(store: FileStore, estimator: DepthEstimator, device: torch.device):
    def handle(job: Job) -> dict:
        req: JobRequest = job.payload
        p = req.params
        img = store.load_image(req.image_id)
        if img is None:
            raise JobError("The image has expired. Please upload it again.")
        if store.has_depth(req.image_id, req.model):
            disp = store.load_depth(req.image_id, req.model)
        else:
            job.stage = "depth"
            disp = estimator.estimate(img, req.model)
            store.save_depth(req.image_id, req.model, disp)

        job.stage = "render"
        rgb = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).to(device)
        H, W = rgb.shape[:2]
        d = resize(disp.to(device), H, W)
        pivot = pivot_disparity(d, p.pivot_x, p.pivot_y, p.pivot_radius)
        frames, _ = render_relief(rgb, d, pivot, p.frames, p.rotation_deg, p.depth_pct, p.axis, p.tear,
                                  p.hole_fill)
        if p.order == "ping_pong" and frames.shape[0] > 2:
            frames = frames[ping_pong(frames.shape[0])]

        job.stage = "encode"
        name = f"{job.id}.{p.format}"
        store.save_result(name, encode_animation(frames, p.fps, p.format))
        return {
            "file": name,
            "depth_file": store.depth_preview_name(req.image_id, req.model),
            "pivot_disparity": pivot,
            "width": W,
            "height": H,
        }

    return handle
