"""Short-lived files on local disk: uploads, cached depth, results. Everything expires after a TTL."""

import re
import secrets
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ID_RE = re.compile(r"^[A-Za-z0-9_-]{22}$")
# Only depth previews and results are ever served publicly.
PUBLIC_RE = re.compile(r"^[A-Za-z0-9_-]{22}(\.(mono|metric)\.depth\.png|\.gif|\.webp)$")


def new_id() -> str:
    return secrets.token_urlsafe(16)  # 22 chars


class FileStore:
    def __init__(self, root: Path, ttl_s: float):
        self.root = Path(root)
        self.ttl_s = ttl_s
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / name

    def image_path(self, image_id: str) -> Path:
        if not ID_RE.match(image_id):
            raise ValueError("bad id")
        return self._path(f"{image_id}.png")

    def depth_path(self, image_id: str, model: str) -> Path:
        return self._path(f"{image_id}.{model}.npy")

    def depth_preview_name(self, image_id: str, model: str) -> str:
        return f"{image_id}.{model}.depth.png"

    def public_path(self, name: str) -> Path | None:
        if not PUBLIC_RE.match(name):
            return None
        p = self._path(name)
        return p if p.is_file() else None

    def save_image(self, image_id: str, img: Image.Image) -> None:
        img.save(self.image_path(image_id), format="PNG", compress_level=1)

    def load_image(self, image_id: str) -> Image.Image | None:
        p = self.image_path(image_id)
        if not p.is_file():
            return None
        p.touch()  # keep alive while in use
        with Image.open(p) as img:
            return img.convert("RGB")

    def has_depth(self, image_id: str, model: str) -> bool:
        return self.depth_path(image_id, model).is_file()

    def save_depth(self, image_id: str, model: str, disp: torch.Tensor) -> None:
        arr = disp.numpy()
        np.save(self.depth_path(image_id, model), arr.astype(np.float16))
        preview = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
        preview.save(self._path(self.depth_preview_name(image_id, model)), format="PNG")

    def load_depth(self, image_id: str, model: str) -> torch.Tensor:
        p = self.depth_path(image_id, model)
        p.touch()
        return torch.from_numpy(np.load(p).astype(np.float32))

    def save_result(self, name: str, data: bytes) -> None:
        self._path(name).write_bytes(data)

    def sweep(self) -> int:
        cutoff = time.time() - self.ttl_s
        removed = 0
        for p in self.root.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
        return removed
