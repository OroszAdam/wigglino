"""Depth Anything 3 (Mono-Large / Metric-Large) inference, matching the ComfyUI v4 workflow.

Uses the official network code (Apache-2.0) and Hugging Face weights, but not
`depth_anything_3.api`, which imports heavy export dependencies (moviepy, open3d, evo, ...).
"""

import logging
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .core.depth import depth_to_disparity, normalize

log = logging.getLogger(__name__)

# key -> (Hugging Face repo, upstream config name)
MODELS = {
    "mono": ("depth-anything/DA3MONO-LARGE", "da3mono-large"),
    "metric": ("depth-anything/DA3METRIC-LARGE", "da3metric-large"),
}
PATCH = 14
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _round_to_patch(x: int) -> int:
    down = (x // PATCH) * PATCH
    return down + PATCH if x - down >= PATCH / 2 else down


def target_size(h: int, w: int, process_res: int) -> tuple[int, int]:
    """Longest side -> process_res, then each side to the nearest multiple of 14."""
    s = process_res / max(h, w)
    return max(PATCH, _round_to_patch(round(h * s))), max(PATCH, _round_to_patch(round(w * s)))


def _load_network(key: str):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    repo, cfg_name = MODELS[key]
    net = create_object(load_config(MODEL_REGISTRY[cfg_name]))
    state = load_file(hf_hub_download(repo, "model.safetensors"))
    state = {k.removeprefix("model."): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return net.eval()


class DepthEstimator:
    """Lazily loads each model once; thread-safe."""

    def __init__(self, device: torch.device, process_res: int = 1024):
        self.device = device
        self.process_res = process_res
        self._nets: dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> list[str]:
        return sorted(self._nets)

    def load(self, key: str) -> torch.nn.Module:
        if key not in MODELS:
            raise ValueError(f"unknown model {key!r}")
        with self._lock:
            if key not in self._nets:
                log.info("loading depth model %s", key)
                self._nets[key] = _load_network(key).to(self.device)
            return self._nets[key]

    @torch.inference_mode()
    def estimate(self, image: Image.Image, key: str) -> torch.Tensor:
        """RGB image -> disparity [H,W] on CPU, 0..1, 1 = near, same size as the image."""
        net = self.load(key)
        W, H = image.size
        th, tw = target_size(H, W, self.process_res)
        resized = image if (tw, th) == (W, H) else image.resize((tw, th), Image.Resampling.LANCZOS)
        x = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None, None]  # [B=1, N=1, 3, h, w]
        x = ((x - _MEAN) / _STD).to(self.device)
        out = net(x)
        depth = F.interpolate(out["depth"].reshape(1, 1, th, tw).float(), size=(H, W),
                              mode="bilinear", align_corners=False)[0, 0].cpu()
        sky = None
        if "sky" in out:
            sky = F.interpolate(out["sky"].reshape(1, 1, th, tw).float(), size=(H, W),
                                mode="bilinear", align_corners=False)[0, 0].cpu()
        return normalize(depth_to_disparity(depth, sky))
