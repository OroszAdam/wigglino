import torch
import torch.nn.functional as F

# Sky probability below this counts as "not sky" (same threshold DA3 uses).
SKY_THRESHOLD = 0.3
# torch.quantile refuses huge inputs; a strided subsample is plenty and deterministic.
_QUANTILE_SAMPLES = 100_000


def _quantiles(values: torch.Tensor, lo: float, hi: float) -> tuple[float, float]:
    step = max(1, values.numel() // _QUANTILE_SAMPLES)
    q = torch.tensor([lo, hi], device=values.device, dtype=values.dtype)
    a, b = torch.quantile(values[::step], q).tolist()
    return a, b


def depth_to_disparity(depth: torch.Tensor, sky: torch.Tensor | None = None,
                       low_q: float = 0.01, high_q: float = 0.99) -> torch.Tensor:
    """DA3 depth [H,W] -> disparity-like map in 0..1 with 1 = near ("v2 style").

    The 1st/99th percentile of non-sky depth map to near/far, so outliers don't flatten the
    rest. Sky pixels (probability >= SKY_THRESHOLD) become 0 (farthest).
    """
    depth = depth.float()
    flat = depth.flatten()
    is_sky = None
    if sky is not None:
        is_sky = sky >= SKY_THRESHOLD
        non_sky = flat[~is_sky.flatten()]
        if non_sky.numel():
            flat = non_sky
    lo, hi = _quantiles(flat, low_q, high_q)
    norm = 1.0 - ((depth - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    if is_sky is not None:
        norm = torch.where(is_sky, torch.zeros_like(norm), norm)
    return norm


def normalize(d: torch.Tensor) -> torch.Tensor:
    """Min/max to 0..1 (all zeros for a constant map)."""
    d = d.float()
    lo, hi = d.min().item(), d.max().item()
    if hi - lo < 1e-8:
        return torch.zeros_like(d)
    return (d - lo) / (hi - lo)


def resize(d: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[H,W] -> [height,width]; bilinear."""
    if d.shape == (height, width):
        return d
    return F.interpolate(d[None, None].float(), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
