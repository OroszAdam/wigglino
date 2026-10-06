import torch


def pivot_disparity(d: torch.Tensor, x: float, y: float, radius_px: int = 4) -> float:
    """Median disparity in a (2r+1)^2 window at (x, y); x, y are fractions of width/height."""
    H, W = d.shape
    cx = round(min(max(x, 0.0), 1.0) * (W - 1))
    cy = round(min(max(y, 0.0), 1.0) * (H - 1))
    r = max(0, int(radius_px))
    window = d[max(0, cy - r): cy + r + 1, max(0, cx - r): cx + r + 1]
    return window.median().item()
