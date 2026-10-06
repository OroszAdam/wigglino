import torch
import torch.nn.functional as F


def push_pull_fill(img: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fill invalid pixels by coarse-to-fine pyramid interpolation. img [C,H,W], valid [H,W]."""
    if valid.all() or not valid.any():
        return img
    w = valid.to(img.dtype)[None]
    levels = [(img * w, w)]
    while max(levels[-1][1].shape[-2:]) > 1:
        c, w = levels[-1]
        k = tuple(2 if s > 1 else 1 for s in w.shape[-2:])
        levels.append((
            F.avg_pool2d(c[None], k, ceil_mode=True)[0],
            F.avg_pool2d(w[None], k, ceil_mode=True)[0],
        ))
    c, w = levels[-1]
    filled = c / w.clamp_min(1e-8)
    for c, w in reversed(levels[:-1]):
        up = F.interpolate(filled[None], size=c.shape[-2:], mode="bilinear", align_corners=False)[0]
        # c is premultiplied by coverage, so known pixels survive exactly at the finest level.
        filled = c + (1 - w) * up
    return filled


def _nearest_valid(valid: torch.Tensor):
    """Per row: column index of the nearest valid pixel to the left / right (inclusive), -1 if none."""
    H, W = valid.shape
    idx = torch.arange(W, device=valid.device).expand(H, W)
    neg = torch.full_like(idx, -1)
    left = torch.where(valid, idx, neg).cummax(dim=1).values
    right_f = torch.where(valid.flip(1), idx, neg).cummax(dim=1).values
    right = torch.where(right_f >= 0, W - 1 - right_f, right_f).flip(1)
    return left, right


def stretch_fill(img: torch.Tensor, disp: torch.Tensor, valid: torch.Tensor, horizontal: bool = True) -> torch.Tensor:
    """Fill holes along rows (or columns) from whichever side is farther (lower disparity).

    Disocclusions reveal background, so stretching the far side avoids smearing the foreground.
    img [C,H,W], disp [H,W], valid [H,W]. Rows/columns with no valid pixel fall back to push-pull.
    """
    if not horizontal:
        return stretch_fill(img.transpose(1, 2), disp.T, valid.T, True).transpose(1, 2)
    C, H, W = img.shape
    left, right = _nearest_valid(valid)
    li, ri = left.clamp_min(0), right.clamp_min(0)
    use_left = (right < 0) | ((left >= 0) & (disp.gather(1, li) <= disp.gather(1, ri)))
    src = torch.where(use_left, li, ri)
    fillable = ~valid & ((left >= 0) | (right >= 0))
    gathered = img.gather(2, src.expand(C, H, W))
    out = torch.where(fillable, gathered, img)
    remaining = ~(valid | fillable)
    if remaining.any():
        out = push_pull_fill(out, ~remaining)
    return out
