import math

import torch

from .fill import push_pull_fill, stretch_fill
from .paths import linear_s

AXES = ("vertical", "horizontal", "circular")
HOLE_FILLS = ("stretch", "push_pull", "none")


def _rasterize_rows(col: torch.Tensor, pos: torch.Tensor, depth: torch.Tensor, tear_d: torch.Tensor,
                    tear: float, valid: torch.Tensor | None = None):
    """Draw each row as a connected polyline: vertex j lands at pos[:, j], nearest depth wins.

    col [C,H,W] attributes, pos/depth/tear_d/valid [H,W]. A segment is skipped (torn) when its
    tear_d jump exceeds tear or either vertex is invalid. Returns (out [C,H,W], hole [H,W]).
    """
    C, H, W = col.shape
    dev = col.device
    x0, x1, z0, z1 = pos[:, :-1], pos[:, 1:], depth[:, :-1], depth[:, 1:]
    c0, c1 = col[:, :, :-1], col[:, :, 1:]
    dx = x1 - x0
    flat = dx.abs() < 1e-6
    safe_dx = torch.where(flat, torch.ones_like(dx), dx)
    start = torch.ceil(torch.minimum(x0, x1))
    span = torch.floor(torch.maximum(x0, x1)) - start + 1
    keep = (tear_d[:, 1:] - tear_d[:, :-1]).abs() <= tear
    if valid is not None:
        keep &= valid[:, 1:] & valid[:, :-1]
    span = torch.where(keep, span, torch.zeros_like(span))
    kmax = int(span.max().item()) if span.numel() else 0
    row_base = (torch.arange(H, device=dev) * W)[:, None]

    def candidates(k):
        p = start + k
        ok = (k < span) & (p >= 0) & (p <= W - 1)
        t = torch.where(flat, torch.zeros_like(dx), ((p - x0) / safe_dx).clamp(0, 1))
        idx = row_base + p.clamp(0, W - 1).long()
        return ok, t, idx, z0 + (z1 - z0) * t

    zbuf = torch.full((H * W,), -math.inf, device=dev)
    for k in range(kmax):
        ok, _, idx, zk = candidates(k)
        zbuf.scatter_reduce_(0, idx[ok], zk[ok], reduce="amax")

    out = torch.zeros(C, H * W, device=dev)
    for k in range(kmax):
        ok, t, idx, zk = candidates(k)
        win = ok & (zk >= zbuf[idx])
        tw = t[win]
        out[:, idx[win]] = c0[:, win] + (c1[:, win] - c0[:, win]) * tw
    hole = torch.isinf(zbuf).reshape(H, W)
    return out.reshape(C, H, W), hole


def _views(frames: int, rotation_deg: float, axis: str) -> list[tuple[float, float, float]]:
    """(tilt angle, motion direction ux, uy) per frame."""
    if axis == "circular":
        tilt = math.radians(rotation_deg / 2.0)
        return [(tilt, math.cos(2 * math.pi * i / frames), math.sin(2 * math.pi * i / frames))
                for i in range(frames)]
    ux, uy = (1.0, 0.0) if axis == "vertical" else (0.0, 1.0)
    return [(math.radians(s * rotation_deg / 2.0), ux, uy) for s in linear_s(frames)]


def _render_view(rgb, d, z, xs, ys, theta, ux, uy, tear):
    """Orthographic rotation by theta about the in-plane axis perpendicular to (ux, uy).

    Done as a row pass (x motion) then a column pass (y motion); exact when motion is axis-aligned.
    """
    H, W = d.shape
    a = (xs - (W - 1) / 2) * ux + (ys - (H - 1) / 2) * uy
    shift = a * (math.cos(theta) - 1) + z * math.sin(theta)
    zview = -a * math.sin(theta) + z * math.cos(theta)
    # Channels: rgb, disparity, pending y shift, view depth.
    col = torch.cat([rgb, d[None], (uy * shift)[None], zview[None]], dim=0)
    hole = torch.zeros(H, W, dtype=torch.bool, device=d.device)
    if abs(ux) > 1e-9:
        col, hole = _rasterize_rows(col, xs + ux * shift, zview, d, tear)
    if abs(uy) > 1e-9:
        out_t, hole_t = _rasterize_rows(col.transpose(1, 2), (ys + col[4]).T, col[5].T, col[3].T, tear, ~hole.T)
        col, hole = out_t.transpose(1, 2), hole_t.T
    return col[:3], col[3], hole


def render_relief(
    image: torch.Tensor,
    disparity: torch.Tensor,
    pivot: float,
    frames: int,
    rotation_deg: float,
    depth_pct: float,
    axis: str = "vertical",
    tear: float = 1.0,
    hole_fill: str = "stretch",
):
    """Turn the disparity map into a relief surface and rotate it slightly per frame.

    image [H,W,C] in 0..1, disparity [H,W] in 0..1 (1 = near). vertical/horizontal: frames swing
    from -rotation/2 to +rotation/2; circular: the relief stays tilted by rotation/2 while the tilt
    direction goes once around. depth_pct is the relief height for disparity 0..1 in % of image
    width. The surface is continuous, so revealed areas stretch instead of opening holes unless the
    depth jump exceeds tear. Returns (frames [N,H,W,3], holes [N,H,W]).
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    if hole_fill not in HOLE_FILLS:
        raise ValueError(f"hole_fill must be one of {HOLE_FILLS}, got {hole_fill!r}")
    rgb = image[..., :3].permute(2, 0, 1).float()
    d = disparity.float()
    H, W = d.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=d.device, dtype=torch.float32),
                            torch.arange(W, device=d.device, dtype=torch.float32), indexing="ij")
    z = depth_pct / 100.0 * W * (d - pivot)

    out_frames, out_holes = [], []
    for theta, ux, uy in _views(frames, rotation_deg, axis):
        rgb_out, d_out, hole = _render_view(rgb, d, z, xs, ys, theta, ux, uy, tear)
        valid = ~hole
        if hole_fill == "stretch":
            rgb_out = stretch_fill(rgb_out, d_out, valid, horizontal=abs(ux) >= abs(uy))
        elif hole_fill == "push_pull":
            rgb_out = push_pull_fill(rgb_out, valid)
        out_frames.append(rgb_out.clamp(0.0, 1.0).permute(1, 2, 0))
        out_holes.append(hole.float())
    return torch.stack(out_frames), torch.stack(out_holes)
