import io

import pytest
import torch
from PIL import Image

from wigglegram.core.depth import depth_to_disparity, normalize, resize
from wigglegram.core.encode import encode_animation
from wigglegram.core.fill import push_pull_fill, stretch_fill
from wigglegram.core.paths import linear_s, ping_pong
from wigglegram.core.pivot import pivot_disparity
from wigglegram.core.relief import render_relief
from wigglegram.depth_model import target_size


def _scene(H=32, W=64, d_b=0.25, d_f=0.75, x0=20, x1=30, y0=8, y1=24):
    """Gray background plane with a red square in front."""
    img = torch.full((H, W, 3), 0.5)
    img[y0:y1, x0:x1] = torch.tensor([1.0, 0.0, 0.0])
    d = torch.full((H, W), d_b)
    d[y0:y1, x0:x1] = d_f
    return img, d


def _red_centre(frame):
    yx = ((frame[..., 0] > 0.9) & (frame[..., 1] < 0.1)).nonzero().float().mean(0)
    return yx[1].item(), yx[0].item()


def test_paths():
    assert linear_s(1) == [0.0]
    assert linear_s(4) == pytest.approx([-1, -1 / 3, 1 / 3, 1])
    assert ping_pong(4) == [0, 1, 2, 3, 2, 1]
    assert ping_pong(2) == [0, 1]


def test_stretch_fill_vertical_and_empty_rows():
    img = torch.zeros(3, 6, 5)
    img[:, 0] = 1.0
    valid = torch.zeros(6, 5, dtype=torch.bool)
    valid[0] = True
    assert torch.allclose(stretch_fill(img, torch.zeros(6, 5), valid, horizontal=False), torch.ones(3, 6, 5))
    # Rows are fully invalid horizontally, so push-pull fallback must still fill them.
    assert torch.allclose(stretch_fill(img, torch.zeros(6, 5), valid, horizontal=True), torch.ones(3, 6, 5),
                          atol=1e-5)


def test_push_pull_keeps_known_and_fills_all():
    torch.manual_seed(0)
    img = torch.rand(3, 17, 23)
    valid = torch.rand(17, 23) > 0.7
    out = push_pull_fill(img * valid, valid)
    assert torch.allclose(out[:, valid], img[:, valid], atol=1e-6)
    assert out.min() >= 0 and out.max() <= 1


def test_relief_flat_plane_barely_changes():
    img = torch.linspace(0, 1, 40).expand(24, 40)[..., None].expand(24, 40, 3).contiguous()
    d = torch.full((24, 40), 0.6)
    frames, holes = render_relief(img, d, pivot=0.6, frames=3, rotation_deg=4, depth_pct=30, hole_fill="none")
    assert frames.shape == (3, 24, 40, 3)
    assert holes[:, :, 1:-1].sum() == 0
    assert torch.allclose(frames[:, :, 1:-1], img[:, 1:-1].expand(3, -1, -1, -1), atol=1e-3)


def test_relief_stretches_instead_of_holes_and_tears_on_request():
    img, d = _scene()
    kw = dict(pivot=0.25, frames=3, rotation_deg=10, depth_pct=200, hole_fill="none")
    frames, holes = render_relief(img, d, **kw)
    cx = _red_centre(img)[0]
    assert _red_centre(frames[0])[0] < cx - 3
    assert _red_centre(frames[2])[0] > cx + 3
    assert abs(_red_centre(frames[1])[0] - cx) < 0.5
    assert holes[:, :, 1:-1].sum() == 0
    _, torn = render_relief(img, d, tear=0.1, **kw)
    assert torn[2, 16, 20:24].all() and torn[0, 16, 26:30].all()


def test_relief_horizontal_axis_moves_vertically():
    img, d = _scene()
    frames, _ = render_relief(img, d, pivot=0.25, frames=2, rotation_deg=10, depth_pct=200, axis="horizontal")
    assert _red_centre(frames[1])[1] > _red_centre(img)[1] + 3
    assert abs(_red_centre(frames[1])[0] - _red_centre(img)[0]) < 0.5


def test_relief_circular_goes_around():
    img, d = _scene()
    frames, holes = render_relief(img, d, pivot=0.25, frames=8, rotation_deg=10, depth_pct=200, axis="circular",
                                  hole_fill="none")
    cx, cy = _red_centre(img)
    pos = [(x - cx, y - cy) for x, y in map(_red_centre, frames)]
    assert pos[0][0] > 3 and abs(pos[0][1]) < 0.5
    assert pos[2][1] > 3 and abs(pos[2][0]) < 0.5
    assert pos[4][0] < -3 and pos[6][1] < -3
    assert holes[:, 1:-1, 1:-1].sum() == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_relief_matches_on_mps():
    img, d = _scene()
    kw = dict(pivot=0.4, frames=3, rotation_deg=8, depth_pct=120, tear=0.3)
    cpu_f, cpu_h = render_relief(img, d, **kw)
    mps_f, mps_h = render_relief(img.to("mps"), d.to("mps"), **kw)
    assert torch.allclose(cpu_f, mps_f.cpu(), atol=1e-4)
    assert torch.equal(cpu_h, mps_h.cpu())


def test_depth_to_disparity_near_is_one_and_sky_is_zero():
    depth = torch.linspace(1, 10, 200).view(10, 20)
    disp = depth_to_disparity(depth)
    assert disp[0, 0] == 1 and disp[-1, -1] == 0
    assert disp.min() >= 0 and disp.max() <= 1
    # Sky pixels are forced far and excluded from the percentile range.
    sky = torch.zeros(10, 20)
    sky[-1] = 1.0
    depth_sky = depth.clone()
    depth_sky[-1] = 1e6
    disp_sky = depth_to_disparity(depth_sky, sky)
    assert (disp_sky[-1] == 0).all()
    assert disp_sky[-2, -1] == 0 and disp_sky[0, 0] == 1


def test_normalize_and_resize():
    assert torch.equal(normalize(torch.full((3, 3), 2.0)), torch.zeros(3, 3))
    n = normalize(torch.tensor([[2.0, 4.0]]))
    assert n.tolist() == [[0.0, 1.0]]
    assert resize(torch.rand(10, 20), 40, 80).shape == (40, 80)


def test_pivot_is_local_median():
    _, d = _scene()
    assert pivot_disparity(d, 25 / 63, 16 / 31, radius_px=2) == pytest.approx(0.75)
    assert pivot_disparity(d, 0.0, 0.0, radius_px=4) == pytest.approx(0.25)
    assert pivot_disparity(d, 1.0, 1.0, radius_px=4) == pytest.approx(0.25)


def test_target_size_matches_da3_rounding():
    # Longest side -> 1024 (rounded to a multiple of 14), aspect kept.
    assert target_size(1140, 1170, 1024) == (994, 1022)
    h, w = target_size(600, 800, 504)
    assert (h, w) == (378, 504)
    assert h % 14 == 0 and w % 14 == 0


@pytest.mark.parametrize("fmt", ["gif", "webp"])
def test_encode_animation_loops_all_frames(fmt):
    frames = torch.rand(5, 12, 16, 3)
    data = encode_animation(frames, fps=24, fmt=fmt)
    with Image.open(io.BytesIO(data)) as img:
        assert img.format == fmt.upper()
        assert img.n_frames == 5
        assert img.size == (16, 12)
    with pytest.raises(ValueError):
        encode_animation(frames, 24, "png")
