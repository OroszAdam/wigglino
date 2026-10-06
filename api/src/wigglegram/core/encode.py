import io

import numpy as np
import torch
from PIL import Image

FORMATS = ("gif", "webp")


def encode_animation(frames: torch.Tensor, fps: float, fmt: str) -> bytes:
    """frames [N,H,W,3] in 0..1 -> looping animated GIF or WebP bytes."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
    arr = np.clip(frames.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil = [Image.fromarray(a) for a in arr]
    duration = int(round(1000.0 / fps))
    buf = io.BytesIO()
    if fmt == "gif":
        # One shared palette avoids per-frame colour flicker.
        palette = pil[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        pal_frames = [f.quantize(palette=palette) for f in pil]
        pal_frames[0].save(buf, format="GIF", save_all=True, append_images=pal_frames[1:],
                           duration=duration, loop=0)
    else:
        pil[0].save(buf, format="WEBP", save_all=True, append_images=pil[1:], duration=duration, loop=0,
                    lossless=False, quality=90, method=4)
    return buf.getvalue()
