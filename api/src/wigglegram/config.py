import os
from dataclasses import dataclass, field
from pathlib import Path


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_rate(spec: str) -> list[tuple[int, float]]:
    """'5/600,30/86400' -> [(5, 600.0), (30, 86400.0)]: at most N requests per S seconds."""
    rules = []
    for part in _list(spec):
        n, s = part.split("/")
        rules.append((int(n), float(s)))
    return rules


@dataclass
class Settings:
    data_dir: Path = Path("/tmp/wigglegram")
    file_ttl_s: float = 3600.0
    max_upload_bytes: int = 10 * 1024 * 1024
    max_input_pixels: int = 40_000_000
    max_side: int = 1280
    max_frames: int = 16
    depth_res: int = 1024
    device: str = "auto"
    workers: int = 1
    max_queue: int = 8
    preload_models: list[str] = field(default_factory=list)
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    allowed_origin_regex: str | None = None
    turnstile_secret: str | None = None
    client_ip_header: str = "x-forwarded-for"
    trusted_proxy_hops: int = 0
    rate_upload: list[tuple[int, float]] = field(default_factory=lambda: parse_rate("10/600,60/86400"))
    rate_depth: list[tuple[int, float]] = field(default_factory=lambda: parse_rate("5/600,30/86400"))
    rate_render: list[tuple[int, float]] = field(default_factory=lambda: parse_rate("30/600,300/86400"))

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        d = cls()
        return cls(
            data_dir=Path(e("DATA_DIR", str(d.data_dir))),
            file_ttl_s=float(e("FILE_TTL_SECONDS", d.file_ttl_s)),
            max_upload_bytes=int(float(e("MAX_UPLOAD_MB", 10)) * 1024 * 1024),
            max_input_pixels=int(e("MAX_INPUT_PIXELS", d.max_input_pixels)),
            max_side=int(e("MAX_SIDE", d.max_side)),
            max_frames=int(e("MAX_FRAMES", d.max_frames)),
            depth_res=int(e("DEPTH_RES", d.depth_res)),
            device=e("DEVICE", d.device),
            workers=int(e("WORKERS", d.workers)),
            max_queue=int(e("MAX_QUEUE", d.max_queue)),
            preload_models=_list(e("PRELOAD_MODELS", "")),
            allowed_origins=_list(e("ALLOWED_ORIGINS", ",".join(d.allowed_origins))),
            allowed_origin_regex=e("ALLOWED_ORIGIN_REGEX") or None,
            turnstile_secret=e("TURNSTILE_SECRET") or None,
            client_ip_header=e("CLIENT_IP_HEADER", d.client_ip_header).lower(),
            trusted_proxy_hops=int(e("TRUSTED_PROXY_HOPS", d.trusted_proxy_hops)),
            rate_upload=parse_rate(e("RATE_UPLOAD", "10/600,60/86400")),
            rate_depth=parse_rate(e("RATE_DEPTH", "5/600,30/86400")),
            rate_render=parse_rate(e("RATE_RENDER", "30/600,300/86400")),
        )
