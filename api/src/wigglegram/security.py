import logging

import httpx
from starlette.requests import Request

log = logging.getLogger(__name__)

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def client_ip(request: Request, header: str, trusted_hops: int) -> str:
    """Caller IP, trusting only the last `trusted_hops` entries of the proxy header.

    Proxies append to the right, so anything further left could be spoofed by the client.
    With trusted_hops=0 the socket peer address is used.
    """
    if trusted_hops > 0:
        raw = request.headers.get(header)
        if raw:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) >= trusted_hops:
                return parts[-trusted_hops]
    return request.client.host if request.client else "unknown"


async def verify_turnstile(http: httpx.AsyncClient, secret: str, token: str, ip: str) -> bool:
    if not token or len(token) > 2048:
        return False
    try:
        r = await http.post(TURNSTILE_VERIFY_URL, data={"secret": secret, "response": token, "remoteip": ip},
                            timeout=10)
        return bool(r.json().get("success"))
    except (httpx.HTTPError, ValueError):
        log.warning("turnstile verification failed", exc_info=True)
        return False
