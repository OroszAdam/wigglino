"""Per-client rate limiting.

`RateLimiter` is the interface the app depends on. `MemoryRateLimiter` is enough for a single
instance; for several instances, implement the same two methods on a shared store (e.g. Redis).
"""

import threading
import time
from typing import Protocol


class RateLimiter(Protocol):
    def hit(self, key: str) -> float | None:
        """Consume one request for key. Returns None if allowed, else seconds until retry."""

    def prune(self) -> None:
        """Drop state for idle keys."""


class MemoryRateLimiter:
    """Token buckets per key; every rule (capacity per period) must have a token left."""

    def __init__(self, rules: list[tuple[int, float]], clock=time.monotonic):
        self.rules = rules
        self.clock = clock
        self._state: dict[str, tuple[list[float], float]] = {}
        self._lock = threading.Lock()

    def _refill(self, key: str, now: float) -> list[float]:
        tokens, last = self._state.get(key, ([float(c) for c, _ in self.rules], now))
        elapsed = now - last
        return [min(float(c), t + elapsed * c / p) for t, (c, p) in zip(tokens, self.rules)]

    def hit(self, key: str) -> float | None:
        with self._lock:
            now = self.clock()
            tokens = self._refill(key, now)
            waits = [(1.0 - t) * p / c for t, (c, p) in zip(tokens, self.rules) if t < 1.0]
            if waits:
                self._state[key] = (tokens, now)
                return max(waits)
            self._state[key] = ([t - 1.0 for t in tokens], now)
            return None

    def prune(self) -> None:
        with self._lock:
            now = self.clock()
            full = [k for k in self._state
                    if all(t >= c for t, (c, _) in zip(self._refill(k, now), self.rules))]
            for k in full:
                del self._state[k]
