def linear_s(n: int) -> list[float]:
    """n evenly spaced offsets from -1 to 1."""
    if n < 1:
        raise ValueError("frames must be >= 1")
    if n == 1:
        return [0.0]
    return [-1.0 + 2.0 * i / (n - 1) for i in range(n)]


def ping_pong(n: int) -> list[int]:
    """0..n-1 then back down, without repeating the endpoints: 0,1,2,3,2,1."""
    return list(range(n)) + list(range(n - 2, 0, -1))
