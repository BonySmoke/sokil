"""Small builders shared by several test modules."""


def hesse_line(rho: float, theta: float) -> dict:
    """A line dict in the shape sokil.util's line helpers pass around."""
    return {
        "rho": float(rho),
        "theta": float(theta),
        "length": 100.0,
        "ref": float(rho),
    }
