from __future__ import annotations

from . import kwave_if
from .sim_setup import SimSetup

__all__ = [
    "SimSetup",
    "run_simulation",
    "kwave_if",
]

try:
    from . import jwave_if
    __all__.append("jwave_if")
    _HAS_JWAVE = True
except ImportError:
    _HAS_JWAVE = False


def run_simulation(*, backend: str = "kwave", **kwargs):
    """Dispatch acoustic simulation to the selected backend.

    Args:
        backend: "kwave", "jwave", or "jwave_cw".
        **kwargs: Forwarded to the backend's run_simulation or run_cw_simulation.

    Returns:
        (xa.Dataset, dict): Simulation results and raw output.
    """
    if backend in ("jwave", "jwave_cw"):
        if not _HAS_JWAVE:
            raise ImportError("jwave is not installed. Install with: uv pip install jwave")
        if backend == "jwave_cw":
            return jwave_if.run_cw_simulation(**kwargs)
        return jwave_if.run_simulation(**kwargs)
    elif backend == "kwave":
        return kwave_if.run_simulation(**kwargs)
    else:
        raise ValueError(f"Unknown simulation backend: {backend!r}. Use 'kwave', 'jwave', or 'jwave_cw'.")
