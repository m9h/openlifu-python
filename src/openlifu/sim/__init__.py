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
        backend: "kwave" or "jwave".
        **kwargs: Forwarded to the backend's run_simulation.

    Returns:
        (xa.Dataset, dict): Simulation results and raw output.
    """
    if backend == "jwave":
        if not _HAS_JWAVE:
            raise ImportError("jwave is not installed. Install with: uv pip install jwave")
        return jwave_if.run_simulation(**kwargs)
    elif backend == "kwave":
        return kwave_if.run_simulation(**kwargs)
    else:
        raise ValueError(f"Unknown simulation backend: {backend!r}. Use 'kwave' or 'jwave'.")
