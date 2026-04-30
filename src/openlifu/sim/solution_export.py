"""Export jwave simulation outputs to openlifu Solution-compatible structures.

Bridges the JAX simulation layer (numpy/xarray pressure fields,
per-element delays + apodizations) to the dict shape that openlifu's
``Solution.from_dict`` consumes for safety analysis, Slicer
visualisation, and hardware programming.

Adapted from ``sbi4dwi/dmipy_jax/biophysics/tus_solution_export.py``.
The xarray Dataset path is largely redundant with what
``openlifu.sim.jwave_if.run_simulation_peak`` already returns directly,
but is preserved as a thin helper for callers that have a raw numpy
peak-pressure array (e.g. an external solver, a custom subsetting
of a larger simulation, or a downstream FWI residual).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xa


def export_simulation_result(
    p_max: np.ndarray,
    coords: dict | xa.Coordinates,
    *,
    p_min: np.ndarray | None = None,
    density: Union[float, np.ndarray] = 1000.0,
    sound_speed: Union[float, np.ndarray] = 1500.0,
) -> xa.Dataset:
    """Wrap a peak-pressure numpy array in an openlifu-format ``xa.Dataset``.

    Returns a dataset with the same variables as
    ``openlifu.sim.jwave_if.run_simulation_peak`` (``p_max`` PPP,
    ``p_min`` |PNP|, ``intensity`` W/cm^2). When ``p_min`` is omitted
    we fall back to the symmetric approximation (|PNP| = |PPP|), which
    is exact for a pure-CW source and a reasonable upper bound for
    short tonebursts.

    Args:
        p_max: Peak positive pressure field (Pa, any spatial shape).
        coords: xarray-compatible coordinates dict or ``xa.Coordinates``.
        p_min: Optional |peak negative pressure| field; if None,
            assumed equal to ``p_max``.
        density, sound_speed: Scalars or arrays used to compute the
            acoustic intensity ``I = p^2 / (2*rho*c)`` (W/cm^2 with the
            1e-4 factor for cm^2 reporting).

    Returns:
        ``xa.Dataset`` with ``p_max``, ``p_min``, ``intensity`` variables.
    """
    p_max_np = np.asarray(p_max, dtype=np.float32)
    p_min_np = (np.asarray(p_min, dtype=np.float32)
                if p_min is not None else p_max_np.copy())

    Z = (np.asarray(density, dtype=np.float32)
         * np.asarray(sound_speed, dtype=np.float32))
    intensity_np = 1e-4 * p_min_np ** 2 / (2.0 * Z)

    p_max_da = xa.DataArray(
        p_max_np, coords=coords, name="p_max",
        attrs={"units": "Pa", "long_name": "PPP"},
    )
    p_min_da = xa.DataArray(
        p_min_np, coords=coords, name="p_min",
        attrs={"units": "Pa", "long_name": "PNP"},
    )
    intensity_da = xa.DataArray(
        intensity_np, coords=coords, name="I",
        attrs={"units": "W/cm^2", "long_name": "Intensity"},
    )
    return xa.Dataset({"p_max": p_max_da, "p_min": p_min_da,
                       "intensity": intensity_da})


def build_solution_dict(
    delays: np.ndarray,
    apodizations: np.ndarray,
    simulation_result: xa.Dataset,
    freq: float,
    *,
    voltage: float = 1.0,
    duration_cycles: float = 20.0,
    target: Optional[Tuple[float, float, float]] = None,
    approved: bool = False,
) -> Dict:
    """Build a dict shaped for ``openlifu.plan.Solution.from_dict``.

    The result is loadable by openlifu's clinical layer for safety
    analysis (MI/TI/intensity), Slicer visualisation, and hardware
    programming (per-element delays + apodisation patterns).

    Args:
        delays: ``(n_foci, n_elements)`` per-element delays in seconds.
            A 1-D ``(n_elements,)`` array is treated as a single focus.
        apodizations: ``(n_foci, n_elements)`` amplitude weights, same
            shape conventions as ``delays``.
        simulation_result: Dataset from ``export_simulation_result`` or
            ``run_simulation_peak``.
        freq: Drive frequency (Hz).
        voltage: Applied voltage (V), forwarded to Solution metadata.
        duration_cycles: Pulse duration expressed in cycles. Default 20
            cycles matches the openlifu reference burst.
        target: Optional ``(x, y, z)`` focal target in metres.
        approved: Clinical-approval flag; defaults to False so the
            downstream Solution loader treats the export as draft.

    Returns:
        A plain dict with keys ``delays``, ``apodizations``, ``voltage``,
        ``pulse``, ``simulation_result``, ``target``, ``approved`` --
        the inputs ``Solution.from_dict`` expects.
    """
    delays_arr = np.asarray(delays)
    apod_arr = np.asarray(apodizations)
    if delays_arr.ndim == 1:
        delays_arr = delays_arr[np.newaxis, :]
    if apod_arr.ndim == 1:
        apod_arr = apod_arr[np.newaxis, :]

    return {
        "delays": delays_arr,
        "apodizations": apod_arr,
        "voltage": voltage,
        "pulse": {
            "frequency": freq,
            "amplitude": 1.0,
            "duration": duration_cycles / freq,
        },
        "simulation_result": simulation_result,
        "target": target,
        "approved": approved,
    }
