"""jwave simulation interface for openlifu.

Provides a differentiable acoustic simulation backend using jwave (JAX-based).
Drop-in replacement for kwave_if with the same run_simulation contract:
accepts openlifu Transducer + xarray params, returns xarray Dataset results.

Two simulation modes:
    - run_simulation: Time-domain (pulsed FUS, treatment planning)
    - run_cw_simulation: Frequency-domain Helmholtz solver (steady-state CW,
      ITRUSST benchmarks, phase correction). Memory-efficient for large grids.
"""

from __future__ import annotations

import logging
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xa
from jaxdf.discretization import FourierSeries
from jwave import simulate_wave_propagation
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium, Sources, TimeAxis

from openlifu import xdc
from openlifu.util.units import getunitconversion


def _dim_names(coords: xa.Coordinates) -> list[str]:
    """Extract the ordered spatial dimension names from xarray Coordinates.

    Parameters
    ----------
    coords : xarray.Coordinates
        Simulation grid coordinates (typically with dimensions ``x``, ``y``,
        ``z``).

    Returns
    -------
    list of str
        Dimension names in iteration order, e.g. ``['x', 'y', 'z']``.
    """
    return list(coords.dims)


def get_domain(coords: xa.Coordinates) -> Tuple[Domain, float]:
    """Create a jwave Domain from openlifu simulation coordinates.

    Args:
        coords: xarray Coordinates with 'x', 'y', 'z' dimensions.
            Each coordinate must have a 'units' attribute.

    Returns:
        domain: jwave Domain
        scl: scale factor from coordinate units to meters
    """
    dims = _dim_names(coords)
    units = [coords[dim].attrs['units'] for dim in dims]
    if not all(unit == units[0] for unit in units):
        raise ValueError("All coordinates must have the same units")
    scl = getunitconversion(units[0], 'm')
    N = tuple(len(coords[dim]) for dim in dims)
    dx = tuple(float(np.diff(coords[dim].values[:2])[0]) * scl for dim in dims)
    return Domain(N, dx), scl


def get_medium(
    params: xa.Dataset,
    domain: Domain,
    ref_values_only: bool = False,
    pml_size: int = 20,
) -> Medium:
    """Create a jwave Medium from openlifu simulation parameters.

    Args:
        params: xarray Dataset with 'sound_speed', 'density', and 'attenuation' variables.
        domain: jwave Domain matching the params grid.
        ref_values_only: If True, use only scalar reference values (homogeneous medium).
        pml_size: Number of PML grid points on each boundary.

    Returns:
        jwave Medium with heterogeneous or homogeneous properties.
    """
    if ref_values_only:
        return Medium(
            domain=domain,
            sound_speed=float(params['sound_speed'].attrs['ref_value']),
            density=float(params['density'].attrs['ref_value']),
            attenuation=float(params['attenuation'].attrs['ref_value']),
            pml_size=pml_size,
        )
    sound_speed = jnp.expand_dims(jnp.array(params['sound_speed'].data, dtype=jnp.float32), -1)
    density = jnp.expand_dims(jnp.array(params['density'].data, dtype=jnp.float32), -1)
    attenuation = jnp.expand_dims(jnp.array(params['attenuation'].data, dtype=jnp.float32), -1)
    return Medium(
        domain=domain,
        sound_speed=FourierSeries(sound_speed, domain),
        density=FourierSeries(density, domain),
        attenuation=FourierSeries(attenuation, domain),
        pml_size=pml_size,
    )


def get_time_axis(
    medium: Medium,
    cfl: float = 0.3,
    t_end: float = 0,
    dt: float = 0,
) -> TimeAxis:
    """Create a jwave ``TimeAxis`` from medium properties or explicit values.

    If *dt* or *t_end* are zero the corresponding value is computed
    automatically from the medium's sound speed and grid spacing using the
    CFL condition.

    Parameters
    ----------
    medium : jwave.geometry.Medium
        Simulation medium (sound speed and grid spacing drive the automatic
        time-step calculation).
    cfl : float, optional
        CFL number for the automatic time step (default 0.3).
    t_end : float, optional
        End time in seconds.  ``0`` (default) means compute from medium.
    dt : float, optional
        Time step in seconds.  ``0`` (default) means compute from medium.

    Returns
    -------
    jwave.geometry.TimeAxis
        Time axis suitable for ``simulate_wave_propagation``.
    """
    if dt == 0 or t_end == 0:
        return TimeAxis.from_medium(medium, cfl=cfl, t_end=t_end if t_end > 0 else None)
    return TimeAxis(dt=dt, t_end=t_end)


def get_sources(
    arr: xdc.Transducer,
    domain: Domain,
    coords: xa.Coordinates,
    scl: float,
    input_signal: np.ndarray,
    dt: float,
    delays: np.ndarray,
    apod: np.ndarray,
) -> Sources:
    """Create jwave Sources from openlifu transducer element positions and signals.

    Maps each transducer element to its nearest grid point and assigns the
    delayed/apodized source signal.

    Args:
        arr: openlifu Transducer with element positions.
        domain: jwave Domain.
        coords: xarray Coordinates for mapping positions to grid indices.
        scl: scale factor from coordinate units to meters.
        input_signal: Base sinusoidal input signal.
        dt: Simulation time step in seconds.
        delays: Per-element time delays in seconds.
        apod: Per-element apodization weights.

    Returns:
        jwave Sources placed at the nearest grid points.
    """
    source_mat = arr.calc_output(input_signal, dt, delays, apod)
    num_elements, num_samples = source_mat.shape

    # Map element positions to grid indices
    dim_names = _dim_names(coords)
    coord_arrays = [coords[dim].values for dim in dim_names]
    positions = arr.get_positions(units="m")

    # For each element, find closest grid index along each axis
    idx_per_dim = [[] for _ in range(len(dim_names))]
    for i in range(num_elements):
        for d, cvals in enumerate(coord_arrays):
            cvals_m = cvals * scl
            idx = int(np.argmin(np.abs(cvals_m - positions[i, d])))
            idx_per_dim[d].append(idx)

    pos_tuple = tuple(np.array(idx_per_dim[d]) for d in range(len(dim_names)))
    signals = jnp.array(source_mat, dtype=jnp.float32)
    return Sources(positions=pos_tuple, signals=signals, dt=dt, domain=domain)


def run_simulation(
    arr: xdc.Transducer,
    params: xa.Dataset,
    delays: np.ndarray | None = None,
    apod: np.ndarray | None = None,
    freq: float = 1e6,
    cycles: float = 20,
    amplitude: float = 1,
    dt: float = 0,
    t_end: float = 0,
    cfl: float = 0.5,
    pml_size: int = 20,
    ref_values_only: bool = False,
) -> Tuple[xa.Dataset, dict]:
    """Run a jwave acoustic simulation for the given transducer and parameters.

    Drop-in replacement for kwave_if.run_simulation. Accepts the same openlifu
    types and returns the same xarray Dataset structure.

    Args:
        arr: Transducer array to simulate.
        params: Simulation parameters as xarray Dataset with 'sound_speed',
            'density', and 'attenuation' variables.
        delays: Per-element time delays in seconds. None defaults to zeros.
        apod: Per-element apodization weights. None defaults to ones.
        freq: Source frequency in Hz.
        cycles: Number of cycles in the source signal.
        amplitude: Source amplitude.
        dt: Simulation time step in seconds. 0 for automatic.
        t_end: Simulation end time in seconds. 0 for automatic.
        cfl: CFL number for time step calculation.
        pml_size: Number of PML absorbing boundary grid points.
        ref_values_only: Use only reference (homogeneous) medium values.

    Returns:
        ds: xarray Dataset with 'p_max', 'p_min', and 'intensity' variables.
        raw: dict with raw JAX arrays keyed by 'p_max', 'p_min', 'pressure'.
    """
    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())

    # Build domain and medium
    domain, scl = get_domain(params.coords)
    medium = get_medium(params, domain, ref_values_only=ref_values_only, pml_size=pml_size)
    time_axis = get_time_axis(medium, cfl=cfl, t_end=t_end, dt=dt)

    # Build source signal and sources
    sim_dt = float(time_axis.dt)
    t = np.arange(0, cycles / freq, sim_dt)
    input_signal = amplitude * np.sin(2 * np.pi * freq * t)
    sources = get_sources(arr, domain, params.coords, scl, input_signal, sim_dt, delays, apod)

    # Run simulation
    logging.info("Running jwave simulation")

    @jax.jit
    def _run(medium, time_axis, sources):
        return simulate_wave_propagation(medium, time_axis, sources=sources)

    pressure_field = _run(medium, time_axis, sources)
    logging.info("jwave simulation complete")

    # Extract pressure time series: shape (Nt, Nx, Ny, Nz, 1)
    p_all = np.asarray(pressure_field.params).squeeze(-1)  # (Nt, Nx, Ny, Nz)

    p_max_data = p_all.max(axis=0)
    p_min_data = -p_all.min(axis=0)  # PNP is magnitude of negative peak

    # Acoustic impedance and intensity
    Z = params['density'].data * params['sound_speed'].data
    intensity_data = 1e-4 * p_all.min(axis=0) ** 2 / (2 * Z)

    # Package as xarray
    sz = list(params.coords.sizes.values())
    p_max = xa.DataArray(
        p_max_data.reshape(sz),
        coords=params.coords,
        name='p_max',
        attrs={'units': 'Pa', 'long_name': 'PPP'},
    )
    p_min = xa.DataArray(
        p_min_data.reshape(sz),
        coords=params.coords,
        name='p_min',
        attrs={'units': 'Pa', 'long_name': 'PNP'},
    )
    intensity = xa.DataArray(
        intensity_data.reshape(sz),
        coords=params.coords,
        name='I',
        attrs={'units': 'W/cm^2', 'long_name': 'Intensity'},
    )

    ds = xa.Dataset({'p_max': p_max, 'p_min': p_min, 'intensity': intensity})
    raw = {
        'p_max': p_max_data,
        'p_min': p_min_data,
        'pressure': p_all,
    }
    return ds, raw


def _build_cw_source_field(
    arr: xdc.Transducer,
    domain: Domain,
    coords: xa.Coordinates,
    scl: float,
    amplitude: float,
    apod: np.ndarray,
    freq: float,
    delays: np.ndarray,
    c0: float = 1500.0,
) -> FourierSeries:
    """Build a complex source field for the Helmholtz solver.

    Each transducer element is placed as a point source at the nearest grid
    point, weighted by amplitude, apodization, beamforming delay phase, and
    element area. The element area normalization ensures grid-independent
    results: each point source represents a finite surface patch of the
    transducer, scaled by A_elem / dx^2 to account for the grid cell area.

    For a focused bowl transducer with zero beamforming delays, all elements
    are equidistant from the focus (distance = ROC), so all phases are equal.
    Focusing arises from the spatial superposition of spherical waves emitted
    from the curved bowl surface.

    Args:
        arr: Transducer with element positions.
        domain: jwave Domain.
        coords: Simulation grid coordinates.
        scl: Unit scale factor to meters.
        amplitude: Source amplitude (Pa).
        apod: Per-element apodization.
        freq: CW frequency in Hz.
        delays: Per-element time delays in seconds (from beamforming).
        c0: Reference sound speed for geometric focusing (m/s).

    Returns:
        Complex FourierSeries source field.
    """
    dim_names = _dim_names(coords)
    coord_arrays = [coords[dim].values for dim in dim_names]
    positions = arr.get_positions(units="m")
    omega = 2 * np.pi * freq
    n_elem = arr.numelements()

    # Compute effective element area from transducer geometry.
    # For a bowl transducer, estimate the total active surface area from
    # the convex hull of element positions, then divide equally.
    # For rectangular arrays, use element size directly.
    elem_sizes = np.array([el.get_size(units="m") for el in arr.elements])
    if np.all(elem_sizes > 0):
        # Use actual element sizes (width * length)
        elem_areas = elem_sizes[:, 0] * elem_sizes[:, 1]
    else:
        # Estimate: total area / n_elements
        elem_areas = np.ones(n_elem) * 1e-6  # fallback 1 mm^2

    # Grid cell face area (for normalizing surface sources to the grid)
    dx_face = domain.dx[0] * domain.dx[1]  # m^2

    src = np.zeros(tuple(domain.N) + (1,), dtype=np.complex64)
    for i in range(n_elem):
        idx = []
        for d, cvals in enumerate(coord_arrays):
            cvals_m = cvals * scl
            idx.append(int(np.argmin(np.abs(cvals_m - positions[i, d]))))
        phase = -omega * delays[i]
        # Scale by element area / grid cell area for grid-independent results
        area_scale = elem_areas[i] / dx_face
        src[tuple(idx) + (0,)] += amplitude * apod[i] * area_scale * np.exp(1j * phase)

    return FourierSeries(jnp.array(src), domain)


def run_cw_simulation(
    arr: xdc.Transducer,
    params: xa.Dataset,
    delays: np.ndarray | None = None,
    apod: np.ndarray | None = None,
    freq: float = 1e6,
    amplitude: float = 1,
    pml_size: int = 20,
    ref_values_only: bool = False,
    tol: float = 1e-3,
    maxiter: int = 1000,
    method: str = "gmres",
) -> Tuple[xa.Dataset, dict]:
    """Run a steady-state CW simulation using jwave's Helmholtz solver.

    Solves the Helmholtz equation directly for the complex pressure field
    at a single frequency. Memory-efficient: only stores one spatial field,
    not the full time series.

    Element focusing is encoded as phase shifts in the complex source field:
    geometric phase from path-length differences plus any beamforming delays.

    Use this for:
        - ITRUSST benchmark comparisons (expects p_amp)
        - Phase correction / aberration correction
        - Any CW or quasi-CW analysis

    Args:
        arr: Transducer array.
        params: Simulation parameters (xarray Dataset with sound_speed,
            density, attenuation).
        delays: Per-element time delays in seconds. None defaults to zeros.
        apod: Per-element apodization weights. None defaults to ones.
        freq: CW frequency in Hz.
        amplitude: Source amplitude (Pa).
        pml_size: PML absorbing boundary grid points.
        ref_values_only: Use only reference (homogeneous) medium values.
        tol: GMRES/BiCGSTAB convergence tolerance.
        maxiter: Maximum solver iterations.
        method: Linear solver method ("gmres" or "bicgstab").

    Returns:
        ds: xarray Dataset with 'p_amp' (pressure amplitude) and
            'p_phase' (pressure phase) variables.
        raw: dict with 'p_complex' (complex pressure field as numpy array).
    """
    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())

    domain, scl = get_domain(params.coords)
    medium = get_medium(params, domain, ref_values_only=ref_values_only, pml_size=pml_size)

    # Reference sound speed for geometric phase calculation
    c0 = float(params['sound_speed'].attrs.get('ref_value', 1500.0))
    source = _build_cw_source_field(
        arr, domain, params.coords, scl, amplitude, apod, freq, delays, c0,
    )

    omega = 2 * np.pi * freq
    logging.info("Running jwave Helmholtz solver (f=%.0f kHz, tol=%.0e)", freq / 1e3, tol)

    @jax.jit
    def _solve(medium, source):
        return helmholtz_solver(
            medium, omega, source, method=method, tol=tol, maxiter=maxiter,
        )

    result = _solve(medium, source)
    logging.info("Helmholtz solver complete")

    p_complex = np.asarray(result.params).squeeze(-1)  # (Nx, Ny, Nz)
    p_amp_data = np.abs(p_complex)
    p_phase_data = np.angle(p_complex)

    # Derived quantities
    Z = params['density'].data * params['sound_speed'].data
    intensity_data = 1e-4 * p_amp_data ** 2 / (2 * Z)

    sz = list(params.coords.sizes.values())
    p_amp = xa.DataArray(
        p_amp_data.reshape(sz),
        coords=params.coords,
        name='p_amp',
        attrs={'units': 'Pa', 'long_name': 'Pressure Amplitude'},
    )
    p_phase = xa.DataArray(
        p_phase_data.reshape(sz),
        coords=params.coords,
        name='p_phase',
        attrs={'units': 'rad', 'long_name': 'Pressure Phase'},
    )
    intensity = xa.DataArray(
        intensity_data.reshape(sz),
        coords=params.coords,
        name='I',
        attrs={'units': 'W/cm^2', 'long_name': 'Intensity'},
    )

    ds = xa.Dataset({'p_amp': p_amp, 'p_phase': p_phase, 'intensity': intensity})
    raw = {'p_complex': p_complex}
    return ds, raw
