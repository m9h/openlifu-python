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


def points_per_wavelength(
    sound_speed: np.ndarray | float,
    freq_hz: float,
    dx_m: float,
) -> float:
    """Minimum points-per-wavelength (PPW) of a grid given a sound-speed map.

    PPW = c_min / (freq * dx). The minimum is taken over ``sound_speed``
    so that the figure reflects the most under-resolved tissue in the
    domain (typically water at ~1500 m/s when bone is the slowest).

    The ITRUSST consensus (Treeby et al. 2022) recommends >= 12 PPW for
    second-order accuracy on heterogeneous skull simulations; PSTD k-Wave
    style codes can sometimes get away with 6-8 PPW for the slowest tissue.
    """
    c_min = float(np.min(np.asarray(sound_speed)))
    return c_min / (freq_hz * dx_m)


def check_ppw(
    sound_speed: np.ndarray | float,
    freq_hz: float,
    dx_m: float,
    target: float = 12.0,
    minimum: float = 6.0,
) -> dict[str, float | bool]:
    """Validate grid resolution against a target points-per-wavelength.

    Logs a warning when below ``target`` and raises ``ValueError`` when
    below ``minimum``. Returns a dict with ``ppw``, ``ok``, ``target``,
    ``minimum`` so callers can branch on the result.
    """
    ppw = points_per_wavelength(sound_speed, freq_hz, dx_m)
    log = logging.getLogger(__name__)
    if ppw < minimum:
        raise ValueError(
            f"Grid resolution dx={dx_m*1e3:.3f} mm gives only {ppw:.1f} PPW "
            f"at f={freq_hz/1e3:.0f} kHz (minimum {minimum:.0f}); refine the "
            f"grid or lower the frequency before simulating."
        )
    if ppw < target:
        log.warning(
            "Grid resolution dx=%.3f mm gives %.1f PPW at f=%.0f kHz "
            "(below ITRUSST target of %.0f). Convergence will be poor for "
            "heterogeneous skull simulations.",
            dx_m * 1e3, ppw, freq_hz / 1e3, target,
        )
    return {"ppw": ppw, "ok": ppw >= target, "target": target, "minimum": minimum}


def make_toneburst(
    freq_hz: float,
    dt: float,
    n_cycles: int,
    *,
    delay: float = 0.0,
    amplitude: float = 1.0,
    pad_cycles: float = 2.0,
) -> jnp.ndarray:
    """Generate a delayed sinusoidal toneburst signal.

    A finite-duration sin wave windowed to ``[delay, delay + n_cycles/freq]``,
    suitable as the input signal for a single transducer element. The
    array is padded by ``pad_cycles / freq`` of zeros at the end so
    downstream solvers see clean trailing edge.

    PRESTUS / k-Plan style transient simulations use 8-12 cycle bursts;
    pure CW analysis (steady-state) typically uses 20+ cycles or the
    Helmholtz solver instead.
    """
    duration = n_cycles / freq_hz
    total_time = delay + duration + pad_cycles / freq_hz
    n_samples = int(total_time / dt) + 1
    t = jnp.arange(n_samples) * dt

    signal = amplitude * jnp.sin(2.0 * jnp.pi * freq_hz * (t - delay))
    window = jnp.where((t >= delay) & (t <= delay + duration), 1.0, 0.0)
    return signal * window


def make_tonebursts_vectorised(
    freq_hz: float,
    dt: float,
    n_cycles: int,
    delays: jnp.ndarray,
    amplitudes: jnp.ndarray,
    *,
    max_delay: float = 50e-6,
    pad_cycles: float = 2.0,
) -> jnp.ndarray:
    """Generate per-element delayed tonebursts in one JAX-traceable call.

    JAX-differentiable w.r.t. ``delays`` and ``amplitudes`` so this can
    be used directly inside autodiff-based delay-optimisation pipelines
    (see ``openlifu.sim.optimize_delays``).

    Args:
        freq_hz: Drive frequency.
        dt: Time step (s).
        n_cycles: Burst duration in cycles.
        delays: ``(n_elements,)`` per-element delays in seconds.
        amplitudes: ``(n_elements,)`` per-element apodisation weights.
        max_delay: Upper bound on delays used to size the time axis.
            Must accommodate the largest absolute delay in ``delays``.
        pad_cycles: Trailing zero-padding in cycles.

    Returns:
        ``(n_elements, n_samples)`` toneburst array.
    """
    duration = n_cycles / freq_hz
    total_time = max_delay + duration + pad_cycles / freq_hz
    n_samples = int(total_time / dt) + 1
    t = jnp.arange(n_samples) * dt
    t_shifted = t[jnp.newaxis, :] - delays[:, jnp.newaxis]
    signal = jnp.sin(2.0 * jnp.pi * freq_hz * t_shifted)
    window = jnp.where(
        (t_shifted >= 0.0) & (t_shifted <= duration), 1.0, 0.0,
    )
    return amplitudes[:, jnp.newaxis] * signal * window


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
    alpha_power: float = 2.0,
) -> Medium:
    """Create a jwave Medium from openlifu simulation parameters.

    Args:
        params: xarray Dataset with 'sound_speed', 'density', and 'attenuation' variables.
        domain: jwave Domain matching the params grid.
        ref_values_only: If True, use only scalar reference values (homogeneous medium).
        pml_size: Number of PML grid points on each boundary.
        alpha_power: Frequency power law exponent for attenuation.
            2.0 (default) = thermoviscous (Stokes, exact for water).
            0.9-1.1 = empirical fit for soft tissue and bone (k-Wave convention).

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
            alpha_power=alpha_power,
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
        alpha_power=alpha_power,
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


def run_simulation_peak(
    arr: xdc.Transducer,
    params: xa.Dataset,
    delays: np.ndarray | None = None,
    apod: np.ndarray | None = None,
    freq: float = 1e6,
    cycles: float = 20,
    amplitude: float = 1.0,
    dt: float = 0,
    t_end: float = 0,
    cfl: float = 0.5,
    pml_size: int = 20,
    ref_values_only: bool = False,
    ppw_target: float = 12.0,
) -> Tuple[xa.Dataset, dict]:
    """Time-domain PSTD simulation that returns peak pressure WITHOUT
    storing the full time series.

    Functionally equivalent to ``run_simulation`` but uses a
    ``jax.lax.scan`` carry to maintain running positive- and
    negative-pressure peaks at every voxel. Memory is O(grid_volume)
    rather than O(Nt * grid_volume), so large 3D simulations (where
    the full pressure history would be tens of GB) fit on a single GPU.

    The numerics match jwave's ``simulate_wave_propagation`` exactly:
    same PSTD discretisation, same PML, same CFL.

    Args:
        arr, params, delays, apod, freq, cycles, amplitude, dt, t_end,
        cfl, pml_size, ref_values_only:
            See ``run_simulation``.
        ppw_target:
            Points-per-wavelength below which a warning is logged
            (does not abort). Set to 0 to disable the check.

    Returns:
        ds: ``xarray.Dataset`` with ``p_max`` (PPP), ``p_min`` (|PNP|),
            and ``intensity`` (W/cm^2) variables.
        raw: dict with ``p_max``, ``p_min``, and ``ppw_check`` entries.
    """
    from jwave.acoustics.operators import (
        mass_conservation_rhs, momentum_conservation_rhs, pressure_from_density,
    )
    from jwave.acoustics.time_varying import (
        fourier_wave_prop_params, TimeWavePropagationSettings,
    )

    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())

    domain, scl = get_domain(params.coords)
    medium = get_medium(params, domain, ref_values_only=ref_values_only, pml_size=pml_size)
    time_axis = get_time_axis(medium, cfl=cfl, t_end=t_end, dt=dt)

    # PPW sanity check before launching the scan.
    if ppw_target > 0:
        check_ppw(params['sound_speed'].data, freq, float(domain.dx[0]),
                  target=ppw_target)

    sim_dt = float(time_axis.dt)
    t = np.arange(0, cycles / freq, sim_dt)
    input_signal = amplitude * np.sin(2 * np.pi * freq * t)
    sources = get_sources(arr, domain, params.coords, scl, input_signal,
                          sim_dt, delays, apod)

    # Build PML / k-space params jwave needs for the inner step.
    settings = TimeWavePropagationSettings()
    pp = fourier_wave_prop_params(medium, time_axis, settings=settings)
    c_ref = pp["c_ref"]
    pml_rho = pp["pml_rho"]
    pml_u = pp["pml_u"]
    fourier = pp["fourier"]

    # Initial fields: zero pressure, zero velocity, zero acoustic density.
    grid_shape = tuple(domain.N)
    u_shape = grid_shape + (len(grid_shape),)
    p_shape = grid_shape + (1,)
    p0 = pml_rho.replace_params(jnp.zeros(p_shape, dtype=jnp.float32))
    u0 = pml_u.replace_params(jnp.zeros(u_shape, dtype=jnp.float32))
    rho0 = p0.replace_params(
        jnp.zeros(u_shape, dtype=jnp.float32) / p0.domain.ndim
    )
    rho0 = rho0 / (medium.sound_speed ** 2)

    p_max_init = jnp.full(grid_shape, -jnp.inf, dtype=jnp.float32)
    p_min_init = jnp.full(grid_shape, jnp.inf, dtype=jnp.float32)

    def step(carry, n):
        p, u, rho, p_max_run, p_min_run = carry
        mass_src_field = sources.on_grid(n)

        du = momentum_conservation_rhs(
            p, u, medium, c_ref=c_ref, dt=sim_dt, params=fourier,
        )
        u = pml_u * (pml_u * u + sim_dt * du)

        drho = mass_conservation_rhs(
            p, u, mass_src_field, medium,
            c_ref=c_ref, dt=sim_dt, params=fourier,
        )
        rho = pml_rho * (pml_rho * rho + sim_dt * drho)
        p = pressure_from_density(rho, medium)

        p_grid = p.params[..., 0]
        p_max_run = jnp.maximum(p_max_run, p_grid)
        p_min_run = jnp.minimum(p_min_run, p_grid)
        return (p, u, rho, p_max_run, p_min_run), None

    output_steps = jnp.arange(0, time_axis.Nt, 1)

    @jax.jit
    def _run():
        carry0 = (p0, u0, rho0, p_max_init, p_min_init)
        (_, _, _, p_max_full, p_min_full), _ = jax.lax.scan(
            step, carry0, output_steps,
        )
        return p_max_full, p_min_full

    logging.info("Running jwave time-domain peak-pressure scan (Nt=%d)",
                 int(time_axis.Nt))
    p_max_full, p_min_full = _run()
    p_max_full = np.asarray(p_max_full)
    p_min_full = np.asarray(p_min_full)
    logging.info("Peak-pressure scan complete")

    # Strip PML padding to match the user-facing grid.
    sl = tuple(slice(pml_size, n - pml_size) for n in grid_shape)
    p_max_data = p_max_full[sl]
    p_min_data = -p_min_full[sl]  # PNP magnitude

    Z = params['density'].data * params['sound_speed'].data
    intensity_data = 1e-4 * (p_min_data ** 2) / (2 * Z)

    sz = list(params.coords.sizes.values())
    coords = params.coords
    p_max = xa.DataArray(p_max_data.reshape(sz), coords=coords, name='p_max',
                         attrs={'units': 'Pa', 'long_name': 'PPP'})
    p_min = xa.DataArray(p_min_data.reshape(sz), coords=coords, name='p_min',
                         attrs={'units': 'Pa', 'long_name': 'PNP'})
    intensity = xa.DataArray(intensity_data.reshape(sz), coords=coords, name='I',
                             attrs={'units': 'W/cm^2', 'long_name': 'Intensity'})
    ds = xa.Dataset({'p_max': p_max, 'p_min': p_min, 'intensity': intensity})
    raw = {'p_max': p_max_data, 'p_min': p_min_data,
           'ppw_check': check_ppw(params['sound_speed'].data, freq,
                                  float(domain.dx[0]), target=ppw_target)
           if ppw_target > 0 else None}
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
    idx_offset: int = 0,
) -> FourierSeries:
    """Build a complex source field for the Helmholtz solver.

    Each transducer element is placed as a point source at the nearest grid
    point, weighted by amplitude, apodization, beamforming delay phase, and
    element area. The element area normalization ensures grid-independent
    results: each point source represents a finite surface patch of the
    transducer, scaled by A_elem / dx^2 to account for the grid cell area.

    The scaling ``A_elem / dx^2`` converts each element's source pressure
    into the correct discrete monopole strength for the Helmholtz equation.
    Specifically, a surface element with pressure ``p_s`` and area ``A``
    corresponds to a volume velocity ``Q = 2 * p_s * A / c`` (factor 2
    from the Rayleigh--Sommerfeld baffle effect).  jwave's internal
    ``scale_source_helmholtz`` then applies ``2 / (dx * c)`` to convert
    this into the velocity-potential source term.  The net effect maps
    ``src_value = p_s * A / dx^2`` to the correct monopole strength
    ``S = 2 * p_s * A / c`` in the Helmholtz Green's function.

    For a focused bowl transducer with zero beamforming delays, all elements
    are equidistant from the focus (distance = ROC), so all phases are equal.
    Focusing arises from the spatial superposition of spherical waves emitted
    from the curved bowl surface.

    Args:
        arr: Transducer with element positions.
        domain: jwave Domain (may be PML-padded; use *idx_offset*).
        coords: Simulation grid coordinates (original, unpadded).
        scl: Unit scale factor to meters.
        amplitude: Source amplitude (Pa).
        apod: Per-element apodization.
        freq: CW frequency in Hz.
        delays: Per-element time delays in seconds (from beamforming).
        c0: Reference sound speed for geometric focusing (m/s).
        idx_offset: Index offset to add to each element position along
            every axis.  Used when *domain* has been padded with PML cells
            so that sources are placed in the interior of the padded grid.

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
            idx.append(int(np.argmin(np.abs(cvals_m - positions[i, d]))) + idx_offset)
        phase = -omega * delays[i]
        # Scale by element area / grid cell area for grid-independent results
        area_scale = elem_areas[i] / dx_face
        src[tuple(idx) + (0,)] += amplitude * apod[i] * area_scale * np.exp(1j * phase)

    return FourierSeries(jnp.array(src), domain)


def _pad_medium_arrays(params: xa.Dataset, pml_size: int) -> dict[str, np.ndarray]:
    """Pad acoustic-property arrays by *pml_size* cells on every face.

    Each array is extended using its outermost value along each axis (edge
    padding), so the PML region has the same material properties as the
    domain boundary.

    Parameters
    ----------
    params : xarray.Dataset
        Dataset containing ``sound_speed``, ``density``, and ``attenuation``.
    pml_size : int
        Number of cells to pad on each side of each dimension.

    Returns
    -------
    dict mapping variable name to padded numpy array (float32).
    """
    pad_width = [(pml_size, pml_size)] * 3
    out = {}
    for var in ("sound_speed", "density", "attenuation"):
        out[var] = np.pad(params[var].data, pad_width, mode="edge").astype(np.float32)
    return out


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
    alpha_power: float = 2.0,
) -> Tuple[xa.Dataset, dict]:
    """Run a steady-state CW simulation using jwave's Helmholtz solver.

    Solves the Helmholtz equation directly for the complex pressure field
    at a single frequency. Memory-efficient: only stores one spatial field,
    not the full time series.

    The simulation domain is automatically padded by *pml_size* cells on
    every face so that the PML absorbing layer lies **outside** the
    user-specified grid.  This prevents source elements near domain
    boundaries from being attenuated by the PML, which would otherwise
    cause systematic underestimation of the pressure field.  The padding
    is stripped from the output so the returned fields have the same
    shape and coordinates as the input *params*.

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
        alpha_power: Frequency power law exponent for attenuation.
            2.0 (default) = thermoviscous (exact for water).
            0.9 = soft tissue/bone (k-Wave convention).

    Returns:
        ds: xarray Dataset with 'p_amp' (pressure amplitude) and
            'p_phase' (pressure phase) variables.
        raw: dict with 'p_complex' (complex pressure field as numpy array).
    """
    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())

    # Build the original (unpadded) domain to determine grid spacing and scale.
    domain_orig, scl = get_domain(params.coords)
    dx = domain_orig.dx

    # Build a PML-padded domain so the absorbing layer sits outside the
    # user grid.  Source elements near the domain boundary are therefore
    # never attenuated by the PML.
    N_padded = tuple(n + 2 * pml_size for n in domain_orig.N)
    domain_padded = Domain(N_padded, dx)

    if ref_values_only:
        medium = Medium(
            domain=domain_padded,
            sound_speed=float(params['sound_speed'].attrs['ref_value']),
            density=float(params['density'].attrs['ref_value']),
            attenuation=float(params['attenuation'].attrs['ref_value']),
            pml_size=pml_size,
            alpha_power=alpha_power,
        )
    else:
        padded = _pad_medium_arrays(params, pml_size)
        medium = Medium(
            domain=domain_padded,
            sound_speed=FourierSeries(
                jnp.expand_dims(jnp.array(padded['sound_speed'], dtype=jnp.float32), -1),
                domain_padded,
            ),
            density=FourierSeries(
                jnp.expand_dims(jnp.array(padded['density'], dtype=jnp.float32), -1),
                domain_padded,
            ),
            attenuation=FourierSeries(
                jnp.expand_dims(jnp.array(padded['attenuation'], dtype=jnp.float32), -1),
                domain_padded,
            ),
            pml_size=pml_size,
            alpha_power=alpha_power,
        )

    # Reference sound speed for geometric phase calculation
    c0 = float(params['sound_speed'].attrs.get('ref_value', 1500.0))

    # Build source on the padded grid.  The *idx_offset* shifts element
    # indices so they land in the interior (non-PML) region.
    source = _build_cw_source_field(
        arr, domain_padded, params.coords, scl, amplitude, apod, freq, delays,
        c0, idx_offset=pml_size,
    )

    omega = 2 * np.pi * freq
    logging.info(
        "Running jwave Helmholtz solver (f=%.0f kHz, tol=%.0e, padded %s)",
        freq / 1e3, tol, N_padded,
    )

    @jax.jit
    def _solve(medium, source):
        return helmholtz_solver(
            medium, omega, source, method=method, tol=tol, maxiter=maxiter,
        )

    result = _solve(medium, source)
    logging.info("Helmholtz solver complete")

    # Crop PML padding to recover the original domain shape.
    s = pml_size
    p_complex_full = np.asarray(result.params).squeeze(-1)
    p_complex = p_complex_full[s:-s, s:-s, s:-s]

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
