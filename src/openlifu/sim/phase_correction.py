"""Gradient-based phase correction using the differentiable Helmholtz solver.

Optimizes per-element transducer delays to maximize pressure at a target
point by backpropagating through jwave's Helmholtz solver via jax.grad.
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xa
from jaxdf.discretization import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium

from openlifu import xdc
from openlifu.sim.jwave_if import _dim_names, get_domain, get_medium

logging.getLogger(__name__)


def _build_differentiable_source(
    positions_idx: list[tuple],
    amplitudes: jnp.ndarray,
    delays: jnp.ndarray,
    omega: float,
    domain: Domain,
) -> FourierSeries:
    """Build a complex CW source field from JAX arrays (fully differentiable).

    Each transducer element is placed at its nearest grid index as a complex
    point source whose phase encodes the beamforming delay.  Because the
    construction uses only JAX primitives, ``jax.grad`` can differentiate
    through it with respect to *delays* and *amplitudes*.

    Parameters
    ----------
    positions_idx : list of tuple
        Grid indices ``(ix, iy, iz)`` for each transducer element.
    amplitudes : jnp.ndarray
        Per-element amplitude weights (shape ``(n_elements,)``).
    delays : jnp.ndarray
        Per-element time delays in seconds (shape ``(n_elements,)``).
    omega : float
        Angular frequency :math:`2\\pi f` in rad/s.
    domain : jwave.geometry.Domain
        Simulation domain (grid size and spacing).

    Returns
    -------
    jaxdf.discretization.FourierSeries
        Complex source field on the simulation grid.
    """
    src = jnp.zeros(tuple(domain.N) + (1,), dtype=jnp.complex64)
    for i, idx in enumerate(positions_idx):
        phase = -omega * delays[i]
        val = amplitudes[i] * jnp.exp(1j * phase)
        src = src.at[idx + (0,)].add(val)
    return FourierSeries(src, domain)


def _element_indices(arr: xdc.Transducer, coords: xa.Coordinates, scl: float):
    """Map transducer element physical positions to nearest grid indices.

    Parameters
    ----------
    arr : openlifu.xdc.Transducer
        Transducer with element positions in metres.
    coords : xarray.Coordinates
        Simulation grid coordinates (must contain dimension names
        matching ``_dim_names``).
    scl : float
        Scale factor from the coordinate units to metres (e.g. 1e-3 for mm).

    Returns
    -------
    list of tuple
        One ``(ix, iy, iz)`` index tuple per element.
    """
    dim_names = _dim_names(coords)
    coord_arrays = [coords[dim].values for dim in dim_names]
    positions = arr.get_positions(units="m")
    indices = []
    for i in range(arr.numelements()):
        idx = tuple(
            int(np.argmin(np.abs(cvals * scl - positions[i, d])))
            for d, cvals in enumerate(coord_arrays)
        )
        indices.append(idx)
    return indices


def compute_focal_gradient(
    arr: xdc.Transducer,
    params: xa.Dataset,
    target_idx: tuple[int, int, int],
    freq: float,
    delays: np.ndarray,
    amplitude: float = 60000.0,
    pml_size: int = 10,
) -> np.ndarray:
    """Compute the gradient of focal pressure amplitude with respect to delays.

    Evaluates a single forward + backward pass through jwave's Helmholtz
    solver via ``jax.grad``.  The returned gradient vector indicates how a
    small change in each element's delay would alter the (negative) pressure
    amplitude at the target, enabling gradient-descent optimisation.

    Parameters
    ----------
    arr : openlifu.xdc.Transducer
        Transducer array.
    params : xarray.Dataset
        Acoustic property maps (``sound_speed``, ``density``,
        ``attenuation``).
    target_idx : tuple of int
        Grid index ``(ix, iy, iz)`` of the focal target.
    freq : float
        CW frequency in Hz.
    delays : numpy.ndarray
        Current per-element delays in seconds, shape ``(n_elements,)``.
    amplitude : float, optional
        Source amplitude in Pa (default 60 000).
    pml_size : int, optional
        PML absorbing boundary thickness in grid points (default 10).

    Returns
    -------
    numpy.ndarray
        Gradient vector of shape ``(n_elements,)`` in units of Pa/s.
    """
    domain, scl = get_domain(params.coords)
    medium = get_medium(params, domain, ref_values_only=True, pml_size=pml_size)
    omega = 2 * np.pi * freq
    elem_idx = _element_indices(arr, params.coords, scl)

    elem_sizes = np.array([el.get_size(units="m") for el in arr.elements])
    elem_areas = elem_sizes[:, 0] * elem_sizes[:, 1]
    dx_face = domain.dx[0] * domain.dx[1]
    amps = jnp.array(amplitude * elem_areas / dx_face, dtype=jnp.float32)

    def focal_pressure(delays_jax):
        source = _build_differentiable_source(elem_idx, amps, delays_jax, omega, domain)
        result = helmholtz_solver(medium, omega, source, tol=1e-3, maxiter=200)
        p_complex = result.params[target_idx + (0,)]
        return -jnp.abs(p_complex)  # negative for maximization

    grad_fn = jax.grad(focal_pressure)
    grad = grad_fn(jnp.array(delays, dtype=jnp.float32))
    return np.asarray(grad)


def optimize_delays(
    arr: xdc.Transducer,
    params: xa.Dataset,
    target_idx: tuple[int, int, int],
    freq: float,
    init_delays: np.ndarray | None = None,
    n_steps: int = 50,
    lr: float = 1e-7,
    amplitude: float = 60000.0,
    pml_size: int = 10,
) -> np.ndarray:
    """Optimise per-element delays to maximise pressure at a target point.

    Performs gradient descent on the negative focal-pressure objective by
    back-propagating through jwave's Helmholtz solver via ``jax.value_and_grad``.
    Each step is JIT-compiled for GPU acceleration.

    Parameters
    ----------
    arr : openlifu.xdc.Transducer
        Transducer array.
    params : xarray.Dataset
        Acoustic property maps (``sound_speed``, ``density``,
        ``attenuation``).
    target_idx : tuple of int
        Grid index ``(ix, iy, iz)`` of the focal target.
    freq : float
        CW frequency in Hz.
    init_delays : numpy.ndarray or None, optional
        Initial per-element delays in seconds.  If ``None``, starts from
        zero delays.
    n_steps : int, optional
        Number of gradient-descent iterations (default 50).
    lr : float, optional
        Learning rate (default 1e-7).
    amplitude : float, optional
        Source amplitude in Pa (default 60 000).
    pml_size : int, optional
        PML absorbing boundary thickness in grid points (default 10).

    Returns
    -------
    numpy.ndarray
        Optimised delay vector in seconds, shape ``(n_elements,)``.
    """
    domain, scl = get_domain(params.coords)
    medium = get_medium(params, domain, ref_values_only=True, pml_size=pml_size)
    omega = 2 * np.pi * freq
    elem_idx = _element_indices(arr, params.coords, scl)

    elem_sizes = np.array([el.get_size(units="m") for el in arr.elements])
    elem_areas = elem_sizes[:, 0] * elem_sizes[:, 1]
    dx_face = domain.dx[0] * domain.dx[1]
    amps = jnp.array(amplitude * elem_areas / dx_face, dtype=jnp.float32)

    if init_delays is None:
        init_delays = np.zeros(arr.numelements())
    delays = jnp.array(init_delays, dtype=jnp.float32)

    @jax.jit
    def loss_and_grad(delays_jax):
        def focal_pressure(d):
            source = _build_differentiable_source(elem_idx, amps, d, omega, domain)
            result = helmholtz_solver(medium, omega, source, tol=1e-3, maxiter=200)
            return -jnp.abs(result.params[target_idx + (0,)])
        return jax.value_and_grad(focal_pressure)(delays_jax)

    for step in range(n_steps):
        loss, grad = loss_and_grad(delays)
        delays = delays - lr * grad
        if step % 10 == 0:
            logging.info("Step %d: loss=%.1f, |grad|=%.1e", step, float(loss), float(jnp.linalg.norm(grad)))

    return np.asarray(delays)
