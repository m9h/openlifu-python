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
    """Build complex source field from JAX arrays (differentiable)."""
    src = jnp.zeros(tuple(domain.N) + (1,), dtype=jnp.complex64)
    for i, idx in enumerate(positions_idx):
        phase = -omega * delays[i]
        val = amplitudes[i] * jnp.exp(1j * phase)
        src = src.at[idx + (0,)].add(val)
    return FourierSeries(src, domain)


def _element_indices(arr: xdc.Transducer, coords: xa.Coordinates, scl: float):
    """Map element positions to grid indices."""
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
    """Compute gradient of focal pressure w.r.t. delays."""
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
    """Optimize per-element delays to maximize pressure at target.

    Uses jax.grad through the Helmholtz solver for gradient descent.

    Returns:
        Optimized delays in seconds.
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
