"""Gradient-based per-element delay optimization for transcranial FUS.

Backpropagates through openlifu's differentiable jwave time-domain
solver (``run_simulation_peak``) to find the per-element delays that
maximise pressure at a focal target voxel through a heterogeneous
skull. This is the differentiable analogue of analytic phase
correction (``openlifu.sim.phase_correction``): the analytic version
uses path-length geometry through a medium-velocity field, while this
version uses the actual wave equation, so it captures multi-path,
diffraction, and finite-bandwidth effects that the geometric
approximation misses.

Adapted from ``sbi4dwi/dmipy_jax/biophysics/tus_optimizer.py`` and
ported to openlifu's xarray-based simulation contract.

Optimizer is a hand-rolled Adam to avoid pulling optax as a hard dep.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xa

from openlifu import xdc
from openlifu.sim.jwave_if import (
    get_domain, get_medium, get_sources, get_time_axis,
)


@dataclass
class OptimizeResult:
    """Outcome of a gradient-based delay optimization run."""

    delays: np.ndarray  # (n_elements,) optimized per-element delays in seconds
    loss_history: list[float]  # negative target pressure per iteration
    final_focal_pressure_pa: float  # peak pressure at target (positive)
    n_iters: int


def _adam_step(
    g: jnp.ndarray, m: jnp.ndarray, v: jnp.ndarray, t: int,
    *, lr: float, beta1: float, beta2: float, eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g * g)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return update, m, v


def optimize_delays(
    arr: xdc.Transducer,
    params: xa.Dataset,
    target_voxel: Tuple[int, int, int],
    freq: float,
    *,
    apod: np.ndarray | None = None,
    initial_delays: np.ndarray | None = None,
    amplitude: float = 60_000.0,
    cycles: float = 8.0,
    cfl: float = 0.3,
    pml_size: int = 8,
    n_iters: int = 50,
    lr: float = 1e-8,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> OptimizeResult:
    """Maximise focal pressure at ``target_voxel`` by adjusting element delays.

    Args:
        arr: openlifu Transducer with element positions.
        params: xarray Dataset with ``sound_speed``, ``density``, and
            ``attenuation`` (a heterogeneous skull medium produced by
            ``HeterogeneousSkullSegmentation`` is the typical input).
        target_voxel: Grid indices ``(i, j, k)`` of the focal target
            inside the simulation domain (after PML padding is stripped,
            i.e. in the user-facing coordinates of ``params``).
        freq: Transducer drive frequency (Hz).
        apod: Per-element apodization weights. Defaults to ones.
        initial_delays: Per-element starting delays in seconds.
            Defaults to zeros (uniform-phase emission).
        amplitude: Source pressure amplitude per element (Pa).
        cycles: Number of cycles in the source toneburst.
        cfl: CFL number for the time axis.
        pml_size: PML grid thickness.
        n_iters: Adam iterations.
        lr: Learning rate. Default 1e-8 is calibrated for delays in
            seconds on a 500 kHz problem; for other frequencies scale
            proportionally to 1/freq.
        beta1, beta2, eps: Adam hyperparameters.

    Returns:
        ``OptimizeResult`` with optimized delays (s), per-iteration loss
        history (negative focal pressure), and final focal pressure (Pa).
    """
    log = logging.getLogger(__name__)
    n_elements = int(arr.numelements())
    if apod is None:
        apod = np.ones(n_elements, dtype=np.float32)
    if initial_delays is None:
        initial_delays = np.zeros(n_elements, dtype=np.float32)

    # Build the static pieces (medium, time axis, source signal grid)
    # once so the gradient loop only re-traces the parts that depend on
    # delays.
    domain, scl = get_domain(params.coords)
    medium = get_medium(params, domain, ref_values_only=False, pml_size=pml_size)
    time_axis = get_time_axis(medium, cfl=cfl)
    sim_dt = float(time_axis.dt)
    t = np.arange(0, cycles / freq, sim_dt)
    base_signal = amplitude * np.sin(2 * np.pi * freq * t)

    # The simulator depends on delays through the per-element source
    # signal. We use jwave's existing forward path (Sources construction
    # + simulate_wave_propagation) inside a jax-traced loss closure so
    # autodiff handles the chain rule through the wave equation.
    from jwave import simulate_wave_propagation

    target_voxel_padded = tuple(int(t) + pml_size for t in target_voxel)

    def loss_fn(delays_jax: jnp.ndarray) -> jnp.ndarray:
        sources = get_sources(
            arr, domain, params.coords, scl,
            base_signal, sim_dt, np.asarray(delays_jax), np.asarray(apod),
        )
        p_field = simulate_wave_propagation(medium, time_axis, sources=sources)
        # p_field has shape (Nt, *grid_shape, 1); we want max over time
        # at the padded target voxel only. Indexing keeps it cheap and
        # spatially sparse (no full peak-pressure map needed).
        i, j, k = target_voxel_padded
        p_traj = p_field.params[:, i, j, k, 0]
        return -jnp.max(jnp.abs(p_traj))

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    delays = jnp.asarray(initial_delays, dtype=jnp.float32)
    m = jnp.zeros_like(delays)
    v = jnp.zeros_like(delays)
    history: list[float] = []
    for step in range(1, n_iters + 1):
        loss_val, grads = grad_fn(delays)
        update, m, v = _adam_step(
            grads, m, v, step,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
        )
        delays = delays - update
        history.append(float(loss_val))
        if step == 1 or step % 10 == 0 or step == n_iters:
            log.info("iter %3d  loss=%.3e  focal_p=%.2f kPa",
                     step, history[-1], -history[-1] / 1e3)

    return OptimizeResult(
        delays=np.asarray(delays),
        loss_history=history,
        final_focal_pressure_pa=float(-history[-1]),
        n_iters=n_iters,
    )
