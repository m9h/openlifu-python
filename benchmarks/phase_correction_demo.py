"""Gradient-based phase correction through skull on A100 GPU.

Demonstrates differentiable tFUS: optimize transducer delays by
backpropagating through jwave's Helmholtz solver to maximize focal
pressure at a target point behind a skull layer.

Usage:
    .venv/bin/modal run benchmarks/phase_correction_demo.py
    .venv/bin/modal run benchmarks/phase_correction_demo.py --dx-mm 0.5
"""
from __future__ import annotations

import logging
import time

import numpy as np
import xarray as xa

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def build_skull_phantom(dx_mm=1.0):
    """Grid with skull layer at x=20-27mm, brain beyond."""
    nx = int(80 / dx_mm) + 1
    ny = int(60 / dx_mm) + 1
    nz = ny

    x = np.arange(nx) * dx_mm
    y = np.arange(ny) * dx_mm - 30.0
    z = np.arange(nz) * dx_mm - 30.0

    coords = {}
    for dim, vals in [("x", x), ("y", y), ("z", z)]:
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        coords[dim] = c
    coords = xa.Coordinates(coords)

    shape = (nx, ny, nz)
    c = np.full(shape, 1500.0, dtype=np.float32)
    rho = np.full(shape, 1000.0, dtype=np.float32)
    alpha = np.full(shape, 0.0, dtype=np.float32)

    skull_start = int(20 / dx_mm)
    skull_end = int(27 / dx_mm)
    c[skull_start:skull_end] = 2800.0
    rho[skull_start:skull_end] = 1850.0
    alpha[skull_start:skull_end] = 4.0

    brain_start = skull_end
    c[brain_start:] = 1560.0
    rho[brain_start:] = 1040.0
    alpha[brain_start:] = 0.3

    return xa.Dataset({
        "sound_speed": xa.DataArray(c, coords=coords, attrs={"units": "m/s", "ref_value": 1500.0}),
        "density": xa.DataArray(rho, coords=coords, attrs={"units": "kg/m^3", "ref_value": 1000.0}),
        "attenuation": xa.DataArray(alpha, coords=coords, attrs={"units": "dB/cm/MHz", "ref_value": 0.0}),
    }), coords


def build_array(n_elements=16):
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer

    # Linear array at x=0, spread in y
    spread = 20e-3  # 20mm total aperture
    positions_y = np.linspace(-spread / 2, spread / 2, n_elements)
    elements = [
        Element(index=i, position=np.array([0., y, 0.]),
                size=np.array([3e-3, 3e-3]), units="m")
        for i, y in enumerate(positions_y)
    ]
    return Transducer(id="phase_demo", elements=elements, frequency=500e3, units="m")


def run_phase_correction(dx_mm=1.0, n_steps=50, n_elements=16, save_png=None):
    from openlifu.sim.jwave_if import run_cw_simulation
    from openlifu.sim.phase_correction import optimize_delays

    params, coords = build_skull_phantom(dx_mm=dx_mm)
    tx = build_array(n_elements=n_elements)

    # Target at x=50mm (behind skull), on-axis
    target_x_idx = int(50 / dx_mm)
    mid_y = params["sound_speed"].shape[1] // 2
    mid_z = params["sound_speed"].shape[2] // 2
    target_idx = (target_x_idx, mid_y, mid_z)

    log.info("Grid: %s, dx=%.1f mm, %d elements", list(params["sound_speed"].shape), dx_mm, n_elements)
    log.info("Target: idx=%s (x=%.0f mm)", target_idx, target_x_idx * dx_mm)

    # 1. Zero delays (no correction)
    log.info("Running with zero delays...")
    ds_zero, _ = run_cw_simulation(arr=tx, params=params, freq=500e3,
                                    amplitude=60000.0, pml_size=10)
    p_zero = float(ds_zero.p_amp.data[target_idx])

    # 2. Random delays (worst case)
    rng = np.random.default_rng(42)
    random_delays = rng.uniform(0, 2e-6, tx.numelements())
    ds_random, _ = run_cw_simulation(arr=tx, params=params, delays=random_delays,
                                      freq=500e3, amplitude=60000.0, pml_size=10)
    p_random = float(ds_random.p_amp.data[target_idx])

    # 3. Optimize delays
    log.info("Optimizing delays (%d steps)...", n_steps)
    t0 = time.perf_counter()
    opt_delays = optimize_delays(
        tx, params, target_idx=target_idx, freq=500e3,
        init_delays=random_delays, n_steps=n_steps, lr=1e-7, pml_size=10,
    )
    opt_time = time.perf_counter() - t0

    ds_opt, _ = run_cw_simulation(arr=tx, params=params, delays=opt_delays,
                                   freq=500e3, amplitude=60000.0, pml_size=10)
    p_opt = float(ds_opt.p_amp.data[target_idx])

    # 4. Water reference (no skull, zero delays)
    water_params, _ = build_skull_phantom(dx_mm=dx_mm)
    water_params["sound_speed"].data[:] = 1500.0
    water_params["density"].data[:] = 1000.0
    water_params["attenuation"].data[:] = 0.0
    ds_water, _ = run_cw_simulation(arr=tx, params=water_params, freq=500e3,
                                     amplitude=60000.0, pml_size=10)
    p_water = float(ds_water.p_amp.data[target_idx])

    recovery = (p_opt - p_random) / (p_water - p_random) * 100 if p_water > p_random else 0

    results = {
        "p_water_kpa": p_water / 1e3,
        "p_zero_delays_kpa": p_zero / 1e3,
        "p_random_delays_kpa": p_random / 1e3,
        "p_optimized_kpa": p_opt / 1e3,
        "improvement_pct": (p_opt - p_random) / max(p_random, 1) * 100,
        "recovery_pct": recovery,
        "opt_time_s": opt_time,
        "n_steps": n_steps,
        "dx_mm": dx_mm,
        "n_elements": n_elements,
        "opt_delays": opt_delays.tolist(),
    }

    log.info("Results:")
    log.info("  Water (no skull):     %.1f kPa", results["p_water_kpa"])
    log.info("  Zero delays:          %.1f kPa", results["p_zero_delays_kpa"])
    log.info("  Random delays:        %.1f kPa", results["p_random_delays_kpa"])
    log.info("  Optimized delays:     %.1f kPa", results["p_optimized_kpa"])
    log.info("  Improvement:          %.1f%%", results["improvement_pct"])
    log.info("  Recovery:             %.1f%%", results["recovery_pct"])
    log.info("  Optimization time:    %.1fs (%d steps)", opt_time, n_steps)

    if save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        x_mm = coords["x"].values
        y_mm = coords["y"].values
        extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
        vmax = max(ds_water.p_amp.data[:, :, mid_z].max(),
                   ds_opt.p_amp.data[:, :, mid_z].max()) / 1e3

        for ax, ds, title in [
            (axes[0, 0], ds_water, f"Water (no skull): {p_water/1e3:.1f} kPa"),
            (axes[0, 1], ds_zero, f"Zero delays: {p_zero/1e3:.1f} kPa"),
            (axes[1, 0], ds_random, f"Random delays: {p_random/1e3:.1f} kPa"),
            (axes[1, 1], ds_opt, f"Optimized: {p_opt/1e3:.1f} kPa"),
        ]:
            im = ax.imshow(ds.p_amp.data[:, :, mid_z].T / 1e3, origin="lower",
                           aspect="auto", extent=extent, cmap="hot", vmin=0, vmax=vmax)
            ax.plot(50, 0, "w+", ms=12, mew=2)
            ax.axvspan(20, 27, alpha=0.2, color="cyan")
            ax.set_title(title)
            ax.set_xlabel("Axial (mm)"); ax.set_ylabel("Lateral (mm)")
            fig.colorbar(im, ax=ax, label="kPa")

        fig.suptitle(f"Phase Correction Through Skull (dx={dx_mm}mm, {n_elements} elements, {n_steps} steps)\n"
                     f"Recovery: {recovery:.0f}%", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_png, dpi=150)
        log.info("Saved %s", save_png)

    return results


try:
    import modal

    app = modal.App("phase-correction")
    img = (
        modal.Image.debian_slim(python_version="3.12").apt_install("git")
        .pip_install(
            "jax[cuda12]>=0.9.0", "jaxdf>=0.3.0",
            "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
            "numpy", "xarray[io]", "matplotlib", "pandas", "scipy", "h5py",
            "nibabel", "scikit-image", "vtk", "trimesh", "pydicom",
            "opencv-contrib-python-headless", "crc", "crcmod", "pyserial",
            "nvidia-ml-py", "OpenEXR", "watchdog", "python-socketio[client]",
            "onnxruntime",
        )
        .add_local_dir("src/openlifu", "/root/pkg/openlifu", copy=True)
        .add_local_file("benchmarks/phase_correction_demo.py",
                        "/root/pkg/benchmarks/phase_correction_demo.py", copy=True)
        .env({"PYTHONPATH": "/root/pkg"})
    )

    @app.function(image=img, gpu="A100", timeout=3600)
    def run_remote(dx_mm: float = 1.0, n_steps: int = 50) -> dict:
        import sys; sys.path.insert(0, "/root/pkg")
        import jax; print(f"JAX: {jax.default_backend()}, {jax.devices()}")
        return run_phase_correction(dx_mm=dx_mm, n_steps=n_steps,
                                     save_png="/tmp/phase_correction.png")

    @app.local_entrypoint()
    def main(dx_mm: float = 1.0, n_steps: int = 50):
        t0 = time.perf_counter()
        print(f"Running phase correction demo on A100 (dx={dx_mm}mm, {n_steps} steps)...")
        r = run_remote.remote(dx_mm=dx_mm, n_steps=n_steps)
        total = time.perf_counter() - t0
        print(f"\nPhase Correction Results:")
        print(f"  Water (reference):  {r['p_water_kpa']:.1f} kPa")
        print(f"  Zero delays:        {r['p_zero_delays_kpa']:.1f} kPa")
        print(f"  Random delays:      {r['p_random_delays_kpa']:.1f} kPa")
        print(f"  Optimized:          {r['p_optimized_kpa']:.1f} kPa")
        print(f"  Improvement:        {r['improvement_pct']:.1f}%")
        print(f"  Recovery:           {r['recovery_pct']:.0f}%")
        print(f"  Opt time:           {r['opt_time_s']:.1f}s")
        print(f"  Total:              {total:.0f}s")
except ImportError:
    pass
