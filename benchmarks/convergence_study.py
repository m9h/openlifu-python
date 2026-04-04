"""Resolution convergence study for ITRUSST BM4 benchmark.

Usage:
    .venv/bin/python benchmarks/convergence_study.py
    .venv/bin/modal run benchmarks/convergence_study.py
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# Reuse BM4 constants
from benchmarks.itrusst_bm4 import (
    FREQ_HZ, SOURCE_PRESSURE, ROC_MM, SKULL_START_MM,
    build_bm4_phantom, build_bowl_transducer,
)

BRAIN_START_MM = SKULL_START_MM + 10.5


def run_single_resolution(dx_mm: float) -> dict:
    from openlifu.sim.jwave_if import run_cw_simulation

    params, coords = build_bm4_phantom(dx_mm=dx_mm)
    tx = build_bowl_transducer()

    t0 = time.perf_counter()
    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=FREQ_HZ, amplitude=SOURCE_PRESSURE,
        pml_size=max(10, int(20 / dx_mm)), tol=1e-4, maxiter=1000,
    )
    sim_time = time.perf_counter() - t0

    mid_z = ds.p_amp.shape[2] // 2
    p_amp_2d = ds.p_amp.data[:, :, mid_z]
    mid_y = p_amp_2d.shape[1] // 2
    axial = p_amp_2d[:, mid_y]

    focus_idx = min(int(round(ROC_MM / dx_mm)), len(axial) - 1)
    focal_pressure = float(axial[focus_idx])

    brain_idx = int(round(BRAIN_START_MM / dx_mm))
    brain_axial = axial[brain_idx:]
    ff_peak = int(np.argmax(brain_axial))
    peak_x_mm = float((brain_idx + ff_peak) * dx_mm)

    return {
        "focal_pressure_pa": focal_pressure,
        "peak_pressure_pa": float(brain_axial[ff_peak]),
        "peak_x_mm": peak_x_mm,
        "grid_shape": list(ds.p_amp.shape),
        "sim_time_s": sim_time,
        "dx_mm": dx_mm,
    }


def run_convergence(dx_values=None, save_png=None):
    if dx_values is None:
        dx_values = [4.0, 2.0, 1.0]

    results = [run_single_resolution(dx) for dx in dx_values]

    print("\n" + "=" * 75)
    print("ITRUSST BM4 RESOLUTION CONVERGENCE")
    print("=" * 75)
    print(f"{'dx':>6}  {'Grid':>18}  {'Focal (kPa)':>12}  {'Peak (kPa)':>12}  {'Peak x':>8}  {'Time':>6}")
    print("-" * 75)
    for r in results:
        shape = "x".join(str(s) for s in r["grid_shape"])
        print(f"{r['dx_mm']:6.1f}  {shape:>18}  {r['focal_pressure_pa']/1e3:12.1f}  "
              f"{r['peak_pressure_pa']/1e3:12.1f}  {r['peak_x_mm']:8.1f}  {r['sim_time_s']:6.1f}")

    if save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        inv_dx = [1/r["dx_mm"] for r in results]
        ax1.plot(inv_dx, [r["focal_pressure_pa"]/1e3 for r in results], "o-", label="Focal")
        ax1.plot(inv_dx, [r["peak_pressure_pa"]/1e3 for r in results], "s--", label="Brain peak")
        ax1.set_xlabel("1/dx (1/mm)"); ax1.set_ylabel("Pressure (kPa)")
        ax1.set_title("Pressure Convergence"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(inv_dx, [r["peak_x_mm"] for r in results], "o-")
        ax2.axhline(ROC_MM, color="gray", ls="--", label=f"ROC={ROC_MM}mm")
        ax2.set_xlabel("1/dx (1/mm)"); ax2.set_ylabel("Peak x (mm)")
        ax2.set_title("Peak Position Convergence"); ax2.legend(); ax2.grid(alpha=0.3)

        fig.suptitle("ITRUSST BM4 Resolution Convergence")
        fig.tight_layout(); fig.savefig(save_png, dpi=150)

    return results


try:
    import modal
    app = modal.App("bm4-convergence")
    img = (
        modal.Image.debian_slim(python_version="3.12").apt_install("git")
        .pip_install(
            "jax[cuda12]>=0.9.0", "jaxdf>=0.3.0",
            "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
            "numpy", "xarray[io]", "matplotlib", "pandas", "scipy", "h5py",
            "nibabel", "scikit-image", "vtk", "trimesh", "pydicom",
            "opencv-contrib-python-headless", "crc", "crcmod", "pyserial",
            "nvidia-ml-py", "OpenEXR", "watchdog", "python-socketio[client]", "onnxruntime",
        )
        .add_local_dir("src/openlifu", "/root/pkg/openlifu", copy=True)
        .add_local_file("benchmarks/convergence_study.py", "/root/pkg/benchmarks/convergence_study.py", copy=True)
        .add_local_file("benchmarks/itrusst_bm4.py", "/root/pkg/benchmarks/itrusst_bm4.py", copy=True)
        .env({"PYTHONPATH": "/root/pkg"})
    )

    @app.function(image=img, gpu="A100", timeout=3600)
    def run_remote():
        import sys; sys.path.insert(0, "/root/pkg")
        import jax; print(f"JAX: {jax.default_backend()}, {jax.devices()}")
        return run_convergence([4.0, 2.0, 1.0, 0.5], save_png="/tmp/convergence.png")

    @app.local_entrypoint()
    def main():
        t0 = time.perf_counter()
        results = run_remote.remote()
        print(f"\nTotal: {time.perf_counter()-t0:.0f}s")
except ImportError:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, nargs="+", default=[4.0, 2.0, 1.0])
    parser.add_argument("--save-png", type=str, default=None)
    args = parser.parse_args()
    run_convergence(sorted(args.dx, reverse=True), args.save_png)
