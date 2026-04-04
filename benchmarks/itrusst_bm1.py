"""ITRUSST Benchmark BM1: Free-field focused bowl in lossless water.

The simplest benchmark — no skull, no attenuation. Validates source
model and solver accuracy against the analytical focal pressure for
a focused bowl transducer.

Analytical focal pressure: p_focus = rho*c*v0*k*a^2/(2*R)
  = 1000*1500*0.04*(2*pi*500e3/1500)*0.032^2/(2*0.064) ≈ 1005 kPa

Usage:
    .venv/bin/python benchmarks/itrusst_bm1.py
    .venv/bin/modal run benchmarks/itrusst_bm1.py
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import xarray as xa

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

FREQ_HZ = 500e3
ROC_MM = 64.0
APERTURE_MM = 64.0
SOURCE_V0 = 0.04
SOURCE_PRESSURE = 1000.0 * 1500.0 * SOURCE_V0  # 60 kPa


def analytical_focal_pressure():
    """Analytical pressure at the geometric focus of a bowl transducer."""
    rho, c, v0 = 1000.0, 1500.0, SOURCE_V0
    k = 2 * np.pi * FREQ_HZ / c
    a = (APERTURE_MM / 2) * 1e-3
    R = ROC_MM * 1e-3
    return rho * c * v0 * k * a**2 / (2 * R)


def build_bm1_phantom(dx_mm: float = 0.5):
    """Uniform lossless water grid."""
    nx = int(120.0 / dx_mm) + 1
    ny = int(70.0 / dx_mm) + 1
    nz = ny

    x = np.arange(nx) * dx_mm
    y = np.arange(ny) * dx_mm - 35.0
    z = np.arange(nz) * dx_mm - 35.0

    coords = {}
    for dim, vals in [("x", x), ("y", y), ("z", z)]:
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        c.attrs["long_name"] = dim.upper()
        coords[dim] = c
    coords = xa.Coordinates(coords)

    shape = (nx, ny, nz)
    params = xa.Dataset({
        "sound_speed": xa.DataArray(np.full(shape, 1500.0, dtype=np.float32), coords=coords,
                                     attrs={"units": "m/s", "ref_value": 1500.0}),
        "density": xa.DataArray(np.full(shape, 1000.0, dtype=np.float32), coords=coords,
                                 attrs={"units": "kg/m^3", "ref_value": 1000.0}),
        "attenuation": xa.DataArray(np.full(shape, 0.0, dtype=np.float32), coords=coords,
                                     attrs={"units": "dB/cm/MHz", "ref_value": 0.0}),
    })
    return params, coords


def build_bowl_transducer(n_elements=256):
    """Reuse the BM4 bowl transducer — same geometry."""
    from benchmarks.itrusst_bm4 import build_bowl_transducer as _build
    return _build(freq_hz=FREQ_HZ, roc_mm=ROC_MM, aperture_mm=APERTURE_MM,
                  n_elements=n_elements)


def run_bm1(dx_mm=1.0, save_png=None):
    from openlifu.sim.jwave_if import run_cw_simulation

    p_analytical = analytical_focal_pressure()
    log.info("Analytical focal pressure: %.1f kPa", p_analytical / 1e3)

    params, coords = build_bm1_phantom(dx_mm=dx_mm)
    tx = build_bowl_transducer()
    log.info("Grid: %s, %d elements", dict(coords.sizes), tx.numelements())

    t0 = time.perf_counter()
    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=FREQ_HZ, amplitude=SOURCE_PRESSURE,
        pml_size=20, tol=1e-4, maxiter=1000,
    )
    sim_time = time.perf_counter() - t0

    mid_z = ds.p_amp.shape[2] // 2
    p_amp = ds.p_amp.data[:, :, mid_z]
    mid_y = p_amp.shape[1] // 2
    axial = p_amp[:, mid_y]
    focus_idx = int(round(64.0 / dx_mm))
    p_focus = axial[focus_idx]

    peak_idx = np.argmax(axial[10:]) + 10
    peak_x = peak_idx * dx_mm

    error_pct = 100 * abs(p_focus - p_analytical) / p_analytical

    log.info("Results:")
    log.info("  Sim time: %.1fs", sim_time)
    log.info("  p_focus (x=64mm): %.1f kPa", p_focus / 1e3)
    log.info("  Analytical:       %.1f kPa", p_analytical / 1e3)
    log.info("  Error:            %.1f%%", error_pct)
    log.info("  Peak on axis:     %.1f kPa at x=%.1f mm", axial[peak_idx] / 1e3, peak_x)

    if save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        x_mm = coords["x"].values
        y_mm = coords["y"].values

        im = ax1.imshow(p_amp.T / 1e3, origin="lower", aspect="auto",
                        extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]], cmap="hot")
        ax1.plot(64, 0, "w+", ms=12, mew=2)
        ax1.set_xlabel("Axial (mm)"); ax1.set_ylabel("Lateral (mm)")
        ax1.set_title("BM1: Free-Field Pressure (kPa)")
        fig.colorbar(im, ax=ax1)

        ax2.plot(x_mm, axial / 1e3, "r-", lw=2, label="Helmholtz")
        ax2.axvline(64, color="k", ls=":", label=f"Focus ({p_focus/1e3:.0f} kPa)")
        ax2.axhline(p_analytical / 1e3, color="b", ls="--", label=f"Analytical ({p_analytical/1e3:.0f} kPa)")
        ax2.set_xlabel("Axial (mm)"); ax2.set_ylabel("p_amp (kPa)")
        ax2.set_title("Axial Profile vs Analytical")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(save_png, dpi=150)
        log.info("Saved %s", save_png)

    return {
        "sim_time_s": sim_time,
        "p_focus_kpa": float(p_focus / 1e3),
        "p_analytical_kpa": float(p_analytical / 1e3),
        "error_pct": float(error_pct),
        "peak_x_mm": float(peak_x),
        "peak_kpa": float(axial[peak_idx] / 1e3),
    }


# Modal deployment
try:
    import modal

    app = modal.App("itrusst-bm1")
    bm1_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install(
            "jax[cuda12]>=0.9.0", "jaxdf>=0.3.0",
            "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
            "numpy", "xarray[io]", "matplotlib", "pandas", "scipy", "h5py",
            "nibabel", "scikit-image", "vtk", "trimesh", "pydicom",
            "opencv-contrib-python-headless", "crc", "crcmod", "pyserial",
            "nvidia-ml-py", "OpenEXR", "watchdog", "python-socketio[client]",
            "onnxruntime",
        )
        .add_local_dir("src/openlifu", "/root/openlifu_pkg/openlifu", copy=True)
        .add_local_file("benchmarks/itrusst_bm1.py", "/root/openlifu_pkg/benchmarks/itrusst_bm1.py", copy=True)
        .add_local_file("benchmarks/itrusst_bm4.py", "/root/openlifu_pkg/benchmarks/itrusst_bm4.py", copy=True)
        .env({"PYTHONPATH": "/root/openlifu_pkg"})
    )

    @app.function(image=bm1_image, gpu="A100", timeout=1800)
    def run_bm1_remote(dx_mm=0.5):
        import sys
        sys.path.insert(0, "/root/openlifu_pkg")
        import jax
        print(f"JAX: {jax.default_backend()}, {jax.devices()}")
        return run_bm1(dx_mm=dx_mm, save_png="/tmp/bm1.png")

    @app.local_entrypoint()
    def modal_main(dx_mm: float = 0.5):
        t0 = time.perf_counter()
        print(f"Running BM1 on A100 (dx={dx_mm} mm)...")
        r = run_bm1_remote.remote(dx_mm=dx_mm)
        total = time.perf_counter() - t0
        print(f"\nBM1 Results:")
        print(f"  Focus:      {r['p_focus_kpa']:.1f} kPa")
        print(f"  Analytical: {r['p_analytical_kpa']:.1f} kPa")
        print(f"  Error:      {r['error_pct']:.1f}%")
        print(f"  Peak:       {r['peak_kpa']:.1f} kPa at x={r['peak_x_mm']:.1f} mm")
        print(f"  Time:       {r['sim_time_s']:.1f}s (total {total:.0f}s)")
except ImportError:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--save-png", type=str, default=None)
    args = parser.parse_args()
    run_bm1(dx_mm=args.dx, save_png=args.save_png)
