"""ITRUSST Benchmark BM4: Multi-layer flat skull with focused bowl source.

Reproduces BM4-SC1 from Aubry et al. (2022) using openlifu's jwave backend.
BM4 is a flat multi-layer skull geometry:
    - Water coupling (0-26 mm)
    - Skin: 4 mm, c=1610, rho=1090, alpha=0.2 dB/cm
    - Cortical bone (outer): 1.5 mm, c=2800, rho=1850, alpha=4.0 dB/cm
    - Trabecular bone: 4 mm, c=2300, rho=1700, alpha=8.0 dB/cm
    - Cortical bone (inner): 1 mm, c=2800, rho=1850, alpha=4.0 dB/cm
    - Brain: remainder, c=1560, rho=1040, alpha=0.3 dB/cm

Source SC1: focused bowl, 64 mm ROC, 64 mm aperture, 500 kHz, v0=0.04 m/s.
Output grid: 0.5 mm spacing, 120 x 70 mm (axial x lateral), 2D central slice.

Reference results from 11 solvers (including jwave) available at:
    https://zenodo.org/records/6020543

Usage:
    # Run locally (CPU, slow)
    .venv/bin/python benchmarks/itrusst_bm4.py

    # Run on Modal A100 GPU
    .venv/bin/modal run benchmarks/itrusst_bm4.py

    # Compare against downloaded reference data
    .venv/bin/python benchmarks/itrusst_bm4.py --ref-path /path/to/KWAVE/PH1-BM4-SC1_KWAVE.mat
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import xarray as xa

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ITRUSST BM4 material properties (Table I, Aubry et al. 2022)
# ---------------------------------------------------------------------------
MATERIALS = {
    "water":      {"c": 1500.0, "rho": 1000.0, "alpha": 0.0},
    "skin":       {"c": 1610.0, "rho": 1090.0, "alpha": 0.2},
    "cortical":   {"c": 2800.0, "rho": 1850.0, "alpha": 4.0},
    "trabecular": {"c": 2300.0, "rho": 1700.0, "alpha": 8.0},
    "brain":      {"c": 1560.0, "rho": 1040.0, "alpha": 0.3},
}

# BM4 geometry: layer boundaries along axial (x) direction from source face
# Source at x=0, skull starts at 26 mm from transducer face
# (but transducer face is at x=0, so layers are offset by transducer-to-skull distance)
SKULL_START_MM = 26.0
LAYERS = [
    # (start_offset_mm, thickness_mm, material)
    (0.0, 4.0, "skin"),
    (4.0, 1.5, "cortical"),     # outer table
    (5.5, 4.0, "trabecular"),
    (9.5, 1.0, "cortical"),     # inner table
]
# Brain fills remainder after skull_start + 10.5 mm = 36.5 mm from source

# Source: focused bowl, 64 mm ROC, 64 mm aperture, 500 kHz
FREQ_HZ = 500e3
ROC_MM = 64.0
APERTURE_MM = 64.0
SOURCE_V0 = 0.04  # m/s surface velocity
# Equivalent source pressure: p0 = rho * c * v0 = 1000 * 1500 * 0.04 = 60 kPa
SOURCE_PRESSURE = MATERIALS["water"]["rho"] * MATERIALS["water"]["c"] * SOURCE_V0

# Output grid: 0.5 mm spacing
DX_MM = 0.5
AXIAL_EXTENT_MM = 120.0    # x: 0 to 120 mm
LATERAL_EXTENT_MM = 70.0   # y: -35 to 35 mm


def build_bm4_phantom(dx_mm: float = DX_MM) -> tuple[xa.Dataset, xa.Coordinates]:
    """Build the BM4 multi-layer skull phantom on a 3D grid.

    Returns acoustic property maps as an xarray Dataset and the coordinates.
    The grid is 3D but thin in z (3 points) for a quasi-2D simulation.
    """
    nx = int(AXIAL_EXTENT_MM / dx_mm) + 1
    ny = int(LATERAL_EXTENT_MM / dx_mm) + 1
    nz = ny  # full 3D (jwave PML requires sufficient z extent)

    x = np.arange(nx) * dx_mm
    y = np.arange(ny) * dx_mm - LATERAL_EXTENT_MM / 2
    z = np.arange(nz) * dx_mm - (nz // 2) * dx_mm

    coords = {}
    for dim, vals in [("x", x), ("y", y), ("z", z)]:
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        c.attrs["long_name"] = dim.upper()
        coords[dim] = c
    coords = xa.Coordinates(coords)

    shape = (nx, ny, nz)
    sound_speed = np.full(shape, MATERIALS["water"]["c"], dtype=np.float32)
    density = np.full(shape, MATERIALS["water"]["rho"], dtype=np.float32)
    attenuation = np.full(shape, MATERIALS["water"]["alpha"], dtype=np.float32)

    # Fill skull layers
    for offset_mm, thickness_mm, mat_name in LAYERS:
        mat = MATERIALS[mat_name]
        x_start = SKULL_START_MM + offset_mm
        x_end = x_start + thickness_mm
        i_start = int(round(x_start / dx_mm))
        i_end = int(round(x_end / dx_mm))
        sound_speed[i_start:i_end, :, :] = mat["c"]
        density[i_start:i_end, :, :] = mat["rho"]
        attenuation[i_start:i_end, :, :] = mat["alpha"]

    # Fill brain (after skull)
    brain_start = SKULL_START_MM + 10.5  # skin(4) + cortical(1.5) + trabecular(4) + cortical(1)
    i_brain = int(round(brain_start / dx_mm))
    mat = MATERIALS["brain"]
    sound_speed[i_brain:, :, :] = mat["c"]
    density[i_brain:, :, :] = mat["rho"]
    attenuation[i_brain:, :, :] = mat["alpha"]

    params = xa.Dataset(
        {
            "sound_speed": xa.DataArray(sound_speed, coords=coords,
                                        attrs={"units": "m/s", "ref_value": 1500.0}),
            "density": xa.DataArray(density, coords=coords,
                                    attrs={"units": "kg/m^3", "ref_value": 1000.0}),
            "attenuation": xa.DataArray(attenuation, coords=coords,
                                        attrs={"units": "dB/cm/MHz", "ref_value": 0.0}),
        }
    )
    return params, coords


def build_bowl_transducer(freq_hz: float = FREQ_HZ,
                          roc_mm: float = ROC_MM,
                          aperture_mm: float = APERTURE_MM,
                          n_elements: int = 256):
    """Build a focused bowl transducer approximated by point elements on the bowl surface.

    The bowl is a spherical cap with center of curvature at (ROC, 0, 0).
    Elements lie on the concave surface, opening toward +x.
    The rear of the bowl (vertex) is at x=0.
    The geometric focus is at (ROC, 0, 0).

    Half-opening angle: sin(theta_max) = (aperture/2) / ROC
    Element positions:
        x = ROC * (1 - cos(theta))
        y = ROC * sin(theta) * cos(phi)
        z = ROC * sin(theta) * sin(phi)
    """
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer

    roc_m = roc_mm * 1e-3
    aperture_m = aperture_mm * 1e-3
    half_angle = np.arcsin(aperture_m / (2 * roc_m))

    # Fibonacci spiral distribution on spherical cap [0, half_angle]
    elements = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_elements):
        # Map uniform distribution [0,1] to cos(theta) in [cos(half_angle), 1]
        cos_min = np.cos(half_angle)
        cos_theta = cos_min + (1 - cos_min) * (i + 0.5) / n_elements
        theta = np.arccos(cos_theta)
        phi = 2 * np.pi * i / golden_ratio

        # Bowl vertex at x=0, focus at x=ROC
        x = roc_m * (1 - np.cos(theta))
        y = roc_m * np.sin(theta) * np.cos(phi)
        z = roc_m * np.sin(theta) * np.sin(phi)

        elements.append(Element(
            index=i,
            position=np.array([x, y, z]),
            size=np.array([1e-3, 1e-3]),
            units="m",
        ))

    log.info("  Bowl: ROC=%.0f mm, aperture=%.0f mm, half_angle=%.1f deg",
             roc_mm, aperture_mm, np.degrees(half_angle))
    log.info("  Element x range: %.1f to %.1f mm",
             min(e.position[0] for e in elements) * 1e3,
             max(e.position[0] for e in elements) * 1e3)
    log.info("  Element y range: %.1f to %.1f mm",
             min(e.position[1] for e in elements) * 1e3,
             max(e.position[1] for e in elements) * 1e3)

    return Transducer(
        id="itrusst_bowl_sc1",
        name="ITRUSST Bowl SC1",
        elements=elements,
        frequency=freq_hz,
        units="m",
    )


def compute_metrics(p_amp: np.ndarray, p_ref: np.ndarray,
                    dx_mm: float, brain_start_idx: int) -> dict:
    """Compute ITRUSST intercomparison metrics.

    Args:
        p_amp: Our pressure amplitude field (2D: axial x lateral).
        p_ref: Reference pressure amplitude field (same shape).
        dx_mm: Grid spacing in mm.
        brain_start_idx: Axial index where brain region starts.

    Returns:
        Dict of metrics matching computeDifferenceMetrics.m
    """
    # Relative L2 error (past the skull exit plane)
    p1 = p_amp[brain_start_idx:]
    p2 = p_ref[brain_start_idx:]
    l2_error = 100 * np.sqrt(np.sum((p1 - p2) ** 2) / np.sum(p2 ** 2))

    # Relative L-inf error
    linf_error = 100 * np.max(np.abs(p1 - p2)) / np.max(p2)

    # Peak pressure in brain
    peak_ours = np.max(p1)
    peak_ref = np.max(p2)
    peak_diff_pct = 100 * abs(peak_ours - peak_ref) / peak_ref

    # Peak position
    idx_ours = np.unravel_index(np.argmax(p1), p1.shape)
    idx_ref = np.unravel_index(np.argmax(p2), p2.shape)
    pos_diff_mm = dx_mm * np.sqrt(sum((a - b) ** 2 for a, b in zip(idx_ours, idx_ref)))

    # FWHM (-6 dB) along axial direction through peak
    def fwhm_1d(profile, dx):
        half_max = np.max(profile) / 2
        above = profile >= half_max
        if not np.any(above):
            return 0.0
        indices = np.where(above)[0]
        return (indices[-1] - indices[0]) * dx

    axial_profile_ours = p1[:, idx_ours[1]]
    axial_profile_ref = p2[:, idx_ref[1]]
    fwhm_axial_ours = fwhm_1d(axial_profile_ours, dx_mm)
    fwhm_axial_ref = fwhm_1d(axial_profile_ref, dx_mm)

    return {
        "l2_error_pct": l2_error,
        "linf_error_pct": linf_error,
        "peak_pressure_ours_kpa": peak_ours / 1e3,
        "peak_pressure_ref_kpa": peak_ref / 1e3,
        "peak_diff_pct": peak_diff_pct,
        "peak_position_diff_mm": pos_diff_mm,
        "fwhm_axial_ours_mm": fwhm_axial_ours,
        "fwhm_axial_ref_mm": fwhm_axial_ref,
    }


def run_bm4(dx_mm: float = 1.0, ref_path: str | None = None, save_png: str | None = None):
    """Run BM4 benchmark simulation and optionally compare against reference."""
    from openlifu.sim.jwave_if import run_cw_simulation

    log.info("Building BM4 phantom (dx=%.1f mm)...", dx_mm)
    params, coords = build_bm4_phantom(dx_mm=dx_mm)

    # Log layer structure
    x = coords["x"].values
    for var in ("sound_speed", "density", "attenuation"):
        arr = params[var].data[:, params[var].shape[1] // 2, 0]
        log.info("  %s along axis: min=%.0f max=%.0f", var, arr.min(), arr.max())

    log.info("Building bowl transducer (f=%.0f kHz, ROC=%.0f mm, D=%.0f mm)...",
             FREQ_HZ / 1e3, ROC_MM, APERTURE_MM)
    tx = build_bowl_transducer()
    log.info("  %d elements", tx.numelements())

    # Use Helmholtz (frequency-domain) solver for steady-state CW pressure.
    # This directly solves for p_amp — no time-stepping, no memory issues.
    log.info("Running jwave Helmholtz solver...")
    t0 = time.perf_counter()
    ds, raw = run_cw_simulation(
        arr=tx,
        params=params,
        freq=FREQ_HZ,
        amplitude=SOURCE_PRESSURE,
        pml_size=10,
        tol=1e-4,
        maxiter=1000,
    )
    sim_time = time.perf_counter() - t0
    log.info("Simulation complete in %.1fs", sim_time)

    # Central 2D slice (z=0)
    mid_z = ds.p_amp.shape[2] // 2
    p_amp_2d = ds.p_amp.data[:, :, mid_z]

    log.info("Results (central slice):")
    log.info("  p_amp: min=%.1f  max=%.1f Pa  (%.1f kPa)",
             p_amp_2d.min(), p_amp_2d.max(), p_amp_2d.max() / 1e3)

    # Brain region metrics (use p_amp for ITRUSST comparison)
    brain_start_mm = SKULL_START_MM + 10.5
    brain_idx = int(round(brain_start_mm / dx_mm))
    p_brain = p_amp_2d[brain_idx:]
    peak_brain = p_brain.max()
    peak_idx = np.unravel_index(np.argmax(p_brain), p_brain.shape)
    peak_x_mm = (brain_idx + peak_idx[0]) * dx_mm
    peak_y_mm = peak_idx[1] * dx_mm - LATERAL_EXTENT_MM / 2
    log.info("  Brain peak (p_amp): %.1f kPa at (%.1f, %.1f) mm", peak_brain / 1e3, peak_x_mm, peak_y_mm)
    log.info("  Expected focus: (%.1f, 0.0) mm", ROC_MM)

    # Compare against reference if provided
    if ref_path:
        log.info("Loading reference data from %s", ref_path)
        try:
            import h5py
            with h5py.File(ref_path, "r") as f:
                p_ref = np.array(f["p_amp"])
            log.info("  Reference shape: %s", p_ref.shape)
        except Exception:
            import scipy.io
            data = scipy.io.loadmat(ref_path)
            p_ref = data["p_amp"]
            log.info("  Reference shape: %s", p_ref.shape)

        # If our grid differs from reference (0.5 mm), note it
        if p_ref.shape != p_amp_2d.shape:
            log.warning("  Grid mismatch: ours=%s ref=%s (resample needed)", p_max_2d.shape, p_ref.shape)
        else:
            metrics = compute_metrics(p_amp_2d, p_ref, dx_mm, brain_idx)
            log.info("  ITRUSST Metrics:")
            for k, v in metrics.items():
                log.info("    %s: %.2f", k, v)

    # Save PNG
    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            ax = axes[0]
            extent = [0, AXIAL_EXTENT_MM, -LATERAL_EXTENT_MM / 2, LATERAL_EXTENT_MM / 2]
            im = ax.imshow(p_amp_2d.T / 1e3, origin="lower", aspect="auto",
                           extent=extent, cmap="hot")
            ax.set_xlabel("Axial (mm)")
            ax.set_ylabel("Lateral (mm)")
            ax.set_title("BM4-SC1: Pressure Amplitude (kPa)")
            ax.axvline(SKULL_START_MM, color="cyan", ls="--", lw=0.8, label="Skull start")
            ax.axvline(brain_start_mm, color="lime", ls="--", lw=0.8, label="Brain start")
            ax.legend(fontsize=8)
            fig.colorbar(im, ax=ax)

            ax = axes[1]
            c_profile = params["sound_speed"].data[:, params["sound_speed"].shape[1] // 2, 0]
            ax.plot(x, c_profile)
            ax.set_xlabel("Axial (mm)")
            ax.set_ylabel("Sound speed (m/s)")
            ax.set_title("BM4: Tissue layers")
            ax.axvline(SKULL_START_MM, color="cyan", ls="--", lw=0.8)
            ax.axvline(brain_start_mm, color="lime", ls="--", lw=0.8)

            fig.suptitle(f"ITRUSST BM4-SC1 (jwave, dx={dx_mm} mm)")
            fig.tight_layout()
            fig.savefig(save_png, dpi=150)
            log.info("Saved figure to %s", save_png)
        except ImportError:
            log.warning("matplotlib not available")

    return {
        "sim_time_s": sim_time,
        "grid_shape": list(ds.p_amp.shape),
        "dx_mm": dx_mm,
        "peak_pressure_kpa": float(peak_brain / 1e3),
        "peak_x_mm": float(peak_x_mm),
        "peak_y_mm": float(peak_y_mm),
        "p_amp_field_max_kpa": float(p_amp_2d.max() / 1e3),
    }


# ---------------------------------------------------------------------------
# Modal deployment
# ---------------------------------------------------------------------------
try:
    import modal

    app = modal.App("itrusst-bm4")

    bm4_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install(
            "jax[cuda12]>=0.9.0",
            "jaxdf>=0.3.0",
            "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
            "numpy", "xarray[io]", "matplotlib", "pandas", "scipy", "h5py",
            "nibabel", "scikit-image", "vtk", "trimesh", "pydicom",
            "opencv-contrib-python-headless", "crc", "crcmod", "pyserial",
            "nvidia-ml-py", "OpenEXR", "watchdog", "python-socketio[client]",
            "onnxruntime",
        )
        .add_local_dir("src/openlifu", "/root/openlifu_pkg/openlifu", copy=True)
        .add_local_file("benchmarks/itrusst_bm4.py", "/root/openlifu_pkg/benchmarks/itrusst_bm4.py", copy=True)
        .env({"PYTHONPATH": "/root/openlifu_pkg"})
    )

    @app.function(image=bm4_image, gpu="A100", timeout=1800)
    def run_bm4_remote(dx_mm: float = 1.0) -> dict:
        """Run BM4 on A100 GPU."""
        import sys
        sys.path.insert(0, "/root/openlifu_pkg")

        import jax
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")

        return run_bm4(dx_mm=dx_mm, save_png="/tmp/bm4.png")

    @app.local_entrypoint()
    def modal_main(dx_mm: float = 1.0):
        wall_start = time.perf_counter()
        print(f"Running ITRUSST BM4 on A100 (dx={dx_mm} mm)...")
        print("=" * 60)
        result = run_bm4_remote.remote(dx_mm=dx_mm)
        wall_total = time.perf_counter() - wall_start

        print("\n" + "=" * 60)
        print("ITRUSST BM4-SC1 RESULTS")
        print("=" * 60)
        print(f"  Grid:           {result['grid_shape']}")
        print(f"  dx:             {result['dx_mm']} mm")
        print(f"  Sim time:       {result['sim_time_s']:.1f}s")
        print(f"  Total time:     {wall_total:.1f}s")
        print(f"  Peak pressure:  {result['peak_pressure_kpa']:.1f} kPa (in brain)")
        print(f"  Peak position:  ({result['peak_x_mm']:.1f}, {result['peak_y_mm']:.1f}) mm")
        print(f"  Field max:      {result['p_amp_field_max_kpa']:.1f} kPa")

except ImportError:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing in mm (default: 1.0)")
    parser.add_argument("--ref-path", type=str, default=None, help="Path to reference .mat file")
    parser.add_argument("--save-png", type=str, default=None, help="Path to save figure")
    args = parser.parse_args()

    result = run_bm4(dx_mm=args.dx, ref_path=args.ref_path, save_png=args.save_png)
    print("\nResults:", result)
