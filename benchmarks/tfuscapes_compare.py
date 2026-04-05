"""Compare jwave Helmholtz solver against TFUScapes k-Wave pressure fields.

Downloads a TFUScapes sample, segments the CT into tissue labels,
runs jwave CW simulation with the same transducer geometry, and
computes comparison metrics against the k-Wave reference.

Usage:
    .venv/bin/modal run benchmarks/tfuscapes_compare.py
"""
from __future__ import annotations

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

FREQ_HZ = 500e3
SOURCE_PRESSURE = 60000.0


def load_tfuscapes_sample(sample_path: str) -> dict:
    """Load a TFUScapes .npz file."""
    d = np.load(sample_path)
    return {"ct": d["ct"], "pmap": d["pmap"], "tr_coords": d["tr_coords"]}


def ct_to_labels(ct: np.ndarray) -> np.ndarray:
    """Segment pseudo-CT volume into tissue labels (openlifu convention).

    Simple threshold-based classification:
        0: water/air (HU < 50)
        1: scalp (50-200 HU)
        2: skull (> 700 HU)
        3: CSF (~200-300 HU, low intensity near ventricles)
        4: gray matter (300-500 HU)
        5: white matter (500-700 HU)
    """
    labels = np.zeros(ct.shape, dtype=np.int32)
    labels[(ct >= 50) & (ct < 200)] = 1    # scalp
    labels[(ct >= 200) & (ct < 350)] = 3   # CSF
    labels[(ct >= 350) & (ct < 500)] = 4   # gray matter
    labels[(ct >= 500) & (ct < 700)] = 5   # white matter
    labels[ct >= 700] = 2                   # skull
    return labels


def build_transducer_from_coords(tr_coords: np.ndarray, dx_mm: float = 1.0):
    """Build an openlifu Transducer from TFUScapes grid index coordinates."""
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer

    # TFUScapes tr_coords are grid indices; convert to meters
    positions_m = tr_coords * dx_mm * 1e-3

    # Subsample if too many elements (16K is excessive for Helmholtz)
    if len(positions_m) > 512:
        step = len(positions_m) // 512
        positions_m = positions_m[::step]

    # Estimate element area from spacing
    n_elem = len(positions_m)
    elem_side = 2e-3  # 2mm elements

    elements = [
        Element(index=i, position=pos, size=np.array([elem_side, elem_side]), units="m")
        for i, pos in enumerate(positions_m)
    ]
    return Transducer(id="tfuscapes", elements=elements, frequency=FREQ_HZ, units="m")


def compare_fields(p_jwave: np.ndarray, p_kwave: np.ndarray) -> dict:
    """Compute comparison metrics between jwave and k-Wave pressure fields."""
    # Normalize to same scale for structural comparison
    mask = p_kwave > p_kwave.max() * 0.01  # ignore background noise

    # L2 relative error
    l2 = 100 * np.sqrt(np.sum((p_jwave[mask] - p_kwave[mask])**2) /
                        np.sum(p_kwave[mask]**2))

    # Peak pressure comparison
    peak_jwave = p_jwave.max()
    peak_kwave = p_kwave.max()
    peak_diff = 100 * abs(peak_jwave - peak_kwave) / peak_kwave

    # Peak position
    idx_jwave = np.unravel_index(np.argmax(p_jwave), p_jwave.shape)
    idx_kwave = np.unravel_index(np.argmax(p_kwave), p_kwave.shape)
    pos_diff_mm = np.sqrt(sum((a - b)**2 for a, b in zip(idx_jwave, idx_kwave)))

    # Correlation
    corr = np.corrcoef(p_jwave[mask].ravel(), p_kwave[mask].ravel())[0, 1]

    return {
        "l2_error_pct": float(l2),
        "peak_jwave": float(peak_jwave),
        "peak_kwave": float(peak_kwave),
        "peak_diff_pct": float(peak_diff),
        "pos_diff_voxels": float(pos_diff_mm),
        "correlation": float(corr),
        "focal_idx_jwave": list(idx_jwave),
        "focal_idx_kwave": list(idx_kwave),
    }


def run_comparison(sample_path: str, save_png: str | None = None) -> dict:
    """Run jwave simulation on a TFUScapes sample and compare to k-Wave."""
    import xarray as xa
    from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation
    from openlifu.sim.jwave_if import run_cw_simulation

    log.info("Loading TFUScapes sample: %s", sample_path)
    data = load_tfuscapes_sample(sample_path)
    ct, pmap, tr_coords = data["ct"], data["pmap"], data["tr_coords"]
    log.info("  CT: %s, pmap: %s, transducer: %d elements",
             ct.shape, pmap.shape, len(tr_coords))

    # Segment CT to tissue labels
    labels = ct_to_labels(ct)
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        log.info("  Label %d: %d voxels (%.1f%%)", u, c, 100*c/labels.size)

    # Build simulation grid (assume 1mm spacing, 256³)
    dx_mm = 1.0
    shape = ct.shape
    coords = {}
    for i, dim in enumerate(("x", "y", "z")):
        vals = np.arange(shape[i]) * dx_mm
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        coords[dim] = c
    coords = xa.Coordinates(coords)

    # Build acoustic property maps
    seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
    volume = xa.DataArray(np.zeros(shape), coords=coords)
    params = seg.seg_params(volume)

    # Build transducer
    tx = build_transducer_from_coords(tr_coords, dx_mm=dx_mm)
    log.info("  Transducer: %d elements (subsampled)", tx.numelements())

    # Run jwave Helmholtz simulation
    log.info("Running jwave Helmholtz solver on 256³ grid...")
    t0 = time.perf_counter()
    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=FREQ_HZ, amplitude=SOURCE_PRESSURE,
        pml_size=15, tol=1e-3, maxiter=500,
    )
    sim_time = time.perf_counter() - t0
    log.info("  Simulation complete in %.1fs", sim_time)

    # Compare
    p_jwave = ds.p_amp.data
    metrics = compare_fields(p_jwave, pmap)
    metrics["sim_time_s"] = sim_time
    metrics["grid_shape"] = list(shape)
    metrics["n_elements"] = tx.numelements()

    log.info("Comparison metrics:")
    for k, v in metrics.items():
        log.info("  %s: %s", k, v)

    if save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mid = shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        im = ax.imshow(pmap[:, :, mid].T, origin="lower", cmap="hot", aspect="auto")
        ax.set_title(f"k-Wave Reference (peak={metrics['peak_kwave']/1e3:.0f} kPa)")
        fig.colorbar(im, ax=ax)

        ax = axes[1]
        im = ax.imshow(p_jwave[:, :, mid].T, origin="lower", cmap="hot", aspect="auto")
        ax.set_title(f"jwave Helmholtz (peak={metrics['peak_jwave']/1e3:.0f} kPa)")
        fig.colorbar(im, ax=ax)

        ax = axes[2]
        im = ax.imshow(ct[:, :, mid].T, origin="lower", cmap="gray", aspect="auto")
        ax.set_title("CT Volume")
        fig.colorbar(im, ax=ax)

        fig.suptitle(f"TFUScapes: jwave vs k-Wave (corr={metrics['correlation']:.3f})")
        fig.tight_layout()
        fig.savefig(save_png, dpi=150)
        log.info("Saved %s", save_png)

    return metrics


# Modal deployment
try:
    import modal

    app = modal.App("tfuscapes-compare")
    img = (
        modal.Image.debian_slim(python_version="3.12").apt_install("git")
        .pip_install(
            "jax[cuda12]>=0.9.0", "jaxdf>=0.3.0",
            "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
            "numpy", "xarray[io]", "matplotlib", "pandas", "scipy", "h5py",
            "nibabel", "scikit-image", "vtk", "trimesh", "pydicom",
            "opencv-contrib-python-headless", "crc", "crcmod", "pyserial",
            "nvidia-ml-py", "OpenEXR", "watchdog", "python-socketio[client]",
            "onnxruntime", "huggingface_hub",
        )
        .add_local_dir("src/openlifu", "/root/pkg/openlifu", copy=True)
        .add_local_file("benchmarks/tfuscapes_compare.py", "/root/pkg/benchmarks/tfuscapes_compare.py", copy=True)
        .env({"PYTHONPATH": "/root/pkg"})
    )

    @app.function(image=img, gpu="A100", timeout=3600)
    def run_comparison_remote(n_samples: int = 1) -> list[dict]:
        import sys
        sys.path.insert(0, "/root/pkg")
        import jax
        print(f"JAX: {jax.default_backend()}, {jax.devices()}")

        from huggingface_hub import hf_hub_download
        results = []
        subjects = ["A00028185", "A00028352", "A00028637"][:n_samples]
        for subj in subjects:
            path = hf_hub_download(
                repo_id="vinkle-srivastav/TFUScapes",
                filename=f"data/{subj}/exp_0.npz",
                repo_type="dataset",
            )
            r = run_comparison(path, save_png=f"/tmp/tfuscapes_{subj}.png")
            r["subject"] = subj
            results.append(r)
        return results

    @app.local_entrypoint()
    def main(n_samples: int = 1):
        t0 = time.perf_counter()
        print(f"Running TFUScapes comparison ({n_samples} samples) on A100...")
        results = run_comparison_remote.remote(n_samples=n_samples)
        total = time.perf_counter() - t0

        print(f"\n{'='*70}")
        print("TFUScapes: jwave vs k-Wave Comparison")
        print(f"{'='*70}")
        for r in results:
            print(f"\n  Subject: {r.get('subject', '?')}")
            print(f"  L2 error:    {r['l2_error_pct']:.1f}%")
            print(f"  Peak jwave:  {r['peak_jwave']/1e3:.0f} kPa")
            print(f"  Peak kwave:  {r['peak_kwave']/1e3:.0f} kPa")
            print(f"  Peak diff:   {r['peak_diff_pct']:.1f}%")
            print(f"  Pos diff:    {r['pos_diff_voxels']:.1f} voxels")
            print(f"  Correlation: {r['correlation']:.4f}")
            print(f"  Sim time:    {r['sim_time_s']:.1f}s")
        print(f"\nTotal: {total:.0f}s")
except ImportError:
    pass
