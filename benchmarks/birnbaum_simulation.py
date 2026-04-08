"""Birnbaum per-patient jwave simulation through expert-segmented skulls.

Runs the Helmholtz solver through real patient skull anatomy from the
Birnbaum dataset (64 subjects, expert 7-class segmentation).

Usage:
    .venv/bin/modal run benchmarks/birnbaum_simulation.py
    .venv/bin/modal run benchmarks/birnbaum_simulation.py --n-subjects 10
"""
from __future__ import annotations

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

BIRNBAUM_TO_OPENLIFU = {0: 0, 1: 0, 2: 0, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
FREQ_HZ = 500e3
SOURCE_PRESSURE = 60000.0


def simulate_subject(label_path: str, subject_id: str, dx_mm: float = 2.0) -> dict:
    """Run CW simulation through one Birnbaum subject's skull."""
    import nibabel as nib
    import xarray as xa
    from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation
    from openlifu.sim.jwave_if import run_cw_simulation
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer

    raw = np.asarray(nib.load(label_path).get_fdata(), dtype=int)
    labels = np.zeros_like(raw, dtype=np.int32)
    for src, dst in BIRNBAUM_TO_OPENLIFU.items():
        labels[raw == src] = dst

    # Downsample if needed
    if dx_mm > 1.0:
        step = int(dx_mm)
        labels = labels[::step, ::step, ::step]

    shape = labels.shape
    coords = xa.Coordinates({
        dim: xa.DataArray(np.arange(shape[i], dtype=float) * dx_mm,
                          dims=[dim], attrs={"units": "mm"})
        for i, dim in enumerate(("x", "y", "z"))
    })

    seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
    volume = xa.DataArray(np.zeros(shape), coords=coords)
    params = seg.seg_params(volume)

    n_skull = int(np.sum(labels == 2))
    n_brain = int(np.sum((labels == 4) | (labels == 5)))

    # Place transducer at the skull surface center
    skull_mask = labels == 2
    if skull_mask.any():
        skull_coords = np.argwhere(skull_mask) * dx_mm * 1e-3
        tx_pos = skull_coords.mean(axis=0)
        tx_pos[0] = skull_coords[:, 0].min() - 5e-3  # 5mm before skull
    else:
        tx_pos = np.array([0., 0., 0.])

    elements = [
        Element(index=i, position=tx_pos + np.array([0., y * 1e-3, 0.]),
                size=np.array([3e-3, 3e-3]), units="m")
        for i, y in enumerate(np.linspace(-15, 15, 16))
    ]
    tx = Transducer(id=f"birnbaum_{subject_id}", elements=elements,
                    frequency=FREQ_HZ, units="m")

    t0 = time.perf_counter()
    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=FREQ_HZ, amplitude=SOURCE_PRESSURE,
        pml_size=8, tol=1e-3, maxiter=300,
    )
    sim_time = time.perf_counter() - t0

    # Focal metrics in brain region
    brain_mask = (labels == 4) | (labels == 5)
    p_brain = ds.p_amp.data[brain_mask] if brain_mask.any() else ds.p_amp.data.ravel()
    p_brain_max = float(p_brain.max()) if len(p_brain) > 0 else 0
    p_brain_mean = float(p_brain.mean()) if len(p_brain) > 0 else 0

    return {
        "subject": subject_id,
        "shape": list(shape),
        "dx_mm": dx_mm,
        "n_skull": n_skull,
        "n_brain": n_brain,
        "p_amp_max_pa": float(ds.p_amp.data.max()),
        "p_brain_max_pa": p_brain_max,
        "p_brain_mean_pa": p_brain_mean,
        "sim_time_s": sim_time,
    }


try:
    import modal

    app = modal.App("birnbaum-sim")
    img = (
        modal.Image.debian_slim(python_version="3.12").apt_install("git")
        .pip_install(
            "jax[cuda12]>=0.9.0", "jaxdf>=0.3.0",
            "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
            "numpy", "xarray[io]", "matplotlib", "pandas", "scipy", "h5py",
            "nibabel", "scikit-image", "vtk", "trimesh", "pydicom",
            "opencv-contrib-python-headless", "crc", "crcmod", "pyserial",
            "nvidia-ml-py", "OpenEXR", "watchdog", "python-socketio[client]",
            "onnxruntime", "kaggle",
        )
        .add_local_dir("src/openlifu", "/root/pkg/openlifu", copy=True)
        .add_local_file("benchmarks/birnbaum_simulation.py",
                        "/root/pkg/benchmarks/birnbaum_simulation.py", copy=True)
        .env({
            "PYTHONPATH": "/root/pkg",
            "KAGGLE_API_TOKEN": "KGAT_3b2c8c3ef4782a7030baee29cf3755a3",
        })
    )

    @app.function(image=img, gpu="A100", timeout=7200, memory=32768)
    def run_subjects(n_subjects: int = 5, dx_mm: float = 2.0) -> list[dict]:
        import sys, subprocess, glob
        sys.path.insert(0, "/root/pkg")
        import jax
        print(f"JAX: {jax.default_backend()}, {jax.devices()}")

        # Download dataset
        subprocess.run(["kaggle", "datasets", "download",
                        "andrewbirnbaum/full-head-mri-and-segmentation-of-stroke-patients",
                        "-p", "/tmp/birnbaum/", "--unzip"], check=True, capture_output=True)

        label_dir = "/tmp/birnbaum/Data/Anonymized_Subjects/Full-Head Segmentation"
        label_files = sorted(glob.glob(f"{label_dir}/*_label_deface.nii"))[:n_subjects]

        results = []
        for i, lf in enumerate(label_files):
            import os
            sid = os.path.basename(lf).split("_label_deface")[0]
            print(f"\n[{i+1}/{len(label_files)}] Simulating {sid}...")
            try:
                r = simulate_subject(lf, sid, dx_mm=dx_mm)
                results.append(r)
                print(f"  p_brain_max={r['p_brain_max_pa']/1e3:.1f} kPa, "
                      f"sim_time={r['sim_time_s']:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")

        print(f"\n{'='*60}")
        print(f"BIRNBAUM PER-PATIENT SIMULATION ({len(results)} subjects)")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['subject']}: brain_max={r['p_brain_max_pa']/1e3:.1f} kPa, "
                  f"time={r['sim_time_s']:.1f}s")

        return results

    @app.local_entrypoint()
    def main(n_subjects: int = 5, dx_mm: float = 2.0):
        t0 = time.perf_counter()
        print(f"Running Birnbaum per-patient simulation ({n_subjects} subjects, dx={dx_mm}mm)...")
        results = run_subjects.remote(n_subjects=n_subjects, dx_mm=dx_mm)
        total = time.perf_counter() - t0
        print(f"\nCompleted {len(results)} subjects in {total:.0f}s")
        for r in results:
            print(f"  {r['subject']}: brain_max={r['p_brain_max_pa']/1e3:.1f} kPa")

except ImportError:
    pass
