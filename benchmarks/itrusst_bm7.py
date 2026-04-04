"""ITRUSST Benchmark BM7: Truncated realistic skull mesh.

Loads MNI152-derived skull STL meshes, rasterizes to a 3D label grid,
and runs the Helmholtz solver through the realistic skull geometry.

Skull data: benchmarks/itrusst_data/intercomparison/skull-stl/
  - skull_outer.stl (175K triangles)
  - skull_inner.stl (92K triangles)
  - affine_transform_v1.mat (visual cortex target orientation)

Usage:
    .venv/bin/python benchmarks/itrusst_bm7.py --dx 2.0
    .venv/bin/modal run benchmarks/itrusst_bm7.py
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import xarray as xa

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

FREQ_HZ = 500e3
SOURCE_PRESSURE = 60000.0  # Pa (rho*c*v0)

STL_DIR = Path(__file__).parent / "itrusst_data" / "intercomparison" / "skull-stl"


def load_skull_meshes(stl_dir: str | None = None):
    """Load skull_outer.stl and skull_inner.stl as trimesh objects."""
    import trimesh
    stl_dir = Path(stl_dir) if stl_dir else STL_DIR
    outer = trimesh.load(stl_dir / "skull_outer.stl")
    inner = trimesh.load(stl_dir / "skull_inner.stl")
    log.info("Loaded outer mesh: %d vertices, %d faces", len(outer.vertices), len(outer.faces))
    log.info("Loaded inner mesh: %d vertices, %d faces", len(inner.vertices), len(inner.faces))
    return outer, inner


def load_affine_transform(stl_dir: str | None = None, target: str = "v1"):
    """Load the affine transform for transducer positioning."""
    import scipy.io as sio
    stl_dir = Path(stl_dir) if stl_dir else STL_DIR
    mat = sio.loadmat(stl_dir / f"affine_transform_{target}.mat")
    return mat["affine_transform"]


def rasterize_skull(outer_mesh, inner_mesh, dx_mm: float = 1.0,
                    domain_mm: tuple | None = None):
    """Rasterize skull STL meshes to a 3D integer label array.

    Labels:
        0 = water (outside outer mesh)
        2 = skull (between outer and inner meshes) — cortical bone
        4 = brain (inside inner mesh) — gray matter

    Args:
        outer_mesh: trimesh of the outer skull surface
        inner_mesh: trimesh of the inner skull surface
        dx_mm: Grid spacing in mm
        domain_mm: Optional (x_range, y_range, z_range) as ((xmin,xmax), ...)
            If None, computed from mesh bounds with padding.

    Returns:
        labels: 3D integer array
        coords: xarray Coordinates
    """
    # Determine domain from mesh bounds
    if domain_mm is None:
        bounds = outer_mesh.bounds  # (2, 3): min, max
        padding = 10.0  # mm
        mins = bounds[0] - padding
        maxs = bounds[1] + padding
    else:
        mins = np.array([d[0] for d in domain_mm])
        maxs = np.array([d[1] for d in domain_mm])

    # Build grid
    axes = []
    for i, dim in enumerate(("x", "y", "z")):
        vals = np.arange(mins[i], maxs[i] + dx_mm, dx_mm)
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        c.attrs["long_name"] = dim.upper()
        axes.append((dim, c, vals))

    coords = xa.Coordinates({dim: c for dim, c, _ in axes})
    shape = tuple(len(v) for _, _, v in axes)

    # Rasterize using trimesh containment checks
    log.info("Rasterizing skull to %s grid (dx=%.1f mm)...", shape, dx_mm)

    # Build query points (subsample if too large)
    xx, yy, zz = np.meshgrid(*[v for _, _, v in axes], indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    log.info("  Checking %d points against outer mesh...", len(points))
    inside_outer = outer_mesh.contains(points)
    log.info("  Checking %d points against inner mesh...", len(points))
    inside_inner = inner_mesh.contains(points)

    labels = np.zeros(shape, dtype=np.int32)
    labels_flat = labels.ravel()
    labels_flat[inside_outer & ~inside_inner] = 2  # skull
    labels_flat[inside_inner] = 4  # brain
    labels = labels_flat.reshape(shape)

    n_water = np.sum(labels == 0)
    n_skull = np.sum(labels == 2)
    n_brain = np.sum(labels == 4)
    log.info("  Labels: water=%d, skull=%d, brain=%d", n_water, n_skull, n_brain)

    return labels, coords


def labels_to_params(labels, coords):
    """Convert integer labels to acoustic property maps."""
    PROPS = {
        0: {"c": 1500.0, "rho": 1000.0, "alpha": 0.0},     # water
        2: {"c": 2800.0, "rho": 1850.0, "alpha": 4.0},     # cortical bone
        4: {"c": 1560.0, "rho": 1040.0, "alpha": 0.3},     # brain (GM)
    }
    shape = labels.shape
    c = np.zeros(shape, dtype=np.float32)
    rho = np.zeros(shape, dtype=np.float32)
    alpha = np.zeros(shape, dtype=np.float32)
    for label, props in PROPS.items():
        mask = labels == label
        c[mask] = props["c"]
        rho[mask] = props["rho"]
        alpha[mask] = props["alpha"]

    return xa.Dataset({
        "sound_speed": xa.DataArray(c, coords=coords, attrs={"units": "m/s", "ref_value": 1500.0}),
        "density": xa.DataArray(rho, coords=coords, attrs={"units": "kg/m^3", "ref_value": 1000.0}),
        "attenuation": xa.DataArray(alpha, coords=coords, attrs={"units": "dB/cm/MHz", "ref_value": 0.0}),
    })


def build_bowl_transducer(n_elements=256):
    from benchmarks.itrusst_bm4 import build_bowl_transducer as _build
    return _build(freq_hz=FREQ_HZ, n_elements=n_elements)


def run_bm7(dx_mm=2.0, save_png=None):
    """Run BM7 benchmark with realistic skull geometry."""
    from openlifu.sim.jwave_if import run_cw_simulation

    outer, inner = load_skull_meshes()

    labels, coords = rasterize_skull(outer, inner, dx_mm=dx_mm)
    params = labels_to_params(labels, coords)

    tx = build_bowl_transducer()
    log.info("Transducer: %d elements at %.0f kHz", tx.numelements(), FREQ_HZ / 1e3)

    t0 = time.perf_counter()
    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=FREQ_HZ, amplitude=SOURCE_PRESSURE,
        pml_size=10, tol=1e-3, maxiter=500,
    )
    sim_time = time.perf_counter() - t0

    mid_z = ds.p_amp.shape[2] // 2
    p_amp = ds.p_amp.data[:, :, mid_z]
    peak = p_amp.max()
    peak_idx = np.unravel_index(np.argmax(p_amp), p_amp.shape)

    log.info("BM7 Results (dx=%.1f mm):", dx_mm)
    log.info("  Grid: %s", list(ds.p_amp.shape))
    log.info("  Sim time: %.1fs", sim_time)
    log.info("  p_amp max: %.1f kPa", peak / 1e3)

    return {
        "sim_time_s": sim_time,
        "grid_shape": list(ds.p_amp.shape),
        "peak_kpa": float(peak / 1e3),
        "n_skull_voxels": int(np.sum(labels == 2)),
        "n_brain_voxels": int(np.sum(labels == 4)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--save-png", type=str, default=None)
    args = parser.parse_args()
    run_bm7(dx_mm=args.dx, save_png=args.save_png)
