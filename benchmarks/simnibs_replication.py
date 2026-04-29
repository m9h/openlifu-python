"""SimNIBS / PRESTUS-style transcranial focused-ultrasound replication.

Runs a CW Helmholtz simulation through a SimNIBS CHARM head model with a
focused-bowl transducer, reporting brain-region focal pressure metrics.
Designed to reproduce the simulation arm of the SimNIBS 4 / PRESTUS
workflow (Kasinadhuni et al., Neumayer et al.) using openlifu's existing
``run_cw_simulation`` rather than k-Wave.

Usage (local):
    .venv/bin/python benchmarks/simnibs_replication.py \\
        --m2m /path/to/m2m_<subject>/ \\
        --target-mm 0 0 0 \\
        --frequency 500e3 --aperture-mm 64 --focal-length-mm 52

Usage (Modal, for cloud-burst when local GPU is busy):
    MODAL_GPU=A100-80GB modal run benchmarks/simnibs_replication.py::run_remote -- \\
        --m2m /path/to/m2m_<subject>/

Output:
    results/simnibs_<subject>_dx<N>mm.json   metrics
    results/simnibs_<subject>_p_amp.nii.gz   peak pressure NIfTI (axial)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# Sonic Concepts H-115 defaults — typical neuromodulation transducer used
# in PRESTUS reference papers. Override on the CLI for other geometries.
DEFAULT_FREQ_HZ = 500e3
DEFAULT_APERTURE_M = 0.064
DEFAULT_FOCAL_LENGTH_M = 0.052
DEFAULT_SOURCE_PA = 60_000.0
DEFAULT_N_BOWL_POINTS = 256


def _bbox_around_skull(labels: np.ndarray, pad_voxels: int = 8) -> tuple[slice, slice, slice]:
    """Tight bbox around skull voxels (label 2) plus padding.

    Cropping the simulation domain to the skull + a small padding keeps
    grid sizes manageable at fine resolutions; the rest of the head
    contributes nothing to a transducer→brain focal calculation.
    """
    skull = labels == 2
    if not skull.any():
        return tuple(slice(None) for _ in range(3))
    coords = np.argwhere(skull)
    lo = np.maximum(coords.min(axis=0) - pad_voxels, 0)
    hi = np.minimum(coords.max(axis=0) + pad_voxels + 1, labels.shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def simulate_subject(
    m2m_path: str | Path,
    subject_id: str,
    dx_mm: float = 0.5,
    frequency_hz: float = DEFAULT_FREQ_HZ,
    aperture_m: float = DEFAULT_APERTURE_M,
    focal_length_m: float = DEFAULT_FOCAL_LENGTH_M,
    source_pa: float = DEFAULT_SOURCE_PA,
    n_bowl_points: int = DEFAULT_N_BOWL_POINTS,
    target_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    crop_to_skull: bool = True,
    tol: float = 1e-3,
    maxiter: int = 300,
) -> dict:
    """Run a CW Helmholtz forward simulation through one SimNIBS head model.

    Args:
        m2m_path: Path to a SimNIBS CHARM ``m2m_<subject>/`` directory or
            directly to a ``tissue_labeling*.nii.gz``.
        subject_id: Tag used in result dict and output filenames.
        dx_mm: Target voxel spacing. CHARM data is native 1 mm; this fn
            nearest-neighbour resamples categorical labels to dx_mm.
        frequency_hz: Transducer drive frequency.
        aperture_m, focal_length_m: Bowl geometry.
        source_pa: Source pressure amplitude (uniform across bowl).
        n_bowl_points: Number of point sources along the bowl surface
            (Fibonacci-sampled).
        target_mm: Focal target offset from the head-model centre, mm.
        crop_to_skull: Crop the simulation domain to the skull bbox + 8
            voxels of padding. Strongly recommended at dx<1 mm.
        tol, maxiter: GMRES solver settings.

    Returns:
        Dict with subject id, grid shape, voxel counts, focal-pressure
        metrics, sim_time_s, and the cropped pressure field (numpy
        array, kPa) for downstream NIfTI export.
    """
    import scipy.ndimage as ndi
    import xarray as xa

    from openlifu.seg.simnibs import (
        load_simnibs_segmentation, remap_charm_to_openlifu,
    )
    from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation
    from openlifu.sim.jwave_if import run_cw_simulation
    from openlifu.xdc.bowl import bowl_transducer_3d
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer

    log.info("Loading SimNIBS segmentation from %s", m2m_path)
    charm = load_simnibs_segmentation(Path(m2m_path))
    labels = remap_charm_to_openlifu(charm)

    # Native CHARM is 1 mm. Nearest-neighbour resample for categorical labels.
    if abs(dx_mm - 1.0) > 1e-6:
        if dx_mm > 1.0:
            step = int(round(dx_mm))
            labels = labels[::step, ::step, ::step]
        else:
            factor = 1.0 / dx_mm
            labels = ndi.zoom(labels, factor, order=0).astype(np.int32)

    if crop_to_skull:
        bbox = _bbox_around_skull(labels)
        labels = labels[bbox]
        log.info("Cropped to skull bbox: %s", labels.shape)

    shape = labels.shape
    coords = xa.Coordinates({
        dim: xa.DataArray(np.arange(shape[i], dtype=float) * dx_mm,
                          dims=[dim], attrs={"units": "mm"})
        for i, dim in enumerate(("x", "y", "z"))
    })

    seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
    volume = xa.DataArray(np.zeros(shape), coords=coords)
    params = seg.seg_params(volume)

    n_skull = int((labels == 2).sum())
    n_brain = int(((labels == 4) | (labels == 5)).sum())

    # Focal target: voxel-space centre + user offset (in mm).
    centre_mm = np.array([s * dx_mm * 0.5 for s in shape])
    focal_mm = centre_mm + np.array(target_mm)
    focal_m = focal_mm * 1e-3

    # Transducer points down the +x axis from the skull-anterior side.
    # If the skull mask is present, place the bowl 5 mm anterior to the
    # min-x extent of the skull (consistent with birnbaum_simulation.py).
    skull_mask = labels == 2
    if skull_mask.any():
        skull_min_x = int(np.argwhere(skull_mask)[:, 0].min())
        bowl_anchor_m = np.array([
            (skull_min_x * dx_mm * 1e-3) - 5e-3,
            focal_m[1],
            focal_m[2],
        ])
    else:
        bowl_anchor_m = focal_m - np.array([focal_length_m, 0, 0])

    # Direction vector pointing from bowl anchor toward the focal point.
    direction = focal_m - bowl_anchor_m
    direction = direction / np.linalg.norm(direction)

    bowl_points = np.asarray(bowl_transducer_3d(
        focal_length=focal_length_m,
        aperture_diameter=aperture_m,
        focal_point=tuple(focal_m),
        direction=tuple(direction),
        n_points=n_bowl_points,
    ))

    elements = [
        Element(index=i, position=p,
                size=np.array([dx_mm * 1e-3, dx_mm * 1e-3]), units="m")
        for i, p in enumerate(bowl_points)
    ]
    tx = Transducer(id=f"simnibs_{subject_id}", elements=elements,
                    frequency=frequency_hz, units="m")

    log.info("Running jwave Helmholtz solver (f=%.1f kHz, dx=%.2f mm, "
             "shape=%s, %d bowl elements)",
             frequency_hz / 1e3, dx_mm, shape, len(elements))
    t0 = time.perf_counter()
    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=frequency_hz, amplitude=source_pa,
        pml_size=8, tol=tol, maxiter=maxiter,
    )
    sim_time = time.perf_counter() - t0

    p_amp = np.asarray(ds.p_amp.data)
    brain_mask = (labels == 4) | (labels == 5)
    p_brain = p_amp[brain_mask] if brain_mask.any() else p_amp.ravel()

    return {
        "subject": subject_id,
        "shape": list(shape),
        "dx_mm": dx_mm,
        "frequency_hz": frequency_hz,
        "aperture_m": aperture_m,
        "focal_length_m": focal_length_m,
        "source_pa": source_pa,
        "n_skull": n_skull,
        "n_brain": n_brain,
        "p_amp_max_pa": float(p_amp.max()),
        "p_brain_max_pa": float(p_brain.max()) if len(p_brain) else 0.0,
        "p_brain_mean_pa": float(p_brain.mean()) if len(p_brain) else 0.0,
        "sim_time_s": sim_time,
        "p_amp_pa": p_amp,  # full field for NIfTI export
    }


def _save_pressure_nifti(p_amp_pa: np.ndarray, dx_mm: float, out_path: Path) -> None:
    """Save the peak (CW amplitude) pressure field as a NIfTI for viewers."""
    import nibabel as nib
    affine = np.eye(4)
    affine[:3, :3] *= dx_mm
    img = nib.Nifti1Image((p_amp_pa / 1e3).astype(np.float32), affine)  # kPa
    img.header.set_xyzt_units("mm")
    nib.save(img, str(out_path))


def main() -> None:
    p = argparse.ArgumentParser(description="SimNIBS / PRESTUS replication via openlifu CW jwave.")
    p.add_argument("--m2m", required=True, help="Path to m2m_<subject>/ directory or tissue_labeling.nii.gz")
    p.add_argument("--subject-id", default=None, help="Tag for results (default: m2m dirname)")
    p.add_argument("--dx-mm", type=float, default=0.5)
    p.add_argument("--frequency-hz", type=float, default=DEFAULT_FREQ_HZ)
    p.add_argument("--aperture-mm", type=float, default=DEFAULT_APERTURE_M * 1e3)
    p.add_argument("--focal-length-mm", type=float, default=DEFAULT_FOCAL_LENGTH_M * 1e3)
    p.add_argument("--source-pa", type=float, default=DEFAULT_SOURCE_PA)
    p.add_argument("--target-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--no-crop", action="store_true", help="Skip skull-bbox crop (full head domain)")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    m2m_path = Path(args.m2m)
    sid = args.subject_id or (m2m_path.name.replace("m2m_", "") if m2m_path.is_dir() else m2m_path.stem)

    r = simulate_subject(
        m2m_path=m2m_path, subject_id=sid, dx_mm=args.dx_mm,
        frequency_hz=args.frequency_hz,
        aperture_m=args.aperture_mm * 1e-3,
        focal_length_m=args.focal_length_mm * 1e-3,
        source_pa=args.source_pa, target_mm=tuple(args.target_mm),
        crop_to_skull=not args.no_crop, tol=args.tol, maxiter=args.maxiter,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"simnibs_{sid}_dx{args.dx_mm}mm"

    p_amp = r.pop("p_amp_pa")
    _save_pressure_nifti(p_amp, args.dx_mm, out_dir / f"{base}_p_amp.nii.gz")
    (out_dir / f"{base}.json").write_text(json.dumps(r, indent=2))

    print(f"\n{'='*60}\nSimNIBS replication: {sid}\n{'='*60}")
    print(f"  shape={r['shape']}, dx={args.dx_mm} mm")
    print(f"  brain p_max = {r['p_brain_max_pa']/1e3:.1f} kPa")
    print(f"  brain p_mean = {r['p_brain_mean_pa']/1e3:.2f} kPa")
    print(f"  domain p_max = {r['p_amp_max_pa']/1e3:.1f} kPa")
    print(f"  sim_time = {r['sim_time_s']:.1f} s")
    print(f"\nSaved {out_dir / (base + '.json')} and {out_dir / (base + '_p_amp.nii.gz')}")


if __name__ == "__main__":
    main()
