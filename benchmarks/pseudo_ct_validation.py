"""Pseudo-CT validation: compare MRI-derived HU against real CT.

Validates the chain: T1 MRI → pseudo-CT (HU) → acoustic properties
using datasets with paired MRI and CT scans.

Supported datasets:
  - SynthRAD2023: 180 paired T1/CT brain scans
  - BabelBrain: 5 subjects with varying skull density ratios
  - Any paired MRI/CT in NIfTI format

Usage:
    .venv/bin/python benchmarks/pseudo_ct_validation.py --mri t1.nii.gz --ct ct.nii.gz
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI volume. Returns (data, affine)."""
    import nibabel as nib
    img = nib.load(path)
    return np.asarray(img.get_fdata(), dtype=np.float32), img.affine


def t1_to_pseudo_ct(t1: np.ndarray, method: str = "plymouth") -> np.ndarray:
    """Convert T1w MRI to pseudo-CT Hounsfield units.

    Plymouth method: inverse relationship (dark in T1 = dense bone = high HU).
    """
    # Normalize T1 to [0, 1]
    t1_norm = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-8)

    if method == "plymouth":
        # T1: bone is dark (low signal) → high HU
        hu = 1700.0 * (1.0 - t1_norm) + 300.0
    elif method == "linear":
        # Simple linear: higher T1 → lower HU (bone is dark in T1)
        hu = 2000.0 * (1.0 - t1_norm)
    else:
        raise ValueError(f"Unknown method: {method}")

    return hu


def compute_skull_mask(ct: np.ndarray, hu_min: float = 300, hu_max: float = 3000) -> np.ndarray:
    """Extract skull mask from CT using HU thresholds."""
    return (ct >= hu_min) & (ct <= hu_max)


def hu_to_acoustic_properties(hu: np.ndarray) -> dict[str, np.ndarray]:
    """Convert HU to acoustic properties using empirical relationships.

    References:
        - Schneider et al. (2000): HU → density
        - Mast (2000): density → sound speed, attenuation
        - Connor et al. (2002): combined relationships
    """
    # Schneider: density from HU
    # rho = 1.0 + 0.000790 * HU (for HU > 0, soft tissue/bone)
    rho = np.where(hu > 0,
                   1000.0 * (1.0 + 0.000790 * hu),
                   1000.0)  # kg/m³

    # Mast: sound speed from density (simplified)
    # c = 1500 + 2.5 * (rho - 1000) for soft tissue
    # c = 1500 + 3.7 * (rho - 1000) for bone
    bone_mask = hu > 300
    c = np.where(bone_mask,
                 1500.0 + 3.7 * (rho - 1000.0),
                 1500.0 + 2.5 * (rho - 1000.0))
    c = np.clip(c, 1400, 4500)

    # Attenuation: roughly proportional to density deviation
    alpha = np.where(bone_mask,
                     4.0 * (rho - 1000.0) / 900.0,  # scale to ~4 dB/cm for cortical
                     0.3 * (rho - 1000.0) / 40.0)     # scale to ~0.3 for brain
    alpha = np.clip(alpha, 0.0, 10.0)

    return {"sound_speed": c, "density": rho, "attenuation": alpha}


def compute_validation_metrics(
    pseudo_hu: np.ndarray,
    real_hu: np.ndarray,
    skull_mask: np.ndarray | None = None,
) -> dict:
    """Compare pseudo-CT against real CT.

    Args:
        pseudo_hu: Predicted HU from MRI
        real_hu: Ground-truth HU from CT
        skull_mask: Optional mask to restrict comparison to skull region

    Returns:
        Dict of metrics: MAE, RMSE, correlation, Dice for bone segmentation
    """
    if skull_mask is not None:
        p = pseudo_hu[skull_mask]
        r = real_hu[skull_mask]
    else:
        # Use all non-zero voxels
        mask = (real_hu > -500) & (pseudo_hu > -500)
        p = pseudo_hu[mask]
        r = real_hu[mask]

    if len(p) == 0 or len(r) == 0:
        return {"error": "No valid voxels for comparison"}

    mae = float(np.mean(np.abs(p - r)))
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    corr = float(np.corrcoef(p.ravel(), r.ravel())[0, 1]) if len(p) > 1 else 0.0

    # Dice coefficient for bone segmentation (HU > 300)
    pred_bone = pseudo_hu > 300
    real_bone = real_hu > 300
    intersection = np.sum(pred_bone & real_bone)
    dice = 2 * intersection / (np.sum(pred_bone) + np.sum(real_bone) + 1e-8)

    # Acoustic property error in skull region
    skull = real_hu > 300
    if skull.any():
        pred_props = hu_to_acoustic_properties(pseudo_hu[skull])
        real_props = hu_to_acoustic_properties(real_hu[skull])
        c_error = float(np.mean(np.abs(pred_props["sound_speed"] - real_props["sound_speed"])))
        rho_error = float(np.mean(np.abs(pred_props["density"] - real_props["density"])))
    else:
        c_error = rho_error = 0.0

    return {
        "mae_hu": mae,
        "rmse_hu": rmse,
        "correlation": corr,
        "dice_bone": float(dice),
        "n_voxels": int(len(p)),
        "c_mae_ms": c_error,
        "rho_mae_kgm3": rho_error,
    }


def run_validation(mri_path: str, ct_path: str, method: str = "plymouth",
                   save_png: str | None = None) -> dict:
    """Full validation pipeline: load MRI/CT, predict pseudo-CT, compare."""
    log.info("Loading MRI: %s", mri_path)
    mri, _ = load_nifti(mri_path)
    log.info("Loading CT: %s", ct_path)
    ct, _ = load_nifti(ct_path)

    if mri.shape != ct.shape:
        log.warning("Shape mismatch: MRI=%s, CT=%s — resampling needed", mri.shape, ct.shape)

    log.info("Computing pseudo-CT (method=%s)...", method)
    pseudo = t1_to_pseudo_ct(mri, method=method)

    log.info("Computing validation metrics...")
    metrics = compute_validation_metrics(pseudo, ct)

    log.info("Results:")
    for k, v in metrics.items():
        log.info("  %s: %s", k, v)

    if save_png:
        _plot_comparison(mri, ct, pseudo, metrics, save_png)

    return metrics


def _plot_comparison(mri, ct, pseudo, metrics, save_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = mri.shape[2] // 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    ax = axes[0]
    ax.imshow(mri[:, :, mid].T, origin="lower", cmap="gray")
    ax.set_title("T1 MRI")

    ax = axes[1]
    im = ax.imshow(ct[:, :, mid].T, origin="lower", cmap="bone", vmin=-200, vmax=2000)
    ax.set_title("Real CT (HU)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    im = ax.imshow(pseudo[:, :, mid].T, origin="lower", cmap="bone", vmin=-200, vmax=2000)
    ax.set_title("Pseudo-CT (HU)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[3]
    diff = pseudo[:, :, mid] - ct[:, :, mid]
    im = ax.imshow(diff.T, origin="lower", cmap="RdBu_r", vmin=-500, vmax=500)
    ax.set_title(f"Difference (MAE={metrics['mae_hu']:.0f} HU)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Pseudo-CT Validation (Dice={metrics['dice_bone']:.3f}, r={metrics['correlation']:.3f})")
    fig.tight_layout()
    fig.savefig(save_png, dpi=150)
    log.info("Saved %s", save_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri", required=True, help="Path to T1 MRI NIfTI")
    parser.add_argument("--ct", required=True, help="Path to CT NIfTI")
    parser.add_argument("--method", default="plymouth", choices=["plymouth", "linear"])
    parser.add_argument("--save-png", default=None)
    args = parser.parse_args()
    run_validation(args.mri, args.ct, args.method, args.save_png)
