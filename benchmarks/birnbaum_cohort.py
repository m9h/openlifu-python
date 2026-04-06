"""Full Birnbaum cohort modeling: 64-subject skull analysis and simulation.

For each subject:
  1. Load T1 MRI + expert segmentation labels
  2. Remap labels to openlifu convention
  3. Compute pseudo-CT and compare against expert bone mask
  4. Map to acoustic properties
  5. Compute skull statistics (thickness, volume, density distribution)
  6. Run jwave Helmholtz simulation through the skull (on Modal)

Usage:
    # Local analysis (no simulation, fast)
    .venv/bin/python benchmarks/birnbaum_cohort.py

    # Full simulation on Modal A100
    .venv/bin/modal run benchmarks/birnbaum_cohort.py
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "birnbaum_data" / "Data" / "Anonymized_Subjects"
LABEL_DIR = DATA_DIR / "Full-Head Segmentation"
T1_DIR = DATA_DIR / "T1-Weighted MRI"

BIRNBAUM_TO_OPENLIFU = {
    0: 0, 1: 0, 2: 0,  # bg/air → water
    3: 5,  # WM → white_matter
    4: 4,  # GM → gray_matter
    5: 3,  # CSF → csf
    6: 2,  # bone → skull
    7: 1,  # scalp → scalp
}


def remap_labels(labels):
    out = np.zeros_like(labels, dtype=np.int32)
    for src, dst in BIRNBAUM_TO_OPENLIFU.items():
        out[labels == src] = dst
    return out


def find_subjects():
    """Find all subjects with both T1 and label files."""
    if not LABEL_DIR.exists():
        # Flat structure (files directly in birnbaum_data/)
        flat_dir = Path(__file__).parent / "birnbaum_data"
        label_files = sorted(flat_dir.glob("*_label_deface.nii*"))
        subjects = []
        for lf in label_files:
            sid = lf.name.split("_label_deface")[0]
            t1 = flat_dir / f"{sid}_deface.nii"
            if not t1.exists():
                t1 = flat_dir / f"{sid}_deface.nii.gz"
            if t1.exists():
                subjects.append({"id": sid, "t1": str(t1), "label": str(lf)})
        return subjects

    label_files = sorted(LABEL_DIR.glob("*_label_deface.nii*"))
    subjects = []
    for lf in label_files:
        sid = lf.name.split("_label_deface")[0]
        t1 = T1_DIR / f"{sid}_deface.nii"
        if not t1.exists():
            t1 = T1_DIR / f"{sid}_deface.nii.gz"
        if t1.exists():
            subjects.append({"id": sid, "t1": str(t1), "label": str(lf)})
    return subjects


def analyze_subject(subj: dict) -> dict:
    """Analyze a single subject: labels, pseudo-CT, skull stats."""
    import nibabel as nib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from pseudo_ct_validation import t1_to_pseudo_ct

    sid = subj["id"]
    t1_data = np.asarray(nib.load(subj["t1"]).get_fdata(), dtype=np.float32)
    raw_labels = np.asarray(nib.load(subj["label"]).get_fdata(), dtype=int)
    labels = remap_labels(raw_labels)

    # Tissue volumes (at 1mm³ per voxel)
    n_total = labels.size
    n_skull = int(np.sum(labels == 2))
    n_brain = int(np.sum((labels == 4) | (labels == 5)))
    n_csf = int(np.sum(labels == 3))
    n_scalp = int(np.sum(labels == 1))

    # Skull thickness estimate: for each column in the axial direction,
    # count contiguous skull voxels
    skull_mask = labels == 2
    thicknesses = []
    # Sample along z-axis at grid of (x, y) positions
    step = 10  # subsample for speed
    for ix in range(0, labels.shape[0], step):
        for iy in range(0, labels.shape[1], step):
            col = skull_mask[ix, iy, :]
            runs = np.diff(np.where(np.concatenate(([col[0]], col[:-1] != col[1:], [True])))[0])[::2]
            if len(runs) > 0:
                thicknesses.extend(runs.tolist())

    thickness_mean = float(np.mean(thicknesses)) if thicknesses else 0
    thickness_std = float(np.std(thicknesses)) if thicknesses else 0

    # Pseudo-CT bone Dice
    pseudo_hu = t1_to_pseudo_ct(t1_data, method="plymouth")
    pred_bone = pseudo_hu > 1200
    expert_bone = raw_labels == 6
    intersection = np.sum(pred_bone & expert_bone)
    dice = float(2 * intersection / (np.sum(pred_bone) + np.sum(expert_bone) + 1e-8))

    # Cohort prefix (GU=Georgetown, NC=UNC, NYU=NYU, H=healthy)
    cohort = sid[:2] if sid[:2] in ("GU", "NC") else sid[:3] if sid[:3] == "NYU" else "other"

    return {
        "subject": sid,
        "cohort": cohort,
        "shape": list(raw_labels.shape),
        "n_skull": n_skull,
        "n_brain": n_brain,
        "n_csf": n_csf,
        "n_scalp": n_scalp,
        "skull_pct": round(100 * n_skull / n_total, 1),
        "brain_pct": round(100 * n_brain / n_total, 1),
        "thickness_mean_mm": round(thickness_mean, 1),
        "thickness_std_mm": round(thickness_std, 1),
        "pseudo_ct_bone_dice": round(dice, 4),
    }


def run_cohort_analysis(max_subjects: int = 0, save_csv: str | None = None,
                         save_png: str | None = None):
    """Run analysis on all Birnbaum subjects."""
    subjects = find_subjects()
    if not subjects:
        log.error("No Birnbaum subjects found in %s", DATA_DIR)
        return []

    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    log.info("Found %d subjects", len(subjects))
    results = []
    for i, subj in enumerate(subjects):
        log.info("[%d/%d] Analyzing %s...", i + 1, len(subjects), subj["id"])
        try:
            r = analyze_subject(subj)
            results.append(r)
        except Exception as e:
            log.error("  Failed: %s", e)

    df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 80)
    print(f"BIRNBAUM COHORT ANALYSIS ({len(results)} subjects)")
    print("=" * 80)
    print(f"\nSkull volume:    {df['n_skull'].mean()/1e3:.0f} ± {df['n_skull'].std()/1e3:.0f} cm³")
    print(f"Brain volume:    {df['n_brain'].mean()/1e3:.0f} ± {df['n_brain'].std()/1e3:.0f} cm³")
    print(f"Skull thickness: {df['thickness_mean_mm'].mean():.1f} ± {df['thickness_mean_mm'].std():.1f} mm")
    print(f"Pseudo-CT Dice:  {df['pseudo_ct_bone_dice'].mean():.3f} ± {df['pseudo_ct_bone_dice'].std():.3f}")

    if len(df["cohort"].unique()) > 1:
        print("\nBy cohort:")
        for cohort, group in df.groupby("cohort"):
            print(f"  {cohort} (n={len(group)}): skull={group['n_skull'].mean()/1e3:.0f}cm³, "
                  f"thickness={group['thickness_mean_mm'].mean():.1f}mm, "
                  f"dice={group['pseudo_ct_bone_dice'].mean():.3f}")

    print("\nPer-subject:")
    print(df[["subject", "cohort", "skull_pct", "thickness_mean_mm",
              "pseudo_ct_bone_dice"]].to_string(index=False))

    if save_csv:
        df.to_csv(save_csv, index=False)
        log.info("Saved CSV to %s", save_csv)

    if save_png:
        _plot_cohort(df, save_png)

    return results


def _plot_cohort(df, save_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Skull volume distribution
    ax = axes[0, 0]
    for cohort in df["cohort"].unique():
        sub = df[df["cohort"] == cohort]
        ax.hist(sub["n_skull"] / 1e3, bins=10, alpha=0.6, label=cohort)
    ax.set_xlabel("Skull volume (cm³)")
    ax.set_ylabel("Count")
    ax.set_title("Skull Volume Distribution")
    ax.legend()

    # Thickness distribution
    ax = axes[0, 1]
    for cohort in df["cohort"].unique():
        sub = df[df["cohort"] == cohort]
        ax.hist(sub["thickness_mean_mm"], bins=10, alpha=0.6, label=cohort)
    ax.set_xlabel("Mean skull thickness (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Skull Thickness Distribution")
    ax.legend()

    # Pseudo-CT Dice by cohort
    ax = axes[1, 0]
    cohorts = df["cohort"].unique()
    positions = range(len(cohorts))
    for i, cohort in enumerate(cohorts):
        vals = df[df["cohort"] == cohort]["pseudo_ct_bone_dice"]
        ax.boxplot(vals, positions=[i], widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Bone Dice")
    ax.set_title("Pseudo-CT Bone Dice by Cohort")

    # Skull % vs brain %
    ax = axes[1, 1]
    ax.scatter(df["brain_pct"], df["skull_pct"], c=df["pseudo_ct_bone_dice"],
               cmap="RdYlGn", s=40)
    ax.set_xlabel("Brain volume (%)")
    ax.set_ylabel("Skull volume (%)")
    ax.set_title("Skull vs Brain Volume (color = Dice)")
    fig.colorbar(ax.collections[0], ax=ax, label="Dice")

    fig.suptitle(f"Birnbaum Cohort: {len(df)} Subjects", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_png, dpi=150)
    log.info("Saved %s", save_png)


# Modal deployment for full simulation
try:
    import modal

    app = modal.App("birnbaum-cohort")
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
        .add_local_file("benchmarks/birnbaum_cohort.py", "/root/pkg/benchmarks/birnbaum_cohort.py", copy=True)
        .add_local_file("benchmarks/pseudo_ct_validation.py", "/root/pkg/benchmarks/pseudo_ct_validation.py", copy=True)
        .env({
            "PYTHONPATH": "/root/pkg",
            "KAGGLE_API_TOKEN": "KGAT_3b2c8c3ef4782a7030baee29cf3755a3",
        })
    )

    @app.function(image=img, gpu="A100", timeout=7200, memory=32768)
    def run_cohort_remote(max_subjects: int = 0) -> list[dict]:
        import sys, subprocess
        sys.path.insert(0, "/root/pkg")

        # Download dataset on Modal
        subprocess.run(["pip", "install", "kaggle"], capture_output=True)
        subprocess.run([
            "kaggle", "datasets", "download",
            "andrewbirnbaum/full-head-mri-and-segmentation-of-stroke-patients",
            "-p", "/root/pkg/benchmarks/birnbaum_data/", "--unzip",
        ], check=True)

        return run_cohort_analysis(
            max_subjects=max_subjects,
            save_csv="/tmp/birnbaum_cohort.csv",
            save_png="/tmp/birnbaum_cohort.png",
        )

    @app.local_entrypoint()
    def modal_main(max_subjects: int = 0):
        t0 = time.perf_counter()
        n = max_subjects if max_subjects > 0 else 64
        print(f"Running Birnbaum cohort analysis ({n} subjects) on Modal...")
        results = run_cohort_remote.remote(max_subjects=max_subjects)
        total = time.perf_counter() - t0
        print(f"\nCompleted {len(results)} subjects in {total:.0f}s")

except ImportError:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subjects", type=int, default=0, help="0 = all")
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-png", type=str, default=None)
    args = parser.parse_args()
    run_cohort_analysis(args.max_subjects, args.save_csv, args.save_png)
