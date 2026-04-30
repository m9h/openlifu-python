#!/usr/bin/env python
"""End-to-end SimNIBS Ernie -> openlifu jwave replication.

Walks the public SimNIBS Ernie example through ``benchmarks/simnibs_replication.py``
on the local machine. Three stages:

  1. Fetch the SimNIBS ``example-dataset`` (Ernie T1 / T2 NIfTIs and
     pre-segmented charm outputs when present) into ``~/data/simnibs-ernie``.
  2. Locate or generate ``m2m_ernie/tissue_labeling.nii.gz``. If the
     example-dataset already ships the pre-segmented m2m we use it as-is;
     otherwise we run ``charm`` from the user's SimNIBS install. ``charm``
     needs ~25-40 min of CPU time and roughly 8 GB of RAM.
  3. Run ``benchmarks/simnibs_replication.py`` against the m2m and write
     metrics + a peak-pressure NIfTI under ``results/ernie/``.

Usage (defaults are sensible)
-----------------------------
    .venv/bin/python scripts/run_ernie_replication.py

Useful flags
------------
    --dx-mm           grid spacing (default 0.5)
    --frequency-hz    transducer drive (default 500e3)
    --aperture-mm     bowl aperture diameter (default 64, H-115)
    --focal-length-mm bowl focal length (default 52, H-115)
    --target-mm x y z focal-target offset from head centre (default 0 0 0)
    --time-domain     use the time-domain PSTD peak-pressure solver
    --skip-charm      bail out instead of running charm if no m2m is found
    --data-dir        where to stage the Ernie example dataset
                      (default ``~/data/simnibs-ernie``)
    --out-dir         where to write the replication outputs
                      (default ``results/ernie``)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)
EXAMPLE_REPO = "https://github.com/simnibs/example-dataset.git"
TISSUE_LABEL_NAMES = (
    "tissue_labeling.nii.gz",
    "tissue_labeling_upsampled.nii.gz",
    "final_tissues.nii.gz",
)


def fetch_example_dataset(data_dir: Path) -> Path:
    """Clone simnibs/example-dataset (or pull if already cloned)."""
    repo_dir = data_dir / "example-dataset"
    if repo_dir.exists():
        LOG.info("example-dataset present at %s; pulling latest", repo_dir)
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"],
                       check=True)
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("cloning %s into %s", EXAMPLE_REPO, repo_dir)
        subprocess.run(["git", "clone", "--depth", "1", EXAMPLE_REPO,
                        str(repo_dir)], check=True)
    return repo_dir


def find_existing_m2m(repo_dir: Path) -> Path | None:
    """Return a pre-segmented m2m_ernie/ if the example-dataset ships one."""
    candidates = [
        repo_dir / "m2m_ernie",
        repo_dir / "ernie" / "m2m_ernie",
        repo_dir / "Ernie" / "m2m_ernie",
    ]
    for c in candidates:
        if c.is_dir() and any((c / n).is_file() for n in TISSUE_LABEL_NAMES):
            LOG.info("found pre-segmented m2m at %s", c)
            return c
    # Walk one level deep for any m2m_*/tissue_labeling*.nii.gz
    for child in repo_dir.iterdir():
        if not child.is_dir():
            continue
        for n in TISSUE_LABEL_NAMES:
            if (child / n).is_file() and child.name.startswith("m2m_"):
                LOG.info("found pre-segmented m2m at %s", child)
                return child
    return None


def find_t1_t2(repo_dir: Path) -> tuple[Path, Path | None]:
    """Locate Ernie T1 (and optional T2) NIfTI in the example-dataset."""
    t1 = None
    t2 = None
    for path in repo_dir.rglob("*.nii*"):
        name = path.name.lower()
        if t1 is None and name.startswith("ernie_t1") or name == "t1.nii.gz":
            t1 = path
        elif t2 is None and (name.startswith("ernie_t2") or name == "t2.nii.gz"):
            t2 = path
    if t1 is None:
        raise FileNotFoundError(
            f"No Ernie T1 NIfTI found under {repo_dir}. The simnibs/"
            "example-dataset layout may have changed; check the repo."
        )
    return t1, t2


def run_charm(work_dir: Path, t1: Path, t2: Path | None,
              subject_id: str = "ernie") -> Path:
    """Run SimNIBS CHARM in ``work_dir`` and return the resulting m2m dir."""
    if shutil.which("charm") is None:
        raise RuntimeError(
            "SimNIBS `charm` executable not found on PATH. Install SimNIBS "
            "(https://simnibs.github.io/simnibs/build/html/installation/install.html)\n"
            "and ensure 'charm' is on your $PATH, or pass --skip-charm to "
            "process a pre-segmented m2m only."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["charm", subject_id, str(t1)]
    if t2 is not None:
        cmd.append(str(t2))
    LOG.info("running charm in %s: %s", work_dir, " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, check=True)
    m2m = work_dir / f"m2m_{subject_id}"
    if not m2m.is_dir():
        raise RuntimeError(f"charm finished but {m2m} does not exist")
    return m2m


def run_replication(m2m_dir: Path, out_dir: Path, *, dx_mm: float,
                    frequency_hz: float, aperture_mm: float,
                    focal_length_mm: float,
                    target_mm: tuple[float, float, float],
                    time_domain: bool,
                    ppw_target: float,
                    extra_args: list[str] | None = None) -> None:
    """Invoke ``benchmarks/simnibs_replication.py`` against ``m2m_dir``."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "benchmarks" / "simnibs_replication.py"
    cmd = [
        sys.executable, str(script),
        "--m2m", str(m2m_dir),
        "--subject-id", "ernie",
        "--dx-mm", str(dx_mm),
        "--frequency-hz", str(frequency_hz),
        "--aperture-mm", str(aperture_mm),
        "--focal-length-mm", str(focal_length_mm),
        "--target-mm", *(str(v) for v in target_mm),
        "--ppw-target", str(ppw_target),
        "--out-dir", str(out_dir),
    ]
    if time_domain:
        cmd.append("--time-domain")
    if extra_args:
        cmd.extend(extra_args)
    LOG.info("running replication: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / "data" / "simnibs-ernie")
    parser.add_argument("--out-dir", type=Path, default=Path("results/ernie"))
    parser.add_argument("--dx-mm", type=float, default=0.5)
    parser.add_argument("--frequency-hz", type=float, default=500e3)
    parser.add_argument("--aperture-mm", type=float, default=64.0)
    parser.add_argument("--focal-length-mm", type=float, default=52.0)
    parser.add_argument("--target-mm", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--time-domain", action="store_true",
                        help="Use time-domain PSTD peak-pressure solver "
                             "instead of CW Helmholtz")
    parser.add_argument("--ppw-target", type=float, default=12.0)
    parser.add_argument("--skip-charm", action="store_true",
                        help="If no pre-segmented m2m is available, fail "
                             "instead of running charm")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_dir = fetch_example_dataset(args.data_dir)
    m2m = find_existing_m2m(repo_dir)
    if m2m is None:
        if args.skip_charm:
            sys.exit(
                "No pre-segmented m2m found in example-dataset and "
                "--skip-charm was set; aborting."
            )
        t1, t2 = find_t1_t2(repo_dir)
        LOG.info("running charm on %s%s", t1, f" + {t2}" if t2 else "")
        m2m = run_charm(args.data_dir / "charm-out", t1, t2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_replication(
        m2m_dir=m2m, out_dir=args.out_dir,
        dx_mm=args.dx_mm, frequency_hz=args.frequency_hz,
        aperture_mm=args.aperture_mm, focal_length_mm=args.focal_length_mm,
        target_mm=tuple(args.target_mm),
        time_domain=args.time_domain, ppw_target=args.ppw_target,
        extra_args=extra,
    )

    LOG.info("done. results in %s", args.out_dir.resolve())


if __name__ == "__main__":
    main()
