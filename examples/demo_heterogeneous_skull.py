"""
Demo: Heterogeneous skull segmentation + jwave simulation.

Demonstrates the two new components on the feature/heterogeneous-skull-segmentation branch:

  1. HeterogeneousSkullSegmentation — builds spatially-varying acoustic property
     maps from integer tissue labels (scalp/skull/CSF/GM/WM).
  2. jwave_if — differentiable JAX-based acoustic simulation backend that
     consumes those heterogeneous property maps.

The script constructs a synthetic layered-skull phantom, generates the property
maps, runs jwave through the heterogeneous medium (and optionally a water-only
reference), then prints summary metrics and saves an axial pressure slice as PNG.

Usage:
    uv run python examples/demo_heterogeneous_skull.py          # heterogeneous
    uv run python examples/demo_heterogeneous_skull.py --water   # water-only comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xa

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Build a synthetic layered-skull phantom (label array)
# ---------------------------------------------------------------------------

def make_layered_skull_labels(shape: tuple[int, int, int], spacing_mm: float = 1.0) -> np.ndarray:
    """Create a 3-D integer label array representing concentric tissue layers.

    The phantom is oriented so that z increases from the transducer face into the
    brain.  Layers (in z):
        0: water coupling (z < 5 mm)
        1: scalp          (5–9 mm)
        2: skull           (9–16 mm)
        3: CSF             (16–18 mm)
        4: gray matter     (18–24 mm)
        5: white matter    (z >= 24 mm)

    Returns an int array with shape *shape*.
    """
    nz = shape[2]
    z_mm = np.arange(nz) * spacing_mm
    labels = np.zeros(shape, dtype=int)

    boundaries = [
        (5.0, 1),   # scalp starts
        (9.0, 2),   # skull starts
        (16.0, 3),  # CSF starts
        (18.0, 4),  # gray matter starts
        (24.0, 5),  # white matter starts
    ]
    for z_thresh, label in boundaries:
        labels[:, :, z_mm >= z_thresh] = label

    return labels


# ---------------------------------------------------------------------------
# 2. Build simulation grid coordinates
# ---------------------------------------------------------------------------

def make_sim_coords(shape, spacing_mm=1.0):
    """Create xarray Coordinates matching the phantom grid."""
    coords = {}
    for i, dim in enumerate(("x", "y", "z")):
        n = shape[i]
        vals = np.arange(n) * spacing_mm - (n // 2) * spacing_mm
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        c.attrs["long_name"] = dim.upper()
        coords[dim] = c
    return xa.Coordinates(coords)


# ---------------------------------------------------------------------------
# 3. Load the example transducer from the test database
# ---------------------------------------------------------------------------

def load_example_transducer():
    from openlifu.xdc.transducer import Transducer
    tx_json = Path(__file__).resolve().parents[1] / "tests/resources/example_db/transducers/example_transducer/example_transducer.json"
    with open(tx_json) as f:
        d = json.load(f)
    return Transducer.from_dict(d)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--water", action="store_true", help="Run with homogeneous water instead of skull layers")
    parser.add_argument("--save-png", type=str, default=None, help="Path to save axial pressure slice PNG")
    args = parser.parse_args()

    # -- Phantom --
    spacing_mm = 1.0
    shape = (61, 61, 65)  # match default SimSetup extents roughly
    labels = make_layered_skull_labels(shape, spacing_mm)
    coords = make_sim_coords(shape, spacing_mm)

    log.info("Phantom shape: %s  spacing: %.1f mm", shape, spacing_mm)
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        log.info("  label %d: %d voxels (%.1f%%)", u, c, 100 * c / labels.size)

    # -- Segmentation → acoustic property maps --
    # Import directly from submodules to avoid the heavy openlifu.__init__ chain
    from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation
    from openlifu.seg.seg_methods.uniform import UniformWater

    if args.water:
        log.info("Using UNIFORM WATER segmentation (homogeneous reference)")
        seg = UniformWater()
        params = seg.ref_params(coords)
    else:
        log.info("Using HETEROGENEOUS SKULL segmentation")
        seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
        volume = xa.DataArray(np.zeros(shape), coords=coords)  # dummy volume (labels override)
        params = seg.seg_params(volume)

    log.info("Acoustic property map variables: %s", list(params.data_vars))
    for var in params.data_vars:
        arr = params[var].data
        log.info("  %s: min=%.1f  max=%.1f  ref=%.1f %s",
                 var, arr.min(), arr.max(),
                 params[var].attrs.get("ref_value", float("nan")),
                 params[var].attrs.get("units", ""))

    # -- jwave simulation --
    try:
        from openlifu.sim.jwave_if import run_simulation as jwave_run_simulation
    except ImportError:
        log.error("jwave/jaxdf is not installed. Install with:  uv add jwave jaxdf")
        sys.exit(1)

    tx = load_example_transducer()
    freq = tx.frequency
    log.info("Transducer: %s  elements: %d  freq: %.1f kHz", tx.name, tx.numelements(), freq / 1e3)

    log.info("Starting jwave simulation (this may take a minute on first run due to JIT)...")
    ds, raw = jwave_run_simulation(
        arr=tx,
        params=params,
        freq=freq,
        cycles=10,
        cfl=0.3,
        pml_size=10,
    )

    # -- Results summary --
    log.info("Simulation complete.")
    for var in ds.data_vars:
        d = ds[var].data
        log.info("  %s: min=%.4g  max=%.4g  mean=%.4g %s",
                 var, d.min(), d.max(), d.mean(), ds[var].attrs.get("units", ""))

    # -- Optional PNG --
    if args.save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            mid_y = shape[1] // 2

            # Pressure map
            ax = axes[0]
            im = ax.imshow(ds["p_max"].data[:, mid_y, :].T, origin="lower", aspect="auto", cmap="hot")
            ax.set_title("Peak Positive Pressure (Pa)")
            ax.set_xlabel("x (voxels)")
            ax.set_ylabel("z (voxels)")
            fig.colorbar(im, ax=ax)

            # Intensity map
            ax = axes[1]
            im = ax.imshow(ds["intensity"].data[:, mid_y, :].T, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title("Intensity (W/cm²)")
            ax.set_xlabel("x (voxels)")
            fig.colorbar(im, ax=ax)

            # Sound speed (to show heterogeneity)
            ax = axes[2]
            im = ax.imshow(params["sound_speed"].data[:, mid_y, :].T, origin="lower", aspect="auto", cmap="gray")
            ax.set_title("Sound Speed (m/s)")
            ax.set_xlabel("x (voxels)")
            fig.colorbar(im, ax=ax)

            fig.suptitle("Heterogeneous Skull" if not args.water else "Homogeneous Water")
            fig.tight_layout()
            fig.savefig(args.save_png, dpi=150)
            log.info("Saved figure to %s", args.save_png)
        except ImportError:
            log.warning("matplotlib not available — skipping PNG output")

    print("\nDone.")


if __name__ == "__main__":
    main()
