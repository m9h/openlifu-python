"""Run openlifu jwave simulations on Modal with A100 GPU.

Deploys the heterogeneous skull demo to a cloud A100 and streams results back.

Prerequisites:
    uv pip install modal
    modal setup  # one-time auth

Usage:
    modal run benchmarks/modal_jwave.py                          # heterogeneous skull demo
    modal run benchmarks/modal_jwave.py --water                  # water-only comparison
    modal run benchmarks/modal_jwave.py --save-png skull.png     # save pressure slice
    modal run benchmarks/modal_jwave.py --test-suite             # run pytest on GPU
"""

import time

import modal

app = modal.App("openlifu-jwave")

# ---------------------------------------------------------------------------
# Container image: JAX with CUDA 12, jwave from git main, openlifu editable.
# ---------------------------------------------------------------------------
openlifu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]>=0.9.0",
        "jaxdf>=0.3.0",
        "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
        "numpy",
        "xarray[io]",
        "matplotlib",
        "pandas",
        "scipy",
        "h5py",
        "nibabel",
        "scikit-image",
        "vtk",
        "trimesh",
        "pydicom",
        "pytest>=7.0",
        "pytest-mock",
        "opencv-contrib-python-headless",
        "crc",
        "crcmod",
        "pyserial",
        "nvidia-ml-py",
        "OpenEXR",
        "watchdog",
        "python-socketio[client]",
        "onnxruntime",
    )
    .add_local_dir("src/openlifu", "/root/openlifu_pkg/openlifu", copy=True)
    .add_local_dir("tests", "/root/openlifu_pkg/tests", copy=True)
    .add_local_dir("examples", "/root/openlifu_pkg/examples", copy=True)
    .env({"PYTHONPATH": "/root/openlifu_pkg"})
)


@app.function(
    image=openlifu_image,
    gpu="A100",
    timeout=1800,
)
def run_demo(water: bool = False, save_png: str | None = None) -> dict:
    """Run heterogeneous skull jwave simulation on A100 GPU."""
    import sys

    sys.path.insert(0, "/root/openlifu_pkg")

    import jax

    backend = jax.default_backend()
    devices = str(jax.devices())
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")

    # GPU smoke test
    import jax.numpy as jnp

    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print(f"GPU smoke test passed: {y.shape}")

    t_start = time.perf_counter()

    import numpy as np
    import xarray as xa

    from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation
    from openlifu.seg.seg_methods.uniform import UniformWater
    from openlifu.sim.jwave_if import run_simulation

    # Build phantom
    spacing_mm = 1.0
    shape = (61, 61, 65)

    coords = {}
    for i, dim in enumerate(("x", "y", "z")):
        n = shape[i]
        vals = np.arange(n) * spacing_mm - (n // 2) * spacing_mm
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        c.attrs["long_name"] = dim.upper()
        coords[dim] = c
    coords = xa.Coordinates(coords)

    if water:
        print("Using UNIFORM WATER (homogeneous)")
        seg = UniformWater()
        params = seg.ref_params(coords)
    else:
        print("Using HETEROGENEOUS SKULL segmentation")
        labels = _make_layered_skull_labels(shape, spacing_mm)
        seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
        volume = xa.DataArray(np.zeros(shape), coords=coords)
        params = seg.seg_params(volume)

    for var in params.data_vars:
        arr = params[var].data
        print(f"  {var}: min={arr.min():.1f}  max={arr.max():.1f}")

    # Load example transducer
    import json
    from pathlib import Path

    from openlifu.xdc.transducer import Transducer

    tx_json = Path("/root/openlifu_pkg/tests/resources/example_db/transducers/example_transducer/example_transducer.json")
    with open(tx_json) as f:
        d = json.load(f)
    tx = Transducer.from_dict(d)
    print(f"Transducer: {tx.name}  elements: {tx.numelements()}  freq: {tx.frequency / 1e3:.1f} kHz")

    # Run simulation
    print("Starting jwave simulation...")
    sim_start = time.perf_counter()
    ds, raw = run_simulation(
        arr=tx,
        params=params,
        freq=tx.frequency,
        cycles=10,
        cfl=0.3,
        pml_size=10,
    )
    sim_time = time.perf_counter() - sim_start
    total_time = time.perf_counter() - t_start

    results = {
        "backend": backend,
        "devices": devices,
        "mode": "water" if water else "heterogeneous",
        "grid_shape": shape,
        "sim_time_s": sim_time,
        "total_time_s": total_time,
    }

    for var in ds.data_vars:
        d = ds[var].data
        results[f"{var}_min"] = float(d.min())
        results[f"{var}_max"] = float(d.max())
        results[f"{var}_mean"] = float(d.mean())
        print(f"  {var}: min={d.min():.4g}  max={d.max():.4g}  mean={d.mean():.4g}")

    print(f"\nSimulation time: {sim_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")

    # Optional PNG
    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            mid_y = shape[1] // 2

            ax = axes[0]
            im = ax.imshow(ds["p_max"].data[:, mid_y, :].T, origin="lower", aspect="auto", cmap="hot")
            ax.set_title("Peak Positive Pressure (Pa)")
            fig.colorbar(im, ax=ax)

            ax = axes[1]
            im = ax.imshow(ds["intensity"].data[:, mid_y, :].T, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title("Intensity (W/cm²)")
            fig.colorbar(im, ax=ax)

            ax = axes[2]
            im = ax.imshow(params["sound_speed"].data[:, mid_y, :].T, origin="lower", aspect="auto", cmap="gray")
            ax.set_title("Sound Speed (m/s)")
            fig.colorbar(im, ax=ax)

            fig.suptitle("Heterogeneous Skull" if not water else "Homogeneous Water")
            fig.tight_layout()
            fig.savefig(save_png, dpi=150)
            print(f"Saved figure to {save_png}")
        except ImportError:
            print("matplotlib not available for PNG output")

    return results


@app.function(
    image=openlifu_image,
    gpu="A100",
    timeout=900,
)
def run_test_suite() -> dict:
    """Run openlifu pytest suite on A100 GPU."""
    import subprocess
    import sys

    sys.path.insert(0, "/root/openlifu_pkg")

    import jax

    backend = jax.default_backend()
    devices = str(jax.devices())
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")

    t_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
        cwd="/root/openlifu_pkg",
        capture_output=True,
        text=True,
        timeout=600,
    )
    test_time = time.perf_counter() - t_start

    print("=" * 70)
    print("PYTEST OUTPUT:")
    print("=" * 70)
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.stderr:
        print(result.stderr[-2000:])

    return {
        "backend": backend,
        "devices": devices,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "test_wall_time_s": test_time,
        "stdout_tail": result.stdout[-3000:],
    }


def _make_layered_skull_labels(shape, spacing_mm=1.0):
    """Synthetic layered skull phantom."""
    import numpy as np

    nz = shape[2]
    z_mm = np.arange(nz) * spacing_mm
    labels = np.zeros(shape, dtype=int)
    for z_thresh, label in [(5.0, 1), (9.0, 2), (16.0, 3), (18.0, 4), (24.0, 5)]:
        labels[:, :, z_mm >= z_thresh] = label
    return labels


@app.local_entrypoint()
def main(
    water: bool = False,
    save_png: str = "",
    test_suite: bool = False,
):
    """Local entrypoint — dispatches to remote A100 function.

    Args:
        water: Run with homogeneous water instead of skull layers.
        save_png: Path to save axial pressure slice PNG (empty = skip).
        test_suite: Run pytest on GPU instead of demo.
    """
    wall_start = time.perf_counter()

    if test_suite:
        print("Running openlifu test suite on A100 GPU...")
        print("=" * 60)
        result = run_test_suite.remote()
        wall_total = time.perf_counter() - wall_start

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"  Backend:    {result['backend']}")
        print(f"  Devices:    {result['devices']}")
        print(f"  Passed:     {result['passed']}")
        print(f"  Test time:  {result['test_wall_time_s']:.1f}s (remote)")
        print(f"  Total time: {wall_total:.1f}s (including Modal overhead)")

    else:
        mode = "water-only" if water else "heterogeneous skull"
        print(f"Running jwave {mode} simulation on A100 GPU...")
        print("=" * 60)
        result = run_demo.remote(
            water=water,
            save_png=save_png or None,
        )
        wall_total = time.perf_counter() - wall_start

        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"  Backend:    {result['backend']}")
        print(f"  Devices:    {result['devices']}")
        print(f"  Mode:       {result['mode']}")
        print(f"  Grid:       {result['grid_shape']}")
        print(f"  Sim time:   {result['sim_time_s']:.1f}s")
        print(f"  Total:      {wall_total:.1f}s (including Modal overhead)")
        print()
        for key in ("p_max", "p_min", "intensity"):
            print(f"  {key}: min={result[f'{key}_min']:.4g}  max={result[f'{key}_max']:.4g}  mean={result[f'{key}_mean']:.4g}")
