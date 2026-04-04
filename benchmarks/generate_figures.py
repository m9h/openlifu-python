"""Generate paper figures on Modal A100 and download them.

Produces:
  fig1_bm4_pressure.png     — BM4 pressure amplitude field (axial slice)
  fig2_bm4_axial.png        — Axial pressure profile vs reference models
  fig3_tissue_layers.png    — Heterogeneous tissue property maps
  fig4_het_vs_water.png     — Heterogeneous skull vs water-only comparison
  fig5_solver_comparison.png — Helmholtz vs time-domain results

Usage:
    .venv/bin/modal run benchmarks/generate_figures.py
"""
import time

import modal

app = modal.App("tfus-figures")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]>=0.9.0", "jaxdf>=0.3.0",
        "jwave @ git+https://github.com/ucl-bug/jwave.git@main",
        "numpy", "xarray[io]", "matplotlib", "pandas", "scipy", "h5py",
        "nibabel", "scikit-image", "vtk", "trimesh", "pydicom",
        "opencv-contrib-python-headless", "crc", "crcmod", "pyserial",
        "nvidia-ml-py", "OpenEXR", "watchdog", "python-socketio[client]",
        "onnxruntime", "requests",
    )
    .add_local_dir("src/openlifu", "/root/openlifu_pkg/openlifu", copy=True)
    .add_local_file("benchmarks/itrusst_bm4.py", "/root/openlifu_pkg/benchmarks/itrusst_bm4.py", copy=True)
    .add_local_dir("tests/resources", "/root/openlifu_pkg/tests/resources", copy=True)
    .env({"PYTHONPATH": "/root/openlifu_pkg"})
)

vol = modal.Volume.from_name("tfus-figures", create_if_missing=True)


@app.function(image=image, gpu="A100", timeout=3600, volumes={"/figures": vol})
def generate_all_figures():
    import sys
    sys.path.insert(0, "/root/openlifu_pkg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import xarray as xa
    import jax

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    from openlifu.sim.jwave_if import run_cw_simulation, run_simulation as run_td_simulation
    from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation
    from openlifu.seg.seg_methods.uniform import UniformWater

    # Import BM4 helpers
    sys.path.insert(0, "/root/openlifu_pkg/benchmarks")
    from itrusst_bm4 import (
        build_bm4_phantom, build_bowl_transducer,
        FREQ_HZ, SOURCE_PRESSURE, SKULL_START_MM,
        MATERIALS, LAYERS, DX_MM, AXIAL_EXTENT_MM, LATERAL_EXTENT_MM,
    )

    outdir = "/figures"

    # ── Shared setup ──
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'figure.facecolor': 'white', 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    # =========================================================================
    # Figure 1: BM4 pressure amplitude field
    # =========================================================================
    print("Generating Figure 1: BM4 pressure field...")
    dx = 0.5
    params, coords = build_bm4_phantom(dx_mm=dx)
    tx = build_bowl_transducer()

    ds_cw, _ = run_cw_simulation(
        arr=tx, params=params, freq=FREQ_HZ, amplitude=SOURCE_PRESSURE,
        pml_size=10, tol=1e-4, maxiter=1000,
    )

    mid_z = ds_cw.p_amp.shape[2] // 2
    p_amp = ds_cw.p_amp.data[:, :, mid_z] / 1e3  # kPa
    x_mm = coords["x"].values
    y_mm = coords["y"].values
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    brain_start = SKULL_START_MM + 10.5

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    im = ax.imshow(p_amp.T, origin="lower", aspect="auto", extent=extent,
                   cmap="hot", vmin=0, vmax=np.percentile(p_amp, 99.5))
    ax.axvline(SKULL_START_MM, color="cyan", ls="--", lw=1, alpha=0.7, label="Skull start")
    ax.axvline(brain_start, color="lime", ls="--", lw=1, alpha=0.7, label="Brain start")
    ax.axvline(64, color="white", ls=":", lw=1, alpha=0.7, label="Geometric focus")
    ax.set_xlabel("Axial distance (mm)")
    ax.set_ylabel("Lateral distance (mm)")
    ax.set_title("ITRUSST BM4-SC1: Pressure Amplitude (Helmholtz, dx=0.5 mm)")
    ax.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(im, ax=ax, label="Pressure amplitude (kPa)")
    fig.savefig(f"{outdir}/fig1_bm4_pressure.png")
    plt.close()
    print(f"  Saved fig1_bm4_pressure.png")

    # =========================================================================
    # Figure 2: Axial pressure profile vs reference
    # =========================================================================
    print("Generating Figure 2: Axial profile comparison...")

    # Download reference data
    import requests, zipfile, os, io
    import scipy.io as sio

    url = "https://zenodo.org/records/6020543/files/INTERCOMPARISON-PH1-DATA-V1.0.zip?download=1"
    zippath = "/tmp/itrusst.zip"
    if not os.path.exists(zippath):
        print("  Downloading reference data from Zenodo...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zippath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8*1024*1024):
                    f.write(chunk)

    z = zipfile.ZipFile(zippath)
    ref_models = {}
    for name in z.namelist():
        if "BM4" not in name or "SC1" not in name or name.endswith("/"):
            continue
        model = name.split("/")[0]
        with z.open(name) as zf:
            data = io.BytesIO(zf.read())
        try:
            mat = sio.loadmat(data)
            p = np.squeeze(mat["p_amp"])
        except Exception:
            import h5py
            data.seek(0)
            tmppath = f"/tmp/{os.path.basename(name)}"
            with open(tmppath, "wb") as f:
                f.write(data.read())
            with h5py.File(tmppath, "r") as hf:
                p = np.squeeze(np.array(hf["p_amp"]))
        ref_models[model] = p

    # Our axial profile through center
    mid_y = p_amp.shape[1] // 2
    our_axial = p_amp[:, mid_y]  # already in kPa

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x_mm, our_axial, "r-", lw=2, label="This work (Helmholtz)")

    # Reference profiles (correct axis orientation: [241,141] = axial-first)
    ref_x = np.linspace(0, 120, 241)
    colors = {"JWAVE": "b", "KWAVE": "g", "STRIDE": "purple",
              "SALVUS": "orange", "OPTIMUS": "brown", "SIM4LIFE": "gray"}
    for model, p_ref in sorted(ref_models.items()):
        if model not in colors:
            continue
        if p_ref.shape[0] == 241:
            mid = p_ref.shape[1] // 2
            profile = p_ref[:, mid] / 1e3
        elif p_ref.shape[1] == 241:
            mid = p_ref.shape[0] // 2
            profile = p_ref[mid, :] / 1e3
        else:
            continue
        ax.plot(ref_x, profile, ls="--", lw=1.2, color=colors[model],
                alpha=0.8, label=f"{model} (ref)")

    ax.axvspan(SKULL_START_MM, brain_start, alpha=0.15, color="gray", label="Skull layers")
    ax.axvline(64, color="k", ls=":", lw=1, alpha=0.5, label="Geometric focus")
    ax.set_xlabel("Axial distance (mm)")
    ax.set_ylabel("Pressure amplitude (kPa)")
    ax.set_title("BM4-SC1: Axial Pressure Profile Comparison")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=0)
    fig.savefig(f"{outdir}/fig2_bm4_axial.png")
    plt.close()
    print(f"  Saved fig2_bm4_axial.png")

    # =========================================================================
    # Figure 3: Tissue property maps
    # =========================================================================
    print("Generating Figure 3: Tissue property maps...")

    c_slice = params["sound_speed"].data[:, params["sound_speed"].shape[1]//2, 0]
    rho_slice = params["density"].data[:, params["density"].shape[1]//2, 0]
    alpha_slice = params["attenuation"].data[:, params["attenuation"].shape[1]//2, 0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(x_mm, c_slice, "b-", lw=1.5)
    ax.set_ylabel("Sound speed (m/s)")
    ax.set_title("BM4 Multi-Layer Skull: Acoustic Properties Along Beam Axis")
    ax.axvspan(SKULL_START_MM, brain_start, alpha=0.15, color="gray")

    ax = axes[1]
    ax.plot(x_mm, rho_slice, "r-", lw=1.5)
    ax.set_ylabel("Density (kg/m³)")
    ax.axvspan(SKULL_START_MM, brain_start, alpha=0.15, color="gray")

    ax = axes[2]
    ax.plot(x_mm, alpha_slice, "g-", lw=1.5)
    ax.set_ylabel("Attenuation (dB/cm/MHz)")
    ax.set_xlabel("Axial distance (mm)")
    ax.axvspan(SKULL_START_MM, brain_start, alpha=0.15, color="gray")

    # Add layer annotations on first subplot
    layers_text = [
        (13, "Water"), (28, "Skin"), (30.5, "Cortical"),
        (33.5, "Trabecular"), (36, "Cort."), (50, "Brain")
    ]
    for xpos, label in layers_text:
        axes[0].annotate(label, xy=(xpos, axes[0].get_ylim()[1]*0.95),
                        fontsize=8, ha="center", va="top", color="navy")

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig3_tissue_layers.png")
    plt.close()
    print(f"  Saved fig3_tissue_layers.png")

    # =========================================================================
    # Figure 4: Heterogeneous skull vs water-only (time-domain, small grid)
    # =========================================================================
    print("Generating Figure 4: Heterogeneous vs water comparison...")

    spacing = 1.0
    shape = (61, 61, 65)

    het_coords = {}
    for i, dim in enumerate(("x", "y", "z")):
        n = shape[i]
        vals = np.arange(n) * spacing - (n // 2) * spacing
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        c.attrs["long_name"] = dim.upper()
        het_coords[dim] = c
    het_coords = xa.Coordinates(het_coords)

    # Heterogeneous layered phantom
    nz = shape[2]
    z_mm = np.arange(nz) * spacing
    labels = np.zeros(shape, dtype=int)
    for z_thresh, label in [(5.0, 1), (9.0, 2), (16.0, 3), (18.0, 4), (24.0, 5)]:
        labels[:, :, z_mm >= z_thresh] = label

    from openlifu.xdc.transducer import Transducer
    from openlifu.xdc.element import Element
    import json
    from pathlib import Path

    tx_json = Path("/root/openlifu_pkg/tests/resources/example_db/transducers/example_transducer/example_transducer.json")
    with open(tx_json) as f:
        d = json.load(f)
    tx_het = Transducer.from_dict(d)

    # Heterogeneous
    seg_het = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
    volume_het = xa.DataArray(np.zeros(shape), coords=het_coords)
    params_het = seg_het.seg_params(volume_het)
    ds_het, _ = run_td_simulation(
        arr=tx_het, params=params_het, freq=tx_het.frequency, cycles=10,
        cfl=0.3, pml_size=10, amplitude=1.0,
    )

    # Water-only
    seg_water = UniformWater()
    params_water = seg_water.ref_params(het_coords)
    ds_water, _ = run_td_simulation(
        arr=tx_het, params=params_water, freq=tx_het.frequency, cycles=10,
        cfl=0.3, pml_size=10, amplitude=1.0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    mid_y = shape[1] // 2
    het_x = het_coords["x"].values
    het_z = het_coords["z"].values
    het_extent = [het_x[0], het_x[-1], het_z[0], het_z[-1]]

    vmax = max(ds_het.p_max.data[:, mid_y, :].max(), ds_water.p_max.data[:, mid_y, :].max())

    ax = axes[0]
    im = ax.imshow(ds_het.p_max.data[:, mid_y, :].T, origin="lower", aspect="auto",
                   extent=het_extent, cmap="hot", vmax=vmax)
    ax.set_title("Heterogeneous Skull")
    ax.set_xlabel("Lateral (mm)"); ax.set_ylabel("Axial (mm)")
    fig.colorbar(im, ax=ax, label="p_max (Pa)")

    ax = axes[1]
    im = ax.imshow(ds_water.p_max.data[:, mid_y, :].T, origin="lower", aspect="auto",
                   extent=het_extent, cmap="hot", vmax=vmax)
    ax.set_title("Homogeneous Water")
    ax.set_xlabel("Lateral (mm)")
    fig.colorbar(im, ax=ax, label="p_max (Pa)")

    ax = axes[2]
    im = ax.imshow(params_het["sound_speed"].data[:, mid_y, :].T, origin="lower", aspect="auto",
                   extent=het_extent, cmap="gray")
    ax.set_title("Sound Speed Map")
    ax.set_xlabel("Lateral (mm)")
    fig.colorbar(im, ax=ax, label="c (m/s)")

    fig.suptitle("Time-Domain Simulation: Heterogeneous vs Homogeneous", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig4_het_vs_water.png")
    plt.close()
    print(f"  Saved fig4_het_vs_water.png")

    # =========================================================================
    # Figure 5: 2D pressure field with focus annotation
    # =========================================================================
    print("Generating Figure 5: Annotated focal field...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full field
    ax = axes[0]
    im = ax.imshow(p_amp.T, origin="lower", aspect="auto", extent=extent,
                   cmap="hot", vmin=0, vmax=np.percentile(p_amp, 99.5))
    ax.axvline(SKULL_START_MM, color="cyan", ls="--", lw=0.8)
    ax.axvline(brain_start, color="lime", ls="--", lw=0.8)
    # Mark focal region
    focus_x = 53.5  # our peak
    ax.plot(focus_x, 0, "w+", ms=15, mew=2)
    ax.annotate(f"Focus: {focus_x:.0f} mm", xy=(focus_x, 2), color="white",
               fontsize=9, ha="center")
    ax.set_xlabel("Axial (mm)")
    ax.set_ylabel("Lateral (mm)")
    ax.set_title("Pressure Amplitude (kPa)")
    fig.colorbar(im, ax=ax)

    # Right: zoom on focal region
    ax = axes[1]
    focus_region = p_amp[int(40/dx):int(80/dx), :]
    zoom_extent = [40, 80, y_mm[0], y_mm[-1]]
    im = ax.imshow(focus_region.T, origin="lower", aspect="auto", extent=zoom_extent,
                   cmap="hot", vmin=0, vmax=np.percentile(focus_region, 99.5))
    ax.plot(focus_x, 0, "w+", ms=15, mew=2)
    ax.axvline(64, color="white", ls=":", lw=1, alpha=0.5)
    ax.annotate("Geometric\nfocus", xy=(64, -30), color="white", fontsize=9, ha="center")
    ax.set_xlabel("Axial (mm)")
    ax.set_ylabel("Lateral (mm)")
    ax.set_title("Focal Region (zoomed)")
    fig.colorbar(im, ax=ax)

    fig.suptitle("ITRUSST BM4-SC1: Helmholtz Solver, dx=0.5 mm", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig5_focal_field.png")
    plt.close()
    print(f"  Saved fig5_focal_field.png")

    vol.commit()
    print("\nAll figures saved to Modal volume 'tfus-figures'.")
    return {"status": "done", "figures": 5}


@app.local_entrypoint()
def main():
    import os

    wall_start = time.perf_counter()
    print("Generating paper figures on A100 GPU...")
    print("=" * 60)
    result = generate_all_figures.remote()
    wall_total = time.perf_counter() - wall_start
    print(f"\nGenerated {result['figures']} figures in {wall_total:.0f}s")

    # Download figures from Modal volume
    outdir = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
    os.makedirs(outdir, exist_ok=True)
    print(f"\nDownloading to {outdir}...")
    for entry in vol.listdir("/"):
        if entry.path.endswith(".png"):
            local = os.path.join(outdir, os.path.basename(entry.path))
            with open(local, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
            print(f"  {os.path.basename(entry.path)}")
    print("Done.")
