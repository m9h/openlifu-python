# Transcranial focused-ultrasound simulation: SimNIBS / PRESTUS replication

This tutorial walks through running a complete tFUS simulation against a
real SimNIBS CHARM-segmented head model using openlifu's jwave backend,
mirroring the PRESTUS workflow (originally MATLAB k-Wave) end-to-end in
differentiable JAX.

The pipeline:

1. Segment a T1 (and optionally T2) MRI with SimNIBS CHARM, producing
   an `m2m_<subject>/` directory with `tissue_labeling.nii.gz`.
2. Load the segmentation, remap CHARM's 11-class labels onto openlifu's
   PRESTUS-compatible 7-class convention.
3. Build a focused-bowl transducer geometry on a spherical cap.
4. Run either a CW Helmholtz solve (steady-state, fastest for
   beam-shape analysis) or a time-domain PSTD solve with peak-pressure
   recording (matches PRESTUS output, captures transient peaks).
5. Write metrics and a peak-pressure NIfTI for visualisation.

## 0. Prerequisites

- A working install of openlifu (`uv sync`) on a JAX-CUDA-enabled host.
  Use `jax[cuda13]` on Blackwell / Grace hardware (DGX Spark GB10),
  `jax[cuda12]` otherwise. The repo pins jwave to the
  `m9h/jwave@feature/configurable-alpha-power` fork; this is required
  until the upstream `alpha_power` PR lands.
- `nibabel`, `scipy`, and `xarray` (already in the openlifu deps).
- For SimNIBS CHARM: a SimNIBS install with `charm` on `$PATH`. Not
  required if you already have a pre-segmented `m2m_<subject>/`.

## 1. Get a CHARM segmentation

Easiest route: the public Ernie example shipped with SimNIBS.

```bash
uv run python scripts/run_ernie_replication.py
```

That script clones `simnibs/example-dataset`, runs `charm` if needed
(~25-40 minutes of CPU time, ~8 GB RAM), and then invokes the
replication benchmark. To skip the CHARM step and use a pre-segmented
m2m you already have:

```bash
uv run python scripts/run_ernie_replication.py --skip-charm \
    --data-dir /path/to/dir/containing/m2m_ernie
```

Or, if you have your own subject:

```bash
charm subj01 t1.nii.gz t2.nii.gz   # writes ./m2m_subj01/
```

## 2. Manual run of the replication benchmark

`benchmarks/simnibs_replication.py` is the scriptable entry point.
Default parameters mirror the Sonic Concepts H-115 transducer used in
the SimNIBS TFUS tutorial (500 kHz, 64 mm aperture, 52 mm focal
length).

```bash
uv run python benchmarks/simnibs_replication.py \
    --m2m /path/to/m2m_subj01 \
    --dx-mm 0.5 \
    --target-mm 0 0 0
```

Flags worth knowing:

| flag | default | what it does |
|---|---|---|
| `--dx-mm` | 0.5 | Grid spacing. The script nearest-neighbour-resamples the 1 mm CHARM labels to keep the physical domain extent constant when `dx<1`. |
| `--frequency-hz` | 500e3 | Source drive frequency. PPW check fails if c_min/(freq*dx) drops below 6. |
| `--aperture-mm` | 64 | Bowl rim diameter. |
| `--focal-length-mm` | 52 | Bowl radius of curvature; centre of curvature = focal point. |
| `--target-mm` | (0, 0, 0) | Focal target offset from the head-model centre, mm. |
| `--time-domain` | off | Use the PSTD peak-pressure solver instead of the CW Helmholtz. Reports both PPP and \|PNP\|. |
| `--cycles` | 20 | Toneburst duration for `--time-domain` mode. |
| `--cfl` | 0.3 | CFL number for the time axis. |
| `--ppw-target` | 12 | Soft target points-per-wavelength; warning below this, hard fail below 6. Set 0 to disable. |
| `--no-crop` | off | Skip the skull-bbox crop and simulate the full head domain. |
| `--maxiter` / `--tol` | 300 / 1e-3 | GMRES settings for the CW Helmholtz solver. |

## 3. What the script does, in code

```python
from pathlib import Path
import numpy as np, xarray as xa

from openlifu.seg.simnibs import (
    load_simnibs_segmentation, remap_charm_to_openlifu,
)
from openlifu.seg.seg_methods.heterogeneous import (
    HeterogeneousSkullSegmentation, PRESTUS_LABEL_TO_MATERIAL,
)
from openlifu.sim.jwave_if import (
    check_ppw, run_cw_simulation, run_simulation_peak,
)
from openlifu.xdc.bowl import bowl_transducer_3d
from openlifu.xdc.element import Element
from openlifu.xdc.transducer import Transducer

dx_mm = 0.5
freq_hz = 500e3

# 1. Load + remap CHARM labels (compact bone -> 2 cortical, spongy -> 6 trabecular)
charm_labels = load_simnibs_segmentation(Path("m2m_subj01"))
labels = remap_charm_to_openlifu(charm_labels)

# 2. Build acoustic property volumes via the PRESTUS-compatible material set
seg = HeterogeneousSkullSegmentation(
    source="labels", label_array=labels,
    label_to_material=PRESTUS_LABEL_TO_MATERIAL,
)
shape = labels.shape
coords = xa.Coordinates({
    d: xa.DataArray(np.arange(shape[i]) * dx_mm, dims=[d], attrs={"units": "mm"})
    for i, d in enumerate(("x", "y", "z"))
})
volume = xa.DataArray(np.zeros(shape), coords=coords)
params = seg.seg_params(volume)

# 3. PPW check: warn / fail-fast on undersampled grids
ppw = check_ppw(np.asarray(params["sound_speed"].data), freq_hz, dx_mm * 1e-3)

# 4. Build a 256-element bowl transducer geometry, focused at the head centre
focal_m = np.array([s * dx_mm * 0.5 for s in shape]) * 1e-3
bowl_pts = np.asarray(bowl_transducer_3d(
    focal_length=0.052, aperture_diameter=0.064,
    focal_point=tuple(focal_m), direction=(1, 0, 0), n_points=256,
))
elements = [
    Element(index=i, position=p, size=np.array([dx_mm*1e-3, dx_mm*1e-3]),
            units="m")
    for i, p in enumerate(bowl_pts)
]
tx = Transducer(id="subj01", elements=elements, frequency=freq_hz, units="m")

# 5a. Steady-state CW Helmholtz (Birnbaum-style; default)
ds_cw, _ = run_cw_simulation(
    arr=tx, params=params, freq=freq_hz, amplitude=60_000.0, pml_size=8,
)

# 5b. ...or transient PSTD with peak-pressure carry (PRESTUS-style)
ds_td, _ = run_simulation_peak(
    arr=tx, params=params, freq=freq_hz, amplitude=60_000.0,
    cycles=20, cfl=0.3, pml_size=8,
)

# Brain-region focal pressure
brain_mask = (labels == 4) | (labels == 5)        # GM + WM (PRESTUS scheme)
brain_p_max_kpa = float(np.asarray(ds_cw.p_amp.data)[brain_mask].max()) / 1e3
print(f"brain p_max = {brain_p_max_kpa:.1f} kPa")
```

## 4. Reading the output

The benchmark writes two files per subject under `--out-dir`:

- `simnibs_<sid>_dx<N>mm_<mode>.json` — config + brain p_max, mean,
  PPW, sim_time, voxel counts (cortical + trabecular skull, brain).
- `simnibs_<sid>_dx<N>mm_<mode>_p_amp.nii.gz` — peak-pressure field
  (kPa) on the cropped simulation grid, viewable in 3D Slicer / FSL.

`<mode>` is `cw` for the Helmholtz solver and `td` for the time-domain
peak-pressure path.

## 5. Choosing a mode

| use case | recommended mode |
|---|---|
| Beam-shape analysis, focal pressure metric, ITRUSST-style benchmarks | CW (`run_cw_simulation`) |
| PRESTUS replication, peak negative pressure (cavitation index, MI) | time-domain (`run_simulation_peak`) |
| Pulsed protocols where transient peaks differ from CW amplitude | time-domain |
| Iterative phase correction (gradient-based delay optimisation) | time-domain via `openlifu.sim.optimize_delays` |
| Quick parameter sweeps (transducer geometry, target placement) | CW |

CW solves the Helmholtz equation directly at a single frequency and is
memory-efficient (one complex spatial field). Time-domain integrates
the wave equation with a CFL-limited time step; the
`run_simulation_peak` carry pattern keeps memory at `O(grid)` rather
than `O(Nt * grid)`, so 256³ grids fit on a single 80 GB GPU.

## 6. Per-element delay optimisation

For aberration correction beyond the analytic geometric model
(`openlifu.sim.phase_correction`), use the gradient-based optimiser
that backpropagates through the time-domain solve:

```python
from openlifu.sim.optimize_delays import optimize_delays

result = optimize_delays(
    arr=tx, params=params, target_voxel=(140, 100, 100),
    freq=freq_hz, n_iters=50, lr=1e-8,
)
print(f"final focal pressure = {result.final_focal_pressure_pa/1e3:.1f} kPa")
print(f"loss curve: {result.loss_history[:5]} ... {result.loss_history[-5:]}")
delays = result.delays  # (n_elements,) seconds, ready for hardware
```

Captures multi-path, diffraction, and finite-bandwidth effects the
geometric path-length model misses, at the cost of ~50× the runtime
(one full forward sim per Adam step).

## 7. Pseudo-CT path (no CHARM)

If you have a T1 only and no T2, skip CHARM and go through the
HU-based pseudo-CT pipeline:

```python
from openlifu.seg.pseudoct import hu_to_acoustic_properties

hu = my_unet_pseudo_ct(t1_volume)            # your trained predictor
props = hu_to_acoustic_properties(hu)        # JAX-differentiable
# props["sound_speed"], props["density"], props["attenuation"]
```

`HeterogeneousSkullSegmentation(source="pseudoct")` also exposes a
crude T1-threshold fallback for treatment-planning previews; for
research-grade results use a learned predictor (e.g. a U-Net trained
against CT-paired T1s) and the `hu_to_acoustic_properties` path.

## 8. Exporting to openlifu Solution format

To hand the result to openlifu's clinical layer (Slicer visualisation,
safety analysis, hardware programming):

```python
from openlifu.sim.solution_export import build_solution_dict

solution = build_solution_dict(
    delays=result.delays,
    apodizations=np.ones(tx.numelements()),
    simulation_result=ds_cw,
    freq=freq_hz,
    target=tuple(focal_m),
)
# Hand `solution` to openlifu.plan.Solution.from_dict(...) when ready.
```

## 9. Caveats and known limits

- **Grid convergence**: the BM1 free-field benchmark (Aubry 2022 BM1)
  shows ~44% error at dx=0.5 mm and ~12% at dx=0.25 mm vs the analytic
  baseline. Per-patient absolutes at dx=0.5 mm in the cohort runs are
  therefore lower bounds; treat the cohort *spread* as the more robust
  reported quantity.
- **Domain cropping**: `--no-crop` simulates the full head and uses a
  lot of memory at fine resolutions. Default skull-bbox crop with 8
  voxels of padding keeps dx=0.25 mm tractable on A100-80GB.
- **CHARM 4+ vs older SimNIBS**: this tutorial assumes the 11-class
  CHARM 4+ output. Older `headreco` outputs use 5 classes; route them
  through `openlifu.seg.seg_methods.heterogeneous.remap_simnibs_labels`
  instead of `remap_charm_to_openlifu`.
- **Single-skull legacy mode**: pass `legacy=True` to
  `remap_charm_to_openlifu` if you specifically need the older
  single-class skull (e.g. to reproduce a pre-PRESTUS openlifu study);
  otherwise stick with the two-class (cortical / trabecular) default.
