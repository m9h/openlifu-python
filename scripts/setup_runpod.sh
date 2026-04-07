#!/bin/bash
# RunPod GPU Setup + Full Overnight Validation
#
# Provisions the openlifu jwave environment on a RunPod GPU pod and runs
# the complete validation suite (BM1-BM7, phase correction, convergence,
# Birnbaum cohort, TFUScapes).
#
# Usage:
#   # From your local machine after creating a RunPod pod:
#   curl -fsSL https://raw.githubusercontent.com/m9h/openlifu-python/feature/heterogeneous-skull-segmentation/scripts/setup_runpod.sh | bash
#
# Or SSH into the pod and run:
#   bash /workspace/openlifu-python/scripts/setup_runpod.sh
#
# Recommended pod config:
#   - GPU: A100 80GB ($1.64/hr) or A40 ($0.39/hr)
#   - Disk: 50GB (for datasets)
#   - Docker image: runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04
#
set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/openlifu-python"
RESULTS_DIR="$WORKSPACE/results_$(date +%Y%m%d_%H%M%S)"

log() { echo "$(date +%H:%M:%S) [RUNPOD] $*"; }

# ============================================================================
# 1. Install uv
# ============================================================================
log "Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# ============================================================================
# 2. Clone repo
# ============================================================================
if [ ! -d "$REPO_DIR" ]; then
    log "Cloning openlifu-python..."
    cd "$WORKSPACE"
    git clone https://github.com/m9h/openlifu-python.git
    cd openlifu-python
    git checkout feature/heterogeneous-skull-segmentation
else
    log "Repo exists, pulling latest..."
    cd "$REPO_DIR"
    git checkout feature/heterogeneous-skull-segmentation
    git pull origin feature/heterogeneous-skull-segmentation
fi

# ============================================================================
# 3. Create venv and install deps
# ============================================================================
log "Setting up Python environment..."
cd "$REPO_DIR"
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" 2>&1 | tail -5

# Fix JAX CUDA if needed (RunPod sometimes needs cu12 → cu12 nudge)
log "Verifying JAX GPU..."
python -c "
import jax
print(f'JAX {jax.__version__}, backend: {jax.default_backend()}, devices: {jax.devices()}')
assert jax.default_backend() == 'gpu', 'No GPU!'
import jax.numpy as jnp
x = jnp.ones((2000,2000)); y = jnp.dot(x,x); y.block_until_ready()
print('GPU OK')
" || {
    log "JAX GPU failed, trying cuda12 fix..."
    uv pip install --force-reinstall "jax[cuda12]>=0.9.0"
    python -c "import jax; assert jax.default_backend()=='gpu'; print('GPU OK after fix')"
}

# ============================================================================
# 4. Download datasets
# ============================================================================
log "Downloading datasets..."
mkdir -p "$RESULTS_DIR"

# TFUScapes sample
if [ ! -f benchmarks/tfuscapes_data/sample_0.npz ]; then
    log "  Downloading TFUScapes sample..."
    uv pip install huggingface_hub 2>/dev/null
    python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('vinkle-srivastav/TFUScapes', 'data/A00028185/exp_0.npz', repo_type='dataset')
shutil.copy(p, 'benchmarks/tfuscapes_data/sample_0.npz')
print('TFUScapes OK')
"
fi

# Birnbaum dataset
if [ ! -d benchmarks/birnbaum_data/Data ]; then
    log "  Downloading Birnbaum dataset (64 subjects)..."
    export KAGGLE_API_TOKEN=KGAT_3b2c8c3ef4782a7030baee29cf3755a3
    uv pip install kaggle 2>/dev/null
    kaggle datasets download andrewbirnbaum/full-head-mri-and-segmentation-of-stroke-patients \
        -p benchmarks/birnbaum_data/ --unzip 2>&1 | tail -3
fi

# ============================================================================
# 5. Full test suite
# ============================================================================
log "Running test suite..."
python -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_DIR/test_results.txt"

# ============================================================================
# 6. Overnight validation runs
# ============================================================================
log "Starting overnight validation runs..."

run_bench() {
    local name="$1"
    local cmd="$2"
    log "  [$name] Starting..."
    local t0=$(date +%s)
    eval "$cmd" 2>&1 | tee "$RESULTS_DIR/${name}.log"
    local t1=$(date +%s)
    log "  [$name] Done in $((t1-t0))s"
}

# BM1 free-field (dx=0.5mm)
run_bench "bm1" "python -c \"
from benchmarks.itrusst_bm1 import run_bm1
import json
r = run_bm1(dx_mm=0.5, save_png='$RESULTS_DIR/bm1.png')
json.dump(r, open('$RESULTS_DIR/bm1.json','w'), indent=2)
print('BM1: focus={p_focus_kpa:.1f} kPa, error={error_pct:.1f}%'.format(**r))
\""

# BM4 multi-layer skull (dx=0.5mm)
run_bench "bm4_05mm" "python -c \"
from benchmarks.itrusst_bm4 import run_bm4
import json
r = run_bm4(dx_mm=0.5, save_png='$RESULTS_DIR/bm4_05mm.png')
json.dump(r, open('$RESULTS_DIR/bm4_05mm.json','w'), indent=2)
\""

# BM4 ultra-fine (dx=0.25mm)
run_bench "bm4_025mm" "python -c \"
from benchmarks.itrusst_bm4 import run_bm4
import json
r = run_bm4(dx_mm=0.25, save_png='$RESULTS_DIR/bm4_025mm.png')
json.dump(r, open('$RESULTS_DIR/bm4_025mm.json','w'), indent=2)
\""

# BM7 realistic skull (dx=1mm then dx=0.5mm)
run_bench "bm7_1mm" "python -c \"
from benchmarks.itrusst_bm7 import run_bm7
import json
r = run_bm7(dx_mm=1.0, save_png='$RESULTS_DIR/bm7_1mm.png')
json.dump(r, open('$RESULTS_DIR/bm7_1mm.json','w'), indent=2)
\""

run_bench "bm7_05mm" "python -c \"
from benchmarks.itrusst_bm7 import run_bm7
import json
r = run_bm7(dx_mm=0.5, save_png='$RESULTS_DIR/bm7_05mm.png')
json.dump(r, open('$RESULTS_DIR/bm7_05mm.json','w'), indent=2)
\""

# Convergence study (dx=4,2,1,0.5,0.25mm)
run_bench "convergence" "python -c \"
from benchmarks.convergence_study import run_convergence
import json
r = run_convergence([4.0, 2.0, 1.0, 0.5, 0.25], save_png='$RESULTS_DIR/convergence.png')
json.dump(r, open('$RESULTS_DIR/convergence.json','w'), indent=2)
\""

# Phase correction (dx=0.5mm, 200 steps)
run_bench "phase_200" "python -c \"
from benchmarks.phase_correction_demo import run_phase_correction
import json
r = run_phase_correction(dx_mm=0.5, n_steps=200, save_png='$RESULTS_DIR/phase_200.png')
r_safe = {k: v for k, v in r.items() if k != 'opt_delays'}
json.dump(r_safe, open('$RESULTS_DIR/phase_200.json','w'), indent=2)
\""

# TFUScapes comparison
if [ -f benchmarks/tfuscapes_data/sample_0.npz ]; then
    run_bench "tfuscapes" "python -c \"
from benchmarks.tfuscapes_compare import run_comparison
import json
r = run_comparison('benchmarks/tfuscapes_data/sample_0.npz', save_png='$RESULTS_DIR/tfuscapes.png')
json.dump(r, open('$RESULTS_DIR/tfuscapes.json','w'), indent=2)
\""
fi

# Birnbaum cohort
if [ -d benchmarks/birnbaum_data/Data ]; then
    run_bench "birnbaum" "python benchmarks/birnbaum_cohort.py \
        --save-csv $RESULTS_DIR/birnbaum.csv \
        --save-png $RESULTS_DIR/birnbaum.png"
fi

# ============================================================================
# 7. Summary
# ============================================================================
log ""
log "=============================================="
log "RUNPOD VALIDATION COMPLETE"
log "=============================================="
log "Results: $RESULTS_DIR"
log ""
ls -lhS "$RESULTS_DIR" | head -20
log ""

# Print key metrics
python -c "
import json, glob, os
print('KEY RESULTS:')
for f in sorted(glob.glob('$RESULTS_DIR/*.json')):
    name = os.path.basename(f).replace('.json','')
    try:
        r = json.load(open(f))
        if isinstance(r, list):
            print(f'  {name}: {len(r)} entries')
        else:
            keys = [k for k in ['p_focus_kpa','peak_kpa','error_pct','recovery_pct','correlation','sim_time_s'] if k in r]
            vals = ', '.join(f'{k}={r[k]}' for k in keys)
            print(f'  {name}: {vals}')
    except: pass
"

log ""
log "Pod can be stopped now. Copy results with:"
log "  runpodctl receive $RESULTS_DIR"
