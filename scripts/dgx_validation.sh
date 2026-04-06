#!/usr/bin/env bash
# DGX Spark Full Validation Suite
#
# Runs the complete jwave integration validation on DGX Spark GPU.
# Covers: test suite, ITRUSST benchmarks, phase correction, cohort analysis.
#
# Usage:
#   ssh dgx-spark
#   bash scripts/dgx_validation.sh          # full run
#   bash scripts/dgx_validation.sh --quick  # tests + BM4 only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

QUICK=false
[[ "${1:-}" == "--quick" ]] && QUICK=true

RESULTS_DIR="$SCRIPT_DIR/results/dgx_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

log() { echo "$(date +%H:%M:%S) [DGX] $*" | tee -a "$RESULTS_DIR/run.log"; }

# ============================================================================
# 1. Environment setup
# ============================================================================
log "Setting up environment..."
if [ ! -d .venv ]; then
    uv venv
    uv pip install -e ".[dev]" 2>&1 | tail -5
fi
source .venv/bin/activate

log "Verifying JAX GPU..."
python -c "
import jax
print(f'JAX {jax.__version__}, backend: {jax.default_backend()}, devices: {jax.devices()}')
assert jax.default_backend() == 'gpu', 'No GPU detected!'
import jax.numpy as jnp
x = jnp.ones((1000,1000)); y = jnp.dot(x,x); y.block_until_ready()
print('GPU matmul OK')
" 2>&1 | tee -a "$RESULTS_DIR/run.log"

# ============================================================================
# 2. Full test suite (48 tests, 3 should flip from xfail to pass)
# ============================================================================
log "Running full test suite..."
python -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_DIR/test_results.txt"
log "Test suite complete."

# ============================================================================
# 3. ITRUSST BM1: Free-field (analytical reference)
# ============================================================================
log "Running BM1 free-field (dx=0.5mm)..."
python -c "
from benchmarks.itrusst_bm1 import run_bm1
r = run_bm1(dx_mm=0.5, save_png='$RESULTS_DIR/bm1.png')
print(f'BM1: focus={r[\"p_focus_kpa\"]:.1f} kPa, analytical={r[\"p_analytical_kpa\"]:.1f} kPa, error={r[\"error_pct\"]:.1f}%')
import json; json.dump(r, open('$RESULTS_DIR/bm1.json','w'), indent=2)
" 2>&1 | tee -a "$RESULTS_DIR/run.log"

# ============================================================================
# 4. ITRUSST BM4: Multi-layer skull (benchmark validation)
# ============================================================================
log "Running BM4 multi-layer skull (dx=0.5mm)..."
python -c "
from benchmarks.itrusst_bm4 import run_bm4
r = run_bm4(dx_mm=0.5, save_png='$RESULTS_DIR/bm4.png')
import json; json.dump(r, open('$RESULTS_DIR/bm4.json','w'), indent=2)
" 2>&1 | tee -a "$RESULTS_DIR/run.log"

if $QUICK; then
    log "Quick mode — skipping remaining benchmarks."
    log "Results saved to $RESULTS_DIR"
    exit 0
fi

# ============================================================================
# 5. ITRUSST BM7: Realistic skull mesh
# ============================================================================
log "Running BM7 realistic skull (dx=1.0mm)..."
python -c "
from benchmarks.itrusst_bm7 import run_bm7
r = run_bm7(dx_mm=1.0, save_png='$RESULTS_DIR/bm7.png')
import json; json.dump(r, open('$RESULTS_DIR/bm7.json','w'), indent=2)
" 2>&1 | tee -a "$RESULTS_DIR/run.log"

# ============================================================================
# 6. Resolution convergence study
# ============================================================================
log "Running convergence study (dx=4,2,1,0.5mm)..."
python -c "
from benchmarks.convergence_study import run_convergence
results = run_convergence(dx_values=[4.0, 2.0, 1.0, 0.5], save_png='$RESULTS_DIR/convergence.png')
import json; json.dump(results, open('$RESULTS_DIR/convergence.json','w'), indent=2)
" 2>&1 | tee -a "$RESULTS_DIR/run.log"

# ============================================================================
# 7. Phase correction through skull
# ============================================================================
log "Running phase correction (dx=0.5mm, 50 steps)..."
python -c "
from benchmarks.phase_correction_demo import run_phase_correction
r = run_phase_correction(dx_mm=0.5, n_steps=50, save_png='$RESULTS_DIR/phase_correction.png')
import json
r_safe = {k: v for k, v in r.items() if k != 'opt_delays'}
json.dump(r_safe, open('$RESULTS_DIR/phase_correction.json','w'), indent=2)
" 2>&1 | tee -a "$RESULTS_DIR/run.log"

# ============================================================================
# 8. Birnbaum cohort (if data available)
# ============================================================================
if [ -d benchmarks/birnbaum_data/Data ]; then
    log "Running Birnbaum cohort analysis (64 subjects)..."
    python benchmarks/birnbaum_cohort.py \
        --save-csv "$RESULTS_DIR/birnbaum_cohort.csv" \
        --save-png "$RESULTS_DIR/birnbaum_cohort.png" \
        2>&1 | tee -a "$RESULTS_DIR/run.log"
else
    log "Birnbaum data not found — skipping cohort analysis."
fi

# ============================================================================
# 9. TFUScapes comparison (if sample available)
# ============================================================================
if [ -f benchmarks/tfuscapes_data/sample_0.npz ]; then
    log "Running TFUScapes comparison..."
    python -c "
from benchmarks.tfuscapes_compare import run_comparison
r = run_comparison('benchmarks/tfuscapes_data/sample_0.npz', save_png='$RESULTS_DIR/tfuscapes.png')
import json; json.dump(r, open('$RESULTS_DIR/tfuscapes.json','w'), indent=2)
" 2>&1 | tee -a "$RESULTS_DIR/run.log"
else
    log "TFUScapes sample not found — skipping."
fi

# ============================================================================
# 10. SCI head model (if sbi4dwi available)
# ============================================================================
SCI_MESH="$HOME/dev/sbi4dwi/data/sci_head/HeadMesh.mat"
if python -c "from openlifu.sim.sci_bridge import load_sci_for_simulation" 2>/dev/null && [ -f "$SCI_MESH" ]; then
    log "Running SCI head model simulation..."
    python -c "
from openlifu.sim.sci_bridge import load_sci_for_simulation
from openlifu.sim.jwave_if import run_cw_simulation
import json, numpy as np

params, coords, sigma = load_sci_for_simulation('$SCI_MESH', dx_mm=2.0)
print(f'SCI grid: {dict(coords.sizes)}')
print(f'Conductivity range: {sigma.min():.3f} - {sigma.max():.3f} S/m')

from openlifu.xdc.element import Element
from openlifu.xdc.transducer import Transducer
el = Element(index=0, position=np.array([0.,0.,0.]), size=np.array([5e-3,5e-3]), units='m')
tx = Transducer(id='test', elements=[el], frequency=500e3, units='m')

ds, _ = run_cw_simulation(arr=tx, params=params, freq=500e3, amplitude=60000.0, pml_size=10)
print(f'SCI p_amp max: {float(ds.p_amp.max()):.0f} Pa')
json.dump({'p_amp_max': float(ds.p_amp.max()), 'grid': list(ds.p_amp.shape)},
          open('$RESULTS_DIR/sci_head.json','w'), indent=2)
" 2>&1 | tee -a "$RESULTS_DIR/run.log"
else
    log "SCI head model or sbi4dwi not available — skipping."
fi

# ============================================================================
# Summary
# ============================================================================
log ""
log "=============================================="
log "DGX VALIDATION COMPLETE"
log "=============================================="
log "Results: $RESULTS_DIR"
log ""
log "Files:"
ls -lh "$RESULTS_DIR" | tee -a "$RESULTS_DIR/run.log"
