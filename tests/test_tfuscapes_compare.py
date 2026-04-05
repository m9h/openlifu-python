"""TDD: TFUScapes jwave vs k-Wave comparison pipeline."""
import numpy as np
import pytest
from pathlib import Path

SAMPLE = Path(__file__).parent.parent / "benchmarks" / "tfuscapes_data" / "sample_0.npz"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_ct_to_labels_produces_all_tissues():
    from benchmarks.tfuscapes_compare import ct_to_labels
    d = np.load(SAMPLE)
    labels = ct_to_labels(d["ct"])
    unique = np.unique(labels)
    # Should have at least water(0), skull(2), and some brain tissue
    assert 0 in unique, "No water/air"
    assert 2 in unique, "No skull"
    assert len(unique) >= 3, f"Too few tissue types: {unique}"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_build_transducer_from_coords():
    from benchmarks.tfuscapes_compare import build_transducer_from_coords
    d = np.load(SAMPLE)
    tx = build_transducer_from_coords(d["tr_coords"])
    # Should subsample from 16K to ~512
    assert tx.numelements() <= 600
    assert tx.numelements() > 10
    positions = tx.get_positions(units="m")
    assert positions.shape[1] == 3
    assert np.all(np.isfinite(positions))


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_compare_fields_metrics():
    """compare_fields should return valid metrics for identical fields."""
    from benchmarks.tfuscapes_compare import compare_fields
    d = np.load(SAMPLE)
    pmap = d["pmap"]
    # Comparing field to itself should give zero error
    metrics = compare_fields(pmap, pmap)
    assert metrics["l2_error_pct"] < 0.01
    assert metrics["peak_diff_pct"] < 0.01
    assert metrics["correlation"] > 0.999
    assert metrics["pos_diff_voxels"] < 0.01
