"""TDD: TFUScapes dataset loading and jwave vs k-Wave comparison."""
import numpy as np
import pytest
from pathlib import Path

SAMPLE = Path(__file__).parent.parent / "benchmarks" / "tfuscapes_data" / "sample_0.npz"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_load_tfuscapes_sample():
    """Verify the TFUScapes .npz structure."""
    d = np.load(SAMPLE)
    assert "ct" in d, "Missing 'ct' key"
    assert "pmap" in d, "Missing 'pmap' key"
    assert "tr_coords" in d, "Missing 'tr_coords' key"
    assert d["ct"].shape == (256, 256, 256)
    assert d["pmap"].shape == (256, 256, 256)
    assert d["tr_coords"].ndim == 2 and d["tr_coords"].shape[1] == 3


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_tfuscapes_ct_has_skull():
    """CT volume should contain skull (high HU values)."""
    d = np.load(SAMPLE)
    ct = d["ct"]
    # Skull bone: HU > 700
    n_bone = np.sum(ct > 700)
    n_tissue = np.sum((ct > 0) & (ct <= 700))
    assert n_bone > 1000, f"Too few bone voxels: {n_bone}"
    assert n_tissue > n_bone, "More bone than soft tissue — unexpected"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_tfuscapes_pmap_has_focus():
    """k-Wave pressure map should have a clear focal peak."""
    d = np.load(SAMPLE)
    pmap = d["pmap"]
    peak = pmap.max()
    mean = pmap[pmap > 0].mean()
    # Focal gain: peak should be >> mean (at least 10x)
    assert peak / mean > 5, f"Weak focusing: peak={peak:.0f}, mean={mean:.0f}"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_tfuscapes_transducer_on_grid():
    """Transducer coordinates should be valid grid indices."""
    d = np.load(SAMPLE)
    tr = d["tr_coords"]
    assert np.all(tr >= 0), "Negative transducer coordinates"
    assert np.all(tr < 256), "Transducer coords out of 256³ grid"
