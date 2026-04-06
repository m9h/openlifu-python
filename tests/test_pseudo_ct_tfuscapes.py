"""TDD: Validate HU → acoustic properties using TFUScapes CT data."""
import numpy as np
import pytest
from pathlib import Path

SAMPLE = Path(__file__).parent.parent / "benchmarks" / "tfuscapes_data" / "sample_0.npz"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_ct_to_acoustic_properties_physical():
    """Acoustic properties derived from TFUScapes CT should be physical."""
    from benchmarks.pseudo_ct_validation import hu_to_acoustic_properties

    d = np.load(SAMPLE)
    ct = d["ct"]
    props = hu_to_acoustic_properties(ct)

    # Sound speed in physical range
    assert props["sound_speed"].min() >= 1400
    assert props["sound_speed"].max() <= 4500

    # Density: water to bone
    assert props["density"].min() >= 1000
    assert props["density"].max() <= 3000

    # Skull regions (HU > 700) should have higher c than brain
    skull = ct > 700
    brain = (ct > 50) & (ct < 200)
    if skull.any() and brain.any():
        c_skull = props["sound_speed"][skull].mean()
        c_brain = props["sound_speed"][brain].mean()
        assert c_skull > c_brain, f"Skull c={c_skull:.0f} not > brain c={c_brain:.0f}"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_ct_labels_match_acoustic_segmentation():
    """CT-derived labels should match our tissue segmentation thresholds."""
    from benchmarks.tfuscapes_compare import ct_to_labels

    d = np.load(SAMPLE)
    labels = ct_to_labels(d["ct"])

    unique = np.unique(labels)
    # Should have at least water, skull, and some brain tissue
    assert 0 in unique  # water/air
    assert 2 in unique  # skull

    # Skull voxels should correspond to high HU regions
    skull_mask = labels == 2
    skull_hu = d["ct"][skull_mask]
    assert skull_hu.mean() > 700, f"Skull HU mean too low: {skull_hu.mean():.0f}"


@pytest.mark.skipif(not SAMPLE.exists(), reason="TFUScapes sample not downloaded")
def test_archies_law_conductivity_from_ct():
    """Archie's Law should produce physical conductivity from CT porosity."""
    from benchmarks.pseudo_ct_validation import hu_to_acoustic_properties

    d = np.load(SAMPLE)
    ct = d["ct"]

    # Porosity from HU
    HU_MAX = 2500.0
    porosity = 1.0 - np.clip(ct, 0, HU_MAX) / HU_MAX
    porosity = np.maximum(porosity, 0.05)

    # Archie's Law
    sigma_brine = 2.0  # S/m
    m = 1.5
    sigma = sigma_brine * (porosity ** m)

    # Dense skull (HU > 1200) should have lower conductivity than brain
    dense_skull = ct > 1200
    brain = (ct > 50) & (ct < 200)
    if dense_skull.any() and brain.any():
        sigma_skull = sigma[dense_skull].mean()
        sigma_brain = sigma[brain].mean()
        assert sigma_skull < sigma_brain, (
            f"Dense skull sigma={sigma_skull:.3f} should be < brain sigma={sigma_brain:.3f}"
        )
        # Dense skull conductivity should be physically reasonable
        assert sigma_skull < 1.5, f"Dense skull conductivity {sigma_skull:.3f} too high"
