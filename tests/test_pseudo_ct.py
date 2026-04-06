"""TDD: Pseudo-CT validation pipeline."""
import numpy as np
import pytest


def test_t1_to_pseudo_ct_range():
    """Pseudo-CT should produce physically reasonable HU values."""
    from benchmarks.pseudo_ct_validation import t1_to_pseudo_ct

    # Synthetic T1: bright tissue, dark bone
    t1 = np.random.rand(20, 20, 20).astype(np.float32) * 1000
    hu = t1_to_pseudo_ct(t1, method="plymouth")

    assert hu.shape == t1.shape
    assert hu.min() >= 200, f"HU too low: {hu.min()}"
    assert hu.max() <= 2100, f"HU too high: {hu.max()}"


def test_t1_dark_regions_give_high_hu():
    """In T1, bone is dark → pseudo-CT should assign high HU to dark regions."""
    from benchmarks.pseudo_ct_validation import t1_to_pseudo_ct

    t1 = np.ones((20, 20, 20), dtype=np.float32) * 500
    t1[8:12, 8:12, 8:12] = 50  # dark region (bone-like)

    hu = t1_to_pseudo_ct(t1, method="plymouth")
    # Dark region should have higher HU than bright region
    assert hu[10, 10, 10] > hu[0, 0, 0], "Dark T1 should map to higher HU"


def test_hu_to_acoustic_properties_physical():
    """Acoustic properties from HU should be in physical range."""
    from benchmarks.pseudo_ct_validation import hu_to_acoustic_properties

    hu = np.array([0, 100, 500, 1000, 2000], dtype=np.float32)
    props = hu_to_acoustic_properties(hu)

    # Sound speed: 1400-4500 m/s
    assert np.all(props["sound_speed"] >= 1400)
    assert np.all(props["sound_speed"] <= 4500)

    # Density: water (1000) to bone (2500)
    assert np.all(props["density"] >= 1000)

    # Bone HU should give higher c than water HU
    assert props["sound_speed"][-1] > props["sound_speed"][0]


def test_compute_validation_metrics_identical():
    """Comparing identical fields should give zero error."""
    from benchmarks.pseudo_ct_validation import compute_validation_metrics

    hu = np.random.rand(20, 20, 20).astype(np.float32) * 2000
    metrics = compute_validation_metrics(hu, hu)

    assert metrics["mae_hu"] < 0.01
    assert metrics["rmse_hu"] < 0.01
    assert metrics["correlation"] > 0.999
    assert metrics["dice_bone"] > 0.999


def test_compute_validation_metrics_with_error():
    """Adding noise should produce nonzero but bounded error."""
    from benchmarks.pseudo_ct_validation import compute_validation_metrics

    rng = np.random.default_rng(42)
    real = rng.uniform(0, 2000, (20, 20, 20)).astype(np.float32)
    pred = real + rng.normal(0, 100, real.shape).astype(np.float32)  # ~100 HU noise

    metrics = compute_validation_metrics(pred, real)

    assert 50 < metrics["mae_hu"] < 200, f"MAE unexpected: {metrics['mae_hu']}"
    assert metrics["correlation"] > 0.8, f"Correlation too low: {metrics['correlation']}"


def test_skull_mask_extraction():
    """Skull mask should isolate bone-density voxels."""
    from benchmarks.pseudo_ct_validation import compute_skull_mask

    ct = np.zeros((20, 20, 20), dtype=np.float32)
    ct[5:15, 5:15, 5:15] = 1000  # bone
    ct[7:13, 7:13, 7:13] = 50    # brain (inside)

    mask = compute_skull_mask(ct, hu_min=300)
    assert mask[10, 10, 5] == True   # bone
    assert mask[10, 10, 10] == False  # brain
    assert mask[0, 0, 0] == False    # air
