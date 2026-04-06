"""TDD: SCI head model bridge between sbi4dwi and openlifu."""
import numpy as np
import pytest
import xarray as xa


def _make_fake_labels(shape=(20, 20, 20)):
    """Synthetic concentric tissue labels matching SCI convention."""
    labels = np.zeros(shape, dtype=np.int32)
    labels[2:-2, 2:-2, 2:-2] = 1   # scalp
    labels[4:-4, 4:-4, 4:-4] = 2   # skull
    labels[5:-5, 5:-5, 5:-5] = 3   # CSF
    labels[6:-6, 6:-6, 6:-6] = 4   # gray matter
    labels[8:-8, 8:-8, 8:-8] = 5   # white matter
    return labels


def test_labels_to_acoustic_and_conductivity():
    """Same labels should produce both acoustic and conductivity maps."""
    from openlifu.sim.sci_bridge import (
        _labels_to_acoustic_params, _labels_to_conductivity, _make_coords,
    )

    labels = _make_fake_labels()
    coords = _make_coords(labels.shape, dx_mm=1.0)

    params = _labels_to_acoustic_params(labels, coords)
    sigma = _labels_to_conductivity(labels)

    # Acoustic properties
    assert params["sound_speed"].data[0, 0, 0] == 1500.0   # water
    assert params["sound_speed"].data[4, 4, 4] == 4080.0   # skull
    assert params["sound_speed"].data[7, 7, 7] == 1560.0   # GM

    # Conductivity (float32 approximate)
    np.testing.assert_allclose(sigma[0, 0, 0], 0.0, atol=1e-6)     # water/air
    np.testing.assert_allclose(sigma[4, 4, 4], 0.01, rtol=1e-5)    # skull
    np.testing.assert_allclose(sigma[7, 7, 7], 0.47, rtol=1e-5)    # GM
    np.testing.assert_allclose(sigma[9, 9, 9], 0.14, rtol=1e-5)    # WM

    # Same shape
    assert params["sound_speed"].shape == sigma.shape


def test_conductivity_values_physical():
    """Conductivity values should be in physical range."""
    from openlifu.sim.sci_bridge import TISSUE_CONDUCTIVITY

    for tissue, sigma in TISSUE_CONDUCTIVITY.items():
        assert sigma >= 0, f"Negative conductivity for tissue {tissue}"
        assert sigma < 5, f"Conductivity too high for tissue {tissue}: {sigma}"

    # CSF should be most conductive
    assert TISSUE_CONDUCTIVITY[3] > TISSUE_CONDUCTIVITY[4]  # CSF > GM
    # Skull should be least conductive (except air)
    assert TISSUE_CONDUCTIVITY[2] < TISSUE_CONDUCTIVITY[4]  # skull < GM


def test_acoustic_and_conductivity_share_geometry():
    """Both property maps should have identical spatial coordinates."""
    from openlifu.sim.sci_bridge import (
        _labels_to_acoustic_params, _labels_to_conductivity, _make_coords,
    )

    labels = _make_fake_labels((30, 25, 20))
    coords = _make_coords(labels.shape, dx_mm=1.5)

    params = _labels_to_acoustic_params(labels, coords)
    sigma = _labels_to_conductivity(labels)

    assert params["sound_speed"].shape == (30, 25, 20)
    assert sigma.shape == (30, 25, 20)
    assert float(params.coords["x"].values[1] - params.coords["x"].values[0]) == 1.5


def test_load_presegmented_npy():
    """Should load pre-rasterized labels from .npy file."""
    from openlifu.sim.sci_bridge import _load_presegmented
    import tempfile, os

    labels = _make_fake_labels()
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, labels)
        tmppath = f.name

    try:
        loaded = _load_presegmented(tmppath)
        np.testing.assert_array_equal(loaded, labels)
    finally:
        os.unlink(tmppath)


def test_bridge_compatible_with_jwave_simulation():
    """Acoustic params from the bridge should work with run_cw_simulation."""
    from openlifu.sim.sci_bridge import (
        _labels_to_acoustic_params, _make_coords,
    )
    from openlifu.sim.jwave_if import run_cw_simulation
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer

    labels = _make_fake_labels((32, 32, 32))
    coords = _make_coords(labels.shape, dx_mm=1.0)
    params = _labels_to_acoustic_params(labels, coords)

    el = Element(index=0, position=np.array([0., 0., 0.]),
                 size=np.array([5e-3, 5e-3]), units="m")
    tx = Transducer(id="test", elements=[el], frequency=500e3, units="m")

    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=500e3,
        amplitude=60000.0, pml_size=5, tol=1e-2, maxiter=50,
    )

    assert "p_amp" in ds.data_vars
    assert ds.p_amp.data.max() > 0
    assert np.all(np.isfinite(ds.p_amp.data))
