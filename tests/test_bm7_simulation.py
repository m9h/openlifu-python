"""TDD: BM7 realistic skull simulation end-to-end."""
import numpy as np
import pytest
from pathlib import Path

STL_DIR = Path(__file__).parent.parent / "benchmarks" / "itrusst_data" / "intercomparison" / "skull-stl"


@pytest.mark.skipif(not STL_DIR.exists(), reason="ITRUSST skull STL data not available")
def test_bm7_labels_to_params_correct_properties():
    """Acoustic property maps should have correct values in each region."""
    from benchmarks.itrusst_bm7 import load_skull_meshes, rasterize_skull, labels_to_params
    outer, inner = load_skull_meshes()
    labels, coords = rasterize_skull(outer, inner, dx_mm=4.0)
    params = labels_to_params(labels, coords)

    # Skull voxels should have cortical bone properties
    skull_mask = labels == 2
    if skull_mask.any():
        assert np.allclose(params["sound_speed"].data[skull_mask], 2800.0)
        assert np.allclose(params["density"].data[skull_mask], 1850.0)

    # Brain voxels should have GM properties
    brain_mask = labels == 4
    if brain_mask.any():
        assert np.allclose(params["sound_speed"].data[brain_mask], 1560.0)


@pytest.mark.skipif(not STL_DIR.exists(), reason="ITRUSST skull STL data not available")
def test_bm7_helmholtz_produces_nonzero_field():
    """CW simulation through realistic skull should produce a nonzero pressure field."""
    from benchmarks.itrusst_bm7 import (
        load_skull_meshes, rasterize_skull, labels_to_params, build_bowl_transducer,
    )
    from openlifu.sim.jwave_if import run_cw_simulation

    outer, inner = load_skull_meshes()
    labels, coords = rasterize_skull(outer, inner, dx_mm=4.0)
    params = labels_to_params(labels, coords)
    tx = build_bowl_transducer(n_elements=64)

    ds, _ = run_cw_simulation(
        arr=tx, params=params, freq=500e3, amplitude=60000.0,
        pml_size=5, tol=1e-2, maxiter=100,
    )
    assert ds.p_amp.data.max() > 0, "Pressure field is all zeros"
    assert np.all(np.isfinite(ds.p_amp.data)), "Non-finite pressure values"


@pytest.mark.skipif(not STL_DIR.exists(), reason="ITRUSST skull STL data not available")
@pytest.mark.xfail(reason="dx=4mm too coarse (1.3 PPW) for reliable attenuation — passes at dx<=1mm on A100")
def test_bm7_skull_attenuates_vs_water():
    """Pressure through realistic skull should be lower than through water."""
    from benchmarks.itrusst_bm7 import (
        load_skull_meshes, rasterize_skull, labels_to_params, build_bowl_transducer,
    )
    from openlifu.sim.jwave_if import run_cw_simulation
    import xarray as xa

    outer, inner = load_skull_meshes()
    labels, coords = rasterize_skull(outer, inner, dx_mm=4.0)

    # Through skull
    params_skull = labels_to_params(labels, coords)
    tx = build_bowl_transducer(n_elements=64)
    ds_skull, _ = run_cw_simulation(
        arr=tx, params=params_skull, freq=500e3, amplitude=60000.0,
        pml_size=5, tol=1e-2, maxiter=100,
    )

    # Water only (same grid)
    shape = labels.shape
    water_params = xa.Dataset({
        "sound_speed": xa.DataArray(np.full(shape, 1500.0, dtype=np.float32), coords=coords,
                                     attrs={"units": "m/s", "ref_value": 1500.0}),
        "density": xa.DataArray(np.full(shape, 1000.0, dtype=np.float32), coords=coords,
                                 attrs={"units": "kg/m^3", "ref_value": 1000.0}),
        "attenuation": xa.DataArray(np.full(shape, 0.0, dtype=np.float32), coords=coords,
                                     attrs={"units": "dB/cm/MHz", "ref_value": 0.0}),
    })
    ds_water, _ = run_cw_simulation(
        arr=tx, params=water_params, freq=500e3, amplitude=60000.0,
        pml_size=5, tol=1e-2, maxiter=100,
    )

    # Skull should attenuate — mean pressure lower
    mean_skull = ds_skull.p_amp.data.mean()
    mean_water = ds_water.p_amp.data.mean()
    assert mean_skull < mean_water, (
        f"Skull should attenuate: skull_mean={mean_skull:.0f} vs water_mean={mean_water:.0f}"
    )
