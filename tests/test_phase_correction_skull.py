"""TDD: Phase correction through heterogeneous skull layers."""
import numpy as np
import pytest
import xarray as xa


def _make_skull_phantom():
    """Small grid with a skull-like layer for phase correction testing."""
    from openlifu.sim import SimSetup
    ss = SimSetup(spacing=1.0, units="mm",
                  x_extent=(-5, 57), y_extent=(-15, 16), z_extent=(-15, 16))
    coords = ss.get_coords()
    shape = tuple(coords.sizes.values())

    # Water everywhere, then add skull layer at x=10-17mm
    c = np.full(shape, 1500.0, dtype=np.float32)
    rho = np.full(shape, 1000.0, dtype=np.float32)
    alpha = np.full(shape, 0.0, dtype=np.float32)

    # Skull layer
    x = coords["x"].values
    skull_mask = (x >= 10) & (x <= 17)
    c[skull_mask, :, :] = 2800.0
    rho[skull_mask, :, :] = 1850.0
    alpha[skull_mask, :, :] = 4.0

    # Brain after skull
    brain_mask = x > 17
    c[brain_mask, :, :] = 1560.0
    rho[brain_mask, :, :] = 1040.0
    alpha[brain_mask, :, :] = 0.3

    params = xa.Dataset({
        "sound_speed": xa.DataArray(c, coords=coords, attrs={"units": "m/s", "ref_value": 1500.0}),
        "density": xa.DataArray(rho, coords=coords, attrs={"units": "kg/m^3", "ref_value": 1000.0}),
        "attenuation": xa.DataArray(alpha, coords=coords, attrs={"units": "dB/cm/MHz", "ref_value": 0.0}),
    })
    return params, coords


def _make_array():
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer
    elements = [
        Element(index=i, position=np.array([0., y * 1e-3, 0.]),
                size=np.array([3e-3, 3e-3]), units="m")
        for i, y in enumerate([-10, -5, 0, 5, 10])
    ]
    return Transducer(id="test_arr", elements=elements, frequency=500e3, units="m")


def test_skull_aberration_reduces_focus():
    """A skull layer should reduce focal pressure compared to water-only."""
    from openlifu.sim.jwave_if import run_cw_simulation
    from openlifu.sim import SimSetup
    from openlifu.seg.seg_methods import UniformWater

    skull_params, coords = _make_skull_phantom()
    tx = _make_array()
    target_idx = (40, 16, 16)  # x=35mm, on-axis, past skull

    # Through skull
    ds_skull, _ = run_cw_simulation(arr=tx, params=skull_params, freq=500e3,
                                     amplitude=60000.0, pml_size=8)
    p_skull = ds_skull.p_amp.data[target_idx]

    # Water only
    ss = SimSetup(spacing=1.0, units="mm",
                  x_extent=(-5, 57), y_extent=(-15, 16), z_extent=(-15, 16))
    water_params = UniformWater().ref_params(ss.get_coords())
    ds_water, _ = run_cw_simulation(arr=tx, params=water_params, freq=500e3,
                                     amplitude=60000.0, pml_size=8)
    p_water = ds_water.p_amp.data[target_idx]

    assert p_skull < p_water, (
        f"Skull should reduce focal pressure: skull={p_skull:.0f} vs water={p_water:.0f}"
    )


def test_phase_correction_recovers_pressure_through_skull():
    """Optimized delays should increase focal pressure through skull."""
    from openlifu.sim.phase_correction import optimize_delays
    from openlifu.sim.jwave_if import run_cw_simulation

    skull_params, coords = _make_skull_phantom()
    tx = _make_array()
    target_idx = (40, 16, 16)

    # Zero delays through skull
    ds_zero, _ = run_cw_simulation(arr=tx, params=skull_params, freq=500e3,
                                    amplitude=60000.0, pml_size=8)
    p_zero = ds_zero.p_amp.data[target_idx]

    # Optimized delays
    opt_delays = optimize_delays(
        tx, skull_params, target_idx=target_idx, freq=500e3,
        n_steps=30, lr=1e-7,
    )

    ds_opt, _ = run_cw_simulation(arr=tx, params=skull_params, delays=opt_delays,
                                   freq=500e3, amplitude=60000.0, pml_size=8)
    p_opt = ds_opt.p_amp.data[target_idx]

    improvement = (p_opt - p_zero) / p_zero * 100
    assert p_opt > p_zero, (
        f"Phase correction should improve: zero={p_zero:.0f}, opt={p_opt:.0f} Pa "
        f"({improvement:+.1f}%)"
    )


def test_gradient_is_finite_through_skull():
    """Gradient should be finite through heterogeneous skull."""
    from openlifu.sim.phase_correction import compute_focal_gradient

    skull_params, _ = _make_skull_phantom()
    tx = _make_array()
    target_idx = (40, 16, 16)

    grad = compute_focal_gradient(
        tx, skull_params, target_idx=target_idx, freq=500e3,
        delays=np.zeros(tx.numelements()),
    )
    assert np.all(np.isfinite(grad)), f"Non-finite gradients: {grad}"
    assert grad.shape == (tx.numelements(),)
