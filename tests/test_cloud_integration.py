"""TDD: Verify jwave results integrate with OpenLIFU cloud/Solution pipeline."""
import numpy as np
import pytest
import xarray as xa


def test_jwave_cw_result_compatible_with_solution():
    """CW simulation result should be storable in a Solution object."""
    from openlifu.plan.solution import Solution
    from openlifu.bf import Pulse, Sequence
    from openlifu.geo import Point
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer
    from openlifu.sim.jwave_if import run_cw_simulation
    from openlifu.sim import SimSetup
    from openlifu.seg.seg_methods import UniformWater

    # Small grid
    ss = SimSetup(spacing=2.0, units="mm",
                  x_extent=(-5, 27), y_extent=(-10, 11), z_extent=(-10, 11))
    coords = ss.get_coords()
    params = UniformWater().ref_params(coords)

    el = Element(index=0, position=np.array([0., 0., 0.]),
                 size=np.array([5e-3, 5e-3]), units="m")
    tx = Transducer(id="test", elements=[el], frequency=500e3, units="m")

    ds, _ = run_cw_simulation(arr=tx, params=params, freq=500e3,
                               amplitude=60000.0, pml_size=5)

    # The CW result has p_amp instead of p_max/p_min.
    # For Solution compatibility, we need to provide simulation_result
    # as an xarray Dataset. Verify structure is valid.
    assert isinstance(ds, xa.Dataset)
    assert "p_amp" in ds.data_vars
    assert "intensity" in ds.data_vars

    # Build a Solution with CW results mapped to the expected fields
    sim_result = xa.Dataset({
        "p_max": ds["p_amp"].rename("p_max"),
        "p_min": ds["p_amp"].rename("p_min"),  # CW: p_amp ≈ p_max ≈ p_min
        "intensity": ds["intensity"],
    })

    target = Point(id="t1", position=np.array([10., 0., 0.]), dims=["x", "y", "z"], units="mm")
    solution = Solution(
        id="test_cw",
        name="CW Test Solution",
        protocol_id="test_proto",
        transducer=tx,
        delays=np.zeros((1, 1)),
        apodizations=np.ones((1, 1)),
        pulse=Pulse(),
        sequence=Sequence(),
        foci=[target],
        target=target,
        simulation_result=sim_result,
    )
    assert solution.id == "test_cw"
    assert solution.simulation_result["p_max"].data.max() > 0


def test_protocol_calc_solution_jwave_cw_backend():
    """protocol.calc_solution with jwave_cw backend should return valid Solution."""
    from openlifu.plan.protocol import Protocol
    from openlifu.sim import SimSetup
    from openlifu.seg.seg_methods import UniformWater
    from openlifu.xdc.element import Element
    from openlifu.xdc.transducer import Transducer
    from openlifu.geo import Point
    from unittest.mock import patch

    ss = SimSetup(spacing=2.0, units="mm",
                  x_extent=(-5, 27), y_extent=(-10, 11), z_extent=(-10, 11))

    el = Element(index=0, position=np.array([0., 0., 0.]),
                 size=np.array([5e-3, 5e-3]), units="m")
    tx = Transducer(id="test", elements=[el], frequency=500e3, units="m")
    target = Point(id="t1", position=np.array([10., 0., 0.]), dims=["x", "y", "z"], units="mm")

    # Mock run_simulation to avoid actual sim in unit test
    mock_ds = xa.Dataset({
        "p_max": xa.DataArray(data=np.ones((17, 11, 11)), dims=["x", "y", "z"],
                               attrs={"units": "Pa"}),
        "p_min": xa.DataArray(data=np.ones((17, 11, 11)), dims=["x", "y", "z"],
                               attrs={"units": "Pa"}),
        "intensity": xa.DataArray(data=np.ones((17, 11, 11)), dims=["x", "y", "z"],
                                   attrs={"units": "W/cm^2"}),
    }, coords={
        "x": xa.DataArray(dims=["x"], data=np.linspace(-5, 27, 17), attrs={"units": "mm"}),
        "y": xa.DataArray(dims=["y"], data=np.linspace(-10, 11, 11), attrs={"units": "mm"}),
        "z": xa.DataArray(dims=["z"], data=np.linspace(-10, 11, 11), attrs={"units": "mm"}),
    })

    protocol = Protocol(id="test", sim_setup=ss, seg_method=UniformWater())

    with patch("openlifu.plan.protocol.run_simulation", return_value=(mock_ds, None)):
        solution, sim_result, analysis = protocol.calc_solution(
            target=target, transducer=tx,
            simulate=True, scale=False,
            sim_backend="jwave_cw",
        )

    assert solution is not None
    assert sim_result is not None
