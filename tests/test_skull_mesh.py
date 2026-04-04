"""TDD: Load and rasterize ITRUSST skull STL meshes for BM7."""
import numpy as np
import pytest
from pathlib import Path

STL_DIR = Path(__file__).parent.parent / "benchmarks" / "itrusst_data" / "intercomparison" / "skull-stl"


@pytest.fixture
def skull_meshes():
    from benchmarks.itrusst_bm7 import load_skull_meshes
    return load_skull_meshes(str(STL_DIR))


@pytest.mark.skipif(not STL_DIR.exists(), reason="ITRUSST skull STL data not available")
def test_load_skull_stl(skull_meshes):
    outer, inner = skull_meshes
    assert outer.vertices.shape[1] == 3
    assert inner.vertices.shape[1] == 3
    assert len(outer.faces) > 100000, f"Outer mesh too small: {len(outer.faces)} faces"
    assert len(inner.faces) > 50000, f"Inner mesh too small: {len(inner.faces)} faces"


@pytest.mark.skipif(not STL_DIR.exists(), reason="ITRUSST skull STL data not available")
def test_rasterize_skull_to_grid(skull_meshes):
    from benchmarks.itrusst_bm7 import rasterize_skull
    outer, inner = skull_meshes
    labels, coords = rasterize_skull(outer, inner, dx_mm=2.0)

    # Should have water (0), skull (2), and brain (4)
    unique = np.unique(labels)
    assert 0 in unique, "No water voxels"
    assert 2 in unique, "No skull voxels"
    assert 4 in unique, "No brain voxels"

    # Skull should be a shell (fewer voxels than brain)
    n_skull = np.sum(labels == 2)
    n_brain = np.sum(labels == 4)
    assert n_skull < n_brain, f"Skull ({n_skull}) should have fewer voxels than brain ({n_brain})"


@pytest.mark.skipif(not STL_DIR.exists(), reason="ITRUSST skull STL data not available")
def test_skull_shell_thickness(skull_meshes):
    from benchmarks.itrusst_bm7 import rasterize_skull
    outer, inner = skull_meshes
    labels, coords = rasterize_skull(outer, inner, dx_mm=2.0)

    # Skull thickness should be reasonable (3-10mm for human skull)
    # Check along a central ray
    mid_y = labels.shape[1] // 2
    mid_z = labels.shape[2] // 2
    ray = labels[:, mid_y, mid_z]
    skull_indices = np.where(ray == 2)[0]
    if len(skull_indices) > 1:
        # Find the first contiguous run of skull voxels (entry wall)
        diffs = np.diff(skull_indices)
        breaks = np.where(diffs > 1)[0]
        if len(breaks) > 0:
            first_run_end = breaks[0] + 1
        else:
            first_run_end = len(skull_indices)
        wall_thickness = first_run_end * 2.0  # mm
        assert wall_thickness > 2, f"Skull wall too thin: {wall_thickness:.0f} mm"
        assert wall_thickness < 20, f"Skull wall too thick: {wall_thickness:.0f} mm"
