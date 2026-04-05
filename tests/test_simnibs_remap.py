"""TDD: SimNIBS label remapping for HeterogeneousSkullSegmentation."""
import numpy as np


def test_remap_simnibs_roundtrip():
    """Remapping should convert all 6 SimNIBS tissue types correctly."""
    from openlifu.seg.seg_methods.heterogeneous import remap_simnibs_labels

    # SimNIBS labels: 0=bg, 1=WM, 2=GM, 3=CSF, 4=bone, 5=skin
    simnibs = np.array([0, 1, 2, 3, 4, 5])
    openlifu = remap_simnibs_labels(simnibs)

    # Expected openlifu: 0=water, 5=WM, 4=GM, 3=CSF, 2=skull, 1=scalp
    expected = np.array([0, 5, 4, 3, 2, 1])
    np.testing.assert_array_equal(openlifu, expected)


def test_remap_preserves_shape():
    from openlifu.seg.seg_methods.heterogeneous import remap_simnibs_labels

    labels = np.random.randint(0, 6, size=(10, 10, 10))
    remapped = remap_simnibs_labels(labels)
    assert remapped.shape == labels.shape


def test_remap_into_heterogeneous_segmentation():
    """Remapped labels should produce valid acoustic property maps."""
    from openlifu.seg.seg_methods.heterogeneous import (
        HeterogeneousSkullSegmentation, remap_simnibs_labels,
    )
    import xarray as xa

    # Fake SimNIBS volume: skin shell, bone, CSF, GM, WM
    shape = (20, 20, 20)
    simnibs_labels = np.zeros(shape, dtype=int)
    simnibs_labels[2:-2, 2:-2, 2:-2] = 5   # skin
    simnibs_labels[4:-4, 4:-4, 4:-4] = 4   # bone
    simnibs_labels[5:-5, 5:-5, 5:-5] = 3   # CSF
    simnibs_labels[6:-6, 6:-6, 6:-6] = 2   # GM
    simnibs_labels[8:-8, 8:-8, 8:-8] = 1   # WM

    openlifu_labels = remap_simnibs_labels(simnibs_labels)

    seg = HeterogeneousSkullSegmentation(source="labels", label_array=openlifu_labels)
    coords = xa.Coordinates({
        dim: xa.DataArray(np.arange(20, dtype=float), dims=[dim], attrs={"units": "mm"})
        for dim in ("x", "y", "z")
    })
    volume = xa.DataArray(np.zeros(shape), coords=coords)
    params = seg.seg_params(volume)

    # Scalp region (index 3 is inside [2:-2] but outside [4:-4])
    assert params["sound_speed"].data[3, 3, 3] == 1610.0  # skin → scalp
    # Skull region (index 4 is inside [4:-4] but outside [5:-5])
    assert params["sound_speed"].data[4, 4, 4] == 4080.0  # bone → skull
    # Brain region
    assert params["sound_speed"].data[7, 7, 7] == 1560.0  # GM
    # Water outside
    assert params["sound_speed"].data[0, 0, 0] == 1500.0
