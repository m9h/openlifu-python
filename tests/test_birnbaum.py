"""TDD: Birnbaum skull dataset loading, remapping, and acoustic simulation."""
import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "benchmarks" / "birnbaum_data"
T1_PATH = DATA_DIR / "GU002_deface.nii"
LABEL_PATH = DATA_DIR / "GU002_label_deface.nii"

# Birnbaum label convention → openlifu convention
BIRNBAUM_TO_OPENLIFU = {
    0: 0,  # background → water
    1: 0,  # air → water
    2: 0,  # air cavities → water (could be 0 or separate)
    3: 5,  # WM → white_matter
    4: 4,  # GM → gray_matter
    5: 3,  # CSF → csf
    6: 2,  # bone → skull
    7: 1,  # scalp → scalp (not present in GU002 apparently)
}


def remap_birnbaum_labels(labels):
    out = np.zeros_like(labels, dtype=np.int32)
    for src, dst in BIRNBAUM_TO_OPENLIFU.items():
        out[labels == src] = dst
    return out


@pytest.mark.skipif(not LABEL_PATH.exists(), reason="Birnbaum data not downloaded")
def test_load_birnbaum_labels():
    """Should load and have expected tissue types."""
    import nibabel as nib
    lab = nib.load(str(LABEL_PATH))
    data = np.asarray(lab.get_fdata(), dtype=int)
    assert data.ndim == 3
    assert 6 in np.unique(data), "No bone label (6)"
    assert data.shape == (186, 222, 220)


@pytest.mark.skipif(not LABEL_PATH.exists(), reason="Birnbaum data not downloaded")
def test_birnbaum_remap():
    """Remapped labels should have skull(2) and GM(4)."""
    import nibabel as nib
    raw = np.asarray(nib.load(str(LABEL_PATH)).get_fdata(), dtype=int)
    remapped = remap_birnbaum_labels(raw)
    assert 2 in np.unique(remapped), "No skull after remap"
    assert 4 in np.unique(remapped), "No GM after remap"
    # Bone count should match
    assert np.sum(remapped == 2) == np.sum(raw == 6)


@pytest.mark.skipif(not LABEL_PATH.exists(), reason="Birnbaum data not downloaded")
def test_birnbaum_to_acoustic_properties():
    """Remapped Birnbaum labels should produce valid acoustic property maps."""
    import nibabel as nib
    from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation
    import xarray as xa

    raw = np.asarray(nib.load(str(LABEL_PATH)).get_fdata(), dtype=int)
    labels = remap_birnbaum_labels(raw)

    seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
    shape = labels.shape
    coords = xa.Coordinates({
        dim: xa.DataArray(np.arange(shape[i], dtype=float), dims=[dim], attrs={"units": "mm"})
        for i, dim in enumerate(("x", "y", "z"))
    })
    volume = xa.DataArray(np.zeros(shape), coords=coords)
    params = seg.seg_params(volume)

    # Skull voxels should have c=4080
    skull = labels == 2
    np.testing.assert_allclose(params["sound_speed"].data[skull], 4080.0)
    # GM voxels should have c=1560
    gm = labels == 4
    np.testing.assert_allclose(params["sound_speed"].data[gm], 1560.0)


@pytest.mark.skipif(not T1_PATH.exists(), reason="Birnbaum T1 not downloaded")
def test_birnbaum_pseudo_ct_vs_expert_labels():
    """Compare pseudo-CT bone prediction against expert skull labels."""
    import nibabel as nib
    from benchmarks.pseudo_ct_validation import t1_to_pseudo_ct

    t1 = np.asarray(nib.load(str(T1_PATH)).get_fdata(), dtype=np.float32)
    raw_labels = np.asarray(nib.load(str(LABEL_PATH)).get_fdata(), dtype=int)

    pseudo = t1_to_pseudo_ct(t1, method="plymouth")
    pred_bone = pseudo > 1200  # high HU threshold for bone
    expert_bone = raw_labels == 6

    # Dice coefficient
    intersection = np.sum(pred_bone & expert_bone)
    dice = 2 * intersection / (np.sum(pred_bone) + np.sum(expert_bone) + 1e-8)

    # The simple threshold method won't be great — just verify it's nonzero
    assert dice > 0.01, f"Bone Dice too low: {dice:.4f}"
    # Report the actual Dice for comparison with PR #436 results
    print(f"\nBirnbaum GU002 pseudo-CT bone Dice: {dice:.4f}")
    print(f"  Expert bone voxels: {expert_bone.sum()}")
    print(f"  Predicted bone voxels: {pred_bone.sum()}")
