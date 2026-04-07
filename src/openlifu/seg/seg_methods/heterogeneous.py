"""
Heterogeneous skull segmentation for transcranial focused ultrasound.

Provides tissue-resolved acoustic property mapping from either:
  - Pre-computed label arrays (from SimNIBS charm, GRACE, SCI mesh, etc.)
  - T1w MRI intensity via pseudo-CT estimation

This is the bridge between sbi4dwi's JAX tissue property pipeline and
openlifu's k-Wave simulation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.seg.material import MATERIALS, PARAM_INFO, Material
from openlifu.seg.seg_method import SegmentationMethod


# Expanded materials dict with tissue types for skull modeling
_HETEROGENEOUS_MATERIALS = {
    "water": Material(name="Water", sound_speed=1500.0, density=1000.0, attenuation=0.0,
                      specific_heat=4182.0, thermal_conductivity=0.598),
    "scalp": Material(name="Scalp", sound_speed=1610.0, density=1090.0, attenuation=3.5,
                      specific_heat=3391.0, thermal_conductivity=0.37),
    "skull": Material(name="Skull", sound_speed=4080.0, density=1900.0, attenuation=4.74,
                      specific_heat=1100.0, thermal_conductivity=0.30),
    "csf": Material(name="CSF", sound_speed=1500.0, density=1000.0, attenuation=0.0,
                    specific_heat=4182.0, thermal_conductivity=0.598),
    "gray_matter": Material(name="Gray Matter", sound_speed=1560.0, density=1040.0, attenuation=5.3,
                            specific_heat=3630.0, thermal_conductivity=0.51),
    "white_matter": Material(name="White Matter", sound_speed=1560.0, density=1040.0, attenuation=5.3,
                             specific_heat=3600.0, thermal_conductivity=0.50),
}

# Map integer tissue labels to material keys (openlifu convention)
_LABEL_TO_MATERIAL = {
    0: "water",
    1: "scalp",
    2: "skull",
    3: "csf",
    4: "gray_matter",
    5: "white_matter",
}

# SimNIBS CHARM/headreco uses a different label convention.
# Remap before passing to HeterogeneousSkullSegmentation(source='labels').
SIMNIBS_TO_OPENLIFU = {
    0: 0,  # background → water
    1: 5,  # WM → white_matter
    2: 4,  # GM → gray_matter
    3: 3,  # CSF → csf
    4: 2,  # bone → skull
    5: 1,  # skin → scalp
}


def remap_simnibs_labels(labels):
    """Convert SimNIBS tissue labels to the openlifu convention.

    SimNIBS CHARM / headreco assigns: 0=background, 1=WM, 2=GM, 3=CSF,
    4=bone, 5=skin.  openlifu expects: 0=water, 1=scalp, 2=skull, 3=CSF,
    4=gray_matter, 5=white_matter.  This function performs a voxel-wise
    relabelling so that downstream ``HeterogeneousSkullSegmentation`` can
    consume the result directly.

    Parameters
    ----------
    labels : numpy.ndarray
        3-D integer array of SimNIBS tissue labels.

    Returns
    -------
    numpy.ndarray
        3-D integer array with labels remapped to the openlifu convention.
        Shape and dtype match *labels*.

    Examples
    --------
    >>> import numpy as np
    >>> from openlifu.seg.seg_methods.heterogeneous import remap_simnibs_labels
    >>> simnibs = np.array([[[0, 4, 5]]])  # bg, bone, skin
    >>> remap_simnibs_labels(simnibs)
    array([[[0, 2, 1]]])
    """
    import numpy as np
    out = np.zeros_like(labels)
    for src, dst in SIMNIBS_TO_OPENLIFU.items():
        out[labels == src] = dst
    return out


@dataclass
class HeterogeneousSkullSegmentation(SegmentationMethod):
    """
    Segmentation method that assigns heterogeneous acoustic properties
    based on tissue labels or pseudo-CT estimation.

    Modes:
        - source='labels': Uses a pre-computed 3D integer label array
          (from SimNIBS charm, GRACE, SCI mesh rasterization, etc.)
        - source='pseudoct': Uses T1w MRI intensity to estimate skull
          acoustic properties via pseudo-CT conversion.

    The _segment() method returns integer labels (for the SegmentationMethod
    contract). The _map_params() method is overridden to produce heterogeneous
    acoustic property arrays based on the label-to-material mapping.
    """

    source: str = "labels"
    label_array: Optional[np.ndarray] = field(default=None, repr=False)
    pseudoct_method: str = "plymouth"

    def __init__(
        self,
        source: str = "labels",
        label_array: Optional[np.ndarray] = None,
        pseudoct_method: str = "plymouth",
        materials: Optional[dict[str, Material]] = None,
        ref_material: str = "water",
    ):
        if materials is None:
            materials = _HETEROGENEOUS_MATERIALS.copy()
        if ref_material not in materials:
            ref_material = "water"
        super().__init__(materials=materials, ref_material=ref_material)
        self.source = source
        self.label_array = label_array
        self.pseudoct_method = pseudoct_method

    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        """
        Produce integer tissue label array.

        For 'labels' mode: returns the stored label array.
        For 'pseudoct' mode: thresholds T1w intensity to produce approximate labels.
        """
        if self.source == "labels" and self.label_array is not None:
            # Clip or pad label_array to match volume shape
            labels = self._fit_labels_to_volume(volume)
            return xa.DataArray(labels, coords=volume.coords)

        elif self.source == "pseudoct":
            return self._segment_pseudoct(volume)

        else:
            # Fallback: all water
            return xa.DataArray(
                np.zeros(volume.shape, dtype=int), coords=volume.coords
            )

    def _fit_labels_to_volume(self, volume: xa.DataArray) -> np.ndarray:
        """Resize or crop ``self.label_array`` to match volume dimensions.

        When the stored label array does not exactly match the simulation
        grid, this method crops overlapping regions or zero-pads missing
        regions so that the result has the same shape as *volume*.

        Parameters
        ----------
        volume : xarray.DataArray
            Reference volume whose ``.shape`` defines the target grid.

        Returns
        -------
        numpy.ndarray
            Integer label array with shape ``volume.shape``.
        """
        labels = np.asarray(self.label_array, dtype=int)
        vol_shape = volume.shape

        if labels.shape == vol_shape:
            return labels

        # Crop or pad to match
        result = np.zeros(vol_shape, dtype=int)
        slices = tuple(
            slice(0, min(labels.shape[i], vol_shape[i]))
            for i in range(len(vol_shape))
        )
        result[slices] = labels[slices]
        return result

    def _segment_pseudoct(self, volume: xa.DataArray) -> xa.DataArray:
        """Approximate tissue segmentation from normalised T1w MRI intensity.

        Applies a simple threshold-based classification to produce integer
        tissue labels.  The thresholds assume the input volume has been
        normalised to [0, 1].  This is intended as a fast approximation
        for treatment-planning previews; for research-quality results use
        pre-computed labels from SimNIBS CHARM or a dedicated pseudo-CT
        model (e.g. Plymouth).

        Threshold scheme (on normalised intensity):

        ============  =================  =========
        Range         Tissue             Label
        ============  =================  =========
        < 0.15        background / air   0
        0.15 -- 0.30  skull              2
        0.30 -- 0.45  CSF                3
        0.45 -- 0.70  gray matter        4
        >= 0.70       white matter       5
        ============  =================  =========

        Parameters
        ----------
        volume : xarray.DataArray
            Normalised T1w MRI volume (values in [0, 1]).

        Returns
        -------
        xarray.DataArray
            Integer tissue labels with the same coordinates as *volume*.
        """
        data = volume.data
        labels = np.full(data.shape, 4, dtype=int)  # default GM
        labels[data < 0.15] = 0   # background
        labels[(data >= 0.15) & (data < 0.3)] = 2  # skull (dark in T1)
        labels[(data >= 0.3) & (data < 0.45)] = 3  # CSF
        labels[(data >= 0.45) & (data < 0.7)] = 4  # gray matter
        labels[data >= 0.7] = 5   # white matter
        return xa.DataArray(labels, coords=volume.coords)

    def _map_params(self, seg: xa.DataArray, materials: dict | None = None):
        """
        Override base class to map integer labels to material properties.

        Uses the _LABEL_TO_MATERIAL mapping to convert SCI-style integer labels
        to the named materials in self.materials.
        """
        materials = self.materials if materials is None else materials
        ref_mat = materials[self.ref_material]
        params = xa.Dataset()

        for param_id in PARAM_INFO:
            info = Material.param_info(param_id)
            param_data = np.full(seg.shape, ref_mat.get_param(param_id))

            for label_int, material_key in _LABEL_TO_MATERIAL.items():
                if material_key in materials:
                    mat = materials[material_key]
                    param_data[seg.data == label_int] = mat.get_param(param_id)

            param = xa.DataArray(
                param_data,
                coords=seg.coords,
                attrs={
                    "units": info["units"],
                    "long_name": info["name"],
                    "ref_value": ref_mat.get_param(param_id),
                },
            )
            params[param_id] = param

        params.attrs["ref_material"] = ref_mat
        return params

    def to_table(self) -> pd.DataFrame:
        records = [
            {"Name": "Type", "Value": "Heterogeneous Skull", "Unit": ""},
            {"Name": "Source", "Value": self.source, "Unit": ""},
        ]
        if self.source == "pseudoct":
            records.append(
                {"Name": "Pseudo-CT Method", "Value": self.pseudoct_method, "Unit": ""}
            )
        records.append(
            {"Name": "Materials", "Value": len(self.materials), "Unit": "tissues"}
        )
        return pd.DataFrame.from_records(records)

    def to_dict(self):
        d = super().to_dict()
        d["source"] = self.source
        d["pseudoct_method"] = self.pseudoct_method
        # Don't serialize label_array (too large) — it must be provided at runtime
        d.pop("label_array", None)
        return d
