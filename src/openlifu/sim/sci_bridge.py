"""Bridge between sbi4dwi SCI head model and openlifu jwave simulation.

Loads the SCI head model via sbi4dwi's loader, rasterizes to a regular
grid, maps tissue labels to both acoustic (for jwave) and electrical
(for EEG/EIT) properties, enabling multi-modal simulation from a single
head model.

The SCI head model (University of Utah) provides:
  - Tetrahedral FEM mesh with tissue labels (8 tissues)
  - T1w, T2w, DTI imaging data
  - 128/256-channel EEG electrode configurations

This bridge creates:
  - Acoustic property maps for jwave (c, rho, alpha) via ITRUSST values
  - Electrical conductivity maps for EEG forward modeling
  - Both from the same geometric substrate

Requires: sbi4dwi package (pip install -e /path/to/sbi4dwi)

Usage:
    from openlifu.sim.sci_bridge import load_sci_for_simulation
    params, coords, conductivity = load_sci_for_simulation(
        mesh_path="HeadMesh.mat",
        dx_mm=1.0,
    )
    # params feeds directly to run_cw_simulation or run_simulation
    # conductivity feeds to neurojax BEM solver or sbi4dwi EIT
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xa

log = logging.getLogger(__name__)

# IT'IS-derived tissue electrical conductivity (S/m) at low frequency
# Same label convention as acoustic properties (SCI Institute)
TISSUE_CONDUCTIVITY = {
    0: 0.0,      # background/air
    1: 0.41,     # scalp
    2: 0.01,     # skull (cortical bone)
    3: 1.71,     # CSF
    4: 0.47,     # gray matter (isotropic)
    5: 0.14,     # white matter (isotropic average; anisotropic via DTI)
}


def load_sci_for_simulation(
    mesh_path: str,
    dx_mm: float = 1.0,
    segmentation_path: str | None = None,
) -> Tuple[xa.Dataset, xa.Coordinates, np.ndarray]:
    """Load SCI head model and prepare for jwave + conductivity simulation.

    Args:
        mesh_path: Path to SCI HeadMesh.mat file.
        dx_mm: Grid spacing for rasterization.
        segmentation_path: Optional path to pre-rasterized segmentation
            (.npy or .nii.gz). If provided, skips mesh rasterization.

    Returns:
        params: xarray Dataset with acoustic properties (for jwave)
        coords: xarray Coordinates
        conductivity: 3D float array of electrical conductivity (for EEG/EIT)
    """
    if segmentation_path and Path(segmentation_path).exists():
        labels = _load_presegmented(segmentation_path)
        shape = labels.shape
        coords = _make_coords(shape, dx_mm)
    else:
        labels, coords = _rasterize_from_mesh(mesh_path, dx_mm)

    params = _labels_to_acoustic_params(labels, coords)
    conductivity = _labels_to_conductivity(labels)

    log.info("SCI head model loaded: shape=%s, dx=%.1f mm", labels.shape, dx_mm)
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        tissue = {0: "water", 1: "scalp", 2: "skull", 3: "CSF", 4: "GM", 5: "WM"}.get(u, f"label_{u}")
        log.info("  %s: %d voxels (%.1f%%)", tissue, c, 100 * c / labels.size)

    return params, coords, conductivity


def _rasterize_from_mesh(mesh_path: str, dx_mm: float):
    """Rasterize the SCI tet mesh to a regular grid."""
    try:
        from dmipy_jax.io.sci_head_loader import load_sci_head_mesh
        from dmipy_jax.biophysics.mesh_rasterizer import rasterize_mesh
    except ImportError:
        raise ImportError(
            "sbi4dwi is required for SCI mesh rasterization. "
            "Install with: pip install -e /path/to/sbi4dwi"
        )

    mesh = load_sci_head_mesh(mesh_path)
    points = np.asarray(mesh["points"])
    cells = np.asarray(mesh["cells"]["tetra"])
    tissue = np.asarray(mesh["cell_data"]["tissue"])

    # Determine grid from mesh bounds
    mins = points.min(axis=0) - 5  # 5mm padding
    maxs = points.max(axis=0) + 5
    shape = tuple(int(np.ceil((maxs[i] - mins[i]) / dx_mm)) for i in range(3))

    log.info("Rasterizing SCI mesh: %d vertices, %d tets -> %s grid (dx=%.1f mm)",
             len(points), len(cells), shape, dx_mm)

    labels = rasterize_mesh(
        points=points,
        cells=cells,
        tissue_labels=tissue,
        grid_shape=shape,
        grid_spacing=dx_mm,
        grid_origin=mins,
    )

    coords = _make_coords(shape, dx_mm, origin=mins)
    return labels, coords


def _load_presegmented(path: str):
    """Load pre-rasterized segmentation volume."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix in (".nii", ".gz"):
        import nibabel as nib
        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def _make_coords(shape, dx_mm, origin=None):
    """Build xarray Coordinates for the grid."""
    coords = {}
    for i, dim in enumerate(("x", "y", "z")):
        o = origin[i] if origin is not None else -(shape[i] // 2) * dx_mm
        vals = o + np.arange(shape[i]) * dx_mm
        c = xa.Variable(dim, vals)
        c.attrs["units"] = "mm"
        c.attrs["long_name"] = dim.upper()
        coords[dim] = c
    return xa.Coordinates(coords)


def _labels_to_acoustic_params(labels, coords):
    """Map tissue labels to acoustic property xarray Dataset."""
    from openlifu.seg.seg_methods.heterogeneous import (
        HeterogeneousSkullSegmentation,
    )
    seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)
    volume = xa.DataArray(np.zeros(labels.shape), coords=coords)
    return seg.seg_params(volume)


def _labels_to_conductivity(labels: np.ndarray) -> np.ndarray:
    """Map tissue labels to electrical conductivity (S/m)."""
    sigma = np.zeros(labels.shape, dtype=np.float32)
    for label, value in TISSUE_CONDUCTIVITY.items():
        sigma[labels == label] = value
    return sigma


def conductivity_from_dti(
    diffusion_tensor: np.ndarray,
    tissue_labels: np.ndarray,
    concentration: float = 150.0,
    temperature: float = 310.15,
) -> np.ndarray:
    """Compute anisotropic conductivity from DTI via Nernst-Einstein.

    For white matter voxels, uses the diffusion tensor to compute an
    anisotropic conductivity tensor. For other tissues, returns isotropic
    conductivity from the lookup table.

    Args:
        diffusion_tensor: (Nx, Ny, Nz, 3, 3) diffusion tensors in m²/s
        tissue_labels: (Nx, Ny, Nz) integer labels
        concentration: Ion concentration in mol/m³ (default: 150 mM NaCl)
        temperature: Temperature in K (default: 37°C)

    Returns:
        (Nx, Ny, Nz, 3, 3) conductivity tensor array in S/m.
        Non-WM voxels have sigma * I (isotropic).
    """
    try:
        from dmipy_jax.biophysics.conductivity import nernst_einstein_conductivity
    except ImportError:
        raise ImportError("sbi4dwi required for DTI-to-conductivity conversion")

    import jax.numpy as jnp

    shape = tissue_labels.shape
    sigma_iso = _labels_to_conductivity(tissue_labels)

    # Start with isotropic conductivity for all tissues
    sigma_tensor = np.zeros(shape + (3, 3), dtype=np.float32)
    for i in range(3):
        sigma_tensor[..., i, i] = sigma_iso

    # Override white matter with DTI-derived anisotropic conductivity
    wm_mask = tissue_labels == 5
    if wm_mask.any() and diffusion_tensor is not None:
        D_wm = jnp.array(diffusion_tensor[wm_mask])
        C_wm = jnp.full(D_wm.shape[0], concentration)
        sigma_wm = np.asarray(nernst_einstein_conductivity(
            D_wm, C_wm, temperature=temperature,
        ))
        sigma_tensor[wm_mask] = sigma_wm

    return sigma_tensor
