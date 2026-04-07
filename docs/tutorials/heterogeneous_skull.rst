Heterogeneous Skull Modeling: From Segmentation to Simulation
=============================================================

This tutorial walks through the complete heterogeneous skull workflow
introduced in the ``feature/heterogeneous-skull-segmentation`` branch:
from tissue segmentation through acoustic property assignment to
full-wave simulation with the jwave backend.

.. contents:: In this tutorial
   :depth: 2
   :local:

Background
----------

Transcranial focused ultrasound (tFUS) must propagate through multiple
tissue layers -- scalp, skull, CSF, gray matter, and white matter -- each
with distinct acoustic properties.  Accurate modeling of these layers is
essential for treatment planning and phase-correction algorithms.

openlifu's heterogeneous skull pipeline supports three label sources:

1. **SimNIBS CHARM / headreco** -- MRI-based segmentation (5 tissue classes)
2. **SCI head model** -- University of Utah FEM mesh (tetrahedral, 8 tissues)
3. **Pseudo-CT from T1w** -- Threshold-based approximation (fast preview)

All sources are remapped to a canonical **openlifu label convention**:

===========  =========
Label value  Tissue
===========  =========
0            Water
1            Scalp
2            Skull
3            CSF
4            Gray matter
5            White matter
===========  =========


Step 1: Preparing Label Arrays
------------------------------

From SimNIBS
~~~~~~~~~~~~

SimNIBS CHARM assigns labels as: 0=background, 1=WM, 2=GM, 3=CSF, 4=bone,
5=skin.  These must be remapped before use:

.. code-block:: python

   import nibabel as nib
   from openlifu.seg.seg_methods.heterogeneous import remap_simnibs_labels

   # Load the SimNIBS segmentation (e.g. from charm output)
   seg_img = nib.load("m2m_subj/final_tissues.nii.gz")
   simnibs_labels = seg_img.get_fdata().astype(int)

   # Remap to openlifu convention
   labels = remap_simnibs_labels(simnibs_labels)

From the SCI Head Model
~~~~~~~~~~~~~~~~~~~~~~~~

The SCI bridge module handles mesh rasterization and label assignment:

.. code-block:: python

   from openlifu.sim.sci_bridge import load_sci_for_simulation

   params, coords, conductivity = load_sci_for_simulation(
       mesh_path="HeadMesh.mat",
       dx_mm=1.0,
   )
   # params is ready for jwave simulation
   # conductivity can be used for EEG forward modeling

For pre-rasterized labels (e.g. from a ``.npy`` file):

.. code-block:: python

   from openlifu.sim.sci_bridge import load_sci_for_simulation

   params, coords, conductivity = load_sci_for_simulation(
       mesh_path="HeadMesh.mat",
       dx_mm=1.0,
       segmentation_path="labels_1mm.npy",  # skip rasterization
   )

From the Birnbaum Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

The Birnbaum dataset uses an 8-class convention.  See
``tests/test_birnbaum.py`` for the remapping:

.. code-block:: python

   import numpy as np

   BIRNBAUM_TO_OPENLIFU = {
       0: 0,  # background -> water
       1: 0,  # air -> water
       2: 0,  # air cavities -> water
       3: 5,  # WM -> white_matter
       4: 4,  # GM -> gray_matter
       5: 3,  # CSF -> csf
       6: 2,  # bone -> skull
       7: 1,  # scalp -> scalp
   }

   def remap_birnbaum_labels(labels):
       out = np.zeros_like(labels, dtype=np.int32)
       for src, dst in BIRNBAUM_TO_OPENLIFU.items():
           out[labels == src] = dst
       return out


Synthetic Phantom (for Testing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick experimentation, build a layered phantom as in the demo script:

.. code-block:: python

   import numpy as np

   def make_layered_skull_labels(shape, spacing_mm=1.0):
       """Concentric layers along the z axis."""
       nz = shape[2]
       z_mm = np.arange(nz) * spacing_mm
       labels = np.zeros(shape, dtype=int)
       for z_thresh, label in [(5, 1), (9, 2), (16, 3), (18, 4), (24, 5)]:
           labels[:, :, z_mm >= z_thresh] = label
       return labels

   labels = make_layered_skull_labels((61, 61, 65))


Step 2: Building Acoustic Property Maps
----------------------------------------

The :class:`~openlifu.seg.seg_methods.heterogeneous.HeterogeneousSkullSegmentation`
class maps integer tissue labels to spatially varying acoustic properties.

.. code-block:: python

   import numpy as np
   import xarray as xa
   from openlifu.seg.seg_methods.heterogeneous import HeterogeneousSkullSegmentation

   # Assume `labels` is a 3D integer array and `coords` are xarray Coordinates

   seg = HeterogeneousSkullSegmentation(source="labels", label_array=labels)

   # Create a dummy volume (the labels override any intensity data)
   volume = xa.DataArray(np.zeros(labels.shape), coords=coords)

   # Generate acoustic property maps
   params = seg.seg_params(volume)

The resulting ``params`` is an ``xarray.Dataset`` with variables:

- ``sound_speed`` -- longitudinal sound speed (m/s)
- ``density`` -- mass density (kg/m^3)
- ``attenuation`` -- absorption coefficient (dB/cm/MHz)
- ``specific_heat`` -- specific heat capacity (J/kg/K)
- ``thermal_conductivity`` -- thermal conductivity (W/m/K)

Each variable carries a ``ref_value`` attribute (the water reference) and
spatially varying data reflecting the tissue labels.

**Default material properties:**

================  ============  ===========  ===========
Tissue            c (m/s)       rho (kg/m3)  alpha (dB/cm/MHz)
================  ============  ===========  ===========
Water             1500          1000         0.0
Scalp             1610          1090         3.5
Skull             4080          1900         4.74
CSF               1500          1000         0.0
Gray matter       1560          1040         5.3
White matter      1560          1040         5.3
================  ============  ===========  ===========


Step 3: Running a jwave Simulation
-----------------------------------

With heterogeneous property maps in hand, run a full-wave simulation using
the jwave backend:

.. code-block:: python

   from openlifu.sim.jwave_if import run_simulation, run_cw_simulation

   # Time-domain simulation (pulsed FUS)
   ds_td, raw_td = run_simulation(
       arr=transducer,
       params=params,
       freq=500e3,
       cycles=10,
       cfl=0.3,
       pml_size=10,
   )

   # Continuous-wave simulation (Helmholtz solver)
   ds_cw, raw_cw = run_cw_simulation(
       arr=transducer,
       params=params,
       freq=500e3,
       amplitude=60000.0,
       pml_size=10,
       tol=1e-3,
   )

Time-domain returns ``p_max``, ``p_min``, and ``intensity``.  CW returns
``p_amp``, ``p_phase``, and ``intensity``.

To compare against a homogeneous water reference:

.. code-block:: python

   ds_water, _ = run_cw_simulation(
       arr=transducer,
       params=params,
       freq=500e3,
       amplitude=60000.0,
       ref_values_only=True,   # uses only the scalar ref_value attributes
   )


Step 4: Inspecting Results
--------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   mid_y = params["sound_speed"].shape[1] // 2

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   axes[0].imshow(ds_cw["p_amp"].data[:, mid_y, :].T,
                  origin="lower", cmap="hot")
   axes[0].set_title("Pressure Amplitude (Pa)")

   axes[1].imshow(ds_cw["intensity"].data[:, mid_y, :].T,
                  origin="lower", cmap="viridis")
   axes[1].set_title("Intensity (W/cm^2)")

   axes[2].imshow(params["sound_speed"].data[:, mid_y, :].T,
                  origin="lower", cmap="gray")
   axes[2].set_title("Sound Speed (m/s)")

   plt.tight_layout()
   plt.show()


Complete Example
----------------

The full demo script is available at ``examples/demo_heterogeneous_skull.py``:

.. code-block:: bash

   uv run python examples/demo_heterogeneous_skull.py
   uv run python examples/demo_heterogeneous_skull.py --water   # homogeneous comparison
   uv run python examples/demo_heterogeneous_skull.py --save-png skull_sim.png


See Also
--------

- :mod:`openlifu.seg.seg_methods.heterogeneous` -- Heterogeneous segmentation API
- :mod:`openlifu.sim.sci_bridge` -- SCI head model bridge
- :mod:`openlifu.sim.jwave_if` -- jwave simulation backend
- :doc:`phase_correction` -- Phase correction tutorial (next step after simulation)
