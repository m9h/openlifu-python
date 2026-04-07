Phase Correction Optimization for Transcranial Focusing
========================================================

This tutorial demonstrates gradient-based phase correction through a skull
layer using openlifu's differentiable Helmholtz solver.  By backpropagating
through the acoustic simulation with ``jax.grad``, we can optimise
per-element transducer delays to maximise focal pressure at a desired
target point.

.. contents:: In this tutorial
   :depth: 2
   :local:


Why Phase Correction?
---------------------

When ultrasound travels through the skull, each tissue layer introduces
phase shifts that distort the focus.  Without correction, the focal
pressure can drop by 50--80% compared to a free-water reference.

Traditional approaches compute delays from the speed-of-sound path length
("time-of-flight" or TOF delays).  openlifu goes further: because the
jwave Helmholtz solver is implemented in JAX, we can differentiate the
focal pressure with respect to the element delays and run gradient descent
to find the true optimum.

This approach accounts for:

- Skull curvature and thickness variation
- Refraction at tissue boundaries
- Interference between elements

TOF delays only account for the first effect.


The Optimization Objective
--------------------------

We minimize the **negative** focal pressure amplitude:

.. math::

   \mathcal{L}(\boldsymbol{\tau}) = -|p(\mathbf{x}_\text{target};\, \boldsymbol{\tau})|

where :math:`p` is the complex pressure field from the Helmholtz equation
and :math:`\boldsymbol{\tau}` is the vector of per-element delays.

The gradient :math:`\nabla_{\boldsymbol{\tau}} \mathcal{L}` is computed
by ``jax.grad`` through the entire Helmholtz solver, and delays are
updated by simple gradient descent:

.. math::

   \boldsymbol{\tau}^{(k+1)} = \boldsymbol{\tau}^{(k)} - \eta \, \nabla_{\boldsymbol{\tau}} \mathcal{L}


Step 1: Set Up the Skull Phantom
---------------------------------

.. code-block:: python

   import numpy as np
   import xarray as xa

   def build_skull_phantom(dx_mm=1.0):
       """Grid with a skull layer at x = 20--27 mm, brain beyond."""
       nx = int(80 / dx_mm) + 1
       ny = nz = int(60 / dx_mm) + 1

       x = np.arange(nx) * dx_mm
       y = np.arange(ny) * dx_mm - 30.0
       z = np.arange(nz) * dx_mm - 30.0

       coords = {}
       for dim, vals in [("x", x), ("y", y), ("z", z)]:
           c = xa.Variable(dim, vals)
           c.attrs["units"] = "mm"
           coords[dim] = c
       coords = xa.Coordinates(coords)

       shape = (nx, ny, nz)
       c = np.full(shape, 1500.0, dtype=np.float32)     # water
       rho = np.full(shape, 1000.0, dtype=np.float32)
       alpha = np.full(shape, 0.0, dtype=np.float32)

       skull_start = int(20 / dx_mm)
       skull_end = int(27 / dx_mm)
       c[skull_start:skull_end] = 2800.0      # skull sound speed
       rho[skull_start:skull_end] = 1850.0
       alpha[skull_start:skull_end] = 4.0      # dB/cm/MHz

       brain_start = skull_end
       c[brain_start:] = 1560.0
       rho[brain_start:] = 1040.0

       return xa.Dataset({
           "sound_speed": xa.DataArray(c, coords=coords,
                                       attrs={"units": "m/s", "ref_value": 1500.0}),
           "density": xa.DataArray(rho, coords=coords,
                                    attrs={"units": "kg/m^3", "ref_value": 1000.0}),
           "attenuation": xa.DataArray(alpha, coords=coords,
                                       attrs={"units": "dB/cm/MHz", "ref_value": 0.0}),
       }), coords

   params, coords = build_skull_phantom(dx_mm=1.0)


Step 2: Create a Transducer
----------------------------

.. code-block:: python

   from openlifu.xdc.element import Element
   from openlifu.xdc.transducer import Transducer

   n_elements = 16
   spread = 20e-3  # 20 mm aperture
   positions_y = np.linspace(-spread / 2, spread / 2, n_elements)

   elements = [
       Element(index=i, position=np.array([0.0, y, 0.0]),
               size=np.array([3e-3, 3e-3]), units="m")
       for i, y in enumerate(positions_y)
   ]
   tx = Transducer(id="phase_demo", elements=elements, frequency=500e3, units="m")


Step 3: Compute Baseline Pressures
-----------------------------------

Before optimisation, measure the focal pressure with zero delays and
with random delays to establish the performance range:

.. code-block:: python

   from openlifu.sim.jwave_if import run_cw_simulation

   target_x_idx = int(50 / 1.0)   # 50 mm behind the transducer
   mid_y = params["sound_speed"].shape[1] // 2
   mid_z = params["sound_speed"].shape[2] // 2
   target_idx = (target_x_idx, mid_y, mid_z)

   # Zero delays
   ds_zero, _ = run_cw_simulation(
       arr=tx, params=params, freq=500e3,
       amplitude=60000.0, pml_size=10,
   )
   p_zero = float(ds_zero.p_amp.data[target_idx])

   # Random delays (worst case)
   rng = np.random.default_rng(42)
   random_delays = rng.uniform(0, 2e-6, tx.numelements())
   ds_random, _ = run_cw_simulation(
       arr=tx, params=params, delays=random_delays,
       freq=500e3, amplitude=60000.0, pml_size=10,
   )
   p_random = float(ds_random.p_amp.data[target_idx])


Step 4: Run Gradient-Based Optimization
----------------------------------------

.. code-block:: python

   from openlifu.sim.phase_correction import optimize_delays

   opt_delays = optimize_delays(
       tx, params,
       target_idx=target_idx,
       freq=500e3,
       init_delays=random_delays,
       n_steps=50,
       lr=1e-7,
       pml_size=10,
   )

   # Measure the optimised focal pressure
   ds_opt, _ = run_cw_simulation(
       arr=tx, params=params, delays=opt_delays,
       freq=500e3, amplitude=60000.0, pml_size=10,
   )
   p_opt = float(ds_opt.p_amp.data[target_idx])

   print(f"Zero delays:     {p_zero/1e3:.1f} kPa")
   print(f"Random delays:   {p_random/1e3:.1f} kPa")
   print(f"Optimised:       {p_opt/1e3:.1f} kPa")


Tuning the Learning Rate
~~~~~~~~~~~~~~~~~~~~~~~~~

The optimal learning rate depends on the grid resolution and transducer
geometry:

- **dx = 1.0 mm**: ``lr = 1e-7`` typically converges in 30--50 steps.
- **dx = 0.5 mm**: Gradients are larger; try ``lr = 5e-8``.
- If focal pressure oscillates, reduce ``lr`` by a factor of 2--5.
- If convergence is too slow, increase ``lr`` or increase ``n_steps``.


Step 5: Using the Gradient Directly
-------------------------------------

For more control, compute the gradient without the optimisation loop:

.. code-block:: python

   from openlifu.sim.phase_correction import compute_focal_gradient

   grad = compute_focal_gradient(
       tx, params,
       target_idx=target_idx,
       freq=500e3,
       delays=np.zeros(tx.numelements()),
   )

   # grad[i] indicates how much a small delay change on element i
   # would increase (negative = decrease) the focal pressure.
   print(f"Gradient norm: {np.linalg.norm(grad):.1e}")


Comparison to TOF Delays
-------------------------

Time-of-flight (TOF) delays assume straight-ray propagation and a
homogeneous medium:

.. code-block:: python

   from openlifu.bf.delay_methods import Direct

   tof_method = Direct(c0=1500.0)
   from openlifu.geo import Point
   target = Point(position=[50.0, 0.0, 0.0], units="mm")
   tof_delays = tof_method.calc_delays(tx, target, params)

TOF is a good initialisation for gradient optimisation.  Starting from
TOF delays instead of random delays typically reduces the number of
gradient steps needed for convergence.


Running on GPU
--------------

Phase correction is computationally intensive.  The ``benchmarks/phase_correction_demo.py``
script includes a `Modal <https://modal.com>`_ deployment for A100 GPU execution:

.. code-block:: bash

   uv run modal run benchmarks/phase_correction_demo.py --dx-mm 0.5 --n-steps 100

Typical runtimes (A100, 500 kHz, 16 elements):

============  ===========  =================
Grid spacing  Grid size    50-step opt time
============  ===========  =================
1.0 mm        81x61x61     ~15 s
0.5 mm        161x121x121  ~60 s
============  ===========  =================


See Also
--------

- :mod:`openlifu.sim.phase_correction` -- Phase correction API reference
- :mod:`openlifu.sim.jwave_if` -- jwave simulation backend
- :doc:`heterogeneous_skull` -- Heterogeneous skull modeling tutorial
- :doc:`jwave_backend` -- jwave backend details
