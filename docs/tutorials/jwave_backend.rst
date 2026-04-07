The jwave Backend: Differentiable Acoustic Simulation
======================================================

This tutorial explains when and how to use the jwave simulation backend,
which provides JAX-based differentiable acoustic simulation as an
alternative to k-Wave.

.. contents:: In this tutorial
   :depth: 2
   :local:


Overview
--------

openlifu supports two acoustic simulation backends:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - k-Wave (``kwave_if``)
     - jwave (``jwave_if``)
   * - Language
     - MATLAB/C++ via k-wave-python
     - Pure JAX (Python)
   * - Mode
     - Time-domain only
     - Time-domain + CW (Helmholtz)
   * - Autodiff
     - No
     - Yes (``jax.grad``)
   * - GPU
     - Via CUDA (k-Wave)
     - Via JAX/XLA (CUDA, ROCm, TPU)
   * - Medium
     - Heterogeneous
     - Heterogeneous
   * - Output
     - ``p_max``, ``p_min``, intensity
     - Same (TD) or ``p_amp``, ``p_phase`` (CW)
   * - Install
     - ``k-wave-python==0.4.0``
     - ``jax``, ``jwave``, ``jaxdf``


When to Use jwave
-----------------

Choose jwave when you need:

1. **Automatic differentiation** -- Phase correction, sensitivity analysis,
   or any optimisation over the pressure field.
2. **CW / Helmholtz solutions** -- Steady-state pressure at a single
   frequency, without simulating the full time series.  This is faster and
   more memory-efficient for ITRUSST benchmarks and aberration correction.
3. **JAX ecosystem integration** -- Composing with other JAX-based tools
   (neural networks, Bayesian inference, differentiable physics).
4. **Reproducibility on CPU** -- jwave runs identically on CPU and GPU;
   k-Wave requires CUDA for GPU acceleration.

Choose k-Wave when you need:

1. **Time-domain accuracy** -- k-Wave's pseudospectral method is the
   gold-standard reference for pulsed-FUS simulations.
2. **Nonlinear propagation** -- k-Wave supports nonlinear acoustics;
   jwave currently does not.
3. **Established validation** -- k-Wave has extensive published validation
   against hydrophone measurements.


CW vs Time-Domain Simulation
-----------------------------

jwave offers both modes through two top-level functions:

Time-Domain: ``run_simulation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solves the wave equation in the time domain using pseudospectral methods.
Returns peak positive pressure (PPP), peak negative pressure (PNP), and
intensity.

.. code-block:: python

   from openlifu.sim.jwave_if import run_simulation

   ds, raw = run_simulation(
       arr=transducer,
       params=params,          # xarray Dataset with sound_speed, density, attenuation
       freq=500e3,             # source frequency
       cycles=20,              # number of cycles in the source burst
       cfl=0.3,                # CFL number for time-step stability
       pml_size=20,            # absorbing boundary thickness
   )

   # ds contains: p_max, p_min, intensity
   print(f"Peak pressure: {ds.p_max.data.max():.0f} Pa")

This mode is appropriate for pulsed FUS treatment planning where the
temporal profile matters (e.g. duty-cycle calculations, MI estimation).


CW / Helmholtz: ``run_cw_simulation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solves the Helmholtz equation directly for the complex pressure field at
a single frequency.  Only one 3D complex array is stored in memory
(vs. the full time series for time-domain).

.. code-block:: python

   from openlifu.sim.jwave_if import run_cw_simulation

   ds, raw = run_cw_simulation(
       arr=transducer,
       params=params,
       freq=500e3,
       amplitude=60000.0,      # source amplitude in Pa
       pml_size=20,
       tol=1e-3,               # GMRES convergence tolerance
       maxiter=1000,
   )

   # ds contains: p_amp, p_phase, intensity
   # raw contains: p_complex (complex pressure field)
   print(f"Focal pressure: {ds.p_amp.data.max():.0f} Pa")

Use this mode for:

- ITRUSST benchmark comparisons (expects ``p_amp``)
- Phase-correction optimisation (differentiable through ``jax.grad``)
- Any steady-state CW analysis


JAX Autodiff Advantages
-----------------------

The key differentiator of jwave is that the entire simulation is
implemented in JAX, making it compatible with ``jax.grad``,
``jax.value_and_grad``, and ``jax.jit``.

Gradient of Focal Pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   from openlifu.sim.phase_correction import compute_focal_gradient

   # Compute d(focal_pressure) / d(delays) in one forward+backward pass
   grad = compute_focal_gradient(
       arr=transducer,
       params=params,
       target_idx=(50, 30, 30),
       freq=500e3,
       delays=current_delays,
   )

This gradient is exact (not finite-difference) and computed in a single
backward pass through the Helmholtz solver.  It enables:

- **Phase correction**: Optimise per-element delays to maximise focal
  pressure (see :doc:`phase_correction`).
- **Sensitivity analysis**: Determine which elements contribute most to
  the focal pressure.
- **Inverse design**: Optimise transducer geometry or medium properties
  jointly with beamforming parameters.

JIT Compilation
~~~~~~~~~~~~~~~

Both ``run_simulation`` and ``run_cw_simulation`` use ``@jax.jit``
internally.  The first call triggers XLA compilation (which can take
10--60 seconds depending on grid size); subsequent calls with the same
grid dimensions are near-instant.

.. code-block:: python

   # First call: slow (JIT compilation)
   ds1, _ = run_cw_simulation(arr=tx, params=params, freq=500e3, ...)

   # Second call with same grid: fast (cached)
   ds2, _ = run_cw_simulation(arr=tx, params=params2, freq=500e3, ...)


Building Blocks
---------------

The jwave interface is composed of several lower-level functions that you
can use independently:

``get_domain``
~~~~~~~~~~~~~~

Creates a jwave ``Domain`` from openlifu xarray coordinates:

.. code-block:: python

   from openlifu.sim.jwave_if import get_domain

   domain, scl = get_domain(params.coords)
   # domain.N = grid shape, domain.dx = grid spacing (meters)
   # scl = unit conversion factor (e.g. 1e-3 for mm -> m)

``get_medium``
~~~~~~~~~~~~~~

Creates a jwave ``Medium`` from openlifu acoustic property maps:

.. code-block:: python

   from openlifu.sim.jwave_if import get_medium

   # Heterogeneous medium
   medium = get_medium(params, domain, pml_size=20)

   # Homogeneous (reference values only)
   medium_ref = get_medium(params, domain, ref_values_only=True, pml_size=20)

``get_sources``
~~~~~~~~~~~~~~~

Maps transducer elements to grid sources with delays and apodisation:

.. code-block:: python

   from openlifu.sim.jwave_if import get_sources

   sources = get_sources(
       arr=transducer,
       domain=domain,
       coords=params.coords,
       scl=scl,
       input_signal=signal,
       dt=time_axis.dt,
       delays=delays,
       apod=apodization,
   )

``get_time_axis``
~~~~~~~~~~~~~~~~~

Creates a time axis from the medium properties or explicit values:

.. code-block:: python

   from openlifu.sim.jwave_if import get_time_axis

   time_axis = get_time_axis(medium, cfl=0.3)
   # or with explicit values:
   time_axis = get_time_axis(medium, dt=1e-8, t_end=100e-6)


Performance Considerations
--------------------------

Grid Size and Memory
~~~~~~~~~~~~~~~~~~~~

The Helmholtz solver (CW mode) stores a single complex 3D field, while
time-domain stores the full pressure time series.  Memory usage:

.. list-table::
   :header-rows: 1

   * - Grid
     - CW memory
     - TD memory (20 cycles at 500 kHz)
   * - 64^3
     - ~2 MB
     - ~200 MB
   * - 128^3
     - ~16 MB
     - ~1.6 GB
   * - 256^3
     - ~128 MB
     - ~13 GB

For large grids (> 128^3), CW mode is strongly preferred unless you
specifically need the time-domain waveform.

GPU Acceleration
~~~~~~~~~~~~~~~~

jwave benefits significantly from GPU acceleration via JAX:

.. list-table::
   :header-rows: 1

   * - Device
     - 64^3 CW
     - 128^3 CW
     - 256^3 CW
   * - CPU (M1 Mac)
     - ~5 s
     - ~30 s
     - ~300 s
   * - A100 40GB
     - ~0.5 s
     - ~2 s
     - ~15 s

To use GPU:

.. code-block:: bash

   # Install JAX with CUDA support
   uv pip install "jax[cuda12]>=0.9.0"


Validating Against k-Wave
--------------------------

The ``benchmarks/tfuscapes_compare.py`` script compares jwave Helmholtz
solutions against k-Wave reference fields from the TFUScapes dataset:

.. code-block:: bash

   uv run modal run benchmarks/tfuscapes_compare.py --n-samples 3

Typical agreement metrics:

- **L2 relative error**: 5--15% (depends on skull complexity)
- **Peak pressure difference**: < 10%
- **Peak position offset**: < 2 voxels
- **Pearson correlation**: > 0.95

The primary source of disagreement is that jwave uses the Helmholtz
(CW) formulation while the k-Wave reference used time-domain simulation.
For CW benchmarks (ITRUSST BM1, BM4, BM7), agreement is typically better.


Installation
------------

The jwave backend requires three packages beyond the base openlifu
dependencies:

.. code-block:: bash

   # CPU only
   uv pip install jax jaxdf
   uv pip install "jwave @ git+https://github.com/ucl-bug/jwave.git@main"

   # With CUDA GPU support
   uv pip install "jax[cuda12]>=0.9.0" jaxdf
   uv pip install "jwave @ git+https://github.com/ucl-bug/jwave.git@main"


See Also
--------

- :mod:`openlifu.sim.jwave_if` -- jwave interface API reference
- :mod:`openlifu.sim.kwave_if` -- k-Wave interface (for comparison)
- :doc:`heterogeneous_skull` -- Heterogeneous skull modeling tutorial
- :doc:`phase_correction` -- Phase correction tutorial
