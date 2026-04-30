"""Hounsfield-unit -> acoustic property maps for pseudo-CT pipelines.

Differentiable JAX implementations of the standard HU -> (sound speed,
density, attenuation) conversions used by transcranial-ultrasound
treatment planning pipelines (Schneider 2000 for density, Mast 2000
for sound speed, empirical bone-fraction interpolation for
attenuation). Intended for use with a learned pseudo-CT predictor
(e.g. a U-Net producing HU from T1 MRI) feeding directly into
``HeterogeneousSkullSegmentation(source='pseudoct')`` or the
heterogeneous medium constructor in ``openlifu.sim.jwave_if``.

Adapted from ``sbi4dwi/dmipy_jax/biophysics/acoustic.py``; the HU->c/rho
relationships are unchanged. Smooth softplus blending is preserved so
the chain is differentiable end-to-end (gradient-based pseudo-CT
training, FWI through skull, etc.).

References
----------
* Schneider et al. 2000, *Phys. Med. Biol.* 45, 459 -- HU -> density.
* Mast 2000, *Acoust. Res. Lett. Online* 1(2), 37 -- density -> sound speed.
* Connor 2002 / NDK constants -- cortical bone attenuation reference.
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def jax_sigmoid(x: jnp.ndarray, sharpness: float = 1.0) -> jnp.ndarray:
    """Smooth sigmoid for differentiable piecewise blending."""
    return jax.nn.sigmoid(x * sharpness)


def hu_to_density(hu: jnp.ndarray) -> jnp.ndarray:
    """Hounsfield unit -> density (kg/m^3) via Schneider 2000 piecewise model.

    Three linear regions blended with sigmoid weights so the result is
    everywhere differentiable:

    * HU <= 0   :  rho = 1000 + 1.0 * HU         (water / soft tissue)
    * 0 < HU <= 1000  :  rho = 1000 + 0.5 * HU   (soft -> bone transition)
    * HU > 1000 :  rho = 1500 + 0.6 * (HU-1000)  (dense cortical bone)

    Output is clipped to ``[1, 2500]`` kg/m^3.
    """
    rho_low = 1000.0 + 1.0 * hu
    rho_mid = 1000.0 + 0.5 * hu
    rho_high = 1500.0 + 0.6 * (hu - 1000.0)

    w1 = jax_sigmoid(-hu, sharpness=0.1)            # HU <= 0
    w3 = jax_sigmoid(hu - 1000.0, sharpness=0.01)   # HU > 1000
    w2 = jnp.maximum(1.0 - w1 - w3, 0.0)

    rho = w1 * rho_low + w2 * rho_mid + w3 * rho_high
    return jnp.clip(rho, 1.0, 2500.0)


def density_to_sound_speed(rho: jnp.ndarray) -> jnp.ndarray:
    """Density -> sound speed (m/s) via Mast 2000.

    ``c = 1.33 * rho + 167``, clipped to [1480, 4500] m/s. Matches the
    fit Mast reports for cortical/trabecular bone in the 0.5-2 MHz
    diagnostic-imaging band.
    """
    c = 1.33 * rho + 167.0
    return jnp.clip(c, 1480.0, 4500.0)


def hu_to_attenuation(hu: jnp.ndarray) -> jnp.ndarray:
    """HU -> attenuation (dB/cm/MHz) via linear soft-tissue->bone blend.

    Interpolates between a soft-tissue baseline (~0.5 dB/cm/MHz) and
    cortical bone (~4.74 dB/cm/MHz, equivalent to 54.553 Np/m/MHz at 1
    MHz) using ``HU/2000`` as the bone fraction. Crude compared to a
    proper density-and-microstructure model, but adequate for the
    heterogeneous-skull regime targeted here.
    """
    alpha_soft = 0.5
    alpha_bone = 4.74
    bone_fraction = jnp.clip(hu / 2000.0, 0.0, 1.0)
    alpha = alpha_soft + (alpha_bone - alpha_soft) * bone_fraction
    return jnp.maximum(alpha, 0.0)


def hu_to_acoustic_properties(hu: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Convenience wrapper: HU -> ``{sound_speed, density, attenuation}``.

    Differentiable end-to-end. Output dicts are shape-preserving in
    ``hu`` and unit-consistent with ``openlifu.seg.material``
    (sound_speed in m/s, density in kg/m^3, attenuation in dB/cm/MHz).
    """
    density = hu_to_density(hu)
    return {
        "sound_speed": density_to_sound_speed(density),
        "density": density,
        "attenuation": hu_to_attenuation(hu),
    }
