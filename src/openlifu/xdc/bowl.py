"""Focused ultrasound transducer geometry for TUS.

Provides:
  - bowl_transducer_3d: Transducers on a spherical cap (bowl)
  - FocusedPhasedArray: Multi-element array with electronic steering
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional


def bowl_transducer_3d(
    focal_length: float,
    aperture_diameter: float,
    focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    n_points: int = 1000,
) -> jnp.ndarray:
    """Generate points on a focused spherical-bowl transducer surface.

    The bowl is the concave radiating face of a focused transducer. Its
    centre of curvature is the focal point, so every point on the surface
    is at distance ``focal_length`` from ``focal_point``. The apex of the
    bowl (the deepest, central back point) lies at
    ``focal_point - direction * focal_length``.

    Args:
        focal_length: Radius of curvature of the bowl, equivalent to the
            distance from any surface point to the focal point (m).
        aperture_diameter: Diameter of the bowl rim (m).
        focal_point: Geometric focus, also the centre of curvature (m).
        direction: Unit vector pointing from the bowl apex toward the
            focal point (i.e., the acoustic axis).
        n_points: Number of point sources to sample on the bowl surface
            using Fibonacci spiral sampling.

    Returns:
        ``(n_points, 3)`` array of transducer positions in metres.
    """
    dir_vec = np.array(direction, dtype=float) / np.linalg.norm(direction)

    # Half-angle subtended by the rim from the centre of curvature
    if aperture_diameter >= 2 * focal_length:
        raise ValueError(
            f"aperture_diameter ({aperture_diameter}) must be < 2*focal_length "
            f"({2*focal_length}); a focused bowl can't have a rim wider than "
            f"its sphere of curvature."
        )
    half_angle = np.arcsin(aperture_diameter / (2 * focal_length))

    # Fibonacci spiral on the unit sphere, restricted to the cap
    # surrounding the -z pole. Building the cap on the -z hemisphere
    # means that after rotating +z onto dir_vec the apex ends up at
    # -dir_vec, and translating by focal_point places it at
    # focal_point - dir_vec * focal_length (i.e. the geometric back
    # of the bowl). Every sampled point sits at unit distance from the
    # origin, so after scaling and translating, every point is at
    # distance focal_length from focal_point.
    indices = np.arange(n_points)
    phi = np.arccos(1 - (1 - np.cos(half_angle)) * (indices / max(n_points, 1)))
    theta = 2 * np.pi * indices * ((1 + np.sqrt(5)) / 2)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = -np.cos(phi)  # cap on -z hemisphere; apex at (0, 0, -1)

    unit_points = np.column_stack([x, y, z])

    # Rotate so the original +z axis aligns with dir_vec.
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(dir_vec, z_axis):
        rotated = unit_points
    elif np.allclose(dir_vec, -z_axis):
        rotated = unit_points * np.array([1.0, 1.0, -1.0])
    else:
        v = np.cross(z_axis, dir_vec)
        s = np.linalg.norm(v)
        c = float(np.dot(z_axis, dir_vec))
        v_skew = np.array([
            [0,    -v[2],  v[1]],
            [v[2],  0,    -v[0]],
            [-v[1], v[0],  0.0],
        ])
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s ** 2))
        rotated = unit_points @ R.T

    # Scale to radius and translate so the sphere centre = focal_point.
    final_points = rotated * focal_length + np.array(focal_point, dtype=float)
    return jnp.array(final_points)
