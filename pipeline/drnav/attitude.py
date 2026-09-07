"""Rotation and attitude helpers.

Conventions used consistently across the whole project:

* **Navigation frame** is local **ENU** (x=East, y=North, z=Up), tangent plane
  anchored at the first valid GNSS fix of a log.
* **Body frame** is x=forward, y=left, z=up.
* Attitude is carried as a 3x3 rotation matrix ``R`` mapping **body -> nav**.
  We deliberately avoid storing quaternions inside the filter: a matrix has no
  sign/order ambiguity, which removes the single most common class of ESKF bug.
  Quaternions are provided only for logging and interchange.
* Gravity is a *nav-frame* vector pointing down: ``[0, 0, -9.80665]``.

An accelerometer measures **specific force**, not acceleration. At rest and
level it reads ``+g`` on its up-axis. The relation used everywhere is::

    f_body = R.T @ (a_nav - g_vec)          # simulate a measurement
    a_nav  = R @ f_body + g_vec             # mechanise a measurement
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

GRAVITY = 9.80665
G_VEC_ENU = np.array([0.0, 0.0, -GRAVITY])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that ``skew(a) @ b == np.cross(a, b)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle, radians) -> 3x3 matrix."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithm map: 3x3 rotation matrix -> rotation vector."""
    return Rotation.from_matrix(R).as_rotvec()


def orthonormalise(R: np.ndarray) -> np.ndarray:
    """Project a drifted matrix back onto SO(3) via SVD.

    Repeated first-order updates slowly break orthonormality; calling this every
    few hundred steps keeps the mechanisation numerically clean.
    """
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:  # guard against a reflection
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def R_from_heading(psi: np.ndarray | float) -> np.ndarray:
    """Body->nav rotation for a level platform with heading ``psi``.

    ``psi`` is measured in radians **clockwise from North**, the usual
    navigation convention. Body x=forward therefore maps to
    ``[sin(psi), cos(psi), 0]`` in ENU.
    """
    c, s = np.cos(psi), np.sin(psi)
    # columns are the body axes expressed in nav coordinates
    fwd = np.array([s, c, 0.0])
    left = np.array([-c, s, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.column_stack([fwd, left, up])


def heading_from_R(R: np.ndarray) -> float:
    """Extract heading (radians clockwise from North) from a body->nav matrix."""
    fwd = R[:, 0]
    return float(np.arctan2(fwd[0], fwd[1]))


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """Body->nav matrix -> quaternion ``[w, x, y, z]`` (for logging only)."""
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def R_from_quat(q: np.ndarray) -> np.ndarray:
    """Quaternion ``[w, x, y, z]`` -> body->nav matrix."""
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def angular_rate_from_R(R_prev: np.ndarray, R_next: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular rate that carries ``R_prev`` to ``R_next`` over ``dt``.

    Used by the simulator to derive a gyroscope signal from a ground-truth
    attitude sequence. ``R_prev.T @ R_next`` is the incremental rotation
    expressed in the body frame, so its log divided by dt is omega_body.
    """
    return log_so3(R_prev.T @ R_next) / dt


def wrap_pi(a: np.ndarray | float):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
