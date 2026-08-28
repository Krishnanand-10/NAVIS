"""
Coordinate Transformations & Rigid Body Kinematics for Inertial Navigation

Conventions:
- ENU (East, North, Up) as default navigation/world frame
- NED (North, East, Down) conversion utilities
- Quaternions follow Hamilton convention: q = [w, x, y, z] (w is real part)
- Body frame: X forward, Y left, Z up (for ENU) or X forward, Y right, Z down (for NED)
- Vector rotations: v_world = R_body_to_world * v_body
"""

import numpy as np
from typing import Tuple, Union


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion or array of quaternions to unit length."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    else:
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        return q / norm


def euler_to_quat(roll: Union[float, np.ndarray],
                    pitch: Union[float, np.ndarray],
                    yaw: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert Euler angles (in radians: roll, pitch, yaw) to unit quaternion q = [w, x, y, z].
    Rotation sequence: Z-Y-X (Yaw -> Pitch -> Roll) representing intrinsic rotations.
    """
    roll = np.asarray(roll, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    if roll.ndim == 0:
        return quat_normalize(np.array([w, x, y, z], dtype=np.float64))
    else:
        q = np.stack([w, x, y, z], axis=-1)
        return quat_normalize(q)


def quat_to_euler(q: np.ndarray) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert unit quaternion q = [w, x, y, z] to Euler angles (roll, pitch, yaw) in radians.
    Sequence: Z-Y-X.
    """
    q = np.asarray(q, dtype=np.float64)
    q = quat_normalize(q)

    if q.ndim == 1:
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion q = [w, x, y, z] to 3x3 rotation matrix R_body_to_world.
    """
    q = np.asarray(q, dtype=np.float64)
    q = quat_normalize(q)

    if q.ndim == 1:
        w, x, y, z = q
        R = np.array([
            [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - w*z),       2.0 * (x*z + w*y)],
            [2.0 * (x*y + w*z),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - w*x)],
            [2.0 * (x*z - w*y),       2.0 * (y*z + w*x),       1.0 - 2.0 * (x*x + y*y)]
        ], dtype=np.float64)
        return R
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        N = q.shape[0]
        R = np.zeros((N, 3, 3), dtype=np.float64)
        R[:, 0, 0] = 1.0 - 2.0 * (y*y + z*z)
        R[:, 0, 1] = 2.0 * (x*y - w*z)
        R[:, 0, 2] = 2.0 * (x*z + w*y)
        R[:, 1, 0] = 2.0 * (x*y + w*z)
        R[:, 1, 1] = 1.0 - 2.0 * (x*x + z*z)
        R[:, 1, 2] = 2.0 * (y*z - w*x)
        R[:, 2, 0] = 2.0 * (x*z - w*y)
        R[:, 2, 1] = 2.0 * (y*z + w*x)
        R[:, 2, 2] = 1.0 - 2.0 * (x*x + y*y)
        return R


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to unit quaternion q = [w, x, y, z].
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return quat_normalize(np.array([w, x, y, z], dtype=np.float64))
    else:
        quats = np.zeros((R.shape[0], 4), dtype=np.float64)
        for i in range(R.shape[0]):
            quats[i] = rot_matrix_to_quat(R[i])
        return quats


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (Hamilton product: q1 * q2).
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    if q1.ndim == 1 and q2.ndim == 1:
        return np.array([w, x, y, z], dtype=np.float64)
    else:
        return np.stack([w, x, y, z], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate q* = [w, -x, -y, -z]."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    else:
        conj = q.copy()
        conj[..., 1:] = -conj[..., 1:]
        return conj


def body_to_world(v_body: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate 3D vector v from body frame to navigation/world frame using orientation quaternion q.
    v_world = R(q) * v_body
    """
    v_body = np.asarray(v_body, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if v_body.ndim == 1 and q.ndim == 1:
        R = quat_to_rot_matrix(q)
        return R @ v_body
    else:
        R = quat_to_rot_matrix(q)
        # Handle batch matrix multiplication
        if R.ndim == 3 and v_body.ndim == 2:
            return np.einsum('nij,nj->ni', R, v_body)
        elif R.ndim == 3 and v_body.ndim == 1:
            return np.einsum('nij,j->ni', R, v_body)
        else:
            return R @ v_body


def world_to_body(v_world: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate 3D vector v from navigation/world frame to body frame using orientation quaternion q.
    v_body = R(q)^T * v_world
    """
    v_world = np.asarray(v_world, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if v_world.ndim == 1 and q.ndim == 1:
        R = quat_to_rot_matrix(q)
        return R.T @ v_world
    else:
        R = quat_to_rot_matrix(q)
        if R.ndim == 3 and v_world.ndim == 2:
            return np.einsum('nji,nj->ni', R, v_world)
        elif R.ndim == 3 and v_world.ndim == 1:
            return np.einsum('nji,j->ni', R, v_world)
        else:
            return R.T @ v_world


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate orientation quaternion with angular velocity vector omega (rad/s) over timestep dt.
    Uses exact closed-form matrix exponential integration:
    dq = [cos(|omega|*dt/2), sin(|omega|*dt/2) * (omega/|omega|)]
    """
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-12:
        return quat_normalize(q)
    
    theta = omega_norm * dt
    half_theta = 0.5 * theta
    axis = omega / omega_norm
    
    dq = np.array([
        np.cos(half_theta),
        np.sin(half_theta) * axis[0],
        np.sin(half_theta) * axis[1],
        np.sin(half_theta) * axis[2]
    ], dtype=np.float64)
    
    return quat_normalize(quat_multiply(q, dq))


def enu_to_ned(data: np.ndarray) -> np.ndarray:
    """
    Convert 3D vector or trajectory from ENU (East, North, Up) to NED (North, East, Down).
    NED_x = ENU_y (North)
    NED_y = ENU_x (East)
    NED_z = -ENU_z (Down)
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return np.array([data[1], data[0], -data[2]], dtype=np.float64)
    else:
        res = np.zeros_like(data)
        res[..., 0] = data[..., 1]
        res[..., 1] = data[..., 0]
        res[..., 2] = -data[..., 2]
        return res


def ned_to_enu(data: np.ndarray) -> np.ndarray:
    """
    Convert 3D vector or trajectory from NED (North, East, Down) to ENU (East, North, Up).
    ENU_x = NED_y (East)
    ENU_y = NED_x (North)
    ENU_z = -NED_z (Up)
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return np.array([data[1], data[0], -data[2]], dtype=np.float64)
    else:
        res = np.zeros_like(data)
        res[..., 0] = data[..., 1]
        res[..., 1] = data[..., 0]
        res[..., 2] = -data[..., 2]
        return res
