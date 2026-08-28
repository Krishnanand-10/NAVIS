"""
Inertial Kinematics & Strapdown Mechanization Equations (SINS)

Provides foundational physics-based integration routines for Strapdown Inertial Navigation:
1. Attitude Integration:
   - 4th-order Runge-Kutta (RK4) Quaternion Integration
   - Midpoint / Exponential Quaternion Integration
   - Forward Euler Quaternion Integration
   - Direct Euler Angle Kinematic Integration (gimbal-lock prone baseline)
2. Specific Force Transformation & Gravity Compensation:
   - Body-to-World acceleration rotation
   - Configurable Earth / Lunar gravity models (WGS-84 / Somigliana)
   - Earth rotation / Coriolis compensation (optional)
3. Translation Mechanization:
   - Trapezoidal / Heun integration (2nd-order accuracy)
   - Velocity-Verlet symplectic integration
   - Forward Euler integration
"""

from typing import Tuple, Union, Optional
import numpy as np

from src.utils.coordinate_transforms import (
    quat_normalize,
    quat_multiply,
    quat_to_rot_matrix,
    body_to_world,
    quat_to_euler,
    euler_to_quat,
)

# Standard gravitational acceleration in ENU frame [m/s^2]
STANDARD_GRAVITY_ENU = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
LUNAR_GRAVITY_ENU = np.array([0.0, 0.0, -1.625], dtype=np.float64)


class GravityModel:
    """Configurable local gravity model."""

    def __init__(self,
                 model_type: str = "standard",
                 latitude_deg: float = 45.0,
                 altitude_m: float = 0.0,
                 custom_g: Optional[np.ndarray] = None):
        self.model_type = model_type.lower()
        if custom_g is not None:
            self.g_vector = np.asarray(custom_g, dtype=np.float64)
        elif self.model_type == "lunar":
            self.g_vector = LUNAR_GRAVITY_ENU.copy()
        elif self.model_type == "wgs84":
            # Somigliana closed-form formula for WGS-84 normal gravity
            lat_rad = np.deg2rad(latitude_deg)
            sin_lat = np.sin(lat_rad)
            sin2_lat = sin_lat ** 2
            # WGS-84 constants
            ge = 9.7803253359  # Equatorial gravity (m/s^2)
            k = 0.00193185265241
            e2 = 0.00669437999014
            g0 = ge * (1.0 + k * sin2_lat) / np.sqrt(1.0 - e2 * sin2_lat)
            # Free-air altitude correction
            g_h = g0 - (3.086e-6 * altitude_m)
            self.g_vector = np.array([0.0, 0.0, -g_h], dtype=np.float64)
        else:
            self.g_vector = STANDARD_GRAVITY_ENU.copy()

    @property
    def vector(self) -> np.ndarray:
        return self.g_vector.copy()

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.g_vector))


# ============================================================================
# 1. ATTITUDE MECHANIZATION ENGINES
# ============================================================================

def _omega_to_pure_quat(omega: np.ndarray) -> np.ndarray:
    """Convert 3D angular velocity vector [wx, wy, wz] into pure quaternion [0, wx, wy, wz]."""
    return np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)


def integrate_attitude_quaternion_euler(q: np.ndarray,
                                        omega: np.ndarray,
                                        dt: float) -> np.ndarray:
    """
    1st-Order Forward Euler Quaternion Integration.
    q_{k+1} = normalize(q_k + 0.5 * dt * (q_k * [0, omega]))
    """
    q = np.asarray(q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    q_omega = _omega_to_pure_quat(omega)
    dq_dt = 0.5 * quat_multiply(q, q_omega)
    q_next = q + dq_dt * dt
    return quat_normalize(q_next)


def integrate_attitude_quaternion_exponential(q: np.ndarray,
                                             omega: np.ndarray,
                                             dt: float) -> np.ndarray:
    """
    Exact closed-form Matrix Exponential Integration assuming constant angular velocity during dt.
    dq = [cos(|omega|*dt/2), sin(|omega|*dt/2) * (omega / |omega|)]
    q_{k+1} = q_k * dq
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


def integrate_attitude_quaternion_midpoint(q: np.ndarray,
                                          omega_prev: np.ndarray,
                                          omega_curr: np.ndarray,
                                          dt: float) -> np.ndarray:
    """
    2nd-Order Midpoint / Heun Quaternion Integration.
    Averages angular rates across the interval for improved dynamic fidelity.
    """
    omega_mid = 0.5 * (np.asarray(omega_prev, dtype=np.float64) + np.asarray(omega_curr, dtype=np.float64))
    return integrate_attitude_quaternion_exponential(q, omega_mid, dt)


def integrate_attitude_quaternion_rk4(q: np.ndarray,
                                     omega_prev: np.ndarray,
                                     omega_curr: np.ndarray,
                                     dt: float) -> np.ndarray:
    """
    4th-Order Runge-Kutta (RK4) Quaternion Integration.
    Yields $O(dt^4)$ truncation error on high-dynamic rotational maneuvers.
    """
    q = np.asarray(q, dtype=np.float64)
    w0 = np.asarray(omega_prev, dtype=np.float64)
    w1 = np.asarray(omega_curr, dtype=np.float64)
    w_mid = 0.5 * (w0 + w1)

    # k1 = 0.5 * q * [0, w0]
    k1 = 0.5 * quat_multiply(q, _omega_to_pure_quat(w0))
    q1 = quat_normalize(q + 0.5 * dt * k1)

    # k2 = 0.5 * q1 * [0, w_mid]
    k2 = 0.5 * quat_multiply(q1, _omega_to_pure_quat(w_mid))
    q2 = quat_normalize(q + 0.5 * dt * k2)

    # k3 = 0.5 * q2 * [0, w_mid]
    k3 = 0.5 * quat_multiply(q2, _omega_to_pure_quat(w_mid))
    q3 = quat_normalize(q + dt * k3)

    # k4 = 0.5 * q3 * [0, w1]
    k4 = 0.5 * quat_multiply(q3, _omega_to_pure_quat(w1))

    # Blend stages
    q_next = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return quat_normalize(q_next)


def integrate_attitude_euler_angles(euler: np.ndarray,
                                    omega: np.ndarray,
                                    dt: float) -> np.ndarray:
    """
    Direct Euler Angle Kinematic Differential Equation Integration (Z-Y-X sequence).
    Demonstrates classical Euler dead reckoning baseline (note: exhibits singularity at pitch = +-90 deg).
    
    [d_roll/dt; d_pitch/dt; d_yaw/dt] = W(roll, pitch) * omega_body
    """
    euler = np.asarray(euler, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    roll, pitch, yaw = euler[0], euler[1], euler[2]
    cos_pitch = np.cos(pitch)
    if np.abs(cos_pitch) < 1e-6:
        # Near gimbal lock singularity, clamp
        cos_pitch = 1e-6 if cos_pitch >= 0 else -1e-6

    tan_pitch = np.tan(pitch)
    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)

    W = np.array([
        [1.0, sin_roll * tan_pitch, cos_roll * tan_pitch],
        [0.0, cos_roll,            -sin_roll],
        [0.0, sin_roll / cos_pitch, cos_roll / cos_pitch]
    ], dtype=np.float64)

    d_euler = W @ omega
    euler_next = euler + d_euler * dt

    # Wrap angles
    euler_next[0] = (euler_next[0] + np.pi) % (2.0 * np.pi) - np.pi
    euler_next[1] = np.clip(euler_next[1], -np.pi / 2.0 + 1e-5, np.pi / 2.0 - 1e-5)
    euler_next[2] = (euler_next[2] + np.pi) % (2.0 * np.pi) - np.pi

    return euler_next


# ============================================================================
# 2. SPECIFIC FORCE & GRAVITY MECHANIZATION
# ============================================================================

def specific_force_to_world_acc(specific_force_body: np.ndarray,
                                q: np.ndarray,
                                gravity: Union[np.ndarray, GravityModel] = STANDARD_GRAVITY_ENU,
                                coriolis_correction: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Transform body-frame specific force into navigation world frame linear acceleration with gravity compensation:
    a_world = R(q) * f_body + g_world - (optional Coriolis)
    """
    f_b = np.asarray(specific_force_body, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if isinstance(gravity, GravityModel):
        g_w = gravity.vector
    else:
        g_w = np.asarray(gravity, dtype=np.float64)

    # Rotate specific force from body to world frame
    f_w = body_to_world(f_b, q)

    # Add gravitational reaction
    a_w = f_w + g_w

    if coriolis_correction is not None:
        a_w -= np.asarray(coriolis_correction, dtype=np.float64)

    return a_w


# ============================================================================
# 3. TRANSLATION (VELOCITY & POSITION) MECHANIZATION ENGINES
# ============================================================================

def integrate_translation_euler(pos: np.ndarray,
                                vel: np.ndarray,
                                acc: np.ndarray,
                                dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    1st-Order Forward Euler Translation Integration:
    v_{k+1} = v_k + a_k * dt
    p_{k+1} = p_k + v_k * dt
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)

    pos_next = pos + vel * dt
    vel_next = vel + acc * dt
    return pos_next, vel_next


def integrate_translation_trapezoidal(pos: np.ndarray,
                                      vel: np.ndarray,
                                      acc_prev: np.ndarray,
                                      acc_curr: np.ndarray,
                                      dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    2nd-Order Trapezoidal / Heun Integration:
    v_{k+1} = v_k + 0.5 * (a_k + a_{k+1}) * dt
    p_{k+1} = p_k + 0.5 * (v_k + v_{k+1}) * dt
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    a0 = np.asarray(acc_prev, dtype=np.float64)
    a1 = np.asarray(acc_curr, dtype=np.float64)

    vel_next = vel + 0.5 * (a0 + a1) * dt
    pos_next = pos + 0.5 * (vel + vel_next) * dt
    return pos_next, vel_next


def integrate_translation_verlet(pos: np.ndarray,
                                 vel: np.ndarray,
                                 acc_prev: np.ndarray,
                                 acc_curr: np.ndarray,
                                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity-Verlet Symplectic Translation Integration (preserves energy bounds in orbital / cyclic motion):
    p_{k+1} = p_k + v_k * dt + 0.5 * a_k * dt^2
    v_{k+1} = v_k + 0.5 * (a_k + a_{k+1}) * dt
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    a0 = np.asarray(acc_prev, dtype=np.float64)
    a1 = np.asarray(acc_curr, dtype=np.float64)

    pos_next = pos + vel * dt + 0.5 * a0 * (dt ** 2)
    vel_next = vel + 0.5 * (a0 + a1) * dt
    return pos_next, vel_next
