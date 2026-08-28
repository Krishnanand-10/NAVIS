"""
Analytical INS Error Dynamics & Drift Propagation

Provides mathematical models for error state propagation in Strapdown Inertial Navigation:
1. 15-State Continuous INS Error Model:
   delta_x = [delta_p (3), delta_v (3), delta_psi (3), delta_b_a (3), delta_b_g (3)]^T
   delta_x_dot = F * delta_x + G * w
2. Skew-Symmetric Cross-Product Matrix Formulation
3. Discrete Covariance Propagation (Lyapunov / Riccati state transition)
4. Theoretical Analytical Drift Curves:
   - Gyro Bias -> Quadratic Velocity & Cubic Position Divergence (O(t^3))
   - Accel Bias -> Linear Velocity & Quadratic Position Divergence (O(t^2))
   - Stochastic Noise -> t^(3/2) and t^(5/2) standard deviation growth
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import numpy as np

from src.utils.coordinate_transforms import quat_to_rot_matrix, body_to_world
from src.simulation.sensor_noise import IMUSpecs, INDUSTRIAL_IMU


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 skew-symmetric matrix [v_x] such that [v_x] * u = v x u.
    """
    v = np.asarray(v, dtype=np.float64)
    return np.array([
        [0.0,    -v[2],   v[1]],
        [v[2],    0.0,   -v[0]],
        [-v[1],   v[0],   0.0]
    ], dtype=np.float64)


@dataclass
class INSErrorState:
    """15-dimensional Strapdown INS Error State Vector."""
    delta_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))   # [m] World ENU
    delta_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))   # [m/s] World ENU
    delta_psi: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))   # [rad] World ENU tilt
    delta_b_a: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))   # [m/s^2] Body frame accel bias
    delta_b_g: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))   # [rad/s] Body frame gyro bias

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([
            self.delta_pos,
            self.delta_vel,
            self.delta_psi,
            self.delta_b_a,
            self.delta_b_g
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "INSErrorState":
        vec = np.asarray(vec, dtype=np.float64)
        return cls(
            delta_pos=vec[0:3],
            delta_vel=vec[3:6],
            delta_psi=vec[6:9],
            delta_b_a=vec[9:12],
            delta_b_g=vec[12:15]
        )


def build_ins_error_jacobian(f_body: np.ndarray,
                             q: np.ndarray,
                             tau_acc: float = 3600.0,
                             tau_gyro: float = 3600.0) -> np.ndarray:
    """
    Construct 15x15 continuous-time system dynamics Jacobian matrix F(t) for INS error state:
    delta_x_dot = F * delta_x + G * w
    
    State indices:
    [0:3]   Position delta_p
    [3:6]   Velocity delta_v
    [6:9]   Attitude misalignment delta_psi
    [9:12]  Accel bias delta_b_a
    [12:15] Gyro bias delta_b_g
    """
    R = quat_to_rot_matrix(q)
    f_world = R @ np.asarray(f_body, dtype=np.float64)
    skew_f = skew_symmetric(f_world)

    F = np.zeros((15, 15), dtype=np.float64)

    # d(delta_p)/dt = delta_v
    F[0:3, 3:6] = np.eye(3)

    # d(delta_v)/dt = -[f_world x] delta_psi + R * delta_b_a
    F[3:6, 6:9] = -skew_f
    F[3:6, 9:12] = R

    # d(delta_psi)/dt = -R * delta_b_g
    F[6:9, 12:15] = -R

    # Gauss-Markov bias correlation decay
    if tau_acc > 0:
        F[9:12, 9:12] = -np.eye(3) / tau_acc
    if tau_gyro > 0:
        F[12:15, 12:15] = -np.eye(3) / tau_gyro

    return F


def propagate_error_covariance(P: np.ndarray,
                               F: np.ndarray,
                               Q_c: np.ndarray,
                               dt: float) -> np.ndarray:
    """
    Propagate 15x15 error covariance matrix P across timestep dt via 1st-order transition matrix approximation:
    Phi = I + F * dt + 0.5 * F^2 * dt^2
    P_{k+1} = Phi * P_k * Phi^T + Q_d
    where Q_d = 0.5 * (Phi * Q_c * Phi^T + Q_c) * dt
    """
    I = np.eye(15, dtype=np.float64)
    F_dt = F * dt
    Phi = I + F_dt + 0.5 * (F_dt @ F_dt)

    Q_d = Q_c * dt
    P_next = Phi @ P @ Phi.T + Q_d
    # Symmetrize
    return 0.5 * (P_next + P_next.T)


class TheoreticalDriftPredictor:
    """
    Analytical Physics-Based INS Drift Predictor.
    Computes theoretical error growth vs time for dead reckoning without external aid.
    """

    def __init__(self,
                 specs: IMUSpecs = INDUSTRIAL_IMU,
                 g_magnitude: float = 9.80665):
        self.specs = specs
        self.g = g_magnitude

    def predict(self,
                timestamps: np.ndarray,
                acc_bias: Optional[np.ndarray] = None,
                gyro_bias: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute deterministic and 1-sigma stochastic error bounds across timestamps.
        
        Deterministic:
          delta_psi_det(t) = b_g * t
          delta_v_det(t) = b_a * t + 0.5 * g * b_g * t^2
          delta_p_det(t) = 0.5 * b_a * t^2 + (1/6) * g * b_g * t^3
          
        Stochastic:
          sigma_psi(t) = gyro_noise_density * sqrt(t)
          sigma_v(t) = sqrt(acc_noise^2 * t + (1/3) * g^2 * gyro_noise^2 * t^3)
          sigma_p(t) = sqrt((1/3) * acc_noise^2 * t^3 + (1/20) * g^2 * gyro_noise^2 * t^5)
        """
        t = np.asarray(timestamps, dtype=np.float64)
        t_rel = t - t[0]

        # Bias magnitudes
        if acc_bias is not None:
            ba = np.linalg.norm(acc_bias)
        else:
            ba = self.specs.acc_turn_on_bias_std

        if gyro_bias is not None:
            bg = np.linalg.norm(gyro_bias)
        else:
            bg = self.specs.gyro_turn_on_bias_std

        # 1. Deterministic error growth
        att_err_det_rad = bg * t_rel
        att_err_det_deg = np.rad2deg(att_err_det_rad)

        vel_err_det_ba = ba * t_rel
        vel_err_det_bg = 0.5 * self.g * bg * (t_rel ** 2)
        vel_err_det_total = vel_err_det_ba + vel_err_det_bg

        pos_err_det_ba = 0.5 * ba * (t_rel ** 2)
        pos_err_det_bg = (1.0 / 6.0) * self.g * bg * (t_rel ** 3)
        pos_err_det_total = pos_err_det_ba + pos_err_det_bg

        # 2. Stochastic 1-sigma bounds (noise density)
        sa = self.specs.acc_noise_density
        sg = self.specs.gyro_noise_density

        sigma_att_rad = sg * np.sqrt(t_rel)
        sigma_att_deg = np.rad2deg(sigma_att_rad)

        sigma_vel = np.sqrt(
            (sa ** 2) * t_rel + (1.0 / 3.0) * (self.g ** 2) * (sg ** 2) * (t_rel ** 3)
        )

        sigma_pos = np.sqrt(
            (1.0 / 3.0) * (sa ** 2) * (t_rel ** 3) + (1.0 / 20.0) * (self.g ** 2) * (sg ** 2) * (t_rel ** 5)
        )

        return {
            "timestamps": t,
            "t_rel": t_rel,
            # Deterministic components
            "pos_error_det_total": pos_err_det_total,
            "pos_error_from_accel_bias": pos_err_det_ba,
            "pos_error_from_gyro_bias": pos_err_det_bg,
            "vel_error_det_total": vel_err_det_total,
            "att_error_det_deg": att_err_det_deg,
            # Stochastic 1-sigma bounds
            "pos_sigma_stochastic": sigma_pos,
            "vel_sigma_stochastic": sigma_vel,
            "att_sigma_stochastic_deg": sigma_att_deg,
            # Combined 3-sigma expected upper bound
            "pos_error_3sigma_bound": pos_err_det_total + 3.0 * sigma_pos,
        }
