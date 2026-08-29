"""
15-State Error-State Extended Kalman Filter (ES-EKF) for Inertial Navigation

Core Navigation Filter:
1. Nominal State:
   - Position: p in R^3 [m] (World ENU)
   - Velocity: v in R^3 [m/s] (World ENU)
   - Attitude: q in SO(3) [w, x, y, z] (Body to World ENU)
   - Accelerometer Bias: b_a in R^3 [m/s^2]
   - Gyroscope Bias: b_g in R^3 [rad/s]

2. True Error State (15-DOF):
   delta_x = [delta_p^T, delta_v^T, delta_theta^T, delta_b_a^T, delta_b_g^T]^T in R^15

3. AI-Adaptive Noise Scaling:
   Dynamically scales process noise Q_k and measurement noise R_k based on
   Module 3 motion context predictions and Module 4 neural uncertainty estimates.

4. Robust Numerical Integration:
   - RK4 nominal quaternion kinematics
   - 2nd-order Taylor expansion for state transition matrix Phi_k
   - Joseph-form covariance update to ensure positive semi-definiteness
   - True SO(3) quaternion injection & error-state reset
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List, Any
import numpy as np

from src.core.kinematics import (
    GravityModel,
    STANDARD_GRAVITY_ENU,
    integrate_attitude_quaternion_rk4,
    integrate_attitude_quaternion_midpoint,
    specific_force_to_world_acc,
)
from src.core.error_dynamics import skew_symmetric
from src.utils.coordinate_transforms import (
    quat_normalize,
    quat_multiply,
    quat_to_rot_matrix,
    quat_to_euler,
    euler_to_quat,
)


@dataclass
class EKFConfig:
    """Configuration parameters for 15-State ES-EKF."""
    # Sensor Noise Spectral Densities (Continuous PSDs)
    accel_noise_sigma: float = 0.08          # Continuous accel white noise [m/s^2 / sqrt(Hz)]
    gyro_noise_sigma: float = 0.005          # Continuous gyro white noise [rad/s / sqrt(Hz)]
    accel_bias_rw_sigma: float = 1e-4        # Accel bias random walk [m/s^3 / sqrt(Hz)]
    gyro_bias_rw_sigma: float = 1e-5         # Gyro bias random walk [rad/s^2 / sqrt(Hz)]

    # Initial State Covariance Standard Deviations
    init_pos_std: float = 0.5                # [m]
    init_vel_std: float = 0.1                # [m/s]
    init_att_std_deg: float = 1.0            # [deg]
    init_acc_bias_std: float = 0.05          # [m/s^2]
    init_gyro_bias_std: float = 0.002        # [rad/s]

    # Bias Correlation Time Constants (Gauss-Markov process)
    tau_accel_bias: float = 3600.0           # [s] (1 hour)
    tau_gyro_bias: float = 3600.0            # [s] (1 hour)

    # Adaptive AI Scaling parameters
    enable_adaptive_q: bool = True
    enable_adaptive_r: bool = True
    min_adaptive_scale: float = 0.2
    max_adaptive_scale: float = 20.0


@dataclass
class EKFState:
    """Nominal Navigation State & 15x15 Error Covariance."""
    pos: np.ndarray                          # (3,) [m] World ENU
    vel: np.ndarray                          # (3,) [m/s] World ENU
    quat: np.ndarray                         # (4,) [w, x, y, z] Body to World
    acc_bias: np.ndarray                     # (3,) [m/s^2]
    gyro_bias: np.ndarray                    # (3,) [rad/s]
    P: np.ndarray                            # (15, 15) Error state covariance
    timestamp: float = 0.0

    @property
    def euler_deg(self) -> np.ndarray:
        """Attitude in Euler angles [roll, pitch, yaw] in degrees."""
        return np.array(quat_to_euler(self.quat), dtype=np.float64)

    @property
    def pos_std(self) -> np.ndarray:
        """1-sigma position standard deviations [m]."""
        return np.sqrt(np.maximum(np.diag(self.P)[0:3], 1e-12))

    @property
    def vel_std(self) -> np.ndarray:
        """1-sigma velocity standard deviations [m/s]."""
        return np.sqrt(np.maximum(np.diag(self.P)[3:6], 1e-12))

    @property
    def att_std_deg(self) -> np.ndarray:
        """1-sigma attitude error standard deviations [deg]."""
        return np.rad2deg(np.sqrt(np.maximum(np.diag(self.P)[6:9], 1e-12)))

    @property
    def acc_bias_std(self) -> np.ndarray:
        """1-sigma accel bias standard deviations [m/s^2]."""
        return np.sqrt(np.maximum(np.diag(self.P)[9:12], 1e-12))

    @property
    def gyro_bias_std(self) -> np.ndarray:
        """1-sigma gyro bias standard deviations [rad/s]."""
        return np.sqrt(np.maximum(np.diag(self.P)[12:15], 1e-12))


class ErrorStateEKF:
    """
    15-State Error-State Extended Kalman Filter (ES-EKF).
    
    Index Map for Error State delta_x (15x1):
      0:3   -> delta_p (Position error [m] in World ENU)
      3:6   -> delta_v (Velocity error [m/s] in World ENU)
      6:9   -> delta_theta (Attitude angle error [rad] in World ENU)
      9:12  -> delta_b_a (Accelerometer bias error [m/s^2] in Body frame)
      12:15 -> delta_b_g (Gyroscope bias error [rad/s] in Body frame)
    """

    def __init__(self,
                 config: Optional[EKFConfig] = None,
                 gravity: Optional[GravityModel] = None,
                 initial_pos: Optional[np.ndarray] = None,
                 initial_vel: Optional[np.ndarray] = None,
                 initial_quat: Optional[np.ndarray] = None,
                 initial_acc_bias: Optional[np.ndarray] = None,
                 initial_gyro_bias: Optional[np.ndarray] = None):
        self.config = config or EKFConfig()
        self.gravity = gravity or GravityModel()

        # Initialize Nominal States
        self.pos = np.asarray(initial_pos, dtype=np.float64) if initial_pos is not None else np.zeros(3)
        self.vel = np.asarray(initial_vel, dtype=np.float64) if initial_vel is not None else np.zeros(3)

        if initial_quat is not None:
            self.quat = quat_normalize(np.asarray(initial_quat, dtype=np.float64))
        else:
            self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.acc_bias = np.asarray(initial_acc_bias, dtype=np.float64) if initial_acc_bias is not None else np.zeros(3)
        self.gyro_bias = np.asarray(initial_gyro_bias, dtype=np.float64) if initial_gyro_bias is not None else np.zeros(3)

        # Initialize 15x15 Error Covariance Matrix P
        self.P = np.zeros((15, 15), dtype=np.float64)
        init_att_std_rad = np.deg2rad(self.config.init_att_std_deg)

        self.P[0:3, 0:3] = (self.config.init_pos_std ** 2) * np.eye(3)
        self.P[3:6, 3:6] = (self.config.init_vel_std ** 2) * np.eye(3)
        self.P[6:9, 6:9] = (init_att_std_rad ** 2) * np.eye(3)
        self.P[9:12, 9:12] = (self.config.init_acc_bias_std ** 2) * np.eye(3)
        self.P[12:15, 12:15] = (self.config.init_gyro_bias_std ** 2) * np.eye(3)

        # Precompute Continuous Process Noise Covariance Q_c (12x12)
        # Noise inputs: w = [w_acc, w_gyro, w_acc_rw, w_gyro_rw]^T
        self.Q_c = np.diag([
            *(self.config.accel_noise_sigma ** 2 * np.ones(3)),
            *(self.config.gyro_noise_sigma ** 2 * np.ones(3)),
            *(self.config.accel_bias_rw_sigma ** 2 * np.ones(3)),
            *(self.config.gyro_bias_rw_sigma ** 2 * np.ones(3)),
        ]).astype(np.float64)

        self.prev_omega = np.zeros(3, dtype=np.float64)
        self.prev_acc_world = np.zeros(3, dtype=np.float64)
        self.timestamp: float = 0.0
        self.adaptive_q_scale: float = 1.0

    @property
    def state(self) -> EKFState:
        """Get snapshot of current nominal state and covariance."""
        return EKFState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            quat=self.quat.copy(),
            acc_bias=self.acc_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            P=self.P.copy(),
            timestamp=self.timestamp,
        )

    # ========================================================================
    # 1. TIME PROPAGATION / PREDICTION STEP
    # ========================================================================

    def predict(self,
                acc_meas: np.ndarray,
                gyro_meas: np.ndarray,
                dt: float,
                timestamp: Optional[float] = None,
                adaptive_scale: float = 1.0) -> EKFState:
        """
        Propagate nominal kinematic states and 15x15 error covariance over interval dt.
        
        Args:
            acc_meas: (3,) Uncorrected accelerometer measurement [m/s^2]
            gyro_meas: (3,) Uncorrected gyroscope measurement [rad/s]
            dt: Sample time interval [s]
            timestamp: Current timestamp [s]
            adaptive_scale: AI-driven process noise multiplier (from Module 3/4)
        """
        if dt <= 0:
            dt = 0.01

        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp += dt

        acc_raw = np.asarray(acc_meas, dtype=np.float64)
        gyro_raw = np.asarray(gyro_meas, dtype=np.float64)

        # 1. Correct sensor measurements with estimated biases
        f_b = acc_raw - self.acc_bias
        omega_b = gyro_raw - self.gyro_bias

        # 2. Nominal Attitude Integration via RK4
        q_prev = self.quat.copy()
        q_next = integrate_attitude_quaternion_rk4(q_prev, self.prev_omega, omega_b, dt)
        self.quat = quat_normalize(q_next)
        R_b2w = quat_to_rot_matrix(self.quat)

        # 3. Specific Force Transformation & Translation Mechanization
        a_w_curr = specific_force_to_world_acc(f_b, self.quat, self.gravity)
        if np.linalg.norm(self.prev_acc_world) == 0:
            self.prev_acc_world = a_w_curr.copy()

        # Trapezoidal translation integration
        v_next = self.vel + 0.5 * (self.prev_acc_world + a_w_curr) * dt
        p_next = self.pos + 0.5 * (self.vel + v_next) * dt

        self.pos = p_next
        self.vel = v_next
        self.prev_acc_world = a_w_curr.copy()
        self.prev_omega = omega_b.copy()

        # 4. Error State Transition Matrix Phi (15x15)
        # Continuous-time Jacobian F_c
        F_c = np.zeros((15, 15), dtype=np.float64)
        F_c[0:3, 3:6] = np.eye(3)                                      # dp/dt = dv
        F_c[3:6, 6:9] = -skew_symmetric(R_b2w @ f_b)                  # dv/dt = -[R * f_b]_x * dtheta
        F_c[3:6, 9:12] = -R_b2w                                        # dv/dt = -R * db_a
        F_c[6:9, 6:9] = -skew_symmetric(R_b2w @ omega_b)              # dtheta/dt = -[R * omega_b]_x * dtheta
        F_c[6:9, 12:15] = -R_b2w                                       # dtheta/dt = -R * db_g
        F_c[9:12, 9:12] = -(1.0 / self.config.tau_accel_bias) * np.eye(3) # Accel bias Gauss-Markov
        F_c[12:15, 12:15] = -(1.0 / self.config.tau_gyro_bias) * np.eye(3) # Gyro bias Gauss-Markov

        # 2nd-order Taylor series approximation: Phi = I + F_c * dt + 0.5 * (F_c * dt)^2
        F_dt = F_c * dt
        Phi = np.eye(15, dtype=np.float64) + F_dt + 0.5 * (F_dt @ F_dt)

        # 5. Continuous Noise Jacobian G_c (15x12)
        G_c = np.zeros((15, 12), dtype=np.float64)
        G_c[3:6, 0:3] = -R_b2w        # accel noise into velocity error
        G_c[6:9, 3:6] = -R_b2w        # gyro noise into attitude error
        G_c[9:12, 6:9] = np.eye(3)    # accel bias random walk
        G_c[12:15, 9:12] = np.eye(3)  # gyro bias random walk

        # Discrete Process Noise Covariance Q_d (15x15)
        # Q_d = G_c * Q_c * G_c^T * dt + 0.5 * (F_c * G_c * Q_c * G_c^T + G_c * Q_c * G_c^T * F_c^T) * dt^2
        GQG = G_c @ self.Q_c @ G_c.T
        Q_d = GQG * dt + 0.5 * (F_c @ GQG + GQG @ F_c.T) * (dt ** 2)

        # Apply AI-Adaptive Covariance Scaling
        if self.config.enable_adaptive_q:
            scale = float(np.clip(adaptive_scale, self.config.min_adaptive_scale, self.config.max_adaptive_scale))
            self.adaptive_q_scale = scale
            Q_d *= scale

        # 6. Covariance Propagation: P = Phi * P * Phi^T + Q_d
        P_pred = Phi @ self.P @ Phi.T + Q_d
        # Ensure exact matrix symmetry
        self.P = 0.5 * (P_pred + P_pred.T)

        return self.state

    # ========================================================================
    # 2. GENERAL MEASUREMENT UPDATE & ERROR-STATE INJECTION
    # ========================================================================

    def update(self,
               z: np.ndarray,
               z_pred: np.ndarray,
               H: np.ndarray,
               R: np.ndarray,
               chi2_threshold: Optional[float] = None) -> Tuple[bool, float, np.ndarray]:
        """
        Execute general Kalman Measurement Update with Joseph-form covariance formulation
        and true SO(3) quaternion injection & error-state reset.
        
        Args:
            z: (M,) Observed measurement vector
            z_pred: (M,) Predicted measurement from nominal state h(x)
            H: (M, 15) Measurement Jacobian matrix
            R: (M, M) Measurement noise covariance matrix
            chi2_threshold: Optional Chi-Square innovation gate for outlier rejection
            
        Returns:
            accepted: bool (True if update passed gating and was applied)
            mahalanobis_dist: float (Normalized innovation distance)
            innovation: (M,) Innovation residual (z - z_pred)
        """
        z_meas = np.asarray(z, dtype=np.float64)
        z_hat = np.asarray(z_pred, dtype=np.float64)
        H_mat = np.asarray(H, dtype=np.float64)
        R_mat = np.asarray(R, dtype=np.float64)

        # 1. Innovation Residual & Innovation Covariance S (MxM)
        innovation = z_meas - z_hat
        S = H_mat @ self.P @ H_mat.T + R_mat
        S = 0.5 * (S + S.T)  # Enforce symmetry

        # 2. Chi-Square Outlier / Spoofing Gating Test
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(innovation.T @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            d2 = float(innovation.T @ S_inv @ innovation)

        if chi2_threshold is not None and d2 > chi2_threshold:
            # Innovation failed gating -> reject measurement
            return False, d2, innovation

        # 3. Kalman Gain K (15xM)
        K = self.P @ H_mat.T @ S_inv

        # 4. Error State Computation delta_x (15x1)
        delta_x = K @ innovation

        # 5. Error State Injection into Nominal State
        # Position correction
        self.pos += delta_x[0:3]

        # Velocity correction
        self.vel += delta_x[3:6]

        # Attitude correction: q = q * [1, 0.5 * delta_theta]^T
        delta_theta = delta_x[6:9]
        dtheta_norm = np.linalg.norm(delta_theta)
        if dtheta_norm > 1e-12:
            # Pure small-angle quaternion
            dq = np.array([np.cos(0.5 * dtheta_norm),
                           *(np.sin(0.5 * dtheta_norm) / dtheta_norm * delta_theta)], dtype=np.float64)
        else:
            dq = np.array([1.0, 0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2]], dtype=np.float64)

        self.quat = quat_normalize(quat_multiply(self.quat, dq))

        # Sensor bias corrections
        self.acc_bias += delta_x[9:12]
        self.gyro_bias += delta_x[12:15]

        # 6. Joseph-Form Covariance Update:
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I15 = np.eye(15, dtype=np.float64)
        IKH = I15 - K @ H_mat
        P_updated = IKH @ self.P @ IKH.T + K @ R_mat @ K.T

        # 7. Error State Reset & Covariance Correction
        # G_reset = diag(I_6, I - 0.5 * [delta_theta]_x, I_6)
        G_reset = np.eye(15, dtype=np.float64)
        G_reset[6:9, 6:9] = np.eye(3) - 0.5 * skew_symmetric(delta_theta)

        P_reset = G_reset @ P_updated @ G_reset.T
        self.P = 0.5 * (P_reset + P_reset.T)

        return True, d2, innovation

    # ========================================================================
    # 3. SPECIALIZED DIRECT MEASUREMENT HANDLERS
    # ========================================================================

    def update_position(self,
                        pos_meas: np.ndarray,
                        pos_cov: Union[float, np.ndarray],
                        chi2_threshold: Optional[float] = 16.27) -> Tuple[bool, float]:
        """
        Update with absolute 3D position measurement (e.g. GPS / GNSS position).
        chi2_threshold default = 16.27 (p = 0.001 for 3-DOF).
        """
        z = np.asarray(pos_meas, dtype=np.float64)
        z_pred = self.pos.copy()

        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        if np.isscalar(pos_cov):
            R = float(pos_cov) * np.eye(3, dtype=np.float64)
        elif pos_cov.ndim == 1:
            R = np.diag(pos_cov).astype(np.float64)
        else:
            R = np.asarray(pos_cov, dtype=np.float64)

        accepted, d2, _ = self.update(z, z_pred, H, R, chi2_threshold=chi2_threshold)
        return accepted, d2

    def update_velocity(self,
                        vel_meas: np.ndarray,
                        vel_cov: Union[float, np.ndarray],
                        chi2_threshold: Optional[float] = 16.27) -> Tuple[bool, float]:
        """
        Update with 3D velocity measurement (e.g. GPS Doppler velocity, AI predicted velocity).
        """
        z = np.asarray(vel_meas, dtype=np.float64)
        z_pred = self.vel.copy()

        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3)

        if np.isscalar(vel_cov):
            R = float(vel_cov) * np.eye(3, dtype=np.float64)
        elif vel_cov.ndim == 1:
            R = np.diag(vel_cov).astype(np.float64)
        else:
            R = np.asarray(vel_cov, dtype=np.float64)

        accepted, d2, _ = self.update(z, z_pred, H, R, chi2_threshold=chi2_threshold)
        return accepted, d2

    def update_zupt(self,
                    zupt_cov_sigma: float = 0.01) -> Tuple[bool, float]:
        """
        Execute Zero-Velocity Update (ZUPT): measurement z = [0, 0, 0]^T.
        """
        z = np.zeros(3, dtype=np.float64)
        R = (zupt_cov_sigma ** 2) * np.eye(3, dtype=np.float64)
        return self.update_velocity(z, R, chi2_threshold=None)

    def update_barometer(self,
                         alt_meas: float,
                         alt_cov_sigma: float = 1.0,
                         chi2_threshold: Optional[float] = 10.83) -> Tuple[bool, float]:
        """
        Update with 1D barometric altitude measurement (ENU z-coordinate).
        """
        z = np.array([float(alt_meas)], dtype=np.float64)
        z_pred = np.array([self.pos[2]], dtype=np.float64)

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 2] = 1.0  # pos_z

        R = np.array([[alt_cov_sigma ** 2]], dtype=np.float64)
        accepted, d2, _ = self.update(z, z_pred, H, R, chi2_threshold=chi2_threshold)
        return accepted, d2

    def update_nhc(self,
                   lateral_cov: float = 0.05,
                   vertical_cov: float = 0.05) -> Tuple[bool, float]:
        """
        Non-Holonomic Constraints (NHC) for wheeled vehicles:
        Enforces zero lateral (y_b) and vertical (z_b) velocity in body frame.
        """
        R_b2w = quat_to_rot_matrix(self.quat)
        R_w2b = R_b2w.T

        # Predicted body velocities
        v_body = R_w2b @ self.vel
        z_pred = np.array([v_body[1], v_body[2]], dtype=np.float64)  # [v_y_body, v_z_body]
        z = np.zeros(2, dtype=np.float64)                           # target = [0, 0]

        # Jacobian H (2x15)
        # dv_body / d_vel = R_w2b
        # dv_body / d_theta = [v_body]_x
        H = np.zeros((2, 15), dtype=np.float64)
        H[0, 3:6] = R_w2b[1, :]
        H[1, 3:6] = R_w2b[2, :]

        H[0, 6:9] = skew_symmetric(v_body)[1, :]
        H[1, 6:9] = skew_symmetric(v_body)[2, :]

        R = np.diag([lateral_cov ** 2, vertical_cov ** 2]).astype(np.float64)
        accepted, d2, _ = self.update(z, z_pred, H, R, chi2_threshold=None)
        return accepted, d2
