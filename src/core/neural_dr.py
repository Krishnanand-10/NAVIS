"""
Neural Dead Reckoning Engine (AI-Augmented Inertial Navigation)

Couples Deep Neural Drift & Bias Estimators (BiLSTM, TCN, or NumPy engines)
with Strapdown Inertial Kinematics:

Correction Modes:
1. VELOCITY_RESIDUAL: AI estimates instantaneous velocity correction vector to damp double-integration drift.
2. DYNAMIC_BIAS: AI dynamically tracks and subtracts 6-DOF accelerometer and gyroscope biases before integration.
3. HYBRID_CORRECTION: Simultaneous sensor bias compensation + velocity residual damping with uncertainty weighting.
4. DIRECT_VELOCITY: AI directly predicts translation velocity (PDR / RoNIN style).

Quantifies benchmark drift reduction relative to:
- Module 2: Classical Uncorrected Dead Reckoning
- Ground Truth Reference Trajectory
"""

import os
import argparse
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.dataset import TrajectoryData
from src.core.kinematics import (
    integrate_attitude_quaternion_rk4,
    integrate_attitude_quaternion_midpoint,
    integrate_attitude_quaternion_euler,
    integrate_attitude_quaternion_exponential,
    specific_force_to_world_acc,
    integrate_translation_trapezoidal,
    integrate_translation_euler,
    integrate_translation_verlet,
    GravityModel,
    STANDARD_GRAVITY_ENU,
)
from src.core.traditional_dr import DeadReckoningEngine, DeadReckoningResult
from src.models.drift_estimator import InertialDriftEstimator, IMUScaler
from src.models.motion_classifier import MotionClassifier, MotionContext
from src.utils.coordinate_transforms import quat_to_euler, euler_to_quat, quat_normalize, quat_to_rot_matrix
from src.utils.metrics import evaluate_trajectory, TrajectoryMetrics


class NeuralCorrectionMode(str, Enum):
    """Operational mode for Neural Dead Reckoning."""
    VELOCITY_RESIDUAL = "velocity_residual"
    DYNAMIC_BIAS = "dynamic_bias"
    HYBRID_CORRECTION = "hybrid"
    DIRECT_VELOCITY = "direct_velocity"

    @classmethod
    def from_str(cls, mode_str: str) -> "NeuralCorrectionMode":
        mode_str = mode_str.lower().strip()
        if "bias" in mode_str:
            return cls.DYNAMIC_BIAS
        elif "direct" in mode_str:
            return cls.DIRECT_VELOCITY
        elif "hybrid" in mode_str:
            return cls.HYBRID_CORRECTION
        return cls.VELOCITY_RESIDUAL


@dataclass
class NeuralDeadReckoningResult:
    """Evaluation result for Neural-Augmented Dead Reckoning."""
    timestamps: np.ndarray                      # (N,) [s]
    pos: np.ndarray                             # (N, 3) [m] AI-corrected position World ENU
    vel: np.ndarray                             # (N, 3) [m/s] AI-corrected velocity World ENU
    quat: np.ndarray                            # (N, 4) [w, x, y, z]
    euler_deg: np.ndarray                       # (N, 3) [roll, pitch, yaw] in degrees
    estimated_acc_bias: np.ndarray              # (N, 3) [m/s^2]
    estimated_gyro_bias: np.ndarray             # (N, 3) [rad/s]
    pos_uncertainty_std: np.ndarray             # (N, 3) [m] Position 1-sigma uncertainty
    vel_uncertainty_std: np.ndarray             # (N, 3) [m/s] Velocity 1-sigma uncertainty
    standard_dr_result: DeadReckoningResult      # Pure uncorrected Module 2 DR baseline
    correction_mode: NeuralCorrectionMode = NeuralCorrectionMode.HYBRID_CORRECTION
    metrics_neural: Optional[TrajectoryMetrics] = None
    metrics_standard: Optional[TrajectoryMetrics] = None
    drift_reduction_pct: float = 0.0
    ground_truth: Optional[Dict[str, np.ndarray]] = None

    def summary_table(self) -> str:
        """Generate side-by-side comparison table."""
        std_ate = self.metrics_standard.ate_rmse if self.metrics_standard else 0.0
        ai_ate = self.metrics_neural.ate_rmse if self.metrics_neural else 0.0
        std_final = self.metrics_standard.final_pos_error if self.metrics_standard else 0.0
        ai_final = self.metrics_neural.final_pos_error if self.metrics_neural else 0.0
        std_drift_rate = self.metrics_standard.drift_percentage if self.metrics_standard else 0.0
        ai_drift_rate = self.metrics_neural.drift_percentage if self.metrics_neural else 0.0

        table = []
        table.append("\n" + "=" * 76)
        table.append(f"  NAVIS Module 4: Neural Dead Reckoning Performance Report [{self.correction_mode.value.upper()}]")
        table.append("=" * 76)
        table.append(f"{'Metric':<32} | {'Pure DR (Mod 2)':<18} | {'Neural DR (Mod 4)':<18}")
        table.append("-" * 76)
        table.append(f"{'ATE RMSE [m]':<32} | {std_ate:<18.4f} | {ai_ate:<18.4f}")
        table.append(f"{'Final Position Error [m]':<32} | {std_final:<18.4f} | {ai_final:<18.4f}")
        table.append(f"{'Drift Rate [% distance]':<32} | {std_drift_rate:<18.4f} | {ai_drift_rate:<18.4f}")
        table.append("-" * 76)
        table.append(f"  ==> POSITIONAL DRIFT REDUCTION: {self.drift_reduction_pct:.2f}% improvement")
        table.append("=" * 76 + "\n")
        return "\n".join(table)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert Neural DR result to pandas DataFrame."""
        df_dict = {
            'timestamp': self.timestamps,
            'pos_x': self.pos[:, 0],
            'pos_y': self.pos[:, 1],
            'pos_z': self.pos[:, 2],
            'vel_x': self.vel[:, 0],
            'vel_y': self.vel[:, 1],
            'vel_z': self.vel[:, 2],
            'roll_deg': self.euler_deg[:, 0],
            'pitch_deg': self.euler_deg[:, 1],
            'yaw_deg': self.euler_deg[:, 2],
            'acc_bias_x': self.estimated_acc_bias[:, 0],
            'acc_bias_y': self.estimated_acc_bias[:, 1],
            'acc_bias_z': self.estimated_acc_bias[:, 2],
            'gyro_bias_x': self.estimated_gyro_bias[:, 0],
            'gyro_bias_y': self.estimated_gyro_bias[:, 1],
            'gyro_bias_z': self.estimated_gyro_bias[:, 2],
            'pos_std_x': self.pos_uncertainty_std[:, 0],
            'pos_std_y': self.pos_uncertainty_std[:, 1],
            'pos_std_z': self.pos_uncertainty_std[:, 2],
            'std_dr_pos_x': self.standard_dr_result.pos[:, 0],
            'std_dr_pos_y': self.standard_dr_result.pos[:, 1],
            'std_dr_pos_z': self.standard_dr_result.pos[:, 2],
        }
        if self.ground_truth is not None and 'pos' in self.ground_truth:
            gt_pos = self.ground_truth['pos']
            df_dict['gt_pos_x'] = gt_pos[:, 0]
            df_dict['gt_pos_y'] = gt_pos[:, 1]
            df_dict['gt_pos_z'] = gt_pos[:, 2]

        return pd.DataFrame(df_dict)

    def plot_comparison(self, save_path: Optional[str] = None, title: Optional[str] = None):
        """Plot comprehensive 2D/3D trajectory comparison with uncertainty bounds."""
        fig = plt.figure(figsize=(16, 10))

        # 1. 2D Trajectory (X vs Y)
        ax1 = fig.add_subplot(2, 2, 1)
        if self.ground_truth is not None and 'pos' in self.ground_truth:
            gt = self.ground_truth['pos']
            ax1.plot(gt[:, 0], gt[:, 1], 'k--', linewidth=2.0, label='Ground Truth')
        ax1.plot(self.standard_dr_result.pos[:, 0], self.standard_dr_result.pos[:, 1],
                 'r:', linewidth=1.8, label='Classical DR (Uncorrected)')
        ax1.plot(self.pos[:, 0], self.pos[:, 1],
                 'b-', linewidth=2.2, label=f'Neural DR ({self.correction_mode.value})')
        ax1.set_xlabel("East (X) [m]")
        ax1.set_ylabel("North (Y) [m]")
        ax1.set_title("2D Trajectory Comparison")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')

        # 2. Position Error vs Time
        ax2 = fig.add_subplot(2, 2, 2)
        if self.ground_truth is not None and 'pos' in self.ground_truth:
            gt = self.ground_truth['pos']
            err_std = np.linalg.norm(self.standard_dr_result.pos - gt, axis=1)
            err_ai = np.linalg.norm(self.pos - gt, axis=1)
            ax2.plot(self.timestamps, err_std, 'r--', label=f'Standard DR Error (End: {err_std[-1]:.1f}m)')
            ax2.plot(self.timestamps, err_ai, 'b-', label=f'Neural DR Error (End: {err_ai[-1]:.1f}m)')
            # Uncertainty envelope
            pos_3d_std = np.linalg.norm(self.pos_uncertainty_std, axis=1)
            ax2.fill_between(self.timestamps, np.maximum(0, err_ai - 2 * pos_3d_std),
                             err_ai + 2 * pos_3d_std, color='blue', alpha=0.15, label=r'$\pm 2\sigma$ Uncertainty')
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Position Error [m]")
        ax2.set_title(f"Cumulative Drift Error (Reduction: {self.drift_reduction_pct:.1f}%)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Velocity Comparison
        ax3 = fig.add_subplot(2, 2, 3)
        v_ai_mag = np.linalg.norm(self.vel, axis=1)
        v_std_mag = np.linalg.norm(self.standard_dr_result.vel, axis=1)
        if self.ground_truth is not None and 'vel' in self.ground_truth:
            gt_v_mag = np.linalg.norm(self.ground_truth['vel'], axis=1)
            ax3.plot(self.timestamps, gt_v_mag, 'k--', label='Ground Truth Speed')
        ax3.plot(self.timestamps, v_std_mag, 'r:', label='Standard DR Speed')
        ax3.plot(self.timestamps, v_ai_mag, 'g-', label='Neural Corrected Speed')
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Speed [m/s]")
        ax3.set_title("Speed Profile Evolution")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Neural Estimated Sensor Biases
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(self.timestamps, self.estimated_acc_bias[:, 0], label='Acc Bias X')
        ax4.plot(self.timestamps, self.estimated_acc_bias[:, 1], label='Acc Bias Y')
        ax4.plot(self.timestamps, self.estimated_acc_bias[:, 2], label='Acc Bias Z')
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Acc Bias [m/s²]")
        ax4.set_title("AI Estimated Dynamic Accelerometer Bias")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Saved comparison plot to: {save_path}")
        plt.close(fig)


# ============================================================================
# 2. NEURAL DEAD RECKONING ENGINE
# ============================================================================

class NeuralDeadReckoningEngine:
    """
    Core Navigation Engine executing Strapdown Inertial Kinematics integrated
    with Deep Neural Drift & Dynamic Bias Estimations.
    """

    def __init__(self,
                 drift_estimator: Optional[InertialDriftEstimator] = None,
                 correction_mode: Union[str, NeuralCorrectionMode] = NeuralCorrectionMode.HYBRID_CORRECTION,
                 velocity_blend_alpha: float = 0.85,
                 bias_blend_alpha: float = 0.70,
                 attitude_integrator: str = "rk4",
                 translation_integrator: str = "trapezoidal",
                 gravity_model: Optional[GravityModel] = None):
        """
        Args:
            drift_estimator: Trained InertialDriftEstimator (if None, initializes default).
            correction_mode: 'velocity_residual', 'dynamic_bias', 'hybrid', or 'direct_velocity'.
            velocity_blend_alpha: Weight factor for blending neural velocity corrections (0.0 to 1.0).
            bias_blend_alpha: Weight factor for dynamic bias correction.
            attitude_integrator: 'rk4', 'midpoint', or 'euler'.
            translation_integrator: 'trapezoidal', 'verlet', or 'euler'.
            gravity_model: GravityModel instance.
        """
        if drift_estimator is None:
            self.drift_estimator = InertialDriftEstimator(model_type="numpy")
        else:
            self.drift_estimator = drift_estimator

        if isinstance(correction_mode, str):
            self.correction_mode = NeuralCorrectionMode.from_str(correction_mode)
        else:
            self.correction_mode = correction_mode

        self.velocity_blend_alpha = float(np.clip(velocity_blend_alpha, 0.0, 1.0))
        self.bias_blend_alpha = float(np.clip(bias_blend_alpha, 0.0, 1.0))
        self.attitude_integrator = attitude_integrator.lower()
        self.translation_integrator = translation_integrator.lower()
        self.gravity = gravity_model or GravityModel()

    def process(self,
                trajectory: TrajectoryData,
                initial_pos: Optional[np.ndarray] = None,
                initial_vel: Optional[np.ndarray] = None,
                initial_quat: Optional[np.ndarray] = None) -> NeuralDeadReckoningResult:
        """
        Run complete Neural Dead Reckoning over an IMU trajectory stream.
        """
        # Run Classical DR baseline for benchmarking
        att_method = "quaternion_rk4" if "rk4" in self.attitude_integrator else "quaternion_midpoint" if "midpoint" in self.attitude_integrator else "quaternion_euler"
        trans_method = "trapezoidal" if "trap" in self.translation_integrator else "forward_euler" if "euler" in self.translation_integrator else "velocity_verlet"

        std_engine = DeadReckoningEngine(
            attitude_method=att_method,
            translation_method=trans_method,
            gravity=self.gravity,
            initial_pos=initial_pos,
            initial_vel=initial_vel,
            initial_quat=initial_quat,
        )
        std_result = std_engine.process_trajectory(
            data=trajectory
        )

        t = trajectory.timestamps
        N = len(t)
        raw_acc = trajectory.acc.copy()
        raw_gyro = trajectory.gyro.copy()
        imu_combined = np.hstack([raw_acc, raw_gyro])

        # Step 1: Predict neural velocity corrections and dynamic biases
        ai_preds = self.drift_estimator.predict_sequence(imu_combined, step_size=1)
        ai_vel = ai_preds['vel']
        ai_acc_bias = ai_preds['acc_bias'] * self.bias_blend_alpha
        ai_gyro_bias = ai_preds['gyro_bias'] * self.bias_blend_alpha
        vel_std = ai_preds['vel_std']

        # Preallocate states
        pos = np.zeros((N, 3), dtype=np.float64)
        vel = np.zeros((N, 3), dtype=np.float64)
        quat = np.zeros((N, 4), dtype=np.float64)
        euler_deg = np.zeros((N, 3), dtype=np.float64)
        pos_cov_std = np.zeros((N, 3), dtype=np.float64)

        # Initialize start states
        if initial_pos is not None:
            pos[0] = initial_pos
        elif trajectory.gt_pos is not None:
            pos[0] = trajectory.gt_pos[0]

        if initial_vel is not None:
            vel[0] = initial_vel
        elif trajectory.gt_vel is not None:
            vel[0] = trajectory.gt_vel[0]

        if initial_quat is not None:
            quat[0] = quat_normalize(initial_quat)
        elif trajectory.gt_quat is not None:
            quat[0] = quat_normalize(trajectory.gt_quat[0])
        else:
            quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        euler_deg[0] = quat_to_euler(quat[0])

        # Step 2: Integrate kinematics with neural augmentation
        q_curr = quat[0].copy()
        p_curr = pos[0].copy()
        v_curr = vel[0].copy()
        pos_var_accum = np.zeros(3, dtype=np.float64)

        # Precompute initial world acceleration
        f_init = raw_acc[0] - (ai_acc_bias[0] if self.correction_mode in (NeuralCorrectionMode.DYNAMIC_BIAS, NeuralCorrectionMode.HYBRID_CORRECTION) else 0.0)
        a_w_prev = specific_force_to_world_acc(f_init, q_curr, self.gravity)
        prev_omega = raw_gyro[0].copy()

        for k in range(1, N):
            dt_k = t[k] - t[k - 1]
            if dt_k <= 0:
                dt_k = 0.01

            omega_meas = raw_gyro[k - 1]
            acc_meas = raw_acc[k - 1]

            # Mode handling for sensor bias subtraction
            if self.correction_mode in (NeuralCorrectionMode.DYNAMIC_BIAS, NeuralCorrectionMode.HYBRID_CORRECTION):
                omega_corr = omega_meas - ai_gyro_bias[k - 1]
                acc_corr = acc_meas - ai_acc_bias[k - 1]
            else:
                omega_corr = omega_meas
                acc_corr = acc_meas

            # Attitude integration
            if self.attitude_integrator == "rk4":
                q_next = integrate_attitude_quaternion_rk4(q_curr, prev_omega, omega_corr, dt_k)
            elif self.attitude_integrator == "midpoint":
                q_next = integrate_attitude_quaternion_midpoint(q_curr, prev_omega, omega_corr, dt_k)
            else:
                q_next = integrate_attitude_quaternion_euler(q_curr, omega_corr, dt_k)

            # World acceleration computation
            a_w_curr = specific_force_to_world_acc(acc_corr, q_next, self.gravity)

            # Translation integration based on correction mode
            if self.correction_mode == NeuralCorrectionMode.DIRECT_VELOCITY:
                # Direct neural velocity integration: AI velocity rotated to world frame
                R_body_to_world = quat_to_rot_matrix(q_next)
                v_world = np.dot(R_body_to_world, ai_vel[k]) if np.linalg.norm(ai_vel[k]) > 0 else ai_vel[k]
                p_next = p_curr + v_world * dt_k
                v_next = v_world

            elif self.correction_mode == NeuralCorrectionMode.VELOCITY_RESIDUAL:
                # Standard double integration + neural velocity damping
                p_next, v_inertial = integrate_translation_trapezoidal(p_curr, v_curr, a_w_prev, a_w_curr, dt_k)
                # Blend with neural velocity
                R_body_to_world = quat_to_rot_matrix(q_next)
                v_target = np.dot(R_body_to_world, ai_vel[k])
                v_next = (1.0 - self.velocity_blend_alpha) * v_inertial + self.velocity_blend_alpha * v_target

            elif self.correction_mode == NeuralCorrectionMode.HYBRID_CORRECTION:
                # Combined bias compensation + velocity residual blending
                p_next, v_inertial = integrate_translation_trapezoidal(p_curr, v_curr, a_w_prev, a_w_curr, dt_k)
                R_body_to_world = quat_to_rot_matrix(q_next)
                v_target = np.dot(R_body_to_world, ai_vel[k])
                # Adaptive blend
                alpha = self.velocity_blend_alpha
                v_next = (1.0 - alpha) * v_inertial + alpha * v_target
                # Position drift correction
                p_next = p_curr + 0.5 * (v_curr + v_next) * dt_k

            else:  # DYNAMIC_BIAS
                p_next, v_next = integrate_translation_trapezoidal(p_curr, v_curr, a_w_prev, a_w_curr, dt_k)

            # Uncertainty accumulation
            pos_var_accum += (vel_std[k] * dt_k) ** 2
            pos_cov_std[k] = np.sqrt(pos_var_accum)

            # Store states
            q_curr = q_next
            p_curr = p_next
            v_curr = v_next
            a_w_prev = a_w_curr
            prev_omega = omega_corr.copy()

            quat[k] = q_curr
            pos[k] = p_curr
            vel[k] = v_curr
            euler_deg[k] = quat_to_euler(q_curr)

        # Performance evaluation vs Ground Truth
        metrics_neural = None
        metrics_standard = None
        drift_reduction_pct = 0.0
        gt_dict = None

        if trajectory.gt_pos is not None:
            gt_pos = trajectory.gt_pos
            gt_vel = trajectory.gt_vel if trajectory.gt_vel is not None else np.zeros_like(gt_pos)
            gt_quat = trajectory.gt_quat if trajectory.gt_quat is not None else np.zeros((N, 4))
            gt_dict = {
                'pos': gt_pos,
                'vel': gt_vel,
                'quat': gt_quat,
            }

            metrics_neural = evaluate_trajectory(
                est_pos=pos,
                gt_pos=gt_pos,
                timestamps=t,
                est_vel=vel,
                gt_vel=gt_vel,
                est_quat=quat,
                gt_quat=gt_quat,
            )

            metrics_standard = evaluate_trajectory(
                est_pos=std_result.pos,
                gt_pos=gt_pos,
                timestamps=t,
                est_vel=std_result.vel,
                gt_vel=gt_vel,
                est_quat=std_result.quat,
                gt_quat=gt_quat,
            )

            if metrics_standard.ate_rmse > 1e-6:
                reduction = (metrics_standard.ate_rmse - metrics_neural.ate_rmse) / metrics_standard.ate_rmse * 100.0
                drift_reduction_pct = float(np.clip(reduction, 0.0, 100.0))

        return NeuralDeadReckoningResult(
            timestamps=t,
            pos=pos,
            vel=vel,
            quat=quat,
            euler_deg=euler_deg,
            estimated_acc_bias=ai_acc_bias,
            estimated_gyro_bias=ai_gyro_bias,
            pos_uncertainty_std=pos_cov_std,
            vel_uncertainty_std=vel_std,
            standard_dr_result=std_result,
            correction_mode=self.correction_mode,
            metrics_neural=metrics_neural,
            metrics_standard=metrics_standard,
            drift_reduction_pct=drift_reduction_pct,
            ground_truth=gt_dict,
        )


def run_neural_dead_reckoning(
    trajectory: TrajectoryData,
    drift_estimator: Optional[InertialDriftEstimator] = None,
    correction_mode: str = "hybrid",
    velocity_blend_alpha: float = 0.85,
    bias_blend_alpha: float = 0.70,
) -> NeuralDeadReckoningResult:
    """
    Convenience function to run Neural Dead Reckoning on an IMU trajectory.
    """
    engine = NeuralDeadReckoningEngine(
        drift_estimator=drift_estimator,
        correction_mode=correction_mode,
        velocity_blend_alpha=velocity_blend_alpha,
        bias_blend_alpha=bias_blend_alpha,
    )
    return engine.process(trajectory)


# ============================================================================
# 3. CLI DEMONSTRATION & BENCHMARKING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run NAVIS Neural Dead Reckoning Benchmark.")
    parser.add_argument("--profile", type=str, default="figure_eight", help="Trajectory motion profile.")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration [s].")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["velocity_residual", "dynamic_bias", "hybrid", "direct_velocity"],
                        help="Neural correction mode.")
    parser.add_argument("--model-type", type=str, default="numpy", choices=["bilstm", "tcn", "numpy"],
                        help="Estimator model type.")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots.")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save comparison plot image.")
    args = parser.parse_args()

    from src.data.loaders import SyntheticDataLoader
    from src.models.train_drift_models import train_drift_estimator

    print(f"\n[*] Generating benchmark {args.profile.upper()} trajectory (Duration: {args.duration}s)...")
    traj = SyntheticDataLoader.create(
        profile=args.profile,
        duration=args.duration,
        dt=0.01,
        imu_grade="consumer",
        seed=42,
    )

    print(f"[*] Initializing Neural Drift Estimator [{args.model_type.upper()}]...")
    estimator = train_drift_estimator(
        model_type=args.model_type,
        epochs=10,
        save_dir="checkpoints/drift_estimator",
        seed=42
    )

    print(f"[*] Running Neural Dead Reckoning in [{args.mode.upper()}] mode...")
    engine = NeuralDeadReckoningEngine(
        drift_estimator=estimator,
        correction_mode=args.mode,
    )
    result = engine.process(traj)

    print(result.summary_table())

    if args.save_plot or args.plot:
        out_path = args.save_plot or f"plots/neural_dr_{args.profile}_{args.mode}.png"
        result.plot_comparison(save_path=out_path)


if __name__ == "__main__":
    main()
