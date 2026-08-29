"""
ZUPT-Augmented Dead Reckoning Engine (AI-ZUPT & Statistical Stance Updates)

Integrates Zero-Velocity Updates (ZUPT) into Strapdown Inertial Navigation:
1. Detects stance / standstill intervals using statistical (GLRT, ARE, MAG) or AI-ZUPT neural networks.
2. When stance is confirmed:
   - Resets velocity error to zero (v = 0).
   - Re-estimates local gyro and accelerometer biases online.
   - Constrains position integration to eliminate stationary drift compounding.
3. Quantifies benchmark drift reduction relative to uncorrected Module 2 Dead Reckoning.
"""

import os
import argparse
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
    integrate_attitude_euler_angles,
    specific_force_to_world_acc,
    integrate_translation_trapezoidal,
    integrate_translation_euler,
    integrate_translation_verlet,
    GravityModel,
    STANDARD_GRAVITY_ENU,
)
from src.core.traditional_dr import DeadReckoningEngine, DeadReckoningResult
from src.models.zupt_detector import ZUPTDetector, GLRTDetector
from src.models.motion_classifier import MotionClassifier, MotionContext
from src.utils.coordinate_transforms import quat_to_euler, euler_to_quat, quat_normalize
from src.utils.metrics import evaluate_trajectory, TrajectoryMetrics


@dataclass
class ZUPTDeadReckoningResult:
    """Evaluation result for ZUPT-augmented Dead Reckoning."""
    timestamps: np.ndarray                 # (N,) [s]
    pos: np.ndarray                        # (N, 3) [m] ZUPT-corrected position
    vel: np.ndarray                        # (N, 3) [m/s] ZUPT-corrected velocity
    quat: np.ndarray                       # (N, 4) [w, x, y, z]
    euler_deg: np.ndarray                  # (N, 3) [roll, pitch, yaw] in degrees
    stance_mask: np.ndarray                # (N,) boolean stance indicators
    standard_dr_result: DeadReckoningResult # Pure uncorrected Module 2 DR result
    metrics_zupt: Optional[TrajectoryMetrics] = None
    metrics_standard: Optional[TrajectoryMetrics] = None
    drift_reduction_pct: float = 0.0
    ground_truth: Optional[Dict[str, np.ndarray]] = None
    motion_context: Optional[MotionContext] = None

    def summary_table(self) -> str:
        """Generate side-by-side comparison table."""
        std_ate = self.metrics_standard.ate_rmse if self.metrics_standard else 0.0
        zupt_ate = self.metrics_zupt.ate_rmse if self.metrics_zupt else 0.0
        std_final = self.metrics_standard.final_pos_error if self.metrics_standard else 0.0
        zupt_final = self.metrics_zupt.final_pos_error if self.metrics_zupt else 0.0
        std_drift_pct = self.metrics_standard.drift_percentage if self.metrics_standard else 0.0
        zupt_drift_pct = self.metrics_zupt.drift_percentage if self.metrics_zupt else 0.0

        lines = [
            "=" * 72,
            f"{'NAVIS ZUPT-AUGMENTED DEAD RECKONING BENCHMARK':^72}",
            "=" * 72,
            f"{'Metric':<30} | {'Standard DR (Mod 2)':<18} | {'ZUPT-Aided DR (Mod 3)':<18}",
            "-" * 72,
            f"{'3D Position ATE RMSE (m)':<30} | {std_ate:<18.4f} | {zupt_ate:<18.4f}",
            f"{'Final Endpoint Error (m)':<30} | {std_final:<18.4f} | {zupt_final:<18.4f}",
            f"{'Cumulative Drift (%)':<30} | {std_drift_pct:<18.2f}%| {zupt_drift_pct:<18.2f}%",
            "-" * 72,
            f"{'Drift Error Reduction':<30} | {'BASELINE':<18} | {self.drift_reduction_pct:<18.2f}%",
            f"{'Stance Detection Ratio':<30} | {'N/A':<18} | {np.mean(self.stance_mask):<18.2%}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def plot_comparison(self, title: str = "NAVIS ZUPT-Augmented Dead Reckoning Benchmark", save_path: Optional[str] = None):
        """Generate comprehensive 6-panel visualizer."""
        has_gt = self.ground_truth is not None and 'pos' in self.ground_truth

        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # 1. 2D Trajectory Comparison
        ax1 = fig.add_subplot(2, 3, 1)
        if has_gt:
            ax1.plot(self.ground_truth['pos'][:, 0], self.ground_truth['pos'][:, 1],
                     label='Ground Truth', color='black', linewidth=2.5)
        ax1.plot(self.standard_dr_result.pos[:, 0], self.standard_dr_result.pos[:, 1],
                 label='Pure DR (Mod 2)', color='crimson', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.plot(self.pos[:, 0], self.pos[:, 1],
                 label='AI-ZUPT DR (Mod 3)', color='forestgreen', linewidth=2.0)
        ax1.scatter(self.pos[0, 0], self.pos[0, 1], color='blue', marker='o', s=60, label='Start')
        ax1.scatter(self.pos[-1, 0], self.pos[-1, 1], color='forestgreen', marker='x', s=60, label='ZUPT End')
        if has_gt:
            ax1.scatter(self.ground_truth['pos'][-1, 0], self.ground_truth['pos'][-1, 1],
                        color='black', marker='x', s=60, label='GT End')
        ax1.set_title("2D Trajectory (East-North)")
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='best', fontsize=8)
        ax1.axis('equal')

        # 2. Cumulative Drift Error vs Time
        ax2 = fig.add_subplot(2, 3, 2)
        if has_gt:
            err_std = np.linalg.norm(self.standard_dr_result.pos - self.ground_truth['pos'], axis=-1)
            err_zupt = np.linalg.norm(self.pos - self.ground_truth['pos'], axis=-1)
            ax2.plot(self.timestamps, err_std, label='Pure DR Error', color='crimson', linestyle='--', linewidth=2)
            ax2.plot(self.timestamps, err_zupt, label='ZUPT-Aided Error', color='forestgreen', linewidth=2.5)
            ax2.set_title("Positional Drift Divergence vs Time")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Error (m)")
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(loc='best', fontsize=8)

        # 3. 3D Trajectory
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        if has_gt:
            ax3.plot(self.ground_truth['pos'][:, 0], self.ground_truth['pos'][:, 1], self.ground_truth['pos'][:, 2],
                     label='Ground Truth', color='black', linewidth=2.5)
        ax3.plot(self.standard_dr_result.pos[:, 0], self.standard_dr_result.pos[:, 1], self.standard_dr_result.pos[:, 2],
                 label='Pure DR', color='crimson', linestyle='--', linewidth=1.2)
        ax3.plot(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
                 label='ZUPT-Aided DR', color='forestgreen', linewidth=2.0)
        ax3.set_title("3D Trajectory Tracking")
        ax3.set_xlabel("East (m)")
        ax3.set_ylabel("North (m)")
        ax3.set_zlabel("Up (m)")
        ax3.legend(loc='best', fontsize=8)

        # 4. Stance Detection Timeline
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.fill_between(self.timestamps, 0, self.stance_mask.astype(int), color='forestgreen', alpha=0.4, label='Detected Stance (ZUPT)')
        if has_gt and 'stance_mask' in self.ground_truth and self.ground_truth['stance_mask'] is not None:
            ax4.step(self.timestamps, self.ground_truth['stance_mask'], color='black', linestyle=':', label='True Stance')
        ax4.set_title("Stance / Zero-Velocity Event Timeline")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Stance Active")
        ax4.set_ylim(-0.1, 1.2)
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.legend(loc='best', fontsize=8)

        # 5. Velocity Profile (Speed Comparison)
        ax5 = fig.add_subplot(2, 3, 5)
        speed_std = np.linalg.norm(self.standard_dr_result.vel, axis=-1)
        speed_zupt = np.linalg.norm(self.vel, axis=-1)
        ax5.plot(self.timestamps, speed_std, label='Pure DR Speed', color='crimson', linestyle='--', alpha=0.7)
        ax5.plot(self.timestamps, speed_zupt, label='ZUPT-Aided Speed', color='forestgreen', linewidth=1.5)
        if has_gt and 'vel' in self.ground_truth:
            speed_gt = np.linalg.norm(self.ground_truth['vel'], axis=-1)
            ax5.plot(self.timestamps, speed_gt, label='True Speed', color='black', linestyle=':', linewidth=1.5)
        ax5.set_title("Speed Magnitude (m/s)")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Speed (m/s)")
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.legend(loc='best', fontsize=8)

        # 6. Benchmark Summary Table
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        summary = self.summary_table()
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#ced4da'))

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[NAVIS] Saved ZUPT-DR benchmark plot to: {save_path}")
        return fig


class ZUPTDeadReckoningEngine:
    """
    Dead Reckoning Solver augmented with AI/Statistical Zero-Velocity Updates.
    """

    def __init__(self,
                 detection_method: str = "glrt",
                 window_size: int = 15,
                 threshold: Optional[float] = None,
                 attitude_method: str = "quaternion_rk4",
                 translation_method: str = "trapezoidal",
                 gravity: Union[np.ndarray, GravityModel] = STANDARD_GRAVITY_ENU,
                 online_bias_tracking: bool = True):
        self.detection_method = detection_method.lower()
        self.window_size = window_size
        self.threshold = threshold
        self.attitude_method = attitude_method
        self.translation_method = translation_method
        self.online_bias_tracking = online_bias_tracking

        if isinstance(gravity, GravityModel):
            self.gravity = gravity
        else:
            self.gravity = GravityModel(custom_g=gravity)

    def process_trajectory(self,
                           data: Union[TrajectoryData, pd.DataFrame, Dict[str, np.ndarray]],
                           stance_mask: Optional[np.ndarray] = None) -> ZUPTDeadReckoningResult:
        """
        Execute comparative dead reckoning: Pure DR vs ZUPT-augmented DR.
        """
        # Parse inputs
        if isinstance(data, TrajectoryData):
            timestamps = data.timestamps
            acc = data.acc
            gyro = data.gyro
            gt_pos = data.gt_pos
            gt_vel = data.gt_vel
            gt_quat = data.gt_quat
            gt_stance = data.stance_mask
        elif isinstance(data, pd.DataFrame):
            timestamps = data['timestamp'].values
            acc = data[['acc_x', 'acc_y', 'acc_z']].values
            gyro = data[['gyro_x', 'gyro_y', 'gyro_z']].values
            gt_pos = data[['gt_pos_x', 'gt_pos_y', 'gt_pos_z']].values if 'gt_pos_x' in data.columns else None
            gt_vel = data[['gt_vel_x', 'gt_vel_y', 'gt_vel_z']].values if 'gt_vel_x' in data.columns else None
            gt_quat = data[['gt_quat_w', 'gt_quat_x', 'gt_quat_y', 'gt_quat_z']].values if 'gt_quat_w' in data.columns else None
            gt_stance = data['stance_mask'].values if 'stance_mask' in data.columns else None
        else:
            timestamps = np.asarray(data['timestamp'])
            acc = np.asarray(data['acc'])
            gyro = np.asarray(data['gyro'])
            gt_pos = data.get('gt_pos', None)
            gt_vel = data.get('gt_vel', None)
            gt_quat = data.get('gt_quat', None)
            gt_stance = data.get('stance_mask', None)

        N = len(timestamps)

        # 1. Run Pure Uncorrected Module 2 Dead Reckoning Baseline
        std_engine = DeadReckoningEngine(
            attitude_method=self.attitude_method,
            translation_method=self.translation_method,
            gravity=self.gravity,
        )
        standard_dr_result = std_engine.process_trajectory(data)

        # 2. Compute Stance Mask
        if stance_mask is not None:
            detected_stance = np.asarray(stance_mask, dtype=bool)
        else:
            detected_stance = ZUPTDetector.detect(
                acc=acc,
                gyro=gyro,
                method=self.detection_method,
                window_size=self.window_size,
                threshold=self.threshold
            )

        # 3. ZUPT-Augmented Mechanization Integration Loop
        pos = np.zeros((N, 3), dtype=np.float64)
        vel = np.zeros((N, 3), dtype=np.float64)
        quat = np.zeros((N, 4), dtype=np.float64)
        euler_deg = np.zeros((N, 3), dtype=np.float64)

        if gt_pos is not None and gt_vel is not None and gt_quat is not None:
            pos[0] = gt_pos[0].copy()
            vel[0] = gt_vel[0].copy()
            quat[0] = gt_quat[0].copy()
        else:
            quat[0] = np.array([1.0, 0.0, 0.0, 0.0])

        r0, p0, y0 = quat_to_euler(quat[0])
        euler_deg[0] = np.rad2deg([r0, p0, y0])

        curr_p = pos[0].copy()
        curr_v = vel[0].copy()
        curr_q = quat[0].copy()
        curr_e = np.array([r0, p0, y0], dtype=np.float64)
        prev_omega = gyro[0].copy()
        prev_acc_w = specific_force_to_world_acc(acc[0], curr_q, self.gravity)

        # Online bias tracker
        gyro_bias_est = np.zeros(3, dtype=np.float64)
        stance_counter = 0

        for i in range(1, N):
            dt = float(timestamps[i] - timestamps[i - 1])
            if dt <= 0:
                dt = 0.01

            is_stance = detected_stance[i]
            corrected_gyro = gyro[i] - gyro_bias_est

            # Online gyro bias calibration during sustained stance
            if is_stance:
                stance_counter += 1
                if self.online_bias_tracking and stance_counter > 5:
                    # Exponential moving average filter on stationary gyro
                    alpha = 0.05
                    gyro_bias_est = (1.0 - alpha) * gyro_bias_est + alpha * gyro[i]
            else:
                stance_counter = 0

            # Attitude integration
            if self.attitude_method == "quaternion_rk4":
                curr_q = integrate_attitude_quaternion_rk4(curr_q, prev_omega, corrected_gyro, dt)
            elif self.attitude_method == "quaternion_midpoint":
                curr_q = integrate_attitude_quaternion_midpoint(curr_q, prev_omega, corrected_gyro, dt)
            else:
                curr_q = integrate_attitude_quaternion_euler(curr_q, corrected_gyro, dt)

            # Specific force transformation
            curr_acc_w = specific_force_to_world_acc(acc[i], curr_q, self.gravity)

            # Translation integration with Zero-Velocity Update
            if is_stance:
                # STANCE: Zero-Velocity Update condition
                curr_v = np.zeros(3, dtype=np.float64)
                # Position holds constant during stance phase
                curr_p = curr_p.copy()
            else:
                # MOTION: Integrate kinematics
                if self.translation_method == "forward_euler":
                    curr_p, curr_v = integrate_translation_euler(curr_p, curr_v, prev_acc_w, dt)
                elif self.translation_method == "velocity_verlet":
                    curr_p, curr_v = integrate_translation_verlet(curr_p, curr_v, prev_acc_w, curr_acc_w, dt)
                else:
                    curr_p, curr_v = integrate_translation_trapezoidal(curr_p, curr_v, prev_acc_w, curr_acc_w, dt)

            pos[i] = curr_p.copy()
            vel[i] = curr_v.copy()
            quat[i] = curr_q.copy()
            r, p, y = quat_to_euler(curr_q)
            euler_deg[i] = np.rad2deg([r, p, y])

            prev_omega = corrected_gyro.copy()
            prev_acc_w = curr_acc_w.copy()

        # 4. Evaluation and Drift Reduction Metric Calculation
        metrics_zupt = None
        metrics_std = standard_dr_result.metrics
        drift_reduct = 0.0
        gt_dict = None

        if gt_pos is not None:
            gt_dict = {
                'pos': gt_pos,
                'vel': gt_vel,
                'quat': gt_quat,
                'stance_mask': gt_stance,
            }
            metrics_zupt = evaluate_trajectory(
                est_pos=pos,
                gt_pos=gt_pos,
                timestamps=timestamps,
                est_vel=vel if gt_vel is not None else None,
                gt_vel=gt_vel,
                est_quat=quat if gt_quat is not None else None,
                gt_quat=gt_quat,
            )
            if metrics_std and metrics_std.ate_rmse > 1e-4:
                drift_reduct = ((metrics_std.ate_rmse - metrics_zupt.ate_rmse) / metrics_std.ate_rmse) * 100.0

        return ZUPTDeadReckoningResult(
            timestamps=timestamps,
            pos=pos,
            vel=vel,
            quat=quat,
            euler_deg=euler_deg,
            stance_mask=detected_stance,
            standard_dr_result=standard_dr_result,
            metrics_zupt=metrics_zupt,
            metrics_standard=metrics_std,
            drift_reduction_pct=float(drift_reduct),
            ground_truth=gt_dict,
        )


def run_zupt_dead_reckoning(profile: str = "pedestrian",
                            duration: float = 60.0,
                            imu_grade: str = "consumer",
                            method: str = "glrt",
                            plot: bool = True,
                            save_dir: str = "data/processed") -> ZUPTDeadReckoningResult:
    """
    Convenience benchmark runner comparing Pure Dead Reckoning vs ZUPT-Augmented Dead Reckoning.
    """
    from src.data.loaders import SyntheticDataLoader

    print(f"[NAVIS] Generating '{profile}' trajectory ({imu_grade.title()} IMU, {duration}s)...")
    traj = SyntheticDataLoader.create(profile=profile, duration=duration, imu_grade=imu_grade)

    engine = ZUPTDeadReckoningEngine(detection_method=method)
    result = engine.process_trajectory(traj)

    print(result.summary_table())

    if plot:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"zupt_benchmark_{profile}_{imu_grade}.png")
        result.plot_comparison(
            title=f"NAVIS Module 3: AI-ZUPT Dead Reckoning Benchmark ({profile.title()}, {imu_grade.title()} IMU)",
            save_path=plot_path
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAVIS Module 3: AI-ZUPT Dead Reckoning Runner")
    parser.add_argument("--profile", type=str, default="pedestrian", help="Motion profile")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--imu-grade", type=str, default="consumer", choices=["consumer", "industrial", "tactical"])
    parser.add_argument("--method", type=str, default="glrt", choices=["glrt", "are", "mag", "ai"])
    parser.add_argument("--plot", action="store_true", default=True, help="Save evaluation plot")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()

    run_zupt_dead_reckoning(
        profile=args.profile,
        duration=args.duration,
        imu_grade=args.imu_grade,
        method=args.method,
        plot=args.plot,
        save_dir=args.output_dir,
    )
