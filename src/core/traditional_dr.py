"""
Classical Strapdown Dead Reckoning (DR) Baseline Engine

Executes pure inertial dead reckoning by integrating raw 6-DOF IMU streams:
- Attitude Integration (Euler, RK4 Quaternion, Midpoint, Matrix Exponential)
- Specific Force Rotation & Gravity Compensation
- Velocity & Position Integration (Forward Euler, Trapezoidal/Heun, Velocity-Verlet)
- Initial Alignment & Leveling (Static Accelerometer/Magnetometer alignment)
- Comparative Error & Drift Benchmarking against Ground Truth
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
    integrate_translation_euler,
    integrate_translation_trapezoidal,
    integrate_translation_verlet,
    GravityModel,
    STANDARD_GRAVITY_ENU,
)
from src.utils.coordinate_transforms import (
    quat_normalize,
    quat_to_euler,
    euler_to_quat,
    quat_to_rot_matrix,
)
from src.utils.metrics import evaluate_trajectory, TrajectoryMetrics


@dataclass
class DeadReckoningResult:
    """Complete estimated trajectory states and performance evaluation from Dead Reckoning."""
    timestamps: np.ndarray             # (N,) [s]
    pos: np.ndarray                    # (N, 3) [m] World ENU
    vel: np.ndarray                    # (N, 3) [m/s] World ENU
    quat: np.ndarray                   # (N, 4) [w, x, y, z]
    euler_deg: np.ndarray              # (N, 3) [roll, pitch, yaw] in degrees
    acc_world: np.ndarray              # (N, 3) [m/s^2] World ENU
    metrics: Optional[TrajectoryMetrics] = None
    ground_truth: Optional[Dict[str, np.ndarray]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert DR estimation result to pandas DataFrame."""
        df_dict = {
            'timestamp': self.timestamps,
            'dr_pos_x': self.pos[:, 0],
            'dr_pos_y': self.pos[:, 1],
            'dr_pos_z': self.pos[:, 2],
            'dr_vel_x': self.vel[:, 0],
            'dr_vel_y': self.vel[:, 1],
            'dr_vel_z': self.vel[:, 2],
            'dr_quat_w': self.quat[:, 0],
            'dr_quat_x': self.quat[:, 1],
            'dr_quat_y': self.quat[:, 2],
            'dr_quat_z': self.quat[:, 3],
            'dr_roll_deg': self.euler_deg[:, 0],
            'dr_pitch_deg': self.euler_deg[:, 1],
            'dr_yaw_deg': self.euler_deg[:, 2],
            'dr_acc_world_x': self.acc_world[:, 0],
            'dr_acc_world_y': self.acc_world[:, 1],
            'dr_acc_world_z': self.acc_world[:, 2],
        }
        if self.ground_truth is not None:
            if 'pos' in self.ground_truth:
                df_dict['gt_pos_x'] = self.ground_truth['pos'][:, 0]
                df_dict['gt_pos_y'] = self.ground_truth['pos'][:, 1]
                df_dict['gt_pos_z'] = self.ground_truth['pos'][:, 2]
                # Instantaneous error
                diff = self.pos - self.ground_truth['pos']
                df_dict['error_pos_3d'] = np.linalg.norm(diff, axis=-1)
                df_dict['error_pos_xy'] = np.linalg.norm(diff[:, :2], axis=-1)
                df_dict['error_pos_z'] = np.abs(diff[:, 2])
            if 'vel' in self.ground_truth:
                df_dict['gt_vel_x'] = self.ground_truth['vel'][:, 0]
                df_dict['gt_vel_y'] = self.ground_truth['vel'][:, 1]
                df_dict['gt_vel_z'] = self.ground_truth['vel'][:, 2]

        return pd.DataFrame(df_dict)

    def plot_comparison(self, title: str = "Dead Reckoning vs Ground Truth", save_path: Optional[str] = None):
        """Generate multi-panel comparison visualization."""
        has_gt = self.ground_truth is not None and 'pos' in self.ground_truth

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # 1. 2D Trajectory (XY)
        ax1 = fig.add_subplot(2, 3, 1)
        if has_gt:
            ax1.plot(self.ground_truth['pos'][:, 0], self.ground_truth['pos'][:, 1],
                     label='Ground Truth', color='black', linewidth=2)
        ax1.plot(self.pos[:, 0], self.pos[:, 1],
                 label='Dead Reckoning', color='crimson', linestyle='--', linewidth=2)
        ax1.scatter(self.pos[0, 0], self.pos[0, 1], color='green', marker='o', s=60, label='Start')
        ax1.scatter(self.pos[-1, 0], self.pos[-1, 1], color='crimson', marker='x', s=60, label='DR End')
        if has_gt:
            ax1.scatter(self.ground_truth['pos'][-1, 0], self.ground_truth['pos'][-1, 1],
                        color='black', marker='x', s=60, label='GT End')
        ax1.set_title("2D Horizontal Trajectory (East-North)")
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='best', fontsize=8)
        ax1.axis('equal')

        # 2. Cumulative Drift Error vs Time
        ax2 = fig.add_subplot(2, 3, 2)
        if has_gt:
            diff = self.pos - self.ground_truth['pos']
            err_3d = np.linalg.norm(diff, axis=-1)
            err_xy = np.linalg.norm(diff[:, :2], axis=-1)
            err_z = np.abs(diff[:, 2])
            ax2.plot(self.timestamps, err_3d, label='3D Position Error', color='crimson', linewidth=2)
            ax2.plot(self.timestamps, err_xy, label='Horizontal (XY) Error', color='darkorange', linestyle='--')
            ax2.plot(self.timestamps, err_z, label='Vertical (Z) Error', color='blue', linestyle=':')
            ax2.set_title("Positional Drift Growth vs Time")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Error (m)")
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(loc='best', fontsize=8)
        else:
            ax2.plot(self.timestamps, self.pos[:, 2], label='Altitude Z', color='blue')
            ax2.set_title("Estimated Altitude (Z)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Z (m)")
            ax2.grid(True, linestyle=':', alpha=0.6)

        # 3. 3D Trajectory
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        if has_gt:
            ax3.plot(self.ground_truth['pos'][:, 0], self.ground_truth['pos'][:, 1], self.ground_truth['pos'][:, 2],
                     label='Ground Truth', color='black', linewidth=2)
        ax3.plot(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
                 label='Dead Reckoning', color='crimson', linestyle='--', linewidth=1.5)
        ax3.set_title("3D Trajectory")
        ax3.set_xlabel("East (m)")
        ax3.set_ylabel("North (m)")
        ax3.set_zlabel("Up (m)")
        ax3.legend(loc='best', fontsize=8)

        # 4. Velocities
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(self.timestamps, self.vel[:, 0], label='DR Vx', color='red', alpha=0.8)
        ax4.plot(self.timestamps, self.vel[:, 1], label='DR Vy', color='green', alpha=0.8)
        ax4.plot(self.timestamps, self.vel[:, 2], label='DR Vz', color='blue', alpha=0.8)
        if has_gt and 'vel' in self.ground_truth:
            ax4.plot(self.timestamps, self.ground_truth['vel'][:, 0], label='GT Vx', color='darkred', linestyle=':')
            ax4.plot(self.timestamps, self.ground_truth['vel'][:, 1], label='GT Vy', color='darkgreen', linestyle=':')
            ax4.plot(self.timestamps, self.ground_truth['vel'][:, 2], label='GT Vz', color='darkblue', linestyle=':')
        ax4.set_title("Velocity Components (m/s)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Velocity (m/s)")
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.legend(loc='best', fontsize=7)

        # 5. Euler Angles (Attitude)
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(self.timestamps, self.euler_deg[:, 0], label='Roll', color='magenta')
        ax5.plot(self.timestamps, self.euler_deg[:, 1], label='Pitch', color='teal')
        ax5.plot(self.timestamps, self.euler_deg[:, 2], label='Yaw', color='goldenrod')
        if has_gt and 'quat' in self.ground_truth:
            r_g, p_g, y_g = quat_to_euler(self.ground_truth['quat'])
            ax5.plot(self.timestamps, np.rad2deg(r_g), color='magenta', linestyle=':', label='GT Roll')
            ax5.plot(self.timestamps, np.rad2deg(p_g), color='teal', linestyle=':', label='GT Pitch')
            ax5.plot(self.timestamps, np.rad2deg(y_g), color='goldenrod', linestyle=':', label='GT Yaw')
        ax5.set_title("Attitude / Euler Angles (deg)")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angle (deg)")
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.legend(loc='best', fontsize=7)

        # 6. Performance Summary Text
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        if self.metrics is not None:
            text = self.metrics.summary_table()
            ax6.text(0.05, 0.95, text, transform=ax6.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#cccccc'))

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[NAVIS] Saved DR comparison plot to: {save_path}")
        return fig


class DeadReckoningEngine:
    """
    Standard Inertial Dead Reckoning Solver.
    
    Supports:
    - Attitude Integration: "quaternion_rk4", "quaternion_midpoint", "quaternion_exponential", "quaternion_euler", "euler_angles"
    - Translation Integration: "trapezoidal", "forward_euler", "velocity_verlet"
    - Static Leveling / Alignment Initialization
    - Pre-calibrated or estimated bias cancellation
    """

    def __init__(self,
                 attitude_method: str = "quaternion_rk4",
                 translation_method: str = "trapezoidal",
                 gravity: Union[np.ndarray, GravityModel] = STANDARD_GRAVITY_ENU,
                 initial_pos: Optional[np.ndarray] = None,
                 initial_vel: Optional[np.ndarray] = None,
                 initial_quat: Optional[np.ndarray] = None,
                 initial_euler: Optional[np.ndarray] = None,
                 acc_bias: Optional[np.ndarray] = None,
                 gyro_bias: Optional[np.ndarray] = None):
        self.attitude_method = attitude_method.lower()
        self.translation_method = translation_method.lower()

        if isinstance(gravity, GravityModel):
            self.gravity = gravity
        else:
            self.gravity = GravityModel(custom_g=gravity)

        self.acc_bias = np.asarray(acc_bias, dtype=np.float64) if acc_bias is not None else np.zeros(3)
        self.gyro_bias = np.asarray(gyro_bias, dtype=np.float64) if gyro_bias is not None else np.zeros(3)

        # Initial state setup
        self.initial_pos = np.asarray(initial_pos, dtype=np.float64) if initial_pos is not None else np.zeros(3)
        self.initial_vel = np.asarray(initial_vel, dtype=np.float64) if initial_vel is not None else np.zeros(3)

        if initial_quat is not None:
            self.initial_quat = quat_normalize(np.asarray(initial_quat, dtype=np.float64))
        elif initial_euler is not None:
            self.initial_quat = euler_to_quat(initial_euler[0], initial_euler[1], initial_euler[2])
        else:
            self.initial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Runtime state
        self.pos = self.initial_pos.copy()
        self.vel = self.initial_vel.copy()
        self.quat = self.initial_quat.copy()
        self.euler = np.array(quat_to_euler(self.quat), dtype=np.float64)
        self.prev_omega = np.zeros(3)
        self.prev_acc_world = np.zeros(3)
        self.is_initialized = False

    def reset(self,
              pos: Optional[np.ndarray] = None,
              vel: Optional[np.ndarray] = None,
              quat: Optional[np.ndarray] = None):
        """Reset internal navigation states."""
        self.pos = np.asarray(pos, dtype=np.float64) if pos is not None else self.initial_pos.copy()
        self.vel = np.asarray(vel, dtype=np.float64) if vel is not None else self.initial_vel.copy()
        self.quat = quat_normalize(np.asarray(quat, dtype=np.float64)) if quat is not None else self.initial_quat.copy()
        self.euler = np.array(quat_to_euler(self.quat), dtype=np.float64)
        self.prev_omega = np.zeros(3)
        self.prev_acc_world = np.zeros(3)
        self.is_initialized = False

    @staticmethod
    def compute_coarse_leveling(stationary_acc: np.ndarray,
                                stationary_mag: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate initial roll, pitch, and yaw from stationary accelerometer and magnetometer averages:
        f_b = -R^T * g = [-g*sin(theta), g*sin(phi)*cos(theta), g*cos(phi)*cos(theta)]^T
        roll = arctan2(f_y, f_z)
        pitch = arctan2(-f_x, sqrt(f_y^2 + f_z^2))
        """
        f = np.mean(np.asarray(stationary_acc, dtype=np.float64), axis=0)
        roll = np.arctan2(f[1], f[2])
        pitch = np.arctan2(-f[0], np.sqrt(f[1] ** 2 + f[2] ** 2))

        yaw = 0.0
        if stationary_mag is not None:
            m = np.mean(np.asarray(stationary_mag, dtype=np.float64), axis=0)
            # Tilt compensation on magnetometer
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            mx_h = m[0] * cp + m[1] * sr * sp + m[2] * cr * sp
            my_h = m[1] * cr - m[2] * sr
            yaw = np.arctan2(-my_h, mx_h)

        return euler_to_quat(roll, pitch, yaw)

    def step(self,
             acc_raw: np.ndarray,
             gyro_raw: np.ndarray,
             dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process single IMU measurement update step:
        Returns: (pos, vel, quat)
        """
        acc = np.asarray(acc_raw, dtype=np.float64) - self.acc_bias
        gyro = np.asarray(gyro_raw, dtype=np.float64) - self.gyro_bias

        if not self.is_initialized:
            self.prev_omega = gyro.copy()
            self.prev_acc_world = specific_force_to_world_acc(acc, self.quat, self.gravity)
            self.is_initialized = True

        # 1. Attitude Integration
        if self.attitude_method == "quaternion_rk4":
            self.quat = integrate_attitude_quaternion_rk4(self.quat, self.prev_omega, gyro, dt)
            self.euler = np.array(quat_to_euler(self.quat), dtype=np.float64)
        elif self.attitude_method == "quaternion_midpoint":
            self.quat = integrate_attitude_quaternion_midpoint(self.quat, self.prev_omega, gyro, dt)
            self.euler = np.array(quat_to_euler(self.quat), dtype=np.float64)
        elif self.attitude_method == "quaternion_exponential":
            self.quat = integrate_attitude_quaternion_exponential(self.quat, gyro, dt)
            self.euler = np.array(quat_to_euler(self.quat), dtype=np.float64)
        elif self.attitude_method == "euler_angles":
            self.euler = integrate_attitude_euler_angles(self.euler, gyro, dt)
            self.quat = euler_to_quat(self.euler[0], self.euler[1], self.euler[2])
        else:  # "quaternion_euler" default
            self.quat = integrate_attitude_quaternion_euler(self.quat, gyro, dt)
            self.euler = np.array(quat_to_euler(self.quat), dtype=np.float64)

        # 2. Specific Force Transformation to Navigation World Frame
        curr_acc_world = specific_force_to_world_acc(acc, self.quat, self.gravity)

        # 3. Translation Mechanization
        if self.translation_method == "forward_euler":
            self.pos, self.vel = integrate_translation_euler(self.pos, self.vel, self.prev_acc_world, dt)
        elif self.translation_method == "velocity_verlet":
            self.pos, self.vel = integrate_translation_verlet(self.pos, self.vel, self.prev_acc_world, curr_acc_world, dt)
        else:  # "trapezoidal" default
            self.pos, self.vel = integrate_translation_trapezoidal(self.pos, self.vel, self.prev_acc_world, curr_acc_world, dt)

        # Update history
        self.prev_omega = gyro.copy()
        self.prev_acc_world = curr_acc_world.copy()

        return self.pos.copy(), self.vel.copy(), self.quat.copy()

    def process_trajectory(self,
                           data: Union[TrajectoryData, pd.DataFrame, Dict[str, np.ndarray]],
                           auto_align: bool = False,
                           align_window_sec: float = 1.0) -> DeadReckoningResult:
        """
        Process entire trajectory sequence in batch mode.
        """
        # Parse inputs
        if isinstance(data, TrajectoryData):
            timestamps = data.timestamps
            acc = data.acc
            gyro = data.gyro
            mag = data.mag
            gt_pos = data.gt_pos
            gt_vel = data.gt_vel
            gt_quat = data.gt_quat
        elif isinstance(data, pd.DataFrame):
            timestamps = data['timestamp'].values
            acc = data[['acc_x', 'acc_y', 'acc_z']].values
            gyro = data[['gyro_x', 'gyro_y', 'gyro_z']].values
            mag = data[['mag_x', 'mag_y', 'mag_z']].values if 'mag_x' in data.columns else None
            gt_pos = data[['gt_pos_x', 'gt_pos_y', 'gt_pos_z']].values if 'gt_pos_x' in data.columns else None
            gt_vel = data[['gt_vel_x', 'gt_vel_y', 'gt_vel_z']].values if 'gt_vel_x' in data.columns else None
            gt_quat = data[['gt_quat_w', 'gt_quat_x', 'gt_quat_y', 'gt_quat_z']].values if 'gt_quat_w' in data.columns else None
        else:
            timestamps = np.asarray(data['timestamp'])
            acc = np.asarray(data['acc'])
            gyro = np.asarray(data['gyro'])
            mag = data.get('mag', None)
            gt_pos = data.get('gt_pos', None)
            gt_vel = data.get('gt_vel', None)
            gt_quat = data.get('gt_quat', None)

        N = len(timestamps)
        assert N >= 2, "Trajectory must contain at least 2 samples"

        # Initialize from ground truth or auto-leveling if specified
        if auto_align:
            dt = float(timestamps[1] - timestamps[0])
            num_samples = max(2, int(align_window_sec / dt))
            init_q = self.compute_coarse_leveling(acc[:num_samples], mag[:num_samples] if mag is not None else None)
            init_p = gt_pos[0] if gt_pos is not None else np.zeros(3)
            init_v = gt_vel[0] if gt_vel is not None else np.zeros(3)
            self.reset(pos=init_p, vel=init_v, quat=init_q)
        elif gt_pos is not None and gt_vel is not None and gt_quat is not None:
            self.reset(pos=gt_pos[0], vel=gt_vel[0], quat=gt_quat[0])
        else:
            self.reset()

        # Output buffers
        out_pos = np.zeros((N, 3), dtype=np.float64)
        out_vel = np.zeros((N, 3), dtype=np.float64)
        out_quat = np.zeros((N, 4), dtype=np.float64)
        out_euler = np.zeros((N, 3), dtype=np.float64)
        out_acc_w = np.zeros((N, 3), dtype=np.float64)

        # Store initial step
        out_pos[0] = self.pos.copy()
        out_vel[0] = self.vel.copy()
        out_quat[0] = self.quat.copy()
        r, p, y = quat_to_euler(self.quat)
        out_euler[0] = np.rad2deg([r, p, y])
        out_acc_w[0] = specific_force_to_world_acc(acc[0] - self.acc_bias, self.quat, self.gravity)

        # Integration loop
        for i in range(1, N):
            dt = float(timestamps[i] - timestamps[i - 1])
            if dt <= 0:
                dt = 0.01

            p_i, v_i, q_i = self.step(acc[i], gyro[i], dt)

            out_pos[i] = p_i
            out_vel[i] = v_i
            out_quat[i] = q_i
            r, p, y = quat_to_euler(q_i)
            out_euler[i] = np.rad2deg([r, p, y])
            out_acc_w[i] = self.prev_acc_world.copy()

        # Evaluation against GT
        gt_dict = None
        metrics = None
        if gt_pos is not None:
            gt_dict = {'pos': gt_pos}
            if gt_vel is not None:
                gt_dict['vel'] = gt_vel
            if gt_quat is not None:
                gt_dict['quat'] = gt_quat

            metrics = evaluate_trajectory(
                est_pos=out_pos,
                gt_pos=gt_pos,
                timestamps=timestamps,
                est_vel=out_vel if gt_vel is not None else None,
                gt_vel=gt_vel,
                est_quat=out_quat if gt_quat is not None else None,
                gt_quat=gt_quat,
            )

        return DeadReckoningResult(
            timestamps=timestamps,
            pos=out_pos,
            vel=out_vel,
            quat=out_quat,
            euler_deg=out_euler,
            acc_world=out_acc_w,
            metrics=metrics,
            ground_truth=gt_dict,
        )


def run_dead_reckoning(dataset_path: Optional[str] = None,
                       profile: str = "urban_driving",
                       duration: float = 60.0,
                       imu_grade: str = "industrial",
                       attitude_method: str = "quaternion_rk4",
                       translation_method: str = "trapezoidal",
                       plot: bool = True,
                       save_dir: str = "data/processed") -> DeadReckoningResult:
    """
    Convenience pipeline to execute Dead Reckoning baseline and print metrics.
    """
    from src.data.loaders import SyntheticDataLoader, CSVTrajectoryLoader

    if dataset_path and os.path.exists(dataset_path):
        print(f"[NAVIS] Loading trajectory from: {dataset_path}")
        traj = CSVTrajectoryLoader.load(dataset_path)
    else:
        print(f"[NAVIS] Generating synthetic trajectory (Profile: {profile}, Grade: {imu_grade}, Duration: {duration}s)...")
        traj = SyntheticDataLoader.create(profile=profile, duration=duration, imu_grade=imu_grade)

    engine = DeadReckoningEngine(
        attitude_method=attitude_method,
        translation_method=translation_method,
    )

    result = engine.process_trajectory(traj)

    if result.metrics:
        print(result.metrics.summary_table())

    if plot:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"dr_benchmark_{profile}_{imu_grade}.png")
        result.plot_comparison(
            title=f"NAVIS Classical Dead Reckoning ({profile.title()}, {imu_grade.title()} IMU)",
            save_path=plot_path
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAVIS Classical Dead Reckoning Baseline Runner")
    parser.add_argument("--dataset", type=str, default=None, help="Path to input CSV dataset")
    parser.add_argument("--profile", type=str, default="urban_driving", help="Motion profile for synthetic run")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--imu-grade", type=str, default="industrial", choices=["consumer", "industrial", "tactical", "space"])
    parser.add_argument("--attitude-method", type=str, default="quaternion_rk4",
                        choices=["quaternion_rk4", "quaternion_midpoint", "quaternion_exponential", "quaternion_euler", "euler_angles"])
    parser.add_argument("--translation-method", type=str, default="trapezoidal",
                        choices=["trapezoidal", "forward_euler", "velocity_verlet"])
    parser.add_argument("--plot", action="store_true", default=True, help="Save evaluation plot")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()

    run_dead_reckoning(
        dataset_path=args.dataset,
        profile=args.profile,
        duration=args.duration,
        imu_grade=args.imu_grade,
        attitude_method=args.attitude_method,
        translation_method=args.translation_method,
        plot=args.plot,
        save_dir=args.output_dir,
    )
