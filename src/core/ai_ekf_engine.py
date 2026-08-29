"""
AI-Enhanced Extended Kalman Filter (AI-EKF) Navigation Engine

Integrates all NAVIS capabilities into an industrial-grade multi-sensor navigation pipeline:
- Module 1: Multi-Sensor Data Stream (IMU, GPS, Baro, Mag, Odo)
- Module 2: Strapdown Inertial Kinematics & Error Dynamics
- Module 3: AI-Assisted Zero-Velocity Updates (AI-ZUPT) & Motion Context Classification
- Module 4: Deep Neural Drift & Sensor Bias Prediction with Aleatoric Uncertainty
- Module 5: 15-State Error-State EKF + Seamless GPS Handoff & Outlier Rejection

Provides:
- Real-time continuous estimation during seamless transitions between GPS-available
  and GPS-denied environments.
- Dynamic covariance bounds (+- 3 sigma uncertainty envelopes).
- Quantitative drift reduction benchmarking against uncorrected DR and standard EKF.
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.dataset import TrajectoryData
from src.core.ekf import ErrorStateEKF, EKFConfig, EKFState
from src.core.handoff_controller import GPSHandoffController, HandoffConfig, HandoffState, HandoffEvent
from src.models.motion_classifier import MotionClassifier, MotionContext
from src.models.zupt_detector import ZUPTDetector, GLRTDetector
from src.models.drift_estimator import InertialDriftEstimator
from src.utils.coordinate_transforms import quat_to_euler, quat_normalize
from src.utils.metrics import evaluate_trajectory, TrajectoryMetrics


@dataclass
class AIEKFResult:
    """Evaluation result and estimated state trajectory from AI-EKF Engine."""
    timestamps: np.ndarray                      # (N,) [s]
    pos: np.ndarray                             # (N, 3) [m] World ENU
    vel: np.ndarray                             # (N, 3) [m/s] World ENU
    quat: np.ndarray                            # (N, 4) [w, x, y, z]
    euler_deg: np.ndarray                       # (N, 3) [roll, pitch, yaw] in degrees
    acc_bias: np.ndarray                        # (N, 3) [m/s^2]
    gyro_bias: np.ndarray                       # (N, 3) [rad/s]
    pos_std_3sigma: np.ndarray                  # (N, 3) [m] 3-sigma position uncertainty
    vel_std_3sigma: np.ndarray                  # (N, 3) [m/s] 3-sigma velocity uncertainty
    handoff_states: List[HandoffState]          # (N,) Handoff operational state
    stance_flags: np.ndarray                    # (N,) Stance indicators
    motion_contexts: List[str]                  # (N,) Motion context labels
    handoff_events: List[HandoffEvent]          # List of transition events
    metrics: Optional[TrajectoryMetrics] = None
    ground_truth: Optional[Dict[str, np.ndarray]] = None

    def summary_table(self) -> str:
        """Generate formatted performance summary table."""
        ate = self.metrics.ate_rmse if self.metrics else 0.0
        final_err = self.metrics.final_pos_error if self.metrics else 0.0
        drift_pct = self.metrics.drift_percentage if self.metrics else 0.0
        drift_rate = self.metrics.drift_rate_mps if self.metrics else 0.0

        outages = [e for e in self.handoff_events if e.event_type == 'outage_start']
        rejections = [e for e in self.handoff_events if e.event_type == 'spoofing_detected']

        table = []
        table.append("\n" + "=" * 76)
        table.append("  NAVIS Module 5: AI-Enhanced Extended Kalman Filter Performance Report")
        table.append("=" * 76)
        table.append(f"{'Performance Metric':<40} | {'AI-EKF System Performance':<30}")
        table.append("-" * 76)
        table.append(f"{'Absolute Trajectory Error (ATE RMSE) [m]':<40} | {ate:<30.4f}")
        table.append(f"{'Final Position Error [m]':<40} | {final_err:<30.4f}")
        table.append(f"{'Cumulative Drift Rate [% distance]':<40} | {drift_pct:<30.4f}")
        table.append(f"{'Drift Speed [m/s]':<40} | {drift_rate:<30.4f}")
        table.append("-" * 76)
        table.append(f"{'Total GPS Outage Events':<40} | {len(outages):<30}")
        table.append(f"{'Spoofing / Outlier Rejections':<40} | {len(rejections):<30}")
        table.append("=" * 76 + "\n")
        return "\n".join(table)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert AI-EKF results to pandas DataFrame."""
        df_dict = {
            'timestamp': self.timestamps,
            'ekf_pos_x': self.pos[:, 0],
            'ekf_pos_y': self.pos[:, 1],
            'ekf_pos_z': self.pos[:, 2],
            'ekf_vel_x': self.vel[:, 0],
            'ekf_vel_y': self.vel[:, 1],
            'ekf_vel_z': self.vel[:, 2],
            'ekf_roll_deg': self.euler_deg[:, 0],
            'ekf_pitch_deg': self.euler_deg[:, 1],
            'ekf_yaw_deg': self.euler_deg[:, 2],
            'ekf_acc_bias_x': self.acc_bias[:, 0],
            'ekf_acc_bias_y': self.acc_bias[:, 1],
            'ekf_acc_bias_z': self.acc_bias[:, 2],
            'ekf_gyro_bias_x': self.gyro_bias[:, 0],
            'ekf_gyro_bias_y': self.gyro_bias[:, 1],
            'ekf_gyro_bias_z': self.gyro_bias[:, 2],
            'pos_3sigma_x': self.pos_std_3sigma[:, 0],
            'pos_3sigma_y': self.pos_std_3sigma[:, 1],
            'pos_3sigma_z': self.pos_std_3sigma[:, 2],
            'handoff_state': [s.value for s in self.handoff_states],
            'stance_flag': self.stance_flags,
            'motion_context': self.motion_contexts,
        }
        if self.ground_truth is not None and 'pos' in self.ground_truth:
            gt_p = self.ground_truth['pos']
            df_dict['gt_pos_x'] = gt_p[:, 0]
            df_dict['gt_pos_y'] = gt_p[:, 1]
            df_dict['gt_pos_z'] = gt_p[:, 2]

        return pd.DataFrame(df_dict)

    def plot_comparison(self, save_path: Optional[str] = None):
        """Generate comprehensive 4-panel AI-EKF navigation diagnostic plot."""
        fig = plt.figure(figsize=(16, 10))

        # 1. 2D Horizontal Trajectory with GPS Outage Regions
        ax1 = fig.add_subplot(2, 2, 1)
        if self.ground_truth is not None and 'pos' in self.ground_truth:
            gt = self.ground_truth['pos']
            ax1.plot(gt[:, 0], gt[:, 1], 'k--', linewidth=2.0, label='Ground Truth')

        # Color-coded trajectory based on GPS state
        locked_mask = np.array([s == HandoffState.GPS_LOCKED for s in self.handoff_states])
        denied_mask = ~locked_mask

        ax1.plot(self.pos[:, 0], self.pos[:, 1], 'b-', linewidth=2.2, label='AI-EKF Trajectory')
        if np.any(denied_mask):
            ax1.scatter(self.pos[denied_mask, 0], self.pos[denied_mask, 1],
                        color='crimson', s=10, alpha=0.6, label='GPS-Denied / AI-DR Zone')

        ax1.set_xlabel("East [m]")
        ax1.set_ylabel("North [m]")
        ax1.set_title("2D Horizontal Trajectory with GPS Denial Zones")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')

        # 2. Position Error vs Time with 3-Sigma Uncertainty Envelope
        ax2 = fig.add_subplot(2, 2, 2)
        if self.ground_truth is not None and 'pos' in self.ground_truth:
            gt = self.ground_truth['pos']
            pos_err = np.linalg.norm(self.pos - gt, axis=1)
            ax2.plot(self.timestamps, pos_err, 'b-', linewidth=2.0, label=f'Position Error (ATE: {self.metrics.ate_rmse:.2f}m)')
            sigma3 = np.linalg.norm(self.pos_std_3sigma, axis=1)
            ax2.fill_between(self.timestamps, 0, sigma3, color='blue', alpha=0.15, label=r'$\pm 3\sigma$ EKF Bounds')

        # Shade GPS outage periods
        in_outage = False
        outage_start = 0.0
        for i, s in enumerate(self.handoff_states):
            if s in (HandoffState.GPS_DENIED, HandoffState.ENTERING_OUTAGE) and not in_outage:
                in_outage = True
                outage_start = self.timestamps[i]
            elif s in (HandoffState.GPS_LOCKED, HandoffState.RECOVERING_GPS) and in_outage:
                in_outage = False
                ax2.axvspan(outage_start, self.timestamps[i], color='red', alpha=0.18, label='GPS Outage Window' if i < 100 else "")

        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Error [m]")
        ax2.set_title("Position Error & Filter Consistency")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # 3. Estimated Accelerometer Dynamic Biases
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(self.timestamps, self.acc_bias[:, 0], label='b_ax (East)')
        ax3.plot(self.timestamps, self.acc_bias[:, 1], label='b_ay (North)')
        ax3.plot(self.timestamps, self.acc_bias[:, 2], label='b_az (Up)')
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Bias [m/s²]")
        ax3.set_title("Online Accelerometer Bias Estimation")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Handoff States & Stance Activity
        ax4 = fig.add_subplot(2, 2, 4)
        state_nums = [0 if s == HandoffState.GPS_LOCKED else 1 if s == HandoffState.RECOVERING_GPS else 2 for s in self.handoff_states]
        ax4.step(self.timestamps, state_nums, 'g-', where='post', label='Handoff State (0: Locked, 1: Recov, 2: Denied)')
        ax4.fill_between(self.timestamps, 0, self.stance_flags, color='purple', alpha=0.3, label='Stance / ZUPT Active')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['GPS Locked', 'Recovering', 'GPS Denied'])
        ax4.set_xlabel("Time [s]")
        ax4.set_title("GPS Handoff State Machine & AI-ZUPT Triggers")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Saved AI-EKF comparison plot to: {save_path}")
        plt.close(fig)


class AIEKFNavigationEngine:
    """
    High-Level AI-Enhanced Navigation Engine.
    Executes end-to-end integration across all sensors and AI modules.
    """

    def __init__(self,
                 ekf_config: Optional[EKFConfig] = None,
                 handoff_config: Optional[HandoffConfig] = None,
                 drift_estimator: Optional[InertialDriftEstimator] = None,
                 motion_classifier: Optional[MotionClassifier] = None,
                 zupt_detector: Optional[ZUPTDetector] = None):
        self.ekf_config = ekf_config or EKFConfig()
        self.handoff_config = handoff_config or HandoffConfig()

        self.drift_estimator = drift_estimator or InertialDriftEstimator(model_type="numpy")
        self.motion_classifier = motion_classifier or MotionClassifier()
        self.zupt_detector = zupt_detector or GLRTDetector()

    def process_trajectory(self,
                           trajectory: TrajectoryData,
                           initial_pos: Optional[np.ndarray] = None,
                           initial_vel: Optional[np.ndarray] = None,
                           initial_quat: Optional[np.ndarray] = None) -> AIEKFResult:
        """
        Execute full AI-EKF navigation pipeline over an IMU and auxiliary sensor dataset.
        """
        t = trajectory.timestamps
        N = len(t)
        raw_acc = trajectory.acc
        raw_gyro = trajectory.gyro
        imu_combined = np.hstack([raw_acc, raw_gyro])

        # Step 1: Precompute AI models (Motion Context, Neural Drift, ZUPT)
        ai_preds = self.drift_estimator.predict_sequence(imu_combined, step_size=1)
        ai_vel = ai_preds['vel']
        ai_vel_std = ai_preds['vel_std']

        # Classify motion context (windowed)
        motion_res = self.motion_classifier.predict_sequence(raw_acc, raw_gyro, use_neural=False)
        contexts = motion_res['contexts']
        context_labels = [MotionContext(int(c)).label_name for c in contexts]

        # Detect stance intervals (Module 3 GLRT/ARE)
        stance_flags = self.zupt_detector.detect(raw_acc, raw_gyro)

        # Step 2: Initialize EKF & Handoff Controller
        p0 = initial_pos if initial_pos is not None else trajectory.gt_pos[0] if trajectory.gt_pos is not None else np.zeros(3)
        v0 = initial_vel if initial_vel is not None else trajectory.gt_vel[0] if trajectory.gt_vel is not None else np.zeros(3)
        q0 = initial_quat if initial_quat is not None else trajectory.gt_quat[0] if trajectory.gt_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])

        ekf = ErrorStateEKF(
            config=self.ekf_config,
            initial_pos=p0,
            initial_vel=v0,
            initial_quat=q0,
        )
        handoff = GPSHandoffController(ekf=ekf, config=self.handoff_config)

        # Preallocate Result Arrays
        pos = np.zeros((N, 3), dtype=np.float64)
        vel = np.zeros((N, 3), dtype=np.float64)
        quat = np.zeros((N, 4), dtype=np.float64)
        euler_deg = np.zeros((N, 3), dtype=np.float64)
        acc_bias = np.zeros((N, 3), dtype=np.float64)
        gyro_bias = np.zeros((N, 3), dtype=np.float64)
        pos_3sigma = np.zeros((N, 3), dtype=np.float64)
        vel_3sigma = np.zeros((N, 3), dtype=np.float64)
        handoff_states: List[HandoffState] = []

        # Record Initial State
        pos[0] = ekf.pos
        vel[0] = ekf.vel
        quat[0] = ekf.quat
        euler_deg[0] = ekf.state.euler_deg
        acc_bias[0] = ekf.acc_bias
        gyro_bias[0] = ekf.gyro_bias
        pos_3sigma[0] = 3.0 * ekf.state.pos_std
        vel_3sigma[0] = 3.0 * ekf.state.vel_std
        handoff_states.append(handoff.state)

        # Step 3: Sequential Time-Marching Loop
        for k in range(1, N):
            dt_k = t[k] - t[k - 1]
            if dt_k <= 0:
                dt_k = 0.01

            # 3.1 Adaptive Process Noise Scale from AI Context & Neural Uncertainty
            curr_context = context_labels[k]
            vel_unc = float(np.mean(ai_vel_std[k]))

            # Dynamic context scale: dynamic maneuvers (aerial/rover) warrant higher Q
            if curr_context in ("aerial", "planetary_rover"):
                adaptive_scale = 1.8 + vel_unc * 2.0
            elif curr_context == "stationary":
                adaptive_scale = 0.4
            else:
                adaptive_scale = 1.0 + vel_unc

            # 3.2 Time Propagation Step
            ekf.predict(
                acc_meas=raw_acc[k],
                gyro_meas=raw_gyro[k],
                dt=dt_k,
                timestamp=t[k],
                adaptive_scale=adaptive_scale
            )

            # 3.3 Sensor Measurement Updates
            # A. Stance / Zero-Velocity Update (ZUPT)
            is_stance = bool(stance_flags[k])
            handoff.process_zupt_update(is_stance=is_stance)

            # B. GPS Position & Velocity Update
            has_gps = (trajectory.gps_valid is not None and bool(trajectory.gps_valid[k]))
            gps_pos = trajectory.gps_pos[k] if (trajectory.gps_pos is not None and has_gps) else None
            gps_vel = trajectory.gps_vel[k] if (trajectory.gps_vel is not None and has_gps) else None

            handoff.process_gps_update(
                timestamp=t[k],
                pos_meas=gps_pos,
                vel_meas=gps_vel,
                is_valid=has_gps
            )

            # C. Neural Velocity Update (when GPS is denied or recovering)
            if not handoff.is_gps_available:
                # Fuse Module 4 Deep Velocity prediction
                v_ai_k = ai_vel[k]
                v_ai_cov = np.diag(ai_vel_std[k] ** 2)
                handoff.process_neural_velocity_update(v_neural=v_ai_k, v_neural_cov=v_ai_cov)

            # D. Barometer Altitude Update (if available)
            if trajectory.baro_alt is not None:
                handoff.process_barometer_update(alt_meas=float(trajectory.baro_alt[k]))

            # E. Non-Holonomic Constraints (if vehicle)
            if curr_context in ("vehicular", "planetary_rover") and not is_stance:
                handoff.process_nhc_update(is_wheeled_vehicle=True)

            # Store Estimated States
            pos[k] = ekf.pos
            vel[k] = ekf.vel
            quat[k] = ekf.quat
            euler_deg[k] = ekf.state.euler_deg
            acc_bias[k] = ekf.acc_bias
            gyro_bias[k] = ekf.gyro_bias
            pos_3sigma[k] = 3.0 * ekf.state.pos_std
            vel_3sigma[k] = 3.0 * ekf.state.vel_std
            handoff_states.append(handoff.state)

        # Step 4: Quantitative Evaluation vs Ground Truth
        metrics = None
        gt_dict = None
        if trajectory.gt_pos is not None:
            gt_dict = {
                'pos': trajectory.gt_pos,
                'vel': trajectory.gt_vel if trajectory.gt_vel is not None else np.zeros_like(pos),
                'quat': trajectory.gt_quat if trajectory.gt_quat is not None else np.zeros((N, 4)),
            }
            metrics = evaluate_trajectory(
                est_pos=pos,
                gt_pos=trajectory.gt_pos,
                timestamps=t,
                est_vel=vel,
                gt_vel=trajectory.gt_vel,
                est_quat=quat,
                gt_quat=trajectory.gt_quat,
            )

        return AIEKFResult(
            timestamps=t,
            pos=pos,
            vel=vel,
            quat=quat,
            euler_deg=euler_deg,
            acc_bias=acc_bias,
            gyro_bias=gyro_bias,
            pos_std_3sigma=pos_3sigma,
            vel_std_3sigma=vel_3sigma,
            handoff_states=handoff_states,
            stance_flags=stance_flags,
            motion_contexts=context_labels,
            handoff_events=handoff.events,
            metrics=metrics,
            ground_truth=gt_dict,
        )


def run_ai_ekf_navigation(
    trajectory: TrajectoryData,
    ekf_config: Optional[EKFConfig] = None,
    handoff_config: Optional[HandoffConfig] = None,
) -> AIEKFResult:
    """Convenience function to run AI-EKF Navigation Engine on a TrajectoryData sequence."""
    engine = AIEKFNavigationEngine(ekf_config=ekf_config, handoff_config=handoff_config)
    return engine.process_trajectory(trajectory)


# ============================================================================
# 4. CLI BENCHMARK ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run NAVIS Module 5 AI-EKF Navigation Benchmark.")
    parser.add_argument("--profile", type=str, default="urban_driving", help="Motion trajectory profile.")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration [s].")
    parser.add_argument("--gps-outage", nargs=2, type=float, default=[20.0, 45.0],
                        metavar=("START", "END"), help="Simulated GPS outage window [s].")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots.")
    parser.add_argument("--save-plot", type=str, default=None, help="Save path for diagnostic plot.")
    args = parser.parse_args()

    from src.data.loaders import SyntheticDataLoader

    outage_windows = [(args.gps_outage[0], args.gps_outage[1])] if args.gps_outage else None
    print(f"\n[*] Generating '{args.profile.upper()}' Trajectory with GPS Outage: {outage_windows}...")
    traj = SyntheticDataLoader.create(
        profile=args.profile,
        duration=args.duration,
        dt=0.01,
        imu_grade="consumer",
        gps_outages=outage_windows,
        seed=42,
    )

    print("[*] Running AI-Enhanced Extended Kalman Filter (AI-EKF)...")
    result = run_ai_ekf_navigation(traj)

    print(result.summary_table())

    if args.save_plot or args.plot:
        out_path = args.save_plot or f"plots/ai_ekf_{args.profile}_outage.png"
        result.plot_comparison(save_path=out_path)


if __name__ == "__main__":
    main()
