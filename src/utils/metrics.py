"""
Navigation Accuracy and Dead Reckoning Error Metrics

Comprehensive performance metrics for evaluating inertial navigation systems:
1. ATE (Absolute Trajectory Error) - RMSE, Mean, Median, Std, Min, Max
2. RPE (Relative Pose / Trajectory Error) - drift rate, drift per km, % drift
3. Attitude & Orientation Error - Geodesic quaternion angle, Euler angle errors
4. Velocity Error - 3D, Horizontal, Vertical
5. Cumulative Drift Dynamics - Instantaneous error growth curve, endpoint error
6. Rigid Body Trajectory Alignment - Umeyama / Kabsch SE(3) SVD alignment
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


@dataclass
class TrajectoryMetrics:
    """Standardized evaluation summary for navigation & dead reckoning pipelines."""
    # Absolute Position Error (m)
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_std: float
    ate_max: float
    ate_horizontal_rmse: float
    ate_vertical_rmse: float
    final_pos_error: float

    # Relative & Drift Metrics
    total_distance_traveled: float
    drift_percentage: float          # (Final Error / Total Distance) * 100%
    drift_rate_mps: float            # Final Error / Duration (m/s)
    drift_per_km: float              # Final Error / (Total Distance / 1000) (m/km)

    # Velocity Error (m/s)
    vel_rmse: Optional[float] = None
    vel_horizontal_rmse: Optional[float] = None
    vel_vertical_rmse: Optional[float] = None

    # Attitude Error (degrees)
    attitude_angle_rmse_deg: Optional[float] = None
    attitude_angle_max_deg: Optional[float] = None
    roll_rmse_deg: Optional[float] = None
    pitch_rmse_deg: Optional[float] = None
    yaw_rmse_deg: Optional[float] = None

    # Duration
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "ATE_RMSE_m": self.ate_rmse,
            "ATE_Mean_m": self.ate_mean,
            "ATE_Median_m": self.ate_median,
            "ATE_Std_m": self.ate_std,
            "ATE_Max_m": self.ate_max,
            "ATE_Horizontal_RMSE_m": self.ate_horizontal_rmse,
            "ATE_Vertical_RMSE_m": self.ate_vertical_rmse,
            "Final_Pos_Error_m": self.final_pos_error,
            "Total_Distance_m": self.total_distance_traveled,
            "Drift_Percentage_%": self.drift_percentage,
            "Drift_Rate_m_per_s": self.drift_rate_mps,
            "Drift_per_km_m": self.drift_per_km,
            "Vel_RMSE_mps": self.vel_rmse,
            "Attitude_RMSE_deg": self.attitude_angle_rmse_deg,
            "Yaw_RMSE_deg": self.yaw_rmse_deg,
            "Duration_s": self.duration_sec,
        }

    def summary_table(self) -> str:
        """Format metrics as a clean text summary table."""
        lines = [
            "=" * 60,
            f"{'NAVIS DEAD RECKONING BENCHMARK REPORT':^60}",
            "=" * 60,
            f"{'Metric':<35} | {'Value':<20}",
            "-" * 60,
            f"{'Duration (s)':<35} | {self.duration_sec:<20.2f}",
            f"{'Total Distance Traveled (m)':<35} | {self.total_distance_traveled:<20.2f}",
            "-" * 60,
            f"{'ATE 3D RMSE (m)':<35} | {self.ate_rmse:<20.4f}",
            f"{'ATE Mean Error (m)':<35} | {self.ate_mean:<20.4f}",
            f"{'ATE Median Error (m)':<35} | {self.ate_median:<20.4f}",
            f"{'ATE Max Error (m)':<35} | {self.ate_max:<20.4f}",
            f"{'ATE Horizontal (XY) RMSE (m)':<35} | {self.ate_horizontal_rmse:<20.4f}",
            f"{'ATE Vertical (Z) RMSE (m)':<35} | {self.ate_vertical_rmse:<20.4f}",
            f"{'Final Endpoint Error (m)':<35} | {self.final_pos_error:<20.4f}",
            "-" * 60,
            f"{'Cumulative Drift (% of distance)':<35} | {self.drift_percentage:<20.2f} %",
            f"{'Drift per km (m/km)':<35} | {self.drift_per_km:<20.2f}",
            f"{'Drift Rate (m/s)':<35} | {self.drift_rate_mps:<20.4f}",
        ]
        if self.vel_rmse is not None:
            lines.append(f"{'Velocity 3D RMSE (m/s)':<35} | {self.vel_rmse:<20.4f}")
        if self.attitude_angle_rmse_deg is not None:
            lines.append(f"{'Attitude Geodesic RMSE (deg)':<35} | {self.attitude_angle_rmse_deg:<20.4f}")
        if self.yaw_rmse_deg is not None:
            lines.append(f"{'Heading / Yaw RMSE (deg)':<35} | {self.yaw_rmse_deg:<20.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_ate(est_pos: np.ndarray, gt_pos: np.ndarray) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE) statistics between estimated and ground truth positions.
    
    Args:
        est_pos: (N, 3) Estimated 3D positions [m]
        gt_pos: (N, 3) Ground truth 3D positions [m]
        
    Returns:
        Dictionary of ATE metrics (RMSE, Mean, Median, Std, Max, Min, Horizontal RMSE, Vertical RMSE)
    """
    est_pos = np.asarray(est_pos, dtype=np.float64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    assert est_pos.shape == gt_pos.shape, f"Shape mismatch: {est_pos.shape} vs {gt_pos.shape}"

    # 3D Euclidean error at each step
    diff = est_pos - gt_pos
    error_3d = np.linalg.norm(diff, axis=-1)  # (N,)

    # Horizontal (XY) and Vertical (Z) errors
    error_xy = np.linalg.norm(diff[:, :2], axis=-1)
    error_z = np.abs(diff[:, 2])

    return {
        "rmse": float(np.sqrt(np.mean(error_3d ** 2))),
        "mean": float(np.mean(error_3d)),
        "median": float(np.median(error_3d)),
        "std": float(np.std(error_3d)),
        "max": float(np.max(error_3d)),
        "min": float(np.min(error_3d)),
        "horizontal_rmse": float(np.sqrt(np.mean(error_xy ** 2))),
        "vertical_rmse": float(np.sqrt(np.mean(error_z ** 2))),
        "final_error": float(error_3d[-1]) if len(error_3d) > 0 else 0.0,
    }


def compute_rpe(est_pos: np.ndarray,
                gt_pos: np.ndarray,
                delta_samples: int = 100) -> Dict[str, float]:
    """
    Compute Relative Pose Error (RPE) over fixed sample intervals.
    Measures local drift per step delta.
    
    Args:
        est_pos: (N, 3) Estimated 3D positions
        gt_pos: (N, 3) Ground truth 3D positions
        delta_samples: Window length in samples
        
    Returns:
        Dictionary of RPE statistics
    """
    est_pos = np.asarray(est_pos, dtype=np.float64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    N = len(est_pos)
    if N <= delta_samples:
        return {"rpe_rmse": 0.0, "rpe_mean": 0.0, "rpe_max": 0.0}

    # Relative displacements
    rel_est = est_pos[delta_samples:] - est_pos[:-delta_samples]
    rel_gt = gt_pos[delta_samples:] - gt_pos[:-delta_samples]
    rel_diff = np.linalg.norm(rel_est - rel_gt, axis=-1)

    return {
        "rpe_rmse": float(np.sqrt(np.mean(rel_diff ** 2))),
        "rpe_mean": float(np.mean(rel_diff)),
        "rpe_max": float(np.max(rel_diff)),
        "delta_samples": delta_samples,
    }


def compute_drift_metrics(est_pos: np.ndarray,
                          gt_pos: np.ndarray,
                          timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute cumulative drift dynamics, drift rate, and drift percentage relative to distance traveled.
    """
    est_pos = np.asarray(est_pos, dtype=np.float64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    N = len(est_pos)

    # Total ground truth path length
    if N > 1:
        step_dists = np.linalg.norm(np.diff(gt_pos, axis=0), axis=-1)
        total_dist = float(np.sum(step_dists))
    else:
        total_dist = 0.0

    final_error = float(np.linalg.norm(est_pos[-1] - gt_pos[-1])) if N > 0 else 0.0

    # Drift as percentage of total distance traveled
    drift_percent = (final_error / total_dist * 100.0) if total_dist > 1e-3 else 0.0
    drift_per_km = (final_error / (total_dist / 1000.0)) if total_dist > 1e-3 else 0.0

    # Drift rate over time
    if timestamps is not None and len(timestamps) > 1:
        duration = float(timestamps[-1] - timestamps[0])
        drift_rate_mps = (final_error / duration) if duration > 1e-6 else 0.0
    else:
        duration = 0.0
        drift_rate_mps = 0.0

    return {
        "total_distance": total_dist,
        "final_error": final_error,
        "drift_percentage": float(drift_percent),
        "drift_per_km": float(drift_per_km),
        "drift_rate_mps": float(drift_rate_mps),
        "duration": duration,
    }


def compute_attitude_error(est_quat: np.ndarray, gt_quat: np.ndarray) -> Dict[str, float]:
    """
    Compute attitude orientation error via geodesic distance on SO(3) and Euler angle errors.
    Geodesic distance between unit quaternions q1 and q2:
    theta = 2 * arccos(|<q1, q2>|)
    """
    from src.utils.coordinate_transforms import quat_normalize, quat_to_euler

    q_est = quat_normalize(np.asarray(est_quat, dtype=np.float64))
    q_gt = quat_normalize(np.asarray(gt_quat, dtype=np.float64))
    assert q_est.shape == q_gt.shape, "Quaternion shapes must match"

    # Inner product (dot product across 4 components)
    dot = np.sum(q_est * q_gt, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle_error_rad = 2.0 * np.arccos(dot)
    angle_error_deg = np.rad2deg(angle_error_rad)

    # Euler angle breakdown
    roll_e, pitch_e, yaw_e = quat_to_euler(q_est)
    roll_g, pitch_g, yaw_g = quat_to_euler(q_gt)

    # Wrap difference to [-pi, pi]
    def wrap_pi(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    d_roll = np.rad2deg(wrap_pi(roll_e - roll_g))
    d_pitch = np.rad2deg(wrap_pi(pitch_e - pitch_g))
    d_yaw = np.rad2deg(wrap_pi(yaw_e - yaw_g))

    return {
        "geodesic_rmse_deg": float(np.sqrt(np.mean(angle_error_deg ** 2))),
        "geodesic_mean_deg": float(np.mean(angle_error_deg)),
        "geodesic_max_deg": float(np.max(angle_error_deg)),
        "roll_rmse_deg": float(np.sqrt(np.mean(d_roll ** 2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(d_pitch ** 2))),
        "yaw_rmse_deg": float(np.sqrt(np.mean(d_yaw ** 2))),
    }


def compute_velocity_error(est_vel: np.ndarray, gt_vel: np.ndarray) -> Dict[str, float]:
    """Compute 3D, Horizontal, and Vertical velocity RMSE."""
    est_vel = np.asarray(est_vel, dtype=np.float64)
    gt_vel = np.asarray(gt_vel, dtype=np.float64)
    diff = est_vel - gt_vel

    diff_3d = np.linalg.norm(diff, axis=-1)
    diff_xy = np.linalg.norm(diff[:, :2], axis=-1)
    diff_z = np.abs(diff[:, 2])

    return {
        "vel_rmse": float(np.sqrt(np.mean(diff_3d ** 2))),
        "vel_mean": float(np.mean(diff_3d)),
        "vel_max": float(np.max(diff_3d)),
        "vel_horizontal_rmse": float(np.sqrt(np.mean(diff_xy ** 2))),
        "vel_vertical_rmse": float(np.sqrt(np.mean(diff_z ** 2))),
    }


def align_trajectories_umeyama(model: np.ndarray,
                              data: np.ndarray,
                              with_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Umeyama / Kabsch rigid body trajectory alignment (SE(3) or Sim(3)).
    Finds optimal Rotation R, Translation t (and optional scale c) such that:
    data_aligned = c * R * model + t minimizes sum of squared errors to data.
    
    Args:
        model: (N, 3) Source trajectory (e.g. estimated)
        data: (N, 3) Target trajectory (e.g. ground truth)
        with_scale: Whether to estimate similarity scale factor
        
    Returns:
        (R, t, scale, model_aligned)
    """
    model = np.asarray(model, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    assert model.shape == data.shape, "Trajectories must have identical shape"
    N, m = model.shape

    # Centroids
    mu_model = np.mean(model, axis=0)
    mu_data = np.mean(data, axis=0)

    # Centered coords
    model_c = model - mu_model
    data_c = data - mu_data

    # Covariance matrix
    H = model_c.T @ data_c / N

    # SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Rotation matrix
    d = np.linalg.det(V @ U.T)
    D = np.eye(m)
    if d < 0:
        D[m - 1, m - 1] = -1.0

    R = V @ D @ U.T

    # Scale factor
    if with_scale:
        var_model = np.var(model_c, axis=0).sum()
        scale = (1.0 / var_model) * np.trace(np.diag(S) @ D)
    else:
        scale = 1.0

    # Translation
    t = mu_data - scale * (R @ mu_model)

    # Aligned source trajectory
    model_aligned = (scale * (R @ model.T)).T + t

    return R, t, scale, model_aligned


def evaluate_trajectory(est_pos: np.ndarray,
                        gt_pos: np.ndarray,
                        timestamps: Optional[np.ndarray] = None,
                        est_vel: Optional[np.ndarray] = None,
                        gt_vel: Optional[np.ndarray] = None,
                        est_quat: Optional[np.ndarray] = None,
                        gt_quat: Optional[np.ndarray] = None) -> TrajectoryMetrics:
    """
    Compute unified full-suite navigation performance metrics.
    """
    ate = compute_ate(est_pos, gt_pos)
    drift = compute_drift_metrics(est_pos, gt_pos, timestamps)

    vel_rmse, vel_h_rmse, vel_v_rmse = None, None, None
    if est_vel is not None and gt_vel is not None:
        vel_stats = compute_velocity_error(est_vel, gt_vel)
        vel_rmse = vel_stats["vel_rmse"]
        vel_h_rmse = vel_stats["vel_horizontal_rmse"]
        vel_v_rmse = vel_stats["vel_vertical_rmse"]

    att_rmse, att_max, r_rmse, p_rmse, y_rmse = None, None, None, None, None
    if est_quat is not None and gt_quat is not None:
        att_stats = compute_attitude_error(est_quat, gt_quat)
        att_rmse = att_stats["geodesic_rmse_deg"]
        att_max = att_stats["geodesic_max_deg"]
        r_rmse = att_stats["roll_rmse_deg"]
        p_rmse = att_stats["pitch_rmse_deg"]
        y_rmse = att_stats["yaw_rmse_deg"]

    return TrajectoryMetrics(
        ate_rmse=ate["rmse"],
        ate_mean=ate["mean"],
        ate_median=ate["median"],
        ate_std=ate["std"],
        ate_max=ate["max"],
        ate_horizontal_rmse=ate["horizontal_rmse"],
        ate_vertical_rmse=ate["vertical_rmse"],
        final_pos_error=ate["final_error"],
        total_distance_traveled=drift["total_distance"],
        drift_percentage=drift["drift_percentage"],
        drift_rate_mps=drift["drift_rate_mps"],
        drift_per_km=drift["drift_per_km"],
        vel_rmse=vel_rmse,
        vel_horizontal_rmse=vel_h_rmse,
        vel_vertical_rmse=vel_v_rmse,
        attitude_angle_rmse_deg=att_rmse,
        attitude_angle_max_deg=att_max,
        roll_rmse_deg=r_rmse,
        pitch_rmse_deg=p_rmse,
        yaw_rmse_deg=y_rmse,
        duration_sec=drift["duration"],
    )
