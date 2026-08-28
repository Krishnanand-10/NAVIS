"""
Unit Tests for Navigation Evaluation Metrics & Trajectory Alignment
"""

import unittest
import numpy as np

from src.utils.metrics import (
    compute_ate,
    compute_rpe,
    compute_drift_metrics,
    compute_attitude_error,
    compute_velocity_error,
    align_trajectories_umeyama,
    evaluate_trajectory,
)
from src.utils.coordinate_transforms import euler_to_quat


class TestMetrics(unittest.TestCase):

    def test_compute_ate_exact_match(self):
        """ATE of identical trajectories must be zero."""
        pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
        ])
        ate = compute_ate(pos, pos)
        self.assertAlmostEqual(ate["rmse"], 0.0, places=6)
        self.assertAlmostEqual(ate["mean"], 0.0, places=6)
        self.assertAlmostEqual(ate["max"], 0.0, places=6)

    def test_compute_ate_constant_offset(self):
        """ATE with known constant displacement."""
        gt_pos = np.zeros((100, 3))
        est_pos = np.ones((100, 3)) * 3.0  # Euclidean distance = sqrt(3^2 + 3^2 + 3^2) = sqrt(27) ≈ 5.19615

        ate = compute_ate(est_pos, gt_pos)
        self.assertAlmostEqual(ate["rmse"], np.sqrt(27), places=4)
        self.assertAlmostEqual(ate["mean"], np.sqrt(27), places=4)

    def test_compute_rpe(self):
        """Verify RPE relative pose difference."""
        gt_pos = np.arange(200).reshape((100, 2))
        gt_pos = np.hstack([gt_pos, np.zeros((100, 1))])
        est_pos = gt_pos + 1.0  # Constant shift

        rpe = compute_rpe(est_pos, gt_pos, delta_samples=10)
        # Relative displacements are identical, so relative error is 0
        self.assertAlmostEqual(rpe["rpe_rmse"], 0.0, places=5)

    def test_compute_drift_percentage(self):
        """Traveled 1000m, final error 10m -> 1.0% drift."""
        t = np.linspace(0, 100, 101)
        gt_pos = np.zeros((101, 3))
        gt_pos[:, 0] = np.linspace(0, 1000, 101)  # 1000m path

        est_pos = gt_pos.copy()
        est_pos[-1, 0] += 10.0  # 10m final error

        drift = compute_drift_metrics(est_pos, gt_pos, timestamps=t)
        self.assertAlmostEqual(drift["total_distance"], 1000.0, places=2)
        self.assertAlmostEqual(drift["final_error"], 10.0, places=2)
        self.assertAlmostEqual(drift["drift_percentage"], 1.0, places=2)
        self.assertAlmostEqual(drift["drift_per_km"], 10.0, places=2)

    def test_compute_attitude_error(self):
        """Compare quaternion geodesic error for 10 degree yaw difference."""
        q_gt = euler_to_quat(0.0, 0.0, 0.0)
        q_est = euler_to_quat(0.0, 0.0, np.deg2rad(10.0))

        q_gt_arr = np.tile(q_gt, (10, 1))
        q_est_arr = np.tile(q_est, (10, 1))

        att_err = compute_attitude_error(q_est_arr, q_gt_arr)
        self.assertAlmostEqual(att_err["geodesic_rmse_deg"], 10.0, places=3)
        self.assertAlmostEqual(att_err["yaw_rmse_deg"], 10.0, places=3)

    def test_umeyama_alignment(self):
        """Recover known rotation and translation via Umeyama SVD alignment."""
        N = 50
        t_true = np.array([10.0, -5.0, 2.0])
        # 90 deg rotation around Z
        R_true = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])

        source = np.random.randn(N, 3) * 10.0
        target = (R_true @ source.T).T + t_true

        R_est, t_est, s_est, source_aligned = align_trajectories_umeyama(source, target)

        np.testing.assert_allclose(R_est, R_true, atol=1e-5)
        np.testing.assert_allclose(t_est, t_true, atol=1e-5)
        np.testing.assert_allclose(source_aligned, target, atol=1e-5)

    def test_evaluate_trajectory_full(self):
        """Verify TrajectoryMetrics dataclass generation."""
        t = np.linspace(0, 10, 101)
        gt_p = np.zeros((101, 3))
        gt_p[:, 0] = np.linspace(0, 100, 101)
        est_p = gt_p.copy()
        est_p[:, 1] += 2.0  # 2m constant lateral offset

        metrics = evaluate_trajectory(est_p, gt_p, timestamps=t)
        self.assertAlmostEqual(metrics.ate_rmse, 2.0, places=3)
        summary = metrics.summary_table()
        self.assertIn("NAVIS DEAD RECKONING BENCHMARK REPORT", summary)


if __name__ == "__main__":
    unittest.main()
