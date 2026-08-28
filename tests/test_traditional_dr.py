"""
Unit Tests for Classical Dead Reckoning Engine & Baseline Solver
"""

import unittest
import numpy as np

from src.core.traditional_dr import (
    DeadReckoningEngine,
    DeadReckoningResult,
    run_dead_reckoning,
)
from src.data.dataset import TrajectoryData
from src.data.loaders import SyntheticDataLoader
from src.utils.coordinate_transforms import euler_to_quat, quat_to_euler


class TestTraditionalDR(unittest.TestCase):

    def test_dr_ideal_stationary(self):
        """Dead Reckoning on ideal noise-free stationary sensor should stay at origin with zero drift."""
        N = 200
        dt = 0.01
        timestamps = np.linspace(0, (N - 1) * dt, N)
        # Sitting flat in ENU: specific force = [0, 0, 9.80665], omega = [0, 0, 0]
        acc = np.tile(np.array([0.0, 0.0, 9.80665]), (N, 1))
        gyro = np.zeros((N, 3))
        gt_pos = np.zeros((N, 3))
        gt_vel = np.zeros((N, 3))
        gt_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))

        traj = TrajectoryData(
            timestamps=timestamps,
            acc=acc,
            gyro=gyro,
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            gt_quat=gt_quat,
        )

        engine = DeadReckoningEngine(attitude_method="quaternion_rk4", translation_method="trapezoidal")
        result = engine.process_trajectory(traj)

        self.assertAlmostEqual(result.metrics.ate_rmse, 0.0, places=4)
        self.assertAlmostEqual(result.metrics.final_pos_error, 0.0, places=4)

    def test_dr_ideal_constant_velocity(self):
        """Dead Reckoning on ideal straight line motion at constant velocity 10 m/s."""
        N = 300
        dt = 0.01
        timestamps = np.linspace(0, (N - 1) * dt, N)
        # Heading East (+X in ENU)
        # Specific force: gravity reaction only
        acc = np.tile(np.array([0.0, 0.0, 9.80665]), (N, 1))
        gyro = np.zeros((N, 3))
        gt_vel = np.tile(np.array([10.0, 0.0, 0.0]), (N, 1))
        gt_pos = np.zeros((N, 3))
        gt_pos[:, 0] = timestamps * 10.0
        gt_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N, 1))

        traj = TrajectoryData(
            timestamps=timestamps,
            acc=acc,
            gyro=gyro,
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            gt_quat=gt_quat,
        )

        engine = DeadReckoningEngine(attitude_method="quaternion_rk4", translation_method="trapezoidal")
        result = engine.process_trajectory(traj)

        self.assertAlmostEqual(result.metrics.ate_rmse, 0.0, places=3)
        self.assertAlmostEqual(result.pos[-1, 0], gt_pos[-1, 0], places=3)

    def test_dr_on_synthetic_noisy_trajectory(self):
        """Dead Reckoning on noisy synthetic data exhibits expected non-zero drift."""
        traj = SyntheticDataLoader.create(
            profile="urban_driving",
            duration=20.0,
            dt=0.01,
            imu_grade="industrial",
            seed=42,
        )

        engine = DeadReckoningEngine(attitude_method="quaternion_rk4", translation_method="trapezoidal")
        result = engine.process_trajectory(traj)

        self.assertIsNotNone(result.metrics)
        self.assertGreater(result.metrics.ate_rmse, 0.0)
        self.assertGreater(result.metrics.total_distance_traveled, 10.0)
        self.assertIsInstance(result.metrics.summary_table(), str)

    def test_coarse_leveling(self):
        """Estimate roll and pitch from stationary accelerometer readings."""
        # Flat: acc = [0, 0, 9.80665] -> roll=0, pitch=0
        flat_acc = np.tile([0.0, 0.0, 9.80665], (50, 1))
        q_est = DeadReckoningEngine.compute_coarse_leveling(flat_acc)
        r, p, y = quat_to_euler(q_est)
        self.assertAlmostEqual(r, 0.0, places=4)
        self.assertAlmostEqual(p, 0.0, places=4)

        # Pitched 30 degrees:
        # f = R^T * [0, 0, g] where R is pitch(30 deg)
        # f_x = -g * sin(30 deg) = -4.903325, f_z = g * cos(30 deg) = 8.4928
        pitch_rad = np.deg2rad(30.0)
        g = 9.80665
        pitched_acc = np.tile([-g * np.sin(pitch_rad), 0.0, g * np.cos(pitch_rad)], (50, 1))
        q_pitch_est = DeadReckoningEngine.compute_coarse_leveling(pitched_acc)
        r, p, y = quat_to_euler(q_pitch_est)
        self.assertAlmostEqual(p, pitch_rad, places=3)
        self.assertAlmostEqual(r, 0.0, places=3)

    def test_different_integration_methods(self):
        """Test all attitude and translation solver permutations."""
        traj = SyntheticDataLoader.create(profile="straight", duration=5.0, dt=0.01, seed=42)

        methods = [
            ("quaternion_rk4", "trapezoidal"),
            ("quaternion_midpoint", "velocity_verlet"),
            ("quaternion_exponential", "forward_euler"),
            ("euler_angles", "trapezoidal"),
        ]

        for att_m, trans_m in methods:
            engine = DeadReckoningEngine(attitude_method=att_m, translation_method=trans_m)
            res = engine.process_trajectory(traj)
            self.assertEqual(len(res.pos), len(traj))
            self.assertFalse(np.any(np.isnan(res.pos)))


if __name__ == "__main__":
    unittest.main()
