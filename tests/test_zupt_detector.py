"""
Unit Tests for Zero-Velocity Detection Suite (GLRT, ARE, MAG, AI-ZUPT)
"""

import unittest
import numpy as np

from src.models.zupt_detector import (
    GLRTDetector,
    AREDetector,
    MAGDetector,
    ZUPTDetector,
    detect_zero_velocity_statistical,
)
from src.simulation.trajectory_gen import TrajectoryGenerator


class TestZUPTDetector(unittest.TestCase):

    def setUp(self):
        # Generate 15s synthetic pedestrian trajectory with realistic heel-strike stance phases
        gen = TrajectoryGenerator(profile="pedestrian", duration=15.0, dt=0.01, imu_grade="industrial", seed=123)
        df = gen.generate()
        self.acc = df[['acc_x', 'acc_y', 'acc_z']].values
        self.gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
        self.true_stance = df['stance_mask'].values.astype(bool)

    def test_glrt_detector_on_pedestrian(self):
        """GLRT detector should accurately identify pedestrian stance phases."""
        detector = GLRTDetector(window_size=9, threshold=200.0)
        pred_stance = detector.detect(self.acc, self.gyro)

        self.assertEqual(len(pred_stance), len(self.acc))
        # Accuracy vs true stance should be high
        accuracy = np.mean(pred_stance == self.true_stance)
        self.assertGreater(accuracy, 0.80)

    def test_are_detector(self):
        """ARE detector on stationary vs active IMU streams."""
        detector = AREDetector(window_size=15, threshold=0.3)
        pred_stance = detector.detect(self.acc)
        self.assertEqual(len(pred_stance), len(self.acc))

    def test_mag_detector(self):
        """MAG detector on angular rates."""
        detector = MAGDetector(window_size=15, threshold=0.1)
        pred_stance = detector.detect(self.gyro)
        self.assertEqual(len(pred_stance), len(self.gyro))

    def test_detector_on_pure_stationary(self):
        """Pure stationary sensor must be detected as stance 100% of the time."""
        N = 200
        flat_acc = np.tile([0.0, 0.0, 9.80665], (N, 1))
        flat_gyro = np.zeros((N, 3))

        glrt = GLRTDetector(window_size=15)
        stance_glrt = glrt.detect(flat_acc, flat_gyro)
        self.assertTrue(np.all(stance_glrt))

        are = AREDetector(window_size=15)
        stance_are = are.detect(flat_acc)
        self.assertTrue(np.all(stance_are))

        mag = MAGDetector(window_size=15)
        stance_mag = mag.detect(flat_gyro)
        self.assertTrue(np.all(stance_mag))

    def test_unified_zupt_factory(self):
        """Test ZUPTDetector.detect dispatcher with various methods."""
        for method in ["glrt", "are", "mag"]:
            mask = ZUPTDetector.detect(self.acc, self.gyro, method=method, window_size=15)
            self.assertEqual(len(mask), len(self.acc))
            self.assertEqual(mask.dtype, bool)


if __name__ == "__main__":
    unittest.main()
