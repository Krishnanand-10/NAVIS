"""
Unit Tests for TrajectoryData Container and Dataset Loaders
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd

from src.data.dataset import TrajectoryData
from src.data.loaders import SyntheticDataLoader, GenericCSVLoader


class TestLoaders(unittest.TestCase):

    def test_trajectory_data_container(self):
        """Test TrajectoryData creation, length, properties, and dataframe round-trip."""
        N = 100
        t = np.linspace(0, 1.0, N)
        acc = np.random.randn(N, 3)
        gyro = np.random.randn(N, 3)
        gt_pos = np.cumsum(acc, axis=0) * 0.01

        data = TrajectoryData(
            timestamps=t,
            acc=acc,
            gyro=gyro,
            gt_pos=gt_pos,
            metadata={"test_key": "test_val"}
        )

        self.assertEqual(len(data), N)
        self.assertAlmostEqual(data.duration, 1.0, places=3)
        self.assertAlmostEqual(data.sampling_rate, 99.0, delta=2.0)

        # To DataFrame
        df = data.to_dataframe()
        self.assertEqual(len(df), N)
        self.assertIn('acc_x', df.columns)
        self.assertIn('gt_pos_x', df.columns)

        # From DataFrame
        data_rec = TrajectoryData.from_dataframe(df)
        self.assertEqual(len(data_rec), N)
        np.testing.assert_allclose(data_rec.acc, acc)
        np.testing.assert_allclose(data_rec.gt_pos, gt_pos)

    def test_synthetic_data_loader(self):
        """Test SyntheticDataLoader generating valid TrajectoryData."""
        traj = SyntheticDataLoader.create(
            profile="figure_eight",
            duration=5.0,
            dt=0.01,
            imu_grade="consumer"
        )
        self.assertIsInstance(traj, TrajectoryData)
        self.assertEqual(len(traj), 500)
        self.assertIsNotNone(traj.mag)
        self.assertIsNotNone(traj.baro_alt)
        self.assertIsNotNone(traj.gt_pos)

    def test_csv_save_and_generic_loader(self):
        """Test saving trajectory to CSV and loading back via GenericCSVLoader."""
        traj = SyntheticDataLoader.create(profile="straight", duration=2.0, dt=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "straight_traj.csv")
            traj.to_csv(csv_path)
            self.assertTrue(os.path.exists(csv_path))

            # Load with GenericCSVLoader
            loaded_traj = GenericCSVLoader.load(csv_path)
            self.assertEqual(len(loaded_traj), len(traj))
            np.testing.assert_allclose(loaded_traj.acc, traj.acc, atol=1e-5)
            np.testing.assert_allclose(loaded_traj.gyro, traj.gyro, atol=1e-5)
            np.testing.assert_allclose(loaded_traj.gt_pos, traj.gt_pos, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
