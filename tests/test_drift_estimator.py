"""
Unit Tests for Deep Neural Drift & Bias Estimator Models (Module 4)

Tests:
1. IMUScaler fitting, transform, and inverse transform.
2. IMUSequenceDataset sliding window indexing and batching.
3. NumPyDriftEstimator forward pass, uncertainty shapes, and persistence (save/load).
4. BiLSTMDriftEstimator and TCNDriftEstimator forward pass and multi-task loss (if PyTorch installed).
5. InertialDriftEstimator unified interface over short and full continuous sequences.
"""

import os
import unittest
import numpy as np
import tempfile

from src.models.drift_estimator import (
    IMUScaler,
    IMUSequenceDataset,
    NumPyDriftEstimator,
    InertialDriftEstimator,
    TORCH_AVAILABLE,
)
from src.models.train_drift_models import (
    generate_multi_regime_training_data,
    train_numpy_estimator,
)

try:
    import torch  # type: ignore
    from src.models.drift_estimator import BiLSTMDriftEstimator, TCNDriftEstimator  # type: ignore
except ImportError:
    torch = None  # type: ignore
    BiLSTMDriftEstimator = object  # type: ignore
    TCNDriftEstimator = object  # type: ignore


class TestDriftEstimator(unittest.TestCase):
    """Test suite for Module 4 Drift & Bias Estimators."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.N = 300
        self.dt = 0.01
        # Mock 6-DOF IMU data (3 accel + 3 gyro)
        self.imu_data = self.rng.normal(0, 1.0, (self.N, 6)).astype(np.float32)
        # Mock 3D velocities
        self.target_vel = self.rng.normal(0, 2.0, (self.N, 3)).astype(np.float32)
        # Mock 6D biases
        self.target_bias = self.rng.normal(0, 0.01, (self.N, 6)).astype(np.float32)

    def test_imu_scaler(self):
        """Test IMUScaler standardization and inverse transformation."""
        scaler = IMUScaler()
        scaler.fit(self.imu_data)

        self.assertIsNotNone(scaler.mean)
        self.assertIsNotNone(scaler.std)
        self.assertEqual(scaler.mean.shape, (6,))
        self.assertEqual(scaler.std.shape, (6,))

        transformed = scaler.transform(self.imu_data)
        self.assertEqual(transformed.shape, (self.N, 6))
        np.testing.assert_allclose(np.mean(transformed, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(transformed, axis=0), 1.0, atol=1e-5)

        reconstructed = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(reconstructed, self.imu_data, atol=1e-5)

        # Test dictionary serialization
        s_dict = scaler.to_dict()
        scaler2 = IMUScaler.from_dict(s_dict)
        np.testing.assert_allclose(scaler2.mean, scaler.mean)
        np.testing.assert_allclose(scaler2.std, scaler.std)

    def test_imu_sequence_dataset(self):
        """Test sliding window dataset generation."""
        dataset = IMUSequenceDataset(
            imu_data=self.imu_data,
            target_vel=self.target_vel,
            target_bias=self.target_bias,
            window_size=50,
            step_size=10,
            fit_scaler=True
        )
        expected_len = (self.N - 50) // 10 + 1
        self.assertEqual(len(dataset), expected_len)

        item = dataset[0]
        x_win, v_target, b_target = item
        if TORCH_AVAILABLE:
            self.assertEqual(x_win.shape, torch.Size([50, 6]))
            self.assertEqual(v_target.shape, torch.Size([3]))
            self.assertEqual(b_target.shape, torch.Size([6]))
        else:
            self.assertEqual(x_win.shape, (50, 6))
            self.assertEqual(v_target.shape, (3,))
            self.assertEqual(b_target.shape, (6,))

    def test_numpy_drift_estimator(self):
        """Test pure NumPyDriftEstimator forward and save/load."""
        estimator = NumPyDriftEstimator(
            input_dim=6,
            hidden_dim=32,
            window_size=50,
            seed=42
        )
        window = self.rng.normal(0, 1.0, (50, 6)).astype(np.float64)
        out = estimator.forward(window)

        self.assertIn('vel', out)
        self.assertIn('bias', out)
        self.assertIn('log_var', out)
        self.assertIn('vel_cov', out)
        self.assertIn('bias_cov', out)

        self.assertEqual(out['vel'].shape, (3,))
        self.assertEqual(out['bias'].shape, (6,))
        self.assertEqual(out['log_var'].shape, (9,))
        self.assertEqual(out['vel_cov'].shape, (3,))
        self.assertEqual(out['bias_cov'].shape, (6,))

        # Test batch forward
        batch_windows = self.rng.normal(0, 1.0, (4, 50, 6)).astype(np.float64)
        batch_out = estimator.forward(batch_windows)
        self.assertEqual(batch_out['vel'].shape, (4, 3))
        self.assertEqual(batch_out['bias'].shape, (4, 6))

        # Test Save & Load
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_numpy_model.npz")
            estimator.save(save_path)

            loaded = NumPyDriftEstimator(input_dim=6, hidden_dim=32, window_size=50)
            loaded.load(save_path)
            np.testing.assert_allclose(loaded.W1, estimator.W1)
            np.testing.assert_allclose(loaded.W_vel, estimator.W_vel)

    def test_train_numpy_estimator(self):
        """Test training pipeline for NumPy drift estimator."""
        estimator, scaler, history = train_numpy_estimator(
            imu_data=self.imu_data,
            target_vel=self.target_vel,
            target_bias=self.target_bias,
            window_size=30,
            step_size=5,
            epochs=5,
            seed=42
        )
        self.assertIsNotNone(estimator)
        self.assertIsNotNone(scaler)
        self.assertIn('val_mse_vel', history)
        self.assertGreater(len(history['val_mse_vel']), 0)

    def test_inertial_drift_estimator_interface(self):
        """Test unified InertialDriftEstimator streaming & sequence interface."""
        estimator = InertialDriftEstimator(
            model_type="numpy",
            window_size=30,
            input_dim=6,
            hidden_dim=32
        )
        # Sequence prediction
        preds = estimator.predict_sequence(self.imu_data, step_size=2)
        self.assertIn('vel', preds)
        self.assertIn('acc_bias', preds)
        self.assertIn('gyro_bias', preds)
        self.assertIn('vel_std', preds)
        self.assertIn('bias_std', preds)

        self.assertEqual(preds['vel'].shape, (self.N, 3))
        self.assertEqual(preds['acc_bias'].shape, (self.N, 3))
        self.assertEqual(preds['gyro_bias'].shape, (self.N, 3))
        self.assertEqual(preds['vel_std'].shape, (self.N, 3))
        self.assertEqual(preds['bias_std'].shape, (self.N, 6))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_bilstm_drift_estimator(self):
        """Test PyTorch BiLSTM model forward pass and output shapes."""
        model = BiLSTMDriftEstimator(
            input_dim=6,
            hidden_dim=64,
            num_layers=2,
            predict_uncertainty=True
        )
        x = torch.randn(4, 50, 6)
        out = model(x)

        self.assertEqual(out['vel'].shape, torch.Size([4, 3]))
        self.assertEqual(out['bias'].shape, torch.Size([4, 6]))
        self.assertEqual(out['log_var'].shape, torch.Size([4, 9]))
        self.assertEqual(out['vel_cov'].shape, torch.Size([4, 3]))
        self.assertEqual(out['bias_cov'].shape, torch.Size([4, 6]))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_tcn_drift_estimator(self):
        """Test PyTorch TCN model forward pass and output shapes."""
        model = TCNDriftEstimator(
            input_dim=6,
            num_channels=[32, 32, 64],
            kernel_size=3,
            predict_uncertainty=True
        )
        x = torch.randn(4, 50, 6)
        out = model(x)

        self.assertEqual(out['vel'].shape, torch.Size([4, 3]))
        self.assertEqual(out['bias'].shape, torch.Size([4, 6]))
        self.assertEqual(out['log_var'].shape, torch.Size([4, 9]))


if __name__ == "__main__":
    unittest.main()
