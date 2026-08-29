"""
Unit Tests for Neural Dead Reckoning Engine (Module 4)

Tests:
1. NeuralDeadReckoningEngine initialization and mode parsing.
2. Trajectory execution across all 4 AI correction modes:
   - VELOCITY_RESIDUAL
   - DYNAMIC_BIAS
   - HYBRID_CORRECTION
   - DIRECT_VELOCITY
3. Verification of trajectory metrics, uncertainty propagation (+- 2 sigma bounds),
   and drift reduction relative to uncorrected DR.
4. Summary table formatting and DataFrame serialization.
5. Convenience wrapper run_neural_dead_reckoning.
"""

import unittest
import numpy as np
import pandas as pd

from src.data.loaders import SyntheticDataLoader
from src.models.drift_estimator import InertialDriftEstimator
from src.core.neural_dr import (
    NeuralDeadReckoningEngine,
    NeuralDeadReckoningResult,
    NeuralCorrectionMode,
    run_neural_dead_reckoning,
)
from src.models.train_drift_models import train_drift_estimator


class TestNeuralDeadReckoning(unittest.TestCase):
    """Test suite for Neural Dead Reckoning Engine."""

    @classmethod
    def setUpClass(cls):
        """Train a lightweight drift estimator once for the test suite."""
        cls.traj = SyntheticDataLoader.create(
            profile="figure_eight",
            duration=15.0,
            dt=0.01,
            imu_grade="consumer",
            seed=42,
        )
        cls.estimator = train_drift_estimator(
            model_type="numpy",
            epochs=5,
            window_size=30,
            save_dir="checkpoints/test_drift_estimator",
            seed=42,
        )

    def test_mode_parsing(self):
        """Test NeuralCorrectionMode string parsing."""
        self.assertEqual(NeuralCorrectionMode.from_str("hybrid"), NeuralCorrectionMode.HYBRID_CORRECTION)
        self.assertEqual(NeuralCorrectionMode.from_str("dynamic_bias"), NeuralCorrectionMode.DYNAMIC_BIAS)
        self.assertEqual(NeuralCorrectionMode.from_str("velocity_residual"), NeuralCorrectionMode.VELOCITY_RESIDUAL)
        self.assertEqual(NeuralCorrectionMode.from_str("direct_velocity"), NeuralCorrectionMode.DIRECT_VELOCITY)

    def test_hybrid_neural_dr(self):
        """Test HYBRID_CORRECTION mode execution and outputs."""
        engine = NeuralDeadReckoningEngine(
            drift_estimator=self.estimator,
            correction_mode=NeuralCorrectionMode.HYBRID_CORRECTION,
            velocity_blend_alpha=0.85,
            bias_blend_alpha=0.70,
        )
        result = engine.process(self.traj)

        self.assertIsInstance(result, NeuralDeadReckoningResult)
        N = len(self.traj.timestamps)

        self.assertEqual(result.pos.shape, (N, 3))
        self.assertEqual(result.vel.shape, (N, 3))
        self.assertEqual(result.quat.shape, (N, 4))
        self.assertEqual(result.euler_deg.shape, (N, 3))
        self.assertEqual(result.estimated_acc_bias.shape, (N, 3))
        self.assertEqual(result.estimated_gyro_bias.shape, (N, 3))
        self.assertEqual(result.pos_uncertainty_std.shape, (N, 3))
        self.assertEqual(result.vel_uncertainty_std.shape, (N, 3))

        # Check that metrics were generated against Ground Truth
        self.assertIsNotNone(result.metrics_neural)
        self.assertIsNotNone(result.metrics_standard)
        self.assertGreaterEqual(result.drift_reduction_pct, 0.0)

        # Summary table formatting
        summary = result.summary_table()
        self.assertIn("NAVIS Module 4", summary)
        self.assertIn("ATE RMSE", summary)

    def test_velocity_residual_mode(self):
        """Test VELOCITY_RESIDUAL mode."""
        engine = NeuralDeadReckoningEngine(
            drift_estimator=self.estimator,
            correction_mode=NeuralCorrectionMode.VELOCITY_RESIDUAL,
        )
        result = engine.process(self.traj)
        self.assertEqual(result.pos.shape, (len(self.traj.timestamps), 3))
        self.assertIsNotNone(result.metrics_neural)

    def test_dynamic_bias_mode(self):
        """Test DYNAMIC_BIAS mode."""
        engine = NeuralDeadReckoningEngine(
            drift_estimator=self.estimator,
            correction_mode=NeuralCorrectionMode.DYNAMIC_BIAS,
        )
        result = engine.process(self.traj)
        self.assertEqual(result.pos.shape, (len(self.traj.timestamps), 3))
        self.assertIsNotNone(result.metrics_neural)

    def test_direct_velocity_mode(self):
        """Test DIRECT_VELOCITY mode."""
        engine = NeuralDeadReckoningEngine(
            drift_estimator=self.estimator,
            correction_mode=NeuralCorrectionMode.DIRECT_VELOCITY,
        )
        result = engine.process(self.traj)
        self.assertEqual(result.pos.shape, (len(self.traj.timestamps), 3))
        self.assertIsNotNone(result.metrics_neural)

    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        engine = NeuralDeadReckoningEngine(drift_estimator=self.estimator)
        result = engine.process(self.traj)
        df = result.to_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.traj.timestamps))
        self.assertIn('pos_x', df.columns)
        self.assertIn('vel_x', df.columns)
        self.assertIn('acc_bias_x', df.columns)
        self.assertIn('pos_std_x', df.columns)
        self.assertIn('std_dr_pos_x', df.columns)
        self.assertIn('gt_pos_x', df.columns)

    def test_run_neural_dr_wrapper(self):
        """Test high-level run_neural_dead_reckoning function."""
        result = run_neural_dead_reckoning(
            trajectory=self.traj,
            drift_estimator=self.estimator,
            correction_mode="hybrid"
        )
        self.assertIsInstance(result, NeuralDeadReckoningResult)
        self.assertEqual(len(result.pos), len(self.traj.timestamps))


if __name__ == "__main__":
    unittest.main()
