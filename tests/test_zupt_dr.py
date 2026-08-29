"""
Integration Tests for ZUPT-Augmented Dead Reckoning Engine
"""

import unittest
import numpy as np

from src.core.zupt_dr import (
    ZUPTDeadReckoningEngine,
    ZUPTDeadReckoningResult,
    run_zupt_dead_reckoning,
)
from src.data.loaders import SyntheticDataLoader


class TestZUPTDeadReckoning(unittest.TestCase):

    def test_pedestrian_drift_reduction(self):
        """ZUPT-aided Dead Reckoning should achieve significant drift reduction (>50%) over pure DR on pedestrian data."""
        traj = SyntheticDataLoader.create(
            profile="pedestrian",
            duration=20.0,
            dt=0.01,
            imu_grade="industrial",
            seed=42,
        )

        engine = ZUPTDeadReckoningEngine(
            detection_method="glrt",
            window_size=15,
            online_bias_tracking=True
        )
        result = engine.process_trajectory(traj)

        self.assertIsNotNone(result.metrics_zupt)
        self.assertIsNotNone(result.metrics_standard)

        # ZUPT ATE RMSE must be strictly lower than standard uncorrected DR
        self.assertLess(result.metrics_zupt.ate_rmse, result.metrics_standard.ate_rmse)

        # Drift reduction should be substantial (typically >50% to >90%)
        self.assertGreater(result.drift_reduction_pct, 50.0)

        # Formatted summary string
        summary = result.summary_table()
        self.assertIn("NAVIS ZUPT-AUGMENTED DEAD RECKONING BENCHMARK", summary)
        self.assertIn("Drift Error Reduction", summary)

    def test_stationary_zero_drift(self):
        """Stationary sensor with ZUPT enabled should have nearly zero drift."""
        traj = SyntheticDataLoader.create(
            profile="stationary",
            duration=10.0,
            dt=0.01,
            imu_grade="consumer",
            seed=99,
        )

        engine = ZUPTDeadReckoningEngine(detection_method="are", window_size=15)
        result = engine.process_trajectory(traj)

        # Stance ratio should be high
        self.assertGreater(np.mean(result.stance_mask), 0.90)
        # Position error should remain tightly bounded
        self.assertLess(result.metrics_zupt.ate_rmse, 0.1)


if __name__ == "__main__":
    unittest.main()
