"""
Unit Tests for GPS Handoff Controller & AI-EKF Navigation Engine (Module 5)

Tests:
1. GPSHandoffController state machine transitions:
   - GPS_LOCKED -> GPS_DENIED -> RECOVERING_GPS -> GPS_LOCKED
2. Smooth recovery blending factor (raised cosine ramp).
3. Spoofing detection and automated outage event generation.
4. Multi-sensor update routing (GPS, ZUPT, Neural Velocity, Baro).
5. AIEKFNavigationEngine end-to-end execution on trajectories with GPS outages.
6. Benchmark comparison against Ground Truth and DataFrame export.
"""

import unittest
import numpy as np
import pandas as pd

from src.core.ekf import ErrorStateEKF, EKFConfig
from src.core.handoff_controller import (
    GPSHandoffController,
    HandoffConfig,
    HandoffState,
    HandoffEvent,
)
from src.core.ai_ekf_engine import (
    AIEKFNavigationEngine,
    AIEKFResult,
    run_ai_ekf_navigation,
)
from src.data.loaders import SyntheticDataLoader


class TestHandoffController(unittest.TestCase):
    """Test suite for GPS Handoff Controller & AI-EKF Engine."""

    def setUp(self):
        self.ekf = ErrorStateEKF(config=EKFConfig())
        self.config = HandoffConfig(
            gps_timeout_sec=1.0,
            transition_duration_sec=2.0,
            consecutive_rejections_limit=3,
        )
        self.handoff = GPSHandoffController(ekf=self.ekf, config=self.config)

    def test_initial_state(self):
        """Test default initial state is GPS_DENIED until first fix."""
        self.assertEqual(self.handoff.state, HandoffState.GPS_DENIED)
        self.assertFalse(self.handoff.is_gps_available)

    def test_gps_lock_and_outage_transition(self):
        """Test transitions from GPS fix -> timeout outage -> recovery."""
        # 1. Provide valid GPS fix at t = 1.0
        accepted, d2 = self.handoff.process_gps_update(
            timestamp=1.0,
            pos_meas=np.array([0.0, 0.0, 0.0]),
            vel_meas=np.array([0.0, 0.0, 0.0]),
            is_valid=True
        )
        self.assertTrue(accepted)
        self.assertEqual(self.handoff.state, HandoffState.RECOVERING_GPS)

        # 2. Advance time past transition window (t = 4.0) with valid fixes
        self.handoff.process_gps_update(
            timestamp=4.0,
            pos_meas=np.array([0.0, 0.0, 0.0]),
            vel_meas=np.array([0.0, 0.0, 0.0]),
            is_valid=True
        )
        self.assertEqual(self.handoff.state, HandoffState.GPS_LOCKED)
        self.assertTrue(self.handoff.is_gps_available)

        # 3. Simulate GPS outage (no GPS update for 2 seconds, t = 6.0)
        self.handoff.process_gps_update(
            timestamp=6.0,
            pos_meas=None,
            is_valid=False
        )
        self.assertEqual(self.handoff.state, HandoffState.GPS_DENIED)
        self.assertFalse(self.handoff.is_gps_available)

    def test_spoofing_detection(self):
        """Test spoofing flag triggered after consecutive Chi^2 rejections."""
        # Establish lock first
        self.handoff.process_gps_update(timestamp=1.0, pos_meas=np.zeros(3), is_valid=True)

        # Send wild spoofed measurements repeatedly
        for k in range(4):
            spoofed_pos = np.array([500.0 + k * 100, 1000.0, 50.0])
            self.handoff.process_gps_update(timestamp=2.0 + k * 0.2, pos_meas=spoofed_pos, is_valid=True)

        # Should transition to GPS_DENIED and record spoofing event
        self.assertEqual(self.handoff.state, HandoffState.GPS_DENIED)
        spoof_events = [e for e in self.handoff.events if e.event_type == 'spoofing_detected']
        self.assertGreater(len(spoof_events), 0)

    def test_smooth_blend_factor(self):
        """Test smooth raised-cosine blend factor computation."""
        self.handoff.state = HandoffState.RECOVERING_GPS
        self.handoff.transition_start_time = 10.0

        # Start of recovery (t = 10.0) -> blend factor = 0.0
        f0 = self.handoff._compute_recovery_blend_factor(10.0)
        self.assertAlmostEqual(f0, 0.0, delta=1e-5)

        # Mid-point of recovery (t = 11.0, duration = 2.0) -> blend factor = 0.5
        f_mid = self.handoff._compute_recovery_blend_factor(11.0)
        self.assertAlmostEqual(f_mid, 0.5, delta=0.05)

        # End of recovery (t = 12.0) -> blend factor = 1.0
        f1 = self.handoff._compute_recovery_blend_factor(12.0)
        self.assertAlmostEqual(f1, 1.0, delta=1e-5)

    def test_ai_ekf_navigation_engine_with_outage(self):
        """Test full AI-EKF Navigation Engine on trajectory with simulated GPS outage."""
        # Generate 20s trajectory with GPS outage between t = 6s and t = 14s
        traj = SyntheticDataLoader.create(
            profile="urban_driving",
            duration=20.0,
            dt=0.01,
            imu_grade="consumer",
            gps_outages=[(6.0, 14.0)],
            seed=42
        )

        engine = AIEKFNavigationEngine()
        result = engine.process_trajectory(traj)

        self.assertIsInstance(result, AIEKFResult)
        self.assertEqual(len(result.pos), len(traj.timestamps))
        self.assertEqual(len(result.vel), len(traj.timestamps))
        self.assertEqual(len(result.handoff_states), len(traj.timestamps))

        # Check performance metrics against Ground Truth
        self.assertIsNotNone(result.metrics)
        self.assertGreater(result.metrics.ate_rmse, 0.0)
        self.assertLess(result.metrics.ate_rmse, 100.0)  # Substantially better than pure DR (>500m)

        # Check summary table and DataFrame serialization
        summary = result.summary_table()
        self.assertIn("NAVIS Module 5", summary)
        self.assertIn("ATE RMSE", summary)

        df = result.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('ekf_pos_x', df.columns)
        self.assertIn('handoff_state', df.columns)
        self.assertIn('pos_3sigma_x', df.columns)


if __name__ == "__main__":
    unittest.main()
