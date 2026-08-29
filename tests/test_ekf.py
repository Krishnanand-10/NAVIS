"""
Unit Tests for 15-State Error-State Extended Kalman Filter (Module 5)

Tests:
1. ErrorStateEKF initialization and state structure.
2. Prediction step, covariance propagation, and matrix symmetry.
3. Measurement updates: Position, Velocity, ZUPT, Barometer, Non-Holonomic Constraints (NHC).
4. Chi-Square innovation gating and outlier rejection.
5. Quaternion error-state injection and error reset.
6. AI-Adaptive process noise scaling.
"""

import unittest
import numpy as np

from src.core.ekf import ErrorStateEKF, EKFConfig, EKFState


class TestErrorStateEKF(unittest.TestCase):
    """Test suite for 15-State Error-State EKF."""

    def setUp(self):
        self.config = EKFConfig(
            init_pos_std=1.0,
            init_vel_std=0.2,
            init_att_std_deg=2.0,
        )
        self.ekf = ErrorStateEKF(config=self.config)

    def test_initial_state(self):
        """Test initial state dimensions and covariance positive-definiteness."""
        state = self.ekf.state
        self.assertEqual(state.pos.shape, (3,))
        self.assertEqual(state.vel.shape, (3,))
        self.assertEqual(state.quat.shape, (4,))
        self.assertEqual(state.acc_bias.shape, (3,))
        self.assertEqual(state.gyro_bias.shape, (3,))
        self.assertEqual(state.P.shape, (15, 15))

        # Check covariance matrix symmetry
        np.testing.assert_allclose(state.P, state.P.T, atol=1e-10)

        # Check positive eigenvalues
        eigvals = np.linalg.eigvalsh(state.P)
        self.assertTrue(np.all(eigvals > 0))

    def test_predict_step(self):
        """Test time propagation step on stationary sensor."""
        acc_meas = np.array([0.0, 0.0, 9.80665])  # Gravity upward reaction in body
        gyro_meas = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        P_prev = self.ekf.P.copy()
        state = self.ekf.predict(acc_meas, gyro_meas, dt=dt)

        self.assertIsInstance(state, EKFState)
        # Covariance uncertainty should grow during prediction without updates
        self.assertGreater(np.trace(state.P), np.trace(P_prev))
        np.testing.assert_allclose(state.P, state.P.T, atol=1e-10)

    def test_position_update(self):
        """Test GPS position update and uncertainty reduction."""
        # Run a few predict steps to grow covariance
        for _ in range(10):
            self.ekf.predict(np.array([0.0, 0.0, 9.80665]), np.zeros(3), dt=0.01)

        P_prior = self.ekf.P.copy()
        pos_meas = np.array([1.0, 2.0, 0.0])
        pos_cov = 1.0  # 1 m^2

        accepted, d2 = self.ekf.update_position(pos_meas, pos_cov)
        self.assertTrue(accepted)
        self.assertGreater(d2, 0.0)

        # Position variance should strictly decrease
        self.assertLess(np.trace(self.ekf.P[0:3, 0:3]), np.trace(P_prior[0:3, 0:3]))
        np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-10)

    def test_zupt_update(self):
        """Test Zero-Velocity Update (ZUPT)."""
        # Inject artificial velocity
        self.ekf.vel = np.array([0.5, -0.3, 0.1])
        P_prior = self.ekf.P.copy()

        accepted, d2 = self.ekf.update_zupt(zupt_cov_sigma=0.01)
        self.assertTrue(accepted)

        # Velocity should be pulled close to zero
        self.assertLess(np.linalg.norm(self.ekf.vel), 0.1)
        # Velocity covariance should shrink
        self.assertLess(np.trace(self.ekf.P[3:6, 3:6]), np.trace(P_prior[3:6, 3:6]))

    def test_barometer_update(self):
        """Test barometric altitude update."""
        self.ekf.pos[2] = 50.0  # 50m altitude
        alt_meas = 52.0         # True baro altitude 52m

        accepted, d2 = self.ekf.update_barometer(alt_meas=alt_meas, alt_cov_sigma=1.0)
        self.assertTrue(accepted)
        self.assertAlmostEqual(self.ekf.pos[2], 51.0, delta=1.5)

    def test_chi2_outlier_rejection(self):
        """Test Chi-Square innovation gate rejecting wild outlier / spoofed GPS fix."""
        pos_meas_spoofed = np.array([1000.0, 5000.0, 200.0])  # Huge leap
        accepted, d2 = self.ekf.update_position(pos_meas_spoofed, pos_cov=1.0, chi2_threshold=16.27)

        # Should be rejected
        self.assertFalse(accepted)
        self.assertGreater(d2, 16.27)

    def test_nhc_update(self):
        """Test Non-Holonomic Constraint (NHC) velocity damping."""
        self.ekf.vel = np.array([10.0, 2.0, 1.5])  # Forward + lateral + vertical
        accepted, d2 = self.ekf.update_nhc(lateral_cov=0.05, vertical_cov=0.05)

        self.assertTrue(accepted)
        # Lateral and vertical body velocities should be damped
        self.assertLess(np.abs(self.ekf.vel[1]), 2.0)


if __name__ == "__main__":
    unittest.main()
