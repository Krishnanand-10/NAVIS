"""
Unit Tests for Analytical INS Error Dynamics & Drift Propagation
"""

import unittest
import numpy as np

from src.core.error_dynamics import (
    skew_symmetric,
    INSErrorState,
    build_ins_error_jacobian,
    propagate_error_covariance,
    TheoreticalDriftPredictor,
)
from src.simulation.sensor_noise import INDUSTRIAL_IMU


class TestErrorDynamics(unittest.TestCase):

    def test_skew_symmetric(self):
        """Cross product matrix equivalence [v_x] * u = v x u."""
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, -5.0, 6.0])

        cross_direct = np.cross(v, u)
        cross_skew = skew_symmetric(v) @ u

        np.testing.assert_allclose(cross_direct, cross_skew, atol=1e-10)

    def test_ins_error_state_dataclass(self):
        """Verify 15-state serialization to and from vector."""
        state = INSErrorState(
            delta_pos=np.array([1.0, 2.0, 3.0]),
            delta_vel=np.array([0.1, 0.2, 0.3]),
            delta_psi=np.array([0.01, 0.02, 0.03]),
            delta_b_a=np.array([0.001, 0.002, 0.003]),
            delta_b_g=np.array([0.0001, 0.0002, 0.0003]),
        )
        vec = state.vector
        self.assertEqual(len(vec), 15)

        reconstructed = INSErrorState.from_vector(vec)
        np.testing.assert_allclose(reconstructed.delta_pos, state.delta_pos)
        np.testing.assert_allclose(reconstructed.delta_b_g, state.delta_b_g)

    def test_ins_error_jacobian_structure(self):
        """Verify structure of 15x15 INS Jacobian."""
        f_body = np.array([0.0, 0.0, 9.80665])
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])

        F = build_ins_error_jacobian(f_body, q_identity)
        self.assertEqual(F.shape, (15, 15))

        # Position derivative wrt velocity should be identity 3x3
        np.testing.assert_allclose(F[0:3, 3:6], np.eye(3))

        # Velocity derivative wrt accel bias should be rotation matrix (identity here)
        np.testing.assert_allclose(F[3:6, 9:12], np.eye(3))

    def test_covariance_propagation(self):
        """Covariance matrix remains symmetric and positive-definite under propagation."""
        P0 = np.eye(15) * 0.1
        f_b = np.array([0.0, 0.0, 9.81])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        F = build_ins_error_jacobian(f_b, q)
        Q = np.eye(15) * 1e-4

        P_next = propagate_error_covariance(P0, F, Q, dt=0.01)

        # Symmetry
        np.testing.assert_allclose(P_next, P_next.T, atol=1e-10)

        # Positive definiteness (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(P_next)
        self.assertTrue(np.all(eigvals > 0))

    def test_theoretical_drift_cubic_growth(self):
        """Verify analytical drift equations match physical O(t^2) accel bias and O(t^3) gyro bias growth."""
        predictor = TheoreticalDriftPredictor(specs=INDUSTRIAL_IMU, g_magnitude=9.80665)
        t = np.linspace(0.0, 60.0, 601)  # 0 to 60 seconds

        acc_bias = np.array([0.01, 0.0, 0.0])      # 0.01 m/s^2
        gyro_bias = np.array([0.0, 0.001, 0.0])    # 0.001 rad/s

        pred = predictor.predict(t, acc_bias=acc_bias, gyro_bias=gyro_bias)

        # At t = 10s:
        # pos_error_acc = 0.5 * 0.01 * 10^2 = 0.5 m
        # pos_error_gyro = (1/6) * 9.80665 * 0.001 * 10^3 = 1.6344 m
        idx_10s = 100
        self.assertAlmostEqual(pred['pos_error_from_accel_bias'][idx_10s], 0.5, places=2)
        self.assertAlmostEqual(pred['pos_error_from_gyro_bias'][idx_10s], 1.6344, places=2)


if __name__ == "__main__":
    unittest.main()
