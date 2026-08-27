"""
Unit Tests for Coordinate Transformations and Rigid-Body Kinematics
"""

import unittest
import numpy as np
from src.utils.coordinate_transforms import (
    euler_to_quat,
    quat_to_euler,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_integrate,
    body_to_world,
    world_to_body,
    enu_to_ned,
    ned_to_enu,
)


class TestCoordinateTransforms(unittest.TestCase):

    def test_euler_quat_roundtrip(self):
        """Test round-trip conversion between Euler angles and Quaternions."""
        test_angles = [
            (0.0, 0.0, 0.0),
            (0.1, -0.2, 1.5),
            (-0.5, 0.3, -2.1),
            (0.0, np.pi / 4, np.pi / 2),
            (np.pi / 6, -np.pi / 3, 0.0),
        ]
        for r, p, y in test_angles:
            q = euler_to_quat(r, p, y)
            self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)
            r_rec, p_rec, y_rec = quat_to_euler(q)
            self.assertAlmostEqual(r, r_rec, places=5)
            self.assertAlmostEqual(p, p_rec, places=5)
            self.assertAlmostEqual(y, y_rec, places=5)

    def test_rot_matrix_roundtrip(self):
        """Test round-trip conversion between Quaternions and 3x3 Rotation Matrices."""
        q_orig = euler_to_quat(0.2, -0.4, 1.1)
        R = quat_to_rot_matrix(q_orig)

        # Check orthogonality: R * R^T = I and det(R) = 1
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)

        q_rec = rot_matrix_to_quat(R)
        # Quaternions q and -q represent same rotation
        dot = np.abs(np.dot(q_orig, q_rec))
        self.assertAlmostEqual(dot, 1.0, places=5)

    def test_vector_rotation(self):
        """Test rotating vectors between Body and World frames."""
        # 90 deg yaw turn (around Z axis)
        q = euler_to_quat(0.0, 0.0, np.pi / 2)
        v_body = np.array([1.0, 0.0, 0.0])  # forward along X in body frame

        # In ENU frame: 90 deg yaw turns forward X into North (Y in ENU)
        v_world = body_to_world(v_body, q)
        np.testing.assert_allclose(v_world, [0.0, 1.0, 0.0], atol=1e-6)

        # Inverse rotation world to body
        v_body_rec = world_to_body(v_world, q)
        np.testing.assert_allclose(v_body_rec, v_body, atol=1e-6)

    def test_quat_integration(self):
        """Test continuous quaternion kinematic integration."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, np.pi / 2])  # 90 deg/s around Z
        dt = 1.0

        q_next = quat_integrate(q0, omega, dt)
        r, p, y = quat_to_euler(q_next)
        self.assertAlmostEqual(y, np.pi / 2, places=5)
        self.assertAlmostEqual(r, 0.0, places=5)
        self.assertAlmostEqual(p, 0.0, places=5)

    def test_enu_ned_conversion(self):
        """Test ENU to NED and NED to ENU coordinate frame conversions."""
        enu_vec = np.array([10.0, 20.0, 5.0])  # East=10, North=20, Up=5
        ned_vec = enu_to_ned(enu_vec)          # North=20, East=10, Down=-5
        np.testing.assert_allclose(ned_vec, [20.0, 10.0, -5.0])

        enu_rec = ned_to_enu(ned_vec)
        np.testing.assert_allclose(enu_rec, enu_vec)


if __name__ == '__main__':
    unittest.main()
