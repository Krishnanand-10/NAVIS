"""
Unit Tests for Inertial Kinematics and Strapdown Mechanization
"""

import unittest
import numpy as np

from src.core.kinematics import (
    integrate_attitude_quaternion_rk4,
    integrate_attitude_quaternion_midpoint,
    integrate_attitude_quaternion_exponential,
    integrate_attitude_quaternion_euler,
    integrate_attitude_euler_angles,
    specific_force_to_world_acc,
    integrate_translation_euler,
    integrate_translation_trapezoidal,
    integrate_translation_verlet,
    GravityModel,
    STANDARD_GRAVITY_ENU,
    LUNAR_GRAVITY_ENU,
)
from src.utils.coordinate_transforms import (
    euler_to_quat,
    quat_to_euler,
    quat_normalize,
)


class TestKinematics(unittest.TestCase):

    def test_gravity_compensation_stationary(self):
        """Stationary sensor sitting flat on ground measures upward normal reaction [0, 0, 9.80665]."""
        f_body = np.array([0.0, 0.0, 9.80665])
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])  # Flat (roll=0, pitch=0, yaw=0)

        a_world = specific_force_to_world_acc(f_body, q_identity, STANDARD_GRAVITY_ENU)
        # Should be exactly zero acceleration
        np.testing.assert_allclose(a_world, np.zeros(3), atol=1e-5)

    def test_gravity_compensation_tilted(self):
        """Stationary sensor pitched 90 degrees up in ENU."""
        # In ENU body frame under Z-Y-X Euler pitching +90 deg: body +X is aligned with -Z, so reaction force is along -X
        q_pitched = euler_to_quat(0.0, np.deg2rad(90.0), 0.0)
        f_body = np.array([-9.80665, 0.0, 0.0])

        a_world = specific_force_to_world_acc(f_body, q_pitched, STANDARD_GRAVITY_ENU)
        np.testing.assert_allclose(a_world, np.zeros(3), atol=1e-4)

    def test_quaternion_integration_constant_rate(self):
        """Integrating constant yaw rate 1 rad/s for pi seconds should rotate exactly 180 degrees."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s around Z
        dt = 0.01
        steps = int(np.pi / dt)

        # 1. Exponential integration
        q_exp = q0.copy()
        for _ in range(steps):
            q_exp = integrate_attitude_quaternion_exponential(q_exp, omega, dt)
        r, p, y = quat_to_euler(q_exp)
        self.assertAlmostEqual(abs(y), np.pi, places=2)

        # 2. RK4 integration
        q_rk4 = q0.copy()
        for _ in range(steps):
            q_rk4 = integrate_attitude_quaternion_rk4(q_rk4, omega, omega, dt)
        r, p, y = quat_to_euler(q_rk4)
        self.assertAlmostEqual(abs(y), np.pi, places=2)

    def test_quaternion_methods_consistency(self):
        """Compare RK4, midpoint, exponential on smooth dynamic motion."""
        q0 = euler_to_quat(0.1, -0.2, 0.5)
        w0 = np.array([0.1, 0.2, -0.3])
        w1 = np.array([0.12, 0.18, -0.28])
        dt = 0.01

        q_rk4 = integrate_attitude_quaternion_rk4(q0, w0, w1, dt)
        q_mid = integrate_attitude_quaternion_midpoint(q0, w0, w1, dt)

        # RK4 and midpoint should be in extremely close agreement for small dt
        np.testing.assert_allclose(q_rk4, q_mid, atol=1e-5)

    def test_euler_angle_kinematics(self):
        """Test Euler angle direct kinematic differential equation integration."""
        euler0 = np.array([0.0, 0.0, 0.0])
        omega = np.array([0.5, 0.0, 0.0])  # Pure roll rate
        dt = 0.01

        euler_next = integrate_attitude_euler_angles(euler0, omega, dt)
        self.assertAlmostEqual(euler_next[0], 0.005, places=5)
        self.assertAlmostEqual(euler_next[1], 0.0, places=5)
        self.assertAlmostEqual(euler_next[2], 0.0, places=5)

    def test_translation_constant_acceleration(self):
        """Constant acceleration a=2 m/s^2 for 5 seconds: v = a*t = 10, p = 0.5*a*t^2 = 25."""
        p0 = np.zeros(3)
        v0 = np.zeros(3)
        a = np.array([2.0, 0.0, 0.0])
        dt = 0.001
        steps = 5000

        p = p0.copy()
        v = v0.copy()
        for _ in range(steps):
            p, v = integrate_translation_trapezoidal(p, v, a, a, dt)

        self.assertAlmostEqual(v[0], 10.0, places=2)
        self.assertAlmostEqual(p[0], 25.0, places=2)

    def test_velocity_verlet_translation(self):
        """Test Velocity-Verlet symplectic integrator under constant acceleration."""
        p0 = np.zeros(3)
        v0 = np.zeros(3)
        a = np.array([0.0, 4.0, 0.0])
        dt = 0.01
        steps = 100

        p = p0.copy()
        v = v0.copy()
        for _ in range(steps):
            p, v = integrate_translation_verlet(p, v, a, a, dt)

        # t = 1.0s, v = 4.0, p = 0.5 * 4 * 1^2 = 2.0
        self.assertAlmostEqual(v[1], 4.0, places=3)
        self.assertAlmostEqual(p[1], 2.0, places=3)

    def test_gravity_models(self):
        """Verify WGS-84 Somigliana and Lunar gravity models."""
        wgs = GravityModel("wgs84", latitude_deg=0.0, altitude_m=0.0)
        # Equatorial gravity approx 9.7803 m/s^2
        self.assertAlmostEqual(wgs.magnitude, 9.7803, places=2)

        lunar = GravityModel("lunar")
        self.assertAlmostEqual(lunar.magnitude, 1.625, places=3)
        np.testing.assert_allclose(lunar.vector, LUNAR_GRAVITY_ENU)


if __name__ == "__main__":
    unittest.main()
