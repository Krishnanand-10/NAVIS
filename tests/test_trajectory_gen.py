"""
Unit Tests for 6-DOF Motion Profiles, Sensor Noise Models, and Trajectory Generator
"""

import unittest
import numpy as np
import pandas as pd

from src.simulation.motion_profiles import (
    StationaryProfile,
    StraightLineProfile,
    FigureEightProfile,
    Helical3DProfile,
    UrbanDrivingProfile,
    PedestrianProfile,
    PlanetaryRoverProfile,
)
from src.simulation.sensor_noise import (
    IMUNoiseModel,
    MagnetometerNoiseModel,
    BarometerNoiseModel,
    OdometryNoiseModel,
    GPSNoiseModel,
    CONSUMER_IMU,
    INDUSTRIAL_IMU,
    TACTICAL_SPACE_IMU,
)
from src.simulation.trajectory_gen import TrajectoryGenerator


class TestTrajectoryGen(unittest.TestCase):

    def test_stationary_profile(self):
        """Test stationary platform properties (zero velocity, 1g specific force)."""
        prof = StationaryProfile(duration=5.0, dt=0.01)
        data = prof.generate()

        self.assertEqual(len(data['timestamps']), 500)
        np.testing.assert_allclose(data['vel'], np.zeros((500, 3)), atol=1e-10)
        np.testing.assert_allclose(data['acc'], np.zeros((500, 3)), atol=1e-10)
        # Stationary platform in ENU measures +9.80665 m/s^2 along upward Z axis in body frame
        np.testing.assert_allclose(data['specific_force'][:, 2], np.full(500, 9.80665), atol=1e-5)
        self.assertTrue(np.all(data['stance_mask']))

    def test_all_motion_profiles_finite_and_shapes(self):
        """Verify all motion profiles execute cleanly, have non-NaN values, and matching dimensions."""
        profiles = [
            StationaryProfile(duration=2.0, dt=0.01),
            StraightLineProfile(duration=2.0, dt=0.01),
            FigureEightProfile(duration=2.0, dt=0.01),
            Helical3DProfile(duration=2.0, dt=0.01),
            UrbanDrivingProfile(duration=5.0, dt=0.01),
            PedestrianProfile(duration=2.0, dt=0.01),
            PlanetaryRoverProfile(duration=2.0, dt=0.01),
        ]

        for prof in profiles:
            gt = prof.generate()
            N = len(gt['timestamps'])
            self.assertEqual(gt['pos'].shape, (N, 3))
            self.assertEqual(gt['vel'].shape, (N, 3))
            self.assertEqual(gt['acc'].shape, (N, 3))
            self.assertEqual(gt['quat'].shape, (N, 4))
            self.assertEqual(gt['omega'].shape, (N, 3))
            self.assertEqual(gt['specific_force'].shape, (N, 3))
            self.assertFalse(np.isnan(gt['pos']).any())
            self.assertFalse(np.isnan(gt['quat']).any())

    def test_pedestrian_stance_phases(self):
        """Verify pedestrian profile generates stance phases where foot velocity is near zero."""
        prof = PedestrianProfile(duration=5.0, dt=0.01)
        gt = prof.generate()
        stance_indices = np.where(gt['stance_mask'])[0]
        self.assertGreater(len(stance_indices), 0)
        # During stance, foot horizontal velocity should be zero
        stance_vel = gt['vel'][stance_indices]
        np.testing.assert_allclose(stance_vel[:, 0], 0.0, atol=1e-1)

    def test_imu_noise_models(self):
        """Verify IMU noise generation and bias drift dynamics."""
        specs = INDUSTRIAL_IMU
        noise_model = IMUNoiseModel(specs=specs, dt=0.01, seed=42)
        N = 1000
        true_acc = np.zeros((N, 3))
        true_acc[:, 2] = 9.80665
        true_omega = np.zeros((N, 3))

        res = noise_model.generate(true_acc, true_omega)
        self.assertEqual(res['acc'].shape, (N, 3))
        self.assertEqual(res['gyro'].shape, (N, 3))
        self.assertEqual(res['acc_bias'].shape, (N, 3))

        # Check noise statistics
        acc_noise_std = np.std(res['acc'] - res['acc_bias'], axis=0)
        self.assertGreater(acc_noise_std[0], 0.0)

    def test_barometer_hypsometric_formula(self):
        """Test barometric formula altitude conversion accuracy."""
        baro = BarometerNoiseModel(noise_std_hpa=0.0, seed=42)
        test_alts = np.array([0.0, 100.0, 500.0, 1000.0, 2500.0])
        pressures = baro.altitude_to_pressure(test_alts)
        # Sea level pressure should be 1013.25
        self.assertAlmostEqual(pressures[0], 1013.25, places=2)
        # Pressure must decrease monotonically with altitude
        self.assertTrue(np.all(np.diff(pressures) < 0))
        # Round trip conversion
        rec_alts = baro.pressure_to_altitude(pressures)
        np.testing.assert_allclose(rec_alts, test_alts, atol=1e-4)

    def test_gps_noise_and_outages(self):
        """Test GPS update rate and outage masking."""
        gps = GPSNoiseModel(update_rate_hz=1.0, outage_windows=[(5.0, 15.0)], seed=42)
        t = np.arange(0.0, 20.0, 0.01)  # 2000 points
        pos = np.zeros((len(t), 3))
        vel = np.zeros((len(t), 3))

        res = gps.generate(t, pos, vel)
        # Valid GPS fixes only once per second (1 Hz) and 0 during [5, 15]
        valid_times = t[res['gps_valid']]
        for vt in valid_times:
            self.assertFalse(5.0 <= vt <= 15.0, f"GPS fix occurred during outage at t={vt}")

    def test_end_to_end_trajectory_generator(self):
        """Test full TrajectoryGenerator pipeline generating unified DataFrame."""
        gen = TrajectoryGenerator(
            profile="urban_driving",
            duration=10.0,
            dt=0.01,
            imu_grade="industrial",
            gps_rate=1.0,
            gps_outages=[(3.0, 7.0)],
            seed=100
        )
        df = gen.generate()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        self.assertIn('acc_x', df.columns)
        self.assertIn('gyro_x', df.columns)
        self.assertIn('mag_x', df.columns)
        self.assertIn('baro_alt', df.columns)
        self.assertIn('gt_pos_x', df.columns)


if __name__ == '__main__':
    unittest.main()
