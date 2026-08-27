"""
Physics-Based Sensor Noise and Degradation Models for Inertial Navigation

Simulates realistic stochastic and systematic errors for:
1. 6-DOF IMU (Accelerometer & Gyroscope):
   - Wideband Gaussian measurement noise (Angle/Velocity Random Walk)
   - In-run bias instability (Brownian motion / 1/f flicker noise)
   - Turn-on bias uncertainty
   - Scale factor error & cross-axis misalignment
   - Temperature gradient drift
2. Magnetometer:
   - Hard-iron (permanent magnet) and Soft-iron (induced magnetic distortion)
   - Earth magnetic field projection
3. Barometer / Altimeter:
   - Standard atmospheric hypsometric formula
   - Pressure noise & slow drift
4. Wheel Odometry / Stride:
   - Slip factor, scale errors, encoder tick quantization
5. GNSS / GPS:
   - Low-frequency fix rate (1-10 Hz) with HDOP/VDOP noise and simulated GPS-denied outage intervals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from src.utils.coordinate_transforms import world_to_body


@dataclass
class IMUSpecs:
    """IMU Noise and Error Specifications."""
    # Accelerometer specs
    acc_noise_density: float = 0.002        # m/s^2 / sqrt(Hz) (White noise)
    acc_random_walk: float = 0.0002         # m/s^3 / sqrt(Hz) (In-run bias instability)
    acc_turn_on_bias_std: float = 0.05      # m/s^2 (Turn-on repeatability 1-sigma)
    acc_scale_misalignment_std: float = 0.001 # 1000 ppm
    acc_temp_coeff: float = 0.001          # (m/s^2) / deg C

    # Gyroscope specs
    gyro_noise_density: float = 0.0003      # rad/s / sqrt(Hz) (Angular Random Walk)
    gyro_random_walk: float = 0.00003       # rad/s^2 / sqrt(Hz) (Rate Random Walk)
    gyro_turn_on_bias_std: float = 0.005    # rad/s (~0.3 deg/s)
    gyro_scale_misalignment_std: float = 0.001 # 1000 ppm
    gyro_temp_coeff: float = 0.0002        # (rad/s) / deg C

    # Name of specification
    name: str = "Standard"


# Standard Industry IMU Presets
CONSUMER_IMU = IMUSpecs(
    name="Consumer_Grade (e.g. MPU9250 / Smartphone)",
    acc_noise_density=0.003,
    acc_random_walk=0.0005,
    acc_turn_on_bias_std=0.1,
    acc_scale_misalignment_std=0.005,
    gyro_noise_density=0.001,
    gyro_random_walk=0.0001,
    gyro_turn_on_bias_std=0.02,
    gyro_scale_misalignment_std=0.005,
)

INDUSTRIAL_IMU = IMUSpecs(
    name="Industrial_Grade (e.g. ADIS16488 / Xsens)",
    acc_noise_density=0.0008,
    acc_random_walk=0.00008,
    acc_turn_on_bias_std=0.015,
    acc_scale_misalignment_std=0.001,
    gyro_noise_density=0.00015,
    gyro_random_walk=0.000015,
    gyro_turn_on_bias_std=0.002,
    gyro_scale_misalignment_std=0.001,
)

TACTICAL_SPACE_IMU = IMUSpecs(
    name="Tactical_Space_Grade (e.g. Honeywell RLG / FOG)",
    acc_noise_density=0.0002,
    acc_random_walk=0.00001,
    acc_turn_on_bias_std=0.002,
    acc_scale_misalignment_std=0.0002,
    gyro_noise_density=0.00003,
    gyro_random_walk=0.000002,
    gyro_turn_on_bias_std=0.0003,
    gyro_scale_misalignment_std=0.0002,
)


class IMUNoiseModel:
    """Simulates realistic 6-DOF IMU sensor errors on specific force and angular velocity."""

    def __init__(self, specs: IMUSpecs = INDUSTRIAL_IMU, dt: float = 0.01, seed: Optional[int] = None):
        self.specs = specs
        self.dt = float(dt)
        self.fs = 1.0 / self.dt
        self.rng = np.random.default_rng(seed)

    def generate(self, specific_force_true: np.ndarray, omega_true: np.ndarray, temp_profile: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Apply noise and bias degradation to ground-truth IMU streams.
        Returns:
        - 'acc': (N, 3) noisy accelerometer readings [m/s^2]
        - 'gyro': (N, 3) noisy gyroscope readings [rad/s]
        - 'acc_bias': (N, 3) true underlying accelerometer bias trajectory [m/s^2]
        - 'gyro_bias': (N, 3) true underlying gyroscope bias trajectory [rad/s]
        """
        N = len(specific_force_true)

        # 1. Turn-on constant bias
        acc_b0 = self.rng.normal(0.0, self.specs.acc_turn_on_bias_std, size=3)
        gyro_b0 = self.rng.normal(0.0, self.specs.gyro_turn_on_bias_std, size=3)

        # 2. In-run bias random walk (Brownian motion)
        # Discrete step sigma = random_walk * sqrt(dt)
        acc_rw_step = self.specs.acc_random_walk * np.sqrt(self.dt)
        gyro_rw_step = self.specs.gyro_random_walk * np.sqrt(self.dt)

        acc_rw = np.cumsum(self.rng.normal(0.0, acc_rw_step, size=(N, 3)), axis=0)
        gyro_rw = np.cumsum(self.rng.normal(0.0, gyro_rw_step, size=(N, 3)), axis=0)

        # 3. Temperature drift
        if temp_profile is None:
            # Default mild ambient temperature rise (e.g. +2 deg C over run)
            temp_delta = np.linspace(0.0, 2.0, N)[:, None]
        else:
            temp_delta = temp_profile[:, None]

        acc_temp_drift = self.specs.acc_temp_coeff * temp_delta
        gyro_temp_drift = self.specs.gyro_temp_coeff * temp_delta

        # Total dynamic bias
        acc_bias = acc_b0 + acc_rw + acc_temp_drift
        gyro_bias = gyro_b0 + gyro_rw + gyro_temp_drift

        # 4. Scale factor & Cross-axis misalignment matrix: M = I + S + E
        M_acc = np.eye(3) + self.rng.normal(0.0, self.specs.acc_scale_misalignment_std, size=(3, 3))
        M_gyro = np.eye(3) + self.rng.normal(0.0, self.specs.gyro_scale_misalignment_std, size=(3, 3))

        # 5. Wideband Gaussian measurement noise
        # Discrete white noise sigma = noise_density * sqrt(fs)
        acc_white_sigma = self.specs.acc_noise_density * np.sqrt(self.fs)
        gyro_white_sigma = self.specs.gyro_noise_density * np.sqrt(self.fs)

        acc_noise = self.rng.normal(0.0, acc_white_sigma, size=(N, 3))
        gyro_noise = self.rng.normal(0.0, gyro_white_sigma, size=(N, 3))

        # Combined signal: raw = M * true + bias + noise
        acc_meas = (specific_force_true @ M_acc.T) + acc_bias + acc_noise
        gyro_meas = (omega_true @ M_gyro.T) + gyro_bias + gyro_noise

        return {
            'acc': acc_meas,
            'gyro': gyro_meas,
            'acc_bias': acc_bias,
            'gyro_bias': gyro_bias,
        }


class MagnetometerNoiseModel:
    """Simulates 3-axis Magnetometer readings with Earth field, hard/soft iron, and noise."""

    def __init__(self,
                 mag_field_world: np.ndarray = np.array([20.0, 5.0, -40.0]),  # microteslas (uT) [North, East, Down] in NED or [East, North, Up] in ENU
                 noise_std: float = 0.5,                                       # uT white noise
                 hard_iron_offset: Optional[np.ndarray] = None,
                 soft_iron_matrix: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        self.B_world = np.asarray(mag_field_world, dtype=np.float64)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

        # Hard-iron offset (permanent magnetic distortion from vehicle chassis / electronics)
        self.hard_iron = hard_iron_offset if hard_iron_offset is not None else self.rng.uniform(-5.0, 5.0, size=3)

        # Soft-iron matrix (permeable metals distorting magnetic field lines)
        if soft_iron_matrix is not None:
            self.soft_iron = np.asarray(soft_iron_matrix, dtype=np.float64)
        else:
            self.soft_iron = np.array([
                [1.05, 0.02, -0.01],
                [0.02, 0.96,  0.03],
                [-0.01, 0.03, 1.02]
            ], dtype=np.float64)

    def generate(self, quat_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate noisy magnetometer stream.
        B_body_true = R(q)^T * B_world
        B_meas = M_soft * B_body_true + b_hard + noise
        """
        N = len(quat_true)
        # Vectorized projection from world to body frame
        B_world_rep = np.tile(self.B_world, (N, 1))
        B_body_true = world_to_body(B_world_rep, quat_true)

        noise = self.rng.normal(0.0, self.noise_std, size=(N, 3))
        B_meas = (B_body_true @ self.soft_iron.T) + self.hard_iron + noise

        return {
            'mag': B_meas,
            'mag_true': B_body_true,
            'hard_iron': self.hard_iron,
            'soft_iron': self.soft_iron,
        }


class BarometerNoiseModel:
    """
    Simulates Barometric Pressure sensor and computes Barometric Altitude.
    Hypsometric formula: P = P0 * (1 - (L*h)/T0)^(g*M / (R0*L))
    """

    def __init__(self,
                 sea_level_pressure: float = 1013.25, # hPa
                 sea_level_temp: float = 288.15,      # K (15 deg C)
                 temp_lapse_rate: float = 0.0065,     # K/m
                 noise_std_hpa: float = 0.08,         # hPa (approx 0.65m altitude noise)
                 seed: Optional[int] = None):
        self.P0 = float(sea_level_pressure)
        self.T0 = float(sea_level_temp)
        self.L = float(temp_lapse_rate)
        self.g = 9.80665
        self.M = 0.0289644
        self.R0 = 8.31432
        self.exponent = (self.g * self.M) / (self.R0 * self.L)  # ~5.25588
        self.inv_exponent = 1.0 / self.exponent

        self.noise_std_hpa = float(noise_std_hpa)
        self.rng = np.random.default_rng(seed)

    def altitude_to_pressure(self, altitude: np.ndarray) -> np.ndarray:
        """Convert true altitude (m) to barometric atmospheric pressure (hPa)."""
        altitude = np.asarray(altitude, dtype=np.float64)
        term = 1.0 - (self.L * altitude) / self.T0
        term = np.maximum(term, 1e-4)
        return self.P0 * (term ** self.exponent)

    def pressure_to_altitude(self, pressure: np.ndarray) -> np.ndarray:
        """Convert pressure (hPa) back to barometric altitude (m)."""
        pressure = np.asarray(pressure, dtype=np.float64)
        ratio = np.maximum(pressure / self.P0, 1e-4)
        return (self.T0 / self.L) * (1.0 - (ratio ** self.inv_exponent))

    def generate(self, altitude_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate barometric pressure and derived noisy altitude.
        """
        N = len(altitude_true)
        p_true = self.altitude_to_pressure(altitude_true)
        # Add white noise and slow thermal drift
        drift = np.cumsum(self.rng.normal(0.0, 0.001, size=N))
        noise = self.rng.normal(0.0, self.noise_std_hpa, size=N)
        p_meas = p_true + drift + noise

        alt_meas = self.pressure_to_altitude(p_meas)

        return {
            'baro_pressure_hpa': p_meas,
            'baro_alt': alt_meas,
            'baro_pressure_true': p_true,
        }


class OdometryNoiseModel:
    """Simulates Wheel Odometry / Forward Speed sensor with slip and quantization."""

    def __init__(self, scale_error: float = 0.01, noise_std: float = 0.05, slip_prob: float = 0.01, seed: Optional[int] = None):
        self.scale_error = float(scale_error)
        self.noise_std = float(noise_std)
        self.slip_prob = float(slip_prob)
        self.rng = np.random.default_rng(seed)

    def generate(self, true_speed: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate noisy forward speed odometry.
        """
        N = len(true_speed)
        noise = self.rng.normal(0.0, self.noise_std, size=N)
        # Scale error
        speed_meas = true_speed * (1.0 + self.scale_error) + noise

        # Inject occasional wheel slip events (speed reads higher due to wheel spin)
        slip_mask = self.rng.random(N) < self.slip_prob
        slip_multiplier = self.rng.uniform(1.2, 1.8, size=N)
        speed_meas[slip_mask] *= slip_multiplier[slip_mask]
        speed_meas = np.maximum(0.0, speed_meas)

        return {
            'odometry_speed': speed_meas,
            'slip_mask': slip_mask,
        }


class GPSNoiseModel:
    """
    Simulates GNSS / GPS fixes with:
    - Downsampled fix rate (e.g. 1 Hz, 5 Hz)
    - Horizontal and Vertical Position noise (1-2m horizontal CEP, 3-5m vertical)
    - Velocity noise (0.05 m/s)
    - Outage / Denied intervals (e.g. tunnels, jamming, indoor)
    """

    def __init__(self,
                 update_rate_hz: float = 1.0,
                 pos_horiz_std: float = 1.5,
                 pos_vert_std: float = 3.0,
                 vel_std: float = 0.05,
                 outage_windows: Optional[List[Tuple[float, float]]] = None,
                 seed: Optional[int] = None):
        self.update_rate_hz = float(update_rate_hz)
        self.pos_horiz_std = float(pos_horiz_std)
        self.pos_vert_std = float(pos_vert_std)
        self.vel_std = float(vel_std)
        self.outage_windows = outage_windows or []
        self.rng = np.random.default_rng(seed)

    def generate(self, timestamps: np.ndarray, pos_true: np.ndarray, vel_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate GPS position, velocity, and availability mask.
        """
        N = len(timestamps)
        dt = timestamps[1] - timestamps[0] if N > 1 else 0.01
        step_interval = max(1, int(round((1.0 / self.update_rate_hz) / dt)))

        gps_valid = np.zeros(N, dtype=bool)
        gps_pos = np.full((N, 3), np.nan, dtype=np.float64)
        gps_vel = np.full((N, 3), np.nan, dtype=np.float64)

        for i in range(0, N, step_interval):
            t = timestamps[i]

            # Check if within any outage window
            in_outage = False
            for (t_start, t_end) in self.outage_windows:
                if t_start <= t <= t_end:
                    in_outage = True
                    break

            if not in_outage:
                gps_valid[i] = True
                # Inject noise
                noise_x = self.rng.normal(0.0, self.pos_horiz_std)
                noise_y = self.rng.normal(0.0, self.pos_horiz_std)
                noise_z = self.rng.normal(0.0, self.pos_vert_std)
                gps_pos[i] = pos_true[i] + np.array([noise_x, noise_y, noise_z])

                vel_noise = self.rng.normal(0.0, self.vel_std, size=3)
                gps_vel[i] = vel_true[i] + vel_noise

        return {
            'gps_valid': gps_valid,
            'gps_pos': gps_pos,
            'gps_vel': gps_vel,
        }
