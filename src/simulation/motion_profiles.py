"""
Analytical Motion Profiles for 6-DOF Trajectory Generation

Provides mathematically continuous ground-truth kinematics:
- Position p(t), Velocity v(t), Acceleration a(t) in World (ENU) Frame
- Orientation q(t) [w, x, y, z] and Angular Velocity omega(t) [rad/s] in Body Frame
- Specific force f_b(t) = R(q)^T * (a_w(t) - g_w)
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np
from src.utils.coordinate_transforms import (
    euler_to_quat,
    quat_to_euler,
    quat_to_rot_matrix,
    world_to_body,
    quat_normalize,
)

# Standard Earth gravity in ENU frame (m/s^2)
GRAVITY_ENU = np.array([0.0, 0.0, -9.80665], dtype=np.float64)


class MotionProfile(ABC):
    """Abstract Base Class for analytical 6-DOF motion profiles."""

    def __init__(self, duration: float = 60.0, dt: float = 0.01, gravity: np.ndarray = GRAVITY_ENU):
        self.duration = float(duration)
        self.dt = float(dt)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.timestamps = np.arange(0.0, self.duration, self.dt)
        self.num_samples = len(self.timestamps)

    @abstractmethod
    def generate(self) -> Dict[str, np.ndarray]:
        """
        Generate ground-truth trajectory.
        Returns dictionary containing:
        - 'timestamps': (N,) array of time in seconds
        - 'pos': (N, 3) position in ENU frame [m]
        - 'vel': (N, 3) velocity in ENU frame [m/s]
        - 'acc': (N, 3) linear acceleration in ENU frame [m/s^2]
        - 'quat': (N, 4) orientation quaternion [w, x, y, z] (Body to World)
        - 'omega': (N, 3) angular velocity in Body frame [rad/s]
        - 'specific_force': (N, 3) true specific force in Body frame [m/s^2]
        - 'stance_mask': (N,) boolean mask indicating stationary/stance phases (for ZUPT)
        """
        pass

    def compute_specific_force(self, acc_world: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        Compute true specific force in body frame:
        f_b = R(q)^T * (a_w - g_w)
        """
        # (a_w - g_w)
        a_minus_g = acc_world - self.gravity  # (N, 3)
        return world_to_body(a_minus_g, quat)


class StationaryProfile(MotionProfile):
    """Stationary / Static sensor platform for calibration and baseline drift testing."""

    def __init__(self, duration: float = 30.0, dt: float = 0.01, initial_pos: Optional[np.ndarray] = None, initial_yaw: float = 0.0, **kwargs):
        super().__init__(duration=duration, dt=dt, **kwargs)
        self.initial_pos = np.zeros(3) if initial_pos is None else np.asarray(initial_pos, dtype=np.float64)
        self.initial_yaw = float(initial_yaw)

    def generate(self) -> Dict[str, np.ndarray]:
        N = self.num_samples
        pos = np.tile(self.initial_pos, (N, 1))
        vel = np.zeros((N, 3), dtype=np.float64)
        acc = np.zeros((N, 3), dtype=np.float64)

        q0 = euler_to_quat(0.0, 0.0, self.initial_yaw)
        quat = np.tile(q0, (N, 1))
        omega = np.zeros((N, 3), dtype=np.float64)
        specific_force = self.compute_specific_force(acc, quat)
        stance_mask = np.ones(N, dtype=bool)

        return {
            'timestamps': self.timestamps,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'quat': quat,
            'omega': omega,
            'specific_force': specific_force,
            'stance_mask': stance_mask,
        }


class StraightLineProfile(MotionProfile):
    """Linear motion with smooth acceleration, cruising, and deceleration phases."""

    def __init__(self, duration: float = 60.0, dt: float = 0.01, max_vel: float = 10.0, accel_time: float = 5.0, heading_rad: float = 0.0, **kwargs):
        super().__init__(duration=duration, dt=dt, **kwargs)
        self.max_vel = float(max_vel)
        self.accel_time = min(float(accel_time), self.duration / 3.0)
        self.heading = float(heading_rad)

    def generate(self) -> Dict[str, np.ndarray]:
        t = self.timestamps
        T = self.duration
        t_acc = self.accel_time
        t_dec = T - t_acc

        s_dot = np.zeros_like(t)
        s_ddot = np.zeros_like(t)
        s = np.zeros_like(t)

        # Acceleration phase (smooth half-sine jerk profile)
        acc_mask = (t < t_acc)
        s_dot[acc_mask] = self.max_vel * (1.0 - np.cos(np.pi * t[acc_mask] / t_acc)) / 2.0
        s_ddot[acc_mask] = self.max_vel * np.pi / (2.0 * t_acc) * np.sin(np.pi * t[acc_mask] / t_acc)

        # Cruise phase
        cruise_mask = (t >= t_acc) & (t < t_dec)
        s_dot[cruise_mask] = self.max_vel
        s_ddot[cruise_mask] = 0.0

        # Deceleration phase
        dec_mask = (t >= t_dec)
        t_rel = t[dec_mask] - t_dec
        s_dot[dec_mask] = self.max_vel * (1.0 + np.cos(np.pi * t_rel / t_acc)) / 2.0
        s_ddot[dec_mask] = -self.max_vel * np.pi / (2.0 * t_acc) * np.sin(np.pi * t_rel / t_acc)

        # Integrate distance
        s = np.cumsum(s_dot) * self.dt

        # Project into 2D heading (ENU: X=East, Y=North)
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)

        pos = np.zeros((self.num_samples, 3), dtype=np.float64)
        pos[:, 0] = s * sin_h  # East
        pos[:, 1] = s * cos_h  # North
        pos[:, 2] = 0.0

        vel = np.zeros((self.num_samples, 3), dtype=np.float64)
        vel[:, 0] = s_dot * sin_h
        vel[:, 1] = s_dot * cos_h

        acc = np.zeros((self.num_samples, 3), dtype=np.float64)
        acc[:, 0] = s_ddot * sin_h
        acc[:, 1] = s_ddot * cos_h

        # Orientation: constant heading (Yaw in ENU: angle from East or North. Here yaw=heading)
        yaw = np.full(self.num_samples, self.heading)
        quat = euler_to_quat(np.zeros_like(yaw), np.zeros_like(yaw), yaw)
        omega = np.zeros((self.num_samples, 3), dtype=np.float64)

        specific_force = self.compute_specific_force(acc, quat)
        stance_mask = (s_dot < 1e-3)

        return {
            'timestamps': self.timestamps,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'quat': quat,
            'omega': omega,
            'specific_force': specific_force,
            'stance_mask': stance_mask,
        }


class FigureEightProfile(MotionProfile):
    """Lemniscate (Figure-8) trajectory with smooth turning, yaw rates, and roll banking."""

    def __init__(self, duration: float = 60.0, dt: float = 0.01, length: float = 40.0, width: float = 20.0, speed: float = 3.0, **kwargs):
        super().__init__(duration=duration, dt=dt, **kwargs)
        self.length = float(length)
        self.width = float(width)
        self.speed = float(speed)

    def generate(self) -> Dict[str, np.ndarray]:
        t = self.timestamps
        # Angular frequency to complete loops
        loop_period = (2.0 * np.pi * np.sqrt(self.length**2 + self.width**2)) / (self.speed * 1.5)
        omega_t = 2.0 * np.pi / loop_period

        # Parametric Lemniscate of Gerono:
        # x(t) = a * sin(omega * t)
        # y(t) = b * sin(omega * t) * cos(omega * t) = (b/2) * sin(2 * omega * t)
        a = self.length / 2.0
        b = self.width

        pos_x = a * np.sin(omega_t * t)
        pos_y = 0.5 * b * np.sin(2.0 * omega_t * t)
        pos_z = np.zeros_like(t)

        vel_x = a * omega_t * np.cos(omega_t * t)
        vel_y = b * omega_t * np.cos(2.0 * omega_t * t)
        vel_z = np.zeros_like(t)

        acc_x = -a * (omega_t**2) * np.sin(omega_t * t)
        acc_y = -2.0 * b * (omega_t**2) * np.sin(2.0 * omega_t * t)
        acc_z = np.zeros_like(t)

        pos = np.stack([pos_x, pos_y, pos_z], axis=-1)
        vel = np.stack([vel_x, vel_y, vel_z], axis=-1)
        acc = np.stack([acc_x, acc_y, acc_z], axis=-1)

        # Yaw angle aligns with velocity vector
        yaw = np.arctan2(vel_y, vel_x)
        # Yaw rate d(yaw)/dt = (vel_x * acc_y - vel_y * acc_x) / (vel_x^2 + vel_y^2)
        v_sq = vel_x**2 + vel_y**2
        v_sq = np.where(v_sq < 1e-8, 1e-8, v_sq)
        yaw_rate = (vel_x * acc_y - vel_y * acc_x) / v_sq

        # Roll banking proportional to lateral acceleration
        # a_lat = v * yaw_rate; tan(roll) = a_lat / g
        roll = np.arctan(np.sqrt(v_sq) * yaw_rate / 9.80665)
        pitch = np.zeros_like(t)

        quat = euler_to_quat(roll, pitch, yaw)

        # Angular velocity in body frame (roll_rate, pitch_rate, yaw_rate)
        # Numerical differentiation of roll for body roll rate
        roll_rate = np.gradient(roll, self.dt)
        omega = np.stack([roll_rate, np.zeros_like(t), yaw_rate], axis=-1)

        specific_force = self.compute_specific_force(acc, quat)
        stance_mask = np.zeros(self.num_samples, dtype=bool)

        return {
            'timestamps': self.timestamps,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'quat': quat,
            'omega': omega,
            'specific_force': specific_force,
            'stance_mask': stance_mask,
        }


class Helical3DProfile(MotionProfile):
    """3D Helical / Drone trajectory with continuous ascent/descent, banking, and pitch angle."""

    def __init__(self, duration: float = 60.0, dt: float = 0.01, radius: float = 15.0, climb_rate: float = 1.0, angular_speed: float = 0.3, **kwargs):
        super().__init__(duration=duration, dt=dt, **kwargs)
        self.radius = float(radius)
        self.climb_rate = float(climb_rate)
        self.omega_s = float(angular_speed)

    def generate(self) -> Dict[str, np.ndarray]:
        t = self.timestamps
        w = self.omega_s
        R = self.radius

        pos_x = R * np.cos(w * t)
        pos_y = R * np.sin(w * t)
        pos_z = self.climb_rate * t

        vel_x = -R * w * np.sin(w * t)
        vel_y = R * w * np.cos(w * t)
        vel_z = np.full_like(t, self.climb_rate)

        acc_x = -R * (w**2) * np.cos(w * t)
        acc_y = -R * (w**2) * np.sin(w * t)
        acc_z = np.zeros_like(t)

        pos = np.stack([pos_x, pos_y, pos_z], axis=-1)
        vel = np.stack([vel_x, vel_y, vel_z], axis=-1)
        acc = np.stack([acc_x, acc_y, acc_z], axis=-1)

        yaw = np.arctan2(vel_y, vel_x)
        # Bank angle (roll) for coordinated turn
        roll = np.full_like(t, np.arctan((R * (w**2)) / 9.80665))
        # Pitch angle from climb rate
        v_horiz = R * w
        pitch = np.full_like(t, np.arctan2(self.climb_rate, v_horiz))

        quat = euler_to_quat(roll, pitch, yaw)
        omega = np.stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, w)], axis=-1)

        specific_force = self.compute_specific_force(acc, quat)
        stance_mask = np.zeros(self.num_samples, dtype=bool)

        return {
            'timestamps': self.timestamps,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'quat': quat,
            'omega': omega,
            'specific_force': specific_force,
            'stance_mask': stance_mask,
        }


class UrbanDrivingProfile(MotionProfile):
    """
    Realistic terrestrial / automotive trajectory featuring:
    - Segments of straight acceleration & cruising
    - Complete stops at intersections/traffic lights (stance phases)
    - 90-degree right and left cornering maneuvers
    """

    def __init__(self, duration: float = 120.0, dt: float = 0.01, cruise_speed: float = 12.0, **kwargs):
        super().__init__(duration=duration, dt=dt, **kwargs)
        self.cruise_speed = float(cruise_speed)

    def generate(self) -> Dict[str, np.ndarray]:
        t = self.timestamps
        N = self.num_samples
        dt = self.dt

        pos = np.zeros((N, 3), dtype=np.float64)
        vel = np.zeros((N, 3), dtype=np.float64)
        acc = np.zeros((N, 3), dtype=np.float64)
        quat = np.zeros((N, 4), dtype=np.float64)
        omega = np.zeros((N, 3), dtype=np.float64)
        stance_mask = np.zeros(N, dtype=bool)

        # Plan schedule of actions based on normalized time intervals
        # e.g., 0-15s: accel & cruise straight East
        # 15-25s: 90 deg turn North
        # 25-45s: cruise North
        # 45-55s: stop at red light (zero velocity)
        # 55-70s: accel & 90 deg turn West
        # 70-95s: cruise West
        # 95-105s: 90 deg turn South & stop
        # 105-120s: stationary

        curr_pos = np.zeros(3)
        curr_speed = 0.0
        curr_yaw = 0.0  # 0 rad = East

        for i in range(N):
            current_time = t[i]
            phase = (current_time / self.duration) * 100.0  # 0 to 100%

            target_speed = self.cruise_speed
            yaw_rate = 0.0

            if phase < 12.0:
                # Accelerate to cruise speed East
                target_speed = self.cruise_speed
                yaw_rate = 0.0
            elif 12.0 <= phase < 22.0:
                # 90 deg turn left (to North, yaw: 0 -> pi/2)
                target_speed = self.cruise_speed * 0.6
                duration_turn = 0.10 * self.duration
                yaw_rate = (np.pi / 2.0) / duration_turn
            elif 22.0 <= phase < 40.0:
                # Cruise North
                target_speed = self.cruise_speed
                yaw_rate = 0.0
            elif 40.0 <= phase < 55.0:
                # Red light stop (decelerate & stationary)
                if phase < 45.0:
                    target_speed = 0.0
                else:
                    target_speed = 0.0
                yaw_rate = 0.0
            elif 55.0 <= phase < 65.0:
                # Accelerate and 90 deg turn left (to West, yaw: pi/2 -> pi)
                target_speed = self.cruise_speed * 0.6
                duration_turn = 0.10 * self.duration
                yaw_rate = (np.pi / 2.0) / duration_turn
            elif 65.0 <= phase < 80.0:
                # Cruise West
                target_speed = self.cruise_speed
                yaw_rate = 0.0
            elif 80.0 <= phase < 90.0:
                # 90 deg turn left (to South, yaw: pi -> 3pi/2)
                target_speed = self.cruise_speed * 0.6
                duration_turn = 0.10 * self.duration
                yaw_rate = (np.pi / 2.0) / duration_turn
            else:
                # Decelerate to stop
                target_speed = 0.0
                yaw_rate = 0.0

            # Smooth speed transition with acceleration limit
            max_acc = 2.5  # m/s^2
            speed_diff = target_speed - curr_speed
            a_tangent = np.clip(speed_diff / dt, -max_acc, max_acc)
            curr_speed += a_tangent * dt
            curr_speed = max(0.0, curr_speed)

            # Update yaw
            curr_yaw += yaw_rate * dt

            # Compute velocity in ENU frame (X=East, Y=North)
            v_x = curr_speed * np.cos(curr_yaw)
            v_y = curr_speed * np.sin(curr_yaw)
            v_z = 0.0

            vel[i] = [v_x, v_y, v_z]
            curr_pos = curr_pos + vel[i] * dt
            pos[i] = curr_pos

            # Lateral acceleration from turning = v * yaw_rate
            a_lat_x = -curr_speed * yaw_rate * np.sin(curr_yaw)
            a_lat_y = curr_speed * yaw_rate * np.cos(curr_yaw)
            # Tangential acceleration
            a_tan_x = a_tangent * np.cos(curr_yaw)
            a_tan_y = a_tangent * np.sin(curr_yaw)

            acc[i] = [a_tan_x + a_lat_x, a_tan_y + a_lat_y, 0.0]

            # Orientation
            quat[i] = euler_to_quat(0.0, 0.0, curr_yaw)
            omega[i] = [0.0, 0.0, yaw_rate]

            if curr_speed < 0.05 and abs(a_tangent) < 0.1:
                stance_mask[i] = True

        specific_force = self.compute_specific_force(acc, quat)

        return {
            'timestamps': self.timestamps,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'quat': quat,
            'omega': omega,
            'specific_force': specific_force,
            'stance_mask': stance_mask,
        }


class PedestrianProfile(MotionProfile):
    """
    Biomechanical human walking motion model with:
    - Step cadence (~1.8 steps/sec)
    - Distinct stance phase (foot stationary, v=0) and swing phase (forward acceleration)
    - Vertical sinusoidal bounce (~0.05 m amplitude)
    - Lateral hip sway oscillation
    - Crucial for validating AI-ZUPT (Zero Velocity Update) algorithms
    """

    def __init__(self, duration: float = 60.0, dt: float = 0.01, step_freq: float = 1.8, step_length: float = 0.75, heading_rad: float = 0.0, **kwargs):
        super().__init__(duration=duration, dt=dt, **kwargs)
        self.step_freq = float(step_freq)
        self.step_length = float(step_length)
        self.heading = float(heading_rad)

    def generate(self) -> Dict[str, np.ndarray]:
        t = self.timestamps
        N = self.num_samples
        dt = self.dt

        step_period = 1.0 / self.step_freq
        stance_ratio = 0.45  # Foot on ground 45% of step cycle

        pos = np.zeros((N, 3), dtype=np.float64)
        vel = np.zeros((N, 3), dtype=np.float64)
        acc = np.zeros((N, 3), dtype=np.float64)
        quat = np.zeros((N, 4), dtype=np.float64)
        omega = np.zeros((N, 3), dtype=np.float64)
        stance_mask = np.zeros(N, dtype=bool)

        curr_pos = np.zeros(3)

        for i in range(N):
            time_in_step = t[i] % step_period
            norm_step_time = time_in_step / step_period

            if norm_step_time < stance_ratio:
                # Stance phase: Foot firmly planted, velocity is 0
                fwd_vel = 0.0
                fwd_acc = 0.0
                z_vel = 0.0
                z_acc = 0.0
                pitch = 0.0
                roll = 0.0
                pitch_rate = 0.0
                stance_mask[i] = True
            else:
                # Swing phase: Foot swings forward
                swing_phase = (norm_step_time - stance_ratio) / (1.0 - stance_ratio)  # 0 to 1
                # Forward velocity is a smooth sine hump with integral = step_length
                # integral of A * sin^2(pi * u) from 0 to 1 = A / 2 => A = 2 * step_length / swing_duration
                swing_duration = step_period * (1.0 - stance_ratio)
                peak_fwd_vel = (2.0 * self.step_length) / swing_duration
                fwd_vel = peak_fwd_vel * np.sin(np.pi * swing_phase)**2

                # Forward acceleration (derivative)
                fwd_acc = (peak_fwd_vel * np.pi / swing_duration) * np.sin(2.0 * np.pi * swing_phase)

                # Vertical foot lift (~5 cm)
                z_vel = (0.05 * np.pi / swing_duration) * np.cos(np.pi * swing_phase)
                z_acc = -(0.05 * (np.pi / swing_duration)**2) * np.sin(np.pi * swing_phase)

                # Foot pitch during swing (toes point up at push-off, down at heel strike)
                pitch = 0.3 * np.sin(2.0 * np.pi * swing_phase)
                pitch_rate = (0.3 * 2.0 * np.pi / swing_duration) * np.cos(2.0 * np.pi * swing_phase)
                roll = 0.05 * np.sin(np.pi * swing_phase)
                stance_mask[i] = False

            # Lateral sway (oscillates at half step frequency)
            sway_phase = 2.0 * np.pi * (self.step_freq / 2.0) * t[i]
            lat_vel = 0.05 * np.cos(sway_phase)
            lat_acc = -0.05 * (np.pi * self.step_freq) * np.sin(sway_phase)

            # Project into world ENU frame along heading
            cos_h = np.cos(self.heading)
            sin_h = np.sin(self.heading)

            v_east = fwd_vel * sin_h + lat_vel * cos_h
            v_north = fwd_vel * cos_h - lat_vel * sin_h
            vel[i] = [v_east, v_north, z_vel]

            curr_pos = curr_pos + vel[i] * dt
            pos[i] = curr_pos

            a_east = fwd_acc * sin_h + lat_acc * cos_h
            a_north = fwd_acc * cos_h - lat_acc * sin_h
            acc[i] = [a_east, a_north, z_acc]

            quat[i] = euler_to_quat(roll, pitch, self.heading)
            omega[i] = [0.0, pitch_rate, 0.0]

        specific_force = self.compute_specific_force(acc, quat)

        return {
            'timestamps': self.timestamps,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'quat': quat,
            'omega': omega,
            'specific_force': specific_force,
            'stance_mask': stance_mask,
        }


class PlanetaryRoverProfile(MotionProfile):
    """
    Planetary / Lunar Rover trajectory (relevant to ISRO Chandrayaan / Gaganyaan / planetary rovers):
    - Low gravity field (e.g. Moon g = 1.62 m/s^2 or Mars g = 3.72 m/s^2)
    - Rough uneven terrain elevation profile (craters, boulders, slopes)
    - Low-speed traversing (0.05 - 0.5 m/s)
    - Rock avoidance slalom turns & micro-vibrations
    - Intermittent wheel slip on loose regolith
    """

    def __init__(self, duration: float = 120.0, dt: float = 0.01, lunar_gravity: bool = True, base_speed: float = 0.25, **kwargs):
        g_val = -1.62 if lunar_gravity else -9.80665
        gravity = np.array([0.0, 0.0, g_val], dtype=np.float64)
        super().__init__(duration=duration, dt=dt, gravity=gravity, **kwargs)
        self.base_speed = float(base_speed)

    def generate(self) -> Dict[str, np.ndarray]:
        t = self.timestamps
        N = self.num_samples
        dt = self.dt

        pos = np.zeros((N, 3), dtype=np.float64)
        vel = np.zeros((N, 3), dtype=np.float64)
        acc = np.zeros((N, 3), dtype=np.float64)
        quat = np.zeros((N, 4), dtype=np.float64)
        omega = np.zeros((N, 3), dtype=np.float64)
        stance_mask = np.zeros(N, dtype=bool)

        curr_pos = np.zeros(3)
        curr_yaw = 0.0

        for i in range(N):
            # Slow weaving motion for boulder avoidance
            yaw_rate = 0.08 * np.sin(0.15 * t[i]) + 0.03 * np.cos(0.04 * t[i])
            curr_yaw += yaw_rate * dt

            # Terrain elevation profile: undulating craters & mounds
            # z(x, y) = 0.3 * sin(0.1 * x) * cos(0.1 * y)
            x_est = curr_pos[0]
            y_est = curr_pos[1]
            z_terrain = 0.4 * np.sin(0.1 * x_est) + 0.2 * np.cos(0.08 * y_est)

            # Terrain slope derivatives (dz/dx, dz/dy) for roll and pitch
            slope_x = 0.04 * np.cos(0.1 * x_est)
            slope_y = -0.016 * np.sin(0.08 * y_est)
            pitch = np.arctan(slope_x * np.cos(curr_yaw) + slope_y * np.sin(curr_yaw))
            roll = np.arctan(-slope_x * np.sin(curr_yaw) + slope_y * np.cos(curr_yaw))

            # Regolith micro-vibrations
            vib_z = 0.005 * np.sin(25.0 * t[i])
            z_pos = z_terrain + vib_z

            speed = self.base_speed * (1.0 + 0.1 * np.sin(0.5 * t[i]))
            v_x = speed * np.cos(curr_yaw)
            v_y = speed * np.sin(curr_yaw)
            v_z = (z_pos - curr_pos[2]) / dt if i > 0 else 0.0

            vel[i] = [v_x, v_y, v_z]
            curr_pos = np.array([curr_pos[0] + v_x * dt, curr_pos[1] + v_y * dt, z_pos])
            pos[i] = curr_pos

            # Acceleration from finite differences
            if i > 0:
                acc[i] = (vel[i] - vel[i - 1]) / dt
            else:
                acc[i] = [0.0, 0.0, 0.0]

            quat[i] = euler_to_quat(roll, pitch, curr_yaw)
            omega[i] = [np.gradient([roll, roll], dt)[0] if i == 0 else (roll - quat_to_euler(quat[i-1])[0])/dt,
                        np.gradient([pitch, pitch], dt)[0] if i == 0 else (pitch - quat_to_euler(quat[i-1])[1])/dt,
                        yaw_rate]

        specific_force = self.compute_specific_force(acc, quat)

        return {
            'timestamps': self.timestamps,
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'quat': quat,
            'omega': omega,
            'specific_force': specific_force,
            'stance_mask': stance_mask,
        }
