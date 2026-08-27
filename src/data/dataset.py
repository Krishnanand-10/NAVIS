"""
Unified Trajectory Data Container for NAVIS

Provides standardized representation for both synthetic and real-world multi-sensor sequences.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class TrajectoryData:
    """
    Standardized container for multi-sensor inertial navigation sequences.
    All angular quantities are in radians, positions/velocities in SI units (m, m/s).
    """
    timestamps: np.ndarray                    # (N,) in seconds
    acc: np.ndarray                           # (N, 3) Body frame specific force [m/s^2]
    gyro: np.ndarray                          # (N, 3) Body frame angular rates [rad/s]

    mag: Optional[np.ndarray] = None          # (N, 3) Magnetometer [uT]
    baro_alt: Optional[np.ndarray] = None     # (N,) Barometric altitude [m]
    baro_pressure: Optional[np.ndarray] = None # (N,) Atmospheric pressure [hPa]
    odometry_speed: Optional[np.ndarray] = None # (N,) Forward speed [m/s]

    gps_pos: Optional[np.ndarray] = None      # (N, 3) GPS position [m] (or NaN when unavailable)
    gps_vel: Optional[np.ndarray] = None      # (N, 3) GPS velocity [m/s]
    gps_valid: Optional[np.ndarray] = None    # (N,) boolean fix status

    # Ground Truth Data
    gt_pos: Optional[np.ndarray] = None       # (N, 3) True position in World ENU frame [m]
    gt_vel: Optional[np.ndarray] = None       # (N, 3) True velocity in World ENU frame [m/s]
    gt_quat: Optional[np.ndarray] = None      # (N, 4) True orientation quaternion [w, x, y, z]
    gt_acc: Optional[np.ndarray] = None       # (N, 3) True linear acceleration [m/s^2]
    gt_omega: Optional[np.ndarray] = None     # (N, 3) True angular velocity [rad/s]
    stance_mask: Optional[np.ndarray] = None  # (N,) boolean stance/stationary indicator

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.acc = np.asarray(self.acc, dtype=np.float64)
        self.gyro = np.asarray(self.gyro, dtype=np.float64)
        assert len(self.timestamps) == len(self.acc) == len(self.gyro), "Array length mismatch in sensor streams"

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0

    @property
    def dt(self) -> float:
        if len(self.timestamps) > 1:
            return float(np.median(np.diff(self.timestamps)))
        return 0.01

    @property
    def sampling_rate(self) -> float:
        dt = self.dt
        return 1.0 / dt if dt > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert TrajectoryData object into a pandas DataFrame."""
        N = len(self)
        d = {
            'timestamp': self.timestamps,
            'acc_x': self.acc[:, 0],
            'acc_y': self.acc[:, 1],
            'acc_z': self.acc[:, 2],
            'gyro_x': self.gyro[:, 0],
            'gyro_y': self.gyro[:, 1],
            'gyro_z': self.gyro[:, 2],
        }

        if self.mag is not None:
            d['mag_x'] = self.mag[:, 0]
            d['mag_y'] = self.mag[:, 1]
            d['mag_z'] = self.mag[:, 2]

        if self.baro_alt is not None:
            d['baro_alt'] = self.baro_alt
        if self.baro_pressure is not None:
            d['baro_pressure_hpa'] = self.baro_pressure
        if self.odometry_speed is not None:
            d['odometry_speed'] = self.odometry_speed

        if self.gps_valid is not None:
            d['gps_valid'] = self.gps_valid
        if self.gps_pos is not None:
            d['gps_pos_x'] = self.gps_pos[:, 0]
            d['gps_pos_y'] = self.gps_pos[:, 1]
            d['gps_pos_z'] = self.gps_pos[:, 2]
        if self.gps_vel is not None:
            d['gps_vel_x'] = self.gps_vel[:, 0]
            d['gps_vel_y'] = self.gps_vel[:, 1]
            d['gps_vel_z'] = self.gps_vel[:, 2]

        if self.gt_pos is not None:
            d['gt_pos_x'] = self.gt_pos[:, 0]
            d['gt_pos_y'] = self.gt_pos[:, 1]
            d['gt_pos_z'] = self.gt_pos[:, 2]
        if self.gt_vel is not None:
            d['gt_vel_x'] = self.gt_vel[:, 0]
            d['gt_vel_y'] = self.gt_vel[:, 1]
            d['gt_vel_z'] = self.gt_vel[:, 2]
        if self.gt_acc is not None:
            d['gt_acc_x'] = self.gt_acc[:, 0]
            d['gt_acc_y'] = self.gt_acc[:, 1]
            d['gt_acc_z'] = self.gt_acc[:, 2]
        if self.gt_quat is not None:
            d['gt_quat_w'] = self.gt_quat[:, 0]
            d['gt_quat_x'] = self.gt_quat[:, 1]
            d['gt_quat_y'] = self.gt_quat[:, 2]
            d['gt_quat_z'] = self.gt_quat[:, 3]
        if self.gt_omega is not None:
            d['gt_omega_x'] = self.gt_omega[:, 0]
            d['gt_omega_y'] = self.gt_omega[:, 1]
            d['gt_omega_z'] = self.gt_omega[:, 2]
        if self.stance_mask is not None:
            d['stance_mask'] = self.stance_mask

        return pd.DataFrame(d)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> 'TrajectoryData':
        """Construct TrajectoryData from a pandas DataFrame."""
        timestamps = df['timestamp'].values
        acc = df[['acc_x', 'acc_y', 'acc_z']].values
        gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

        mag = df[['mag_x', 'mag_y', 'mag_z']].values if 'mag_x' in df else None
        baro_alt = df['baro_alt'].values if 'baro_alt' in df else None
        baro_pressure = df['baro_pressure_hpa'].values if 'baro_pressure_hpa' in df else None
        odometry_speed = df['odometry_speed'].values if 'odometry_speed' in df else None

        gps_valid = df['gps_valid'].values if 'gps_valid' in df else None
        gps_pos = df[['gps_pos_x', 'gps_pos_y', 'gps_pos_z']].values if 'gps_pos_x' in df else None
        gps_vel = df[['gps_vel_x', 'gps_vel_y', 'gps_vel_z']].values if 'gps_vel_x' in df else None

        gt_pos = df[['gt_pos_x', 'gt_pos_y', 'gt_pos_z']].values if 'gt_pos_x' in df else None
        gt_vel = df[['gt_vel_x', 'gt_vel_y', 'gt_vel_z']].values if 'gt_vel_x' in df else None
        gt_acc = df[['gt_acc_x', 'gt_acc_y', 'gt_acc_z']].values if 'gt_acc_x' in df else None
        gt_quat = df[['gt_quat_w', 'gt_quat_x', 'gt_quat_y', 'gt_quat_z']].values if 'gt_quat_w' in df else None
        gt_omega = df[['gt_omega_x', 'gt_omega_y', 'gt_omega_z']].values if 'gt_omega_x' in df else None
        stance_mask = df['stance_mask'].values if 'stance_mask' in df else None

        return cls(
            timestamps=timestamps,
            acc=acc,
            gyro=gyro,
            mag=mag,
            baro_alt=baro_alt,
            baro_pressure=baro_pressure,
            odometry_speed=odometry_speed,
            gps_valid=gps_valid,
            gps_pos=gps_pos,
            gps_vel=gps_vel,
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            gt_acc=gt_acc,
            gt_quat=gt_quat,
            gt_omega=gt_omega,
            stance_mask=stance_mask,
            metadata=metadata or {}
        )

    def to_csv(self, filepath: str) -> None:
        """Save trajectory data to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> 'TrajectoryData':
        """Load trajectory data from CSV."""
        df = pd.read_csv(filepath)
        return cls.from_dataframe(df, metadata=metadata)
