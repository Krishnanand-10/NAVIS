"""
Real & Synthetic Dataset Loaders for NAVIS

Parsers for:
1. Synthetic Trajectories (NAVIS Generator)
2. Oxford Inertial Odometry Dataset (OxIOD)
3. KITTI Raw IMU/GPS (oxts)
4. RoNIN / RIDI Pedestrian Inertial Datasets
5. Generic / Custom CSV Loaders (Auto-column detection)
"""

import os
import glob
from typing import Dict, Optional, List, Union, Any, Tuple
import numpy as np
import pandas as pd

from src.data.dataset import TrajectoryData
from src.simulation.trajectory_gen import TrajectoryGenerator
from src.utils.coordinate_transforms import (
    euler_to_quat,
    ned_to_enu,
    quat_normalize,
)


class SyntheticDataLoader:
    """Loader / Factory for generating on-the-fly or loading saved synthetic trajectories."""

    @staticmethod
    def create(profile: str = "urban_driving",
               duration: float = 60.0,
               dt: float = 0.01,
               imu_grade: str = "industrial",
               gps_rate: float = 1.0,
               gps_outages: Optional[List[Tuple[float, float]]] = None,
               seed: Optional[int] = 42,
               **profile_kwargs) -> TrajectoryData:
        """Generate a new synthetic TrajectoryData instance."""
        gen = TrajectoryGenerator(
            profile=profile,
            duration=duration,
            dt=dt,
            imu_grade=imu_grade,
            gps_rate=gps_rate,
            gps_outages=gps_outages,
            seed=seed,
            **profile_kwargs
        )
        df = gen.generate()
        meta = {
            "source": "synthetic",
            "profile": profile,
            "duration": duration,
            "dt": dt,
            "imu_grade": imu_grade,
        }
        return TrajectoryData.from_dataframe(df, metadata=meta)

    @staticmethod
    def load(filepath: str) -> TrajectoryData:
        """Load a saved synthetic CSV file."""
        return TrajectoryData.from_csv(filepath, metadata={"source": "synthetic_file", "filepath": filepath})


class OxIODLoader:
    """
    Loader for Oxford Inertial Odometry Dataset (OxIOD).
    Reference: 'OxIOD: The Dataset for Deep Inertial Odometry' (Chen et al.)
    Standard format:
    - imu.csv: Time, wx, wy, wz, ax, ay, az, mag_x, mag_y, mag_z
    - vi.csv: Time, Header_seq, Header_stamp, Translation_x, Translation_y, Translation_z, Rotation_x, Rotation_y, Rotation_z, Rotation_w
    """

    @staticmethod
    def load(imu_path: str, vi_path: Optional[str] = None) -> TrajectoryData:
        df_imu = pd.read_csv(imu_path)
        
        # Standardize IMU column names
        # OxIOD standard columns: Time, Angular Velocity X, Y, Z, Linear Acceleration X, Y, Z, Magnetometer X, Y, Z
        time_col = [c for c in df_imu.columns if 'time' in c.lower() or 'timestamp' in c.lower()][0]
        t = df_imu[time_col].values
        t = t - t[0]  # Normalize time to start at 0

        # Gyro
        gyro_cols = [c for c in df_imu.columns if 'gyro' in c.lower() or 'angular velocity' in c.lower() or 'w' in c.lower()]
        if len(gyro_cols) >= 3:
            gyro = df_imu[gyro_cols[:3]].values
        else:
            gyro = df_imu.iloc[:, 1:4].values

        # Accel
        acc_cols = [c for c in df_imu.columns if 'acc' in c.lower() or 'linear acceleration' in c.lower()]
        if len(acc_cols) >= 3:
            acc = df_imu[acc_cols[:3]].values
        else:
            acc = df_imu.iloc[:, 4:7].values

        # Magnetometer if available
        mag = None
        mag_cols = [c for c in df_imu.columns if 'mag' in c.lower()]
        if len(mag_cols) >= 3:
            mag = df_imu[mag_cols[:3]].values

        # Ground truth from Vicon if provided
        gt_pos = None
        gt_quat = None
        gt_vel = None

        if vi_path and os.path.exists(vi_path):
            df_vi = pd.read_csv(vi_path)
            t_vi = df_vi.iloc[:, 0].values
            t_vi = t_vi - t_vi[0]
            pos_vi = df_vi.iloc[:, 3:6].values
            quat_vi = df_vi.iloc[:, [9, 6, 7, 8]].values  # reorder to [w, x, y, z]

            # Interpolate Ground Truth to match IMU timestamps
            from scipy.interpolate import interp1d
            interp_pos = interp1d(t_vi, pos_vi, axis=0, fill_value="extrapolate")
            gt_pos = interp_pos(t)

            interp_quat = interp1d(t_vi, quat_vi, axis=0, fill_value="extrapolate")
            gt_quat = quat_normalize(interp_quat(t))

            # Numerical velocity
            dt_step = np.gradient(t)
            gt_vel = np.gradient(gt_pos, axis=0) / dt_step[:, None]

        return TrajectoryData(
            timestamps=t,
            acc=acc,
            gyro=gyro,
            mag=mag,
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            gt_quat=gt_quat,
            metadata={"dataset": "OxIOD", "imu_path": imu_path}
        )


class KITTILoader:
    """
    Loader for KITTI Raw / Odometry (OXTS IMU-GPS packets).
    OXTS format contains 30 columns:
    lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ax, ay, az, af, al, au, wx, wy, wz, pos_accuracy, vel_accuracy, ...
    """

    @staticmethod
    def load_oxts_directory(oxts_dir: str, timestamps_file: Optional[str] = None) -> TrajectoryData:
        files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))
        if not files:
            raise FileNotFoundError(f"No OXTS files found in {oxts_dir}")

        records = []
        for f in files:
            with open(f, 'r') as fp:
                line = fp.readline().strip()
                if line:
                    records.append([float(x) for x in line.split()])

        data = np.array(records)
        N = len(data)

        # Timestamps
        if timestamps_file and os.path.exists(timestamps_file):
            t_df = pd.read_csv(timestamps_file, header=None)
            # Parse datetime or float timestamps
            t_vals = pd.to_datetime(t_df[0]).astype('int64') / 1e9
            timestamps = (t_vals - t_vals.iloc[0]).values
        else:
            timestamps = np.arange(N) * 0.1  # KITTI default 10 Hz

        # In OXTS:
        # ax, ay, az are cols 11, 12, 13 (forward, left, up specific force)
        # wx, wy, wz are cols 17, 18, 19 (angular rates around x, y, z)
        acc = data[:, 11:14]
        gyro = data[:, 17:20]

        # Orientation: roll (col 3), pitch (col 4), yaw (col 5)
        roll = data[:, 3]
        pitch = data[:, 4]
        yaw = data[:, 5]
        quat = euler_to_quat(roll, pitch, yaw)

        # Velocities: ve (East, col 7), vn (North, col 6), vu (Up, col 10)
        gt_vel = np.stack([data[:, 7], data[:, 6], data[:, 10]], axis=-1)

        # Integrate velocity for relative position in ENU frame
        dt_vals = np.gradient(timestamps)
        gt_pos = np.cumsum(gt_vel * dt_vals[:, None], axis=0)

        # Forward speed odometry
        vf = data[:, 8]

        return TrajectoryData(
            timestamps=timestamps,
            acc=acc,
            gyro=gyro,
            odometry_speed=vf,
            gt_pos=gt_pos,
            gt_vel=gt_vel,
            gt_quat=quat,
            metadata={"dataset": "KITTI_OXTS", "dir": oxts_dir}
        )


class RoNINLoader:
    """
    Loader for RoNIN (Robust Neural Inertial Navigation) dataset sequences.
    Standard fields: gyro, acc (uncalibrated or calibrated), ground truth position.
    """

    @staticmethod
    def load(csv_or_h5_path: str) -> TrajectoryData:
        df = pd.read_csv(csv_or_h5_path)
        
        time_col = [c for c in df.columns if 'time' in c.lower()][0]
        timestamps = df[time_col].values
        timestamps = timestamps - timestamps[0]

        acc_cols = [c for c in df.columns if 'acc' in c.lower()][:3]
        gyro_cols = [c for c in df.columns if 'gyro' in c.lower()][:3]

        acc = df[acc_cols].values
        gyro = df[gyro_cols].values

        gt_pos = None
        pos_cols = [c for c in df.columns if 'pos' in c.lower() or 'gt_p' in c.lower()][:3]
        if len(pos_cols) >= 3:
            gt_pos = df[pos_cols].values

        return TrajectoryData(
            timestamps=timestamps,
            acc=acc,
            gyro=gyro,
            gt_pos=gt_pos,
            metadata={"dataset": "RoNIN", "path": csv_or_h5_path}
        )


class GenericCSVLoader:
    """
    Flexible CSV Loader with automated column mapping or user-supplied dictionary.
    Supports smartphone sensor loggers (e.g., Sensor Logger App, Physics Toolbox, Pixhawk logs).
    """

    DEFAULT_COLUMN_MAPPINGS = {
        'timestamp': ['timestamp', 'time', 't', 'time_s', 'epoch'],
        'acc_x': ['acc_x', 'accel_x', 'ax', 'linear_acceleration_x', 'acceleration_x'],
        'acc_y': ['acc_y', 'accel_y', 'ay', 'linear_acceleration_y', 'acceleration_y'],
        'acc_z': ['acc_z', 'accel_z', 'az', 'linear_acceleration_z', 'acceleration_z'],
        'gyro_x': ['gyro_x', 'angular_velocity_x', 'wx', 'gx', 'gyr_x', 'rotation_rate_x'],
        'gyro_y': ['gyro_y', 'angular_velocity_y', 'wy', 'gy', 'gyr_y', 'rotation_rate_y'],
        'gyro_z': ['gyro_z', 'angular_velocity_z', 'wz', 'gz', 'gyr_z', 'rotation_rate_z'],
        'mag_x': ['mag_x', 'magnetometer_x', 'mx', 'b_x'],
        'mag_y': ['mag_y', 'magnetometer_y', 'my', 'b_y'],
        'mag_z': ['mag_z', 'magnetometer_z', 'mz', 'b_z'],
        'baro_alt': ['baro_alt', 'altitude', 'baro_height', 'alt'],
        'odometry_speed': ['odometry_speed', 'speed', 'velocity', 'wheel_speed', 'v_fwd'],
        'gt_pos_x': ['gt_pos_x', 'gt_x', 'pos_x', 'true_x'],
        'gt_pos_y': ['gt_pos_y', 'gt_y', 'pos_y', 'true_y'],
        'gt_pos_z': ['gt_pos_z', 'gt_z', 'pos_z', 'true_z'],
    }

    @classmethod
    def load(cls, filepath: str, column_mapping: Optional[Dict[str, str]] = None) -> TrajectoryData:
        """
        Load any CSV file and automatically map headers to standard NAVIS TrajectoryData format.
        """
        df = pd.read_csv(filepath)
        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        mapping = {}
        target_map = column_mapping or {}

        # Resolve column names
        for standard_key, aliases in cls.DEFAULT_COLUMN_MAPPINGS.items():
            if standard_key in target_map:
                user_col = target_map[standard_key]
                if user_col in df.columns:
                    mapping[standard_key] = user_col
            else:
                for alias in aliases:
                    if alias in cols_lower:
                        mapping[standard_key] = cols_lower[alias]
                        break

        if 'timestamp' not in mapping or 'acc_x' not in mapping or 'gyro_x' not in mapping:
            raise ValueError(f"CSV file '{filepath}' is missing essential IMU columns (timestamp, acc_x, gyro_x). Available columns: {list(df.columns)}")

        # Build TrajectoryData
        t = df[mapping['timestamp']].values
        if np.issubdtype(t.dtype, np.datetime64) or isinstance(t[0], str):
            t = (pd.to_datetime(t).astype('int64') / 1e9).values
        t = t - t[0]

        acc = df[[mapping['acc_x'], mapping['acc_y'], mapping['acc_z']]].values
        gyro = df[[mapping['gyro_x'], mapping['gyro_y'], mapping['gyro_z']]].values

        mag = None
        if all(k in mapping for k in ['mag_x', 'mag_y', 'mag_z']):
            mag = df[[mapping['mag_x'], mapping['mag_y'], mapping['mag_z']]].values

        baro_alt = df[mapping['baro_alt']].values if 'baro_alt' in mapping else None
        odometry_speed = df[mapping['odometry_speed']].values if 'odometry_speed' in mapping else None

        gt_pos = None
        if all(k in mapping for k in ['gt_pos_x', 'gt_pos_y', 'gt_pos_z']):
            gt_pos = df[[mapping['gt_pos_x'], mapping['gt_pos_y'], mapping['gt_pos_z']]].values

        return TrajectoryData(
            timestamps=t,
            acc=acc,
            gyro=gyro,
            mag=mag,
            baro_alt=baro_alt,
            odometry_speed=odometry_speed,
            gt_pos=gt_pos,
            metadata={"source": "generic_csv", "filepath": filepath}
        )
