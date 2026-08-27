"""
Unified Trajectory Generator Engine for NAVIS

Orchestrates analytical kinematics, sensor noise models (6-DOF IMU, Mag, Baro, Odo, GPS),
and generates benchmark datasets for Dead Reckoning and Sensor Fusion.
"""

import os
import argparse
from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulation.motion_profiles import (
    MotionProfile,
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
    IMUSpecs,
    CONSUMER_IMU,
    INDUSTRIAL_IMU,
    TACTICAL_SPACE_IMU,
)

PROFILE_REGISTRY = {
    "stationary": StationaryProfile,
    "straight": StraightLineProfile,
    "figure_eight": FigureEightProfile,
    "helical_3d": Helical3DProfile,
    "urban_driving": UrbanDrivingProfile,
    "pedestrian": PedestrianProfile,
    "planetary_rover": PlanetaryRoverProfile,
}

IMU_GRADE_REGISTRY = {
    "consumer": CONSUMER_IMU,
    "industrial": INDUSTRIAL_IMU,
    "tactical": TACTICAL_SPACE_IMU,
    "space": TACTICAL_SPACE_IMU,
}


class TrajectoryGenerator:
    """End-to-end 6-DOF Trajectory & Multi-Sensor Stream Simulator."""

    def __init__(self,
                 profile: str = "urban_driving",
                 duration: float = 60.0,
                 dt: float = 0.01,
                 imu_grade: str = "industrial",
                 gps_rate: float = 1.0,
                 gps_outages: Optional[List[Tuple[float, float]]] = None,
                 seed: Optional[int] = 42,
                 **profile_kwargs):
        self.profile_name = profile.lower()
        if self.profile_name not in PROFILE_REGISTRY:
            raise ValueError(f"Unknown profile: {profile}. Available: {list(PROFILE_REGISTRY.keys())}")

        self.duration = duration
        self.dt = dt
        self.seed = seed

        # Instantiate motion profile
        profile_cls = PROFILE_REGISTRY[self.profile_name]
        self.motion_profile: MotionProfile = profile_cls(duration=duration, dt=dt, **profile_kwargs)

        # Instantiate sensor noise models
        self.imu_specs = IMU_GRADE_REGISTRY.get(imu_grade.lower(), INDUSTRIAL_IMU)
        self.imu_noise = IMUNoiseModel(specs=self.imu_specs, dt=dt, seed=seed)
        self.mag_noise = MagnetometerNoiseModel(seed=seed + 1 if seed else None)
        self.baro_noise = BarometerNoiseModel(seed=seed + 2 if seed else None)
        self.odo_noise = OdometryNoiseModel(seed=seed + 3 if seed else None)
        self.gps_noise = GPSNoiseModel(update_rate_hz=gps_rate, outage_windows=gps_outages, seed=seed + 4 if seed else None)

    def generate(self) -> pd.DataFrame:
        """
        Generate complete trajectory dataframe containing ground truth and noisy sensor streams.
        """
        # 1. Ground truth kinematics
        gt = self.motion_profile.generate()
        t = gt['timestamps']
        N = len(t)

        # 2. Noisy 6-DOF IMU
        imu_data = self.imu_noise.generate(gt['specific_force'], gt['omega'])

        # 3. Magnetometer
        mag_data = self.mag_noise.generate(gt['quat'])

        # 4. Barometer (altitude is z position in ENU)
        baro_data = self.baro_noise.generate(gt['pos'][:, 2])

        # 5. Odometry (true horizontal or total speed)
        speed_true = np.linalg.norm(gt['vel'], axis=-1)
        odo_data = self.odo_noise.generate(speed_true)

        # 6. GPS
        gps_data = self.gps_noise.generate(t, gt['pos'], gt['vel'])

        # Assemble unified DataFrame
        df_dict = {
            'timestamp': t,
            # Noisy IMU Measurements (Body Frame)
            'acc_x': imu_data['acc'][:, 0],
            'acc_y': imu_data['acc'][:, 1],
            'acc_z': imu_data['acc'][:, 2],
            'gyro_x': imu_data['gyro'][:, 0],
            'gyro_y': imu_data['gyro'][:, 1],
            'gyro_z': imu_data['gyro'][:, 2],
            # Magnetometer (Body Frame, uT)
            'mag_x': mag_data['mag'][:, 0],
            'mag_y': mag_data['mag'][:, 1],
            'mag_z': mag_data['mag'][:, 2],
            # Barometer
            'baro_pressure_hpa': baro_data['baro_pressure_hpa'],
            'baro_alt': baro_data['baro_alt'],
            # Wheel Odometry
            'odometry_speed': odo_data['odometry_speed'],
            # GPS / GNSS
            'gps_valid': gps_data['gps_valid'],
            'gps_pos_x': gps_data['gps_pos'][:, 0],
            'gps_pos_y': gps_data['gps_pos'][:, 1],
            'gps_pos_z': gps_data['gps_pos'][:, 2],
            'gps_vel_x': gps_data['gps_vel'][:, 0],
            'gps_vel_y': gps_data['gps_vel'][:, 1],
            'gps_vel_z': gps_data['gps_vel'][:, 2],
            # Ground Truth States (World ENU Frame)
            'gt_pos_x': gt['pos'][:, 0],
            'gt_pos_y': gt['pos'][:, 1],
            'gt_pos_z': gt['pos'][:, 2],
            'gt_vel_x': gt['vel'][:, 0],
            'gt_vel_y': gt['vel'][:, 1],
            'gt_vel_z': gt['vel'][:, 2],
            'gt_acc_x': gt['acc'][:, 0],
            'gt_acc_y': gt['acc'][:, 1],
            'gt_acc_z': gt['acc'][:, 2],
            'gt_quat_w': gt['quat'][:, 0],
            'gt_quat_x': gt['quat'][:, 1],
            'gt_quat_y': gt['quat'][:, 2],
            'gt_quat_z': gt['quat'][:, 3],
            'gt_omega_x': gt['omega'][:, 0],
            'gt_omega_y': gt['omega'][:, 1],
            'gt_omega_z': gt['omega'][:, 2],
            'gt_acc_bias_x': imu_data['acc_bias'][:, 0],
            'gt_acc_bias_y': imu_data['acc_bias'][:, 1],
            'gt_acc_bias_z': imu_data['acc_bias'][:, 2],
            'gt_gyro_bias_x': imu_data['gyro_bias'][:, 0],
            'gt_gyro_bias_y': imu_data['gyro_bias'][:, 1],
            'gt_gyro_bias_z': imu_data['gyro_bias'][:, 2],
            'stance_mask': gt['stance_mask'],
        }

        return pd.DataFrame(df_dict)


def plot_trajectory_summary(df: pd.DataFrame, title: str = "Trajectory Summary", save_path: Optional[str] = None):
    """Generate multi-panel overview plot of trajectory and sensor streams."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. 2D / 3D Position Plot
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(df['gt_pos_x'], df['gt_pos_y'], label='Ground Truth', color='blue', linewidth=2)
    valid_gps = df[df['gps_valid']]
    if len(valid_gps) > 0:
        ax1.scatter(valid_gps['gps_pos_x'], valid_gps['gps_pos_y'], label='GPS Fixes', color='red', s=15, alpha=0.7)
    ax1.scatter(df['gt_pos_x'].iloc[0], df['gt_pos_y'].iloc[0], color='green', marker='o', s=60, label='Start')
    ax1.scatter(df['gt_pos_x'].iloc[-1], df['gt_pos_y'].iloc[-1], color='black', marker='x', s=60, label='End')
    ax1.set_title('2D Trajectory (East-North)')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best')

    # 2. 3D Elevation / Altitude
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(df['timestamp'], df['gt_pos_z'], label='True Altitude', color='blue', linewidth=1.5)
    ax2.plot(df['timestamp'], df['baro_alt'], label='Barometer Altitude', color='orange', alpha=0.6, linewidth=1)
    ax2.set_title('Altitude / Elevation Profile')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Altitude [m]')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best')

    # 3. Specific Force / Accelerometer Readings
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(df['timestamp'], df['acc_x'], label='Acc X', alpha=0.7, linewidth=0.8)
    ax3.plot(df['timestamp'], df['acc_y'], label='Acc Y', alpha=0.7, linewidth=0.8)
    ax3.plot(df['timestamp'], df['acc_z'], label='Acc Z', alpha=0.7, linewidth=0.8)
    ax3.set_title('Raw Accelerometer Readings (Body Frame)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Specific Force [m/s²]')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right')

    # 4. Gyroscope Readings
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(df['timestamp'], np.rad2deg(df['gyro_x']), label='Gyro X', alpha=0.7, linewidth=0.8)
    ax4.plot(df['timestamp'], np.rad2deg(df['gyro_y']), label='Gyro Y', alpha=0.7, linewidth=0.8)
    ax4.plot(df['timestamp'], np.rad2deg(df['gyro_z']), label='Gyro Z', alpha=0.7, linewidth=0.8)
    ax4.set_title('Raw Gyroscope Readings (Body Frame)')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angular Rate [deg/s]')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='upper right')

    # 5. Velocity & Stance Phases
    ax5 = fig.add_subplot(2, 3, 5)
    speed = np.sqrt(df['gt_vel_x']**2 + df['gt_vel_y']**2 + df['gt_vel_z']**2)
    ax5.plot(df['timestamp'], speed, label='True Speed', color='purple', linewidth=1.5)
    ax5.plot(df['timestamp'], df['odometry_speed'], label='Odometry Speed', color='green', alpha=0.5, linewidth=1)
    if 'stance_mask' in df:
        stance_times = df['timestamp'][df['stance_mask']]
        if len(stance_times) > 0:
            ax5.scatter(stance_times, np.zeros_like(stance_times), color='red', marker='|', s=20, label='Stance/ZUPT')
    ax5.set_title('Speed & Zero-Velocity Phases')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Speed [m/s]')
    ax5.grid(True, linestyle='--', alpha=0.6)
    ax5.legend(loc='best')

    # 6. Sensor Bias Drift
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(df['timestamp'], df['gt_acc_bias_x'], label='Acc Bias X', linestyle='--')
    ax6.plot(df['timestamp'], df['gt_acc_bias_y'], label='Acc Bias Y', linestyle='--')
    ax6.plot(df['timestamp'], df['gt_acc_bias_z'], label='Acc Bias Z', linestyle='--')
    ax6.set_title('In-Run Accelerometer Bias Drift')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Bias [m/s²]')
    ax6.grid(True, linestyle='--', alpha=0.6)
    ax6.legend(loc='best')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="NAVIS 6-DOF Synthetic Trajectory Generator")
    parser.add_argument("--profile", type=str, default="urban_driving", choices=list(PROFILE_REGISTRY.keys()),
                        help="Motion profile type")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (seconds), e.g. 0.01 for 100 Hz")
    parser.add_argument("--imu", type=str, default="industrial", choices=list(IMU_GRADE_REGISTRY.keys()),
                        help="IMU sensor noise grade")
    parser.add_argument("--gps_rate", type=float, default=1.0, help="GPS update frequency in Hz")
    parser.add_argument("--gps_outage", type=str, default="20,40", help="Comma-separated GPS outage start and end times in sec (e.g. 20,40)")
    parser.add_argument("--out", type=str, default="data/processed/trajectory_sample.csv", help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Generate and save preview plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatability")

    args = parser.parse_args()

    gps_outages = []
    if args.gps_outage:
        try:
            parts = [float(x.strip()) for x in args.gps_outage.split(",") if x.strip()]
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    gps_outages.append((parts[i], parts[i+1]))
        except Exception as e:
            print(f"Warning: Could not parse gps_outage '{args.gps_outage}': {e}")

    print(f" Generating '{args.profile}' trajectory ({args.duration}s @ {1.0/args.dt:.0f} Hz, IMU: {args.imu})...")
    gen = TrajectoryGenerator(
        profile=args.profile,
        duration=args.duration,
        dt=args.dt,
        imu_grade=args.imu,
        gps_rate=args.gps_rate,
        gps_outages=gps_outages,
        seed=args.seed
    )

    df = gen.generate()

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f" Successfully saved {len(df)} samples to {args.out}")

    if args.plot:
        plot_out = os.path.splitext(args.out)[0] + ".png"
        plot_trajectory_summary(df, title=f"NAVIS Trajectory: {args.profile.upper()} ({args.imu})", save_path=plot_out)
        print(f" Preview plot saved to {plot_out}")


if __name__ == "__main__":
    main()
