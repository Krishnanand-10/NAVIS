"""
NAVIS Simulation Package
"""
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

__all__ = [
    "MotionProfile",
    "StationaryProfile",
    "StraightLineProfile",
    "FigureEightProfile",
    "Helical3DProfile",
    "UrbanDrivingProfile",
    "PedestrianProfile",
    "PlanetaryRoverProfile",
    "IMUNoiseModel",
    "MagnetometerNoiseModel",
    "BarometerNoiseModel",
    "OdometryNoiseModel",
    "GPSNoiseModel",
    "IMUSpecs",
    "CONSUMER_IMU",
    "INDUSTRIAL_IMU",
    "TACTICAL_SPACE_IMU",
    "TrajectoryGenerator",
    "plot_trajectory_summary",
]


def __getattr__(name):
    if name in ("TrajectoryGenerator", "plot_trajectory_summary"):
        from src.simulation.trajectory_gen import TrajectoryGenerator, plot_trajectory_summary
        if name == "TrajectoryGenerator":
            return TrajectoryGenerator
        return plot_trajectory_summary
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
