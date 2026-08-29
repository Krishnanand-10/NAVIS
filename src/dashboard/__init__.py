"""
NAVIS Dashboard Package: Interactive 3D Trajectory Visualizer & Analytics Suite
"""

from src.dashboard.server import generate_dashboard_data, launch_dashboard
from src.dashboard.export_report import export_static_report

__all__ = [
    "generate_dashboard_data",
    "launch_dashboard",
    "export_static_report",
]
