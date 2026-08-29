"""
Offline Standalone HTML Report Exporter for NAVIS

Compiles a self-contained, standalone interactive HTML report with embedded
3D WebGL trajectory visualizer, error curves, telemetry replay, and benchmark
scorecards. Does not require a live backend server to view.
"""

import os
import json
import argparse
from typing import Optional

from src.dashboard.server import generate_dashboard_data


def export_static_report(output_path: str = "reports/navis_interactive_report.html",
                         profile: str = "urban_driving",
                         duration: float = 60.0,
                         imu_grade: str = "consumer",
                         gps_outage_start: float = 20.0,
                         gps_outage_end: float = 45.0,
                         seed: int = 42) -> str:
    """
    Generate and export a self-contained offline HTML report.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    # 1. Generate simulation payload
    outage = (gps_outage_start, gps_outage_end) if gps_outage_start < gps_outage_end else None
    data = generate_dashboard_data(
        profile=profile,
        duration=duration,
        dt=0.02,
        imu_grade=imu_grade,
        gps_outage=outage,
        seed=seed,
    )

    # 2. Read template HTML, CSS, JS
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()

    with open(os.path.join(static_dir, "style.css"), "r", encoding="utf-8") as f:
        css = f.read()

    with open(os.path.join(static_dir, "app.js"), "r", encoding="utf-8") as f:
        js = f.read()

    # 3. Embed pre-computed JSON dataset directly into JS script
    data_json = json.dumps(data)
    embedded_script = f"""
    <script>
        window.__PRELOADED_NAVIS_DATA__ = {data_json};
    </script>
    """

    # Inline CSS and JS
    html = html.replace('<link rel="stylesheet" href="style.css">', f'<style>\n{css}\n</style>')
    html = html.replace('<script src="app.js"></script>', f'{embedded_script}\n<script>\n{js}\n</script>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[+] Successfully exported standalone interactive report to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export standalone NAVIS interactive report.")
    parser.add_argument("--output", type=str, default="reports/navis_interactive_report.html", help="Output file path.")
    parser.add_argument("--profile", type=str, default="urban_driving", help="Motion trajectory profile.")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration [s].")
    parser.add_argument("--imu-grade", type=str, default="consumer", help="IMU grade preset.")
    parser.add_argument("--gps-outage", nargs=2, type=float, default=[20.0, 45.0], metavar=("START", "END"), help="GPS outage window.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()

    export_static_report(
        output_path=args.output,
        profile=args.profile,
        duration=args.duration,
        imu_grade=args.imu_grade,
        gps_outage_start=args.gps_outage[0],
        gps_outage_end=args.gps_outage[1],
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
