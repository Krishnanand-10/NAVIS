"""
NAVIS Interactive 3D Trajectory & Drift Comparison Dashboard Server

Provides a high-performance REST API and static web server to power the
real-time 3D navigation dashboard:
- Synchronized multi-engine simulation runner (Modules 1, 2, 3, 4, 5)
- Automated metric scorecards & drift error comparisons
- Zero-dependency Python http.server backend with JSON API endpoints
- Static file serving (HTML, CSS, JS, Plotly/WebGL)
"""

import os
import json
import argparse
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional, Tuple, Union, List, Any
import numpy as np

from src.data.loaders import SyntheticDataLoader
from src.core.traditional_dr import DeadReckoningEngine
from src.core.zupt_dr import ZUPTDeadReckoningEngine
from src.core.neural_dr import run_neural_dead_reckoning
from src.core.ai_ekf_engine import run_ai_ekf_navigation
from src.models.drift_estimator import InertialDriftEstimator


# ============================================================================
# 1. MULTI-ENGINE SIMULATION PIPELINE & DATA GENERATOR
# ============================================================================

def generate_dashboard_data(
    profile: str = "urban_driving",
    duration: float = 60.0,
    dt: float = 0.02,
    imu_grade: str = "consumer",
    gps_outage: Optional[Tuple[float, float]] = (20.0, 45.0),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute full synchronized multi-engine navigation simulation across all 5 NAVIS modules.
    
    Returns:
        Structured JSON-serializable dictionary containing:
        - Timestamps & sensor telemetry (IMU, GPS validity, Baro)
        - 3D trajectories: Ground Truth, Classical Dead Reckoning, AI-ZUPT, Neural DR, AI-EKF
        - Attitude Euler angles (roll, pitch, yaw) & velocity profiles
        - Online estimated biases & +-3 sigma covariance uncertainty envelopes
        - Multi-engine comparative benchmark scorecards (ATE RMSE, Final Drift, % Reduction)
    """
    outage_list = [gps_outage] if gps_outage and gps_outage[0] < gps_outage[1] else None

    # 1. Module 1: Multi-Sensor Data Stream Generation
    traj = SyntheticDataLoader.create(
        profile=profile,
        duration=duration,
        dt=dt,
        imu_grade=imu_grade,
        gps_outages=outage_list,
        seed=seed,
    )
    t = traj.timestamps
    N = len(t)

    # 2. Module 2: Classical Dead Reckoning Baseline
    std_engine = DeadReckoningEngine(attitude_method="quaternion_rk4", translation_method="trapezoidal")
    mod2_result = std_engine.process_trajectory(traj)

    # 3. Module 3: ZUPT-Augmented Dead Reckoning
    zupt_engine = ZUPTDeadReckoningEngine(detection_method="glrt")
    mod3_result = zupt_engine.process_trajectory(traj)

    # 4. Module 4: Deep Neural Drift & Bias Estimator
    drift_est = InertialDriftEstimator(model_type="numpy", window_size=30)
    mod4_result = run_neural_dead_reckoning(traj, drift_estimator=drift_est, correction_mode="hybrid")

    # 5. Module 5: AI-Enhanced Extended Kalman Filter (AI-EKF)
    mod5_result = run_ai_ekf_navigation(traj)

    # Extract Ground Truth
    gt_pos = traj.gt_pos if traj.gt_pos is not None else np.zeros((N, 3))
    gt_vel = traj.gt_vel if traj.gt_vel is not None else np.zeros((N, 3))

    # Downsample factor for fluid browser rendering if sequence is long
    step = 1 if N <= 1500 else int(np.ceil(N / 1500))
    idx = slice(0, N, step)

    # Assemble Comparative Benchmark Scorecards
    def extract_score(name: str, metrics: Any, color: str) -> Dict[str, Any]:
        if metrics is None:
            return {"name": name, "ate_rmse": 0.0, "final_err": 0.0, "drift_pct": 0.0, "reduction_pct": 0.0, "color": color}
        return {
            "name": name,
            "ate_rmse": float(round(metrics.ate_rmse, 3)),
            "final_err": float(round(metrics.final_pos_error, 3)),
            "drift_pct": float(round(metrics.drift_percentage, 2)),
            "drift_rate_mps": float(round(metrics.drift_rate_mps, 3)),
            "color": color,
        }

    base_ate = mod2_result.metrics.ate_rmse if mod2_result.metrics else 1.0
    score_mod2 = extract_score("Classical Dead Reckoning", mod2_result.metrics, "#ef4444")
    score_mod2["reduction_pct"] = 0.0

    score_mod3 = extract_score("AI-ZUPT Velocity Clamping", mod3_result.metrics_zupt, "#f59e0b")
    score_mod3["reduction_pct"] = float(round(max(0.0, (base_ate - score_mod3["ate_rmse"]) / base_ate * 100.0), 2))

    score_mod4 = extract_score("Deep Neural Drift Estimator", mod4_result.metrics_neural, "#06b6d4")
    score_mod4["reduction_pct"] = float(round(max(0.0, (base_ate - score_mod4["ate_rmse"]) / base_ate * 100.0), 2))

    score_mod5 = extract_score("AI-Enhanced EKF Fusion", mod5_result.metrics, "#10b981")
    score_mod5["reduction_pct"] = float(round(max(0.0, (base_ate - score_mod5["ate_rmse"]) / base_ate * 100.0), 2))

    # Cumulative Error Curves vs Time
    err_mod2 = np.linalg.norm(mod2_result.pos[idx] - gt_pos[idx], axis=1)
    err_mod3 = np.linalg.norm(mod3_result.pos[idx] - gt_pos[idx], axis=1)
    err_mod4 = np.linalg.norm(mod4_result.pos[idx] - gt_pos[idx], axis=1)
    err_mod5 = np.linalg.norm(mod5_result.pos[idx] - gt_pos[idx], axis=1)

    payload = {
        "metadata": {
            "profile": profile,
            "duration": float(duration),
            "dt": float(dt),
            "imu_grade": imu_grade,
            "gps_outage": list(gps_outage) if gps_outage else None,
            "total_samples": int(len(t[idx])),
        },
        "scorecards": [score_mod2, score_mod3, score_mod4, score_mod5],
        "timestamps": t[idx].tolist(),
        "trajectories": {
            "ground_truth": {
                "x": gt_pos[idx, 0].tolist(),
                "y": gt_pos[idx, 1].tolist(),
                "z": gt_pos[idx, 2].tolist(),
            },
            "classical_dr": {
                "x": mod2_result.pos[idx, 0].tolist(),
                "y": mod2_result.pos[idx, 1].tolist(),
                "z": mod2_result.pos[idx, 2].tolist(),
            },
            "zupt_dr": {
                "x": mod3_result.pos[idx, 0].tolist(),
                "y": mod3_result.pos[idx, 1].tolist(),
                "z": mod3_result.pos[idx, 2].tolist(),
            },
            "neural_dr": {
                "x": mod4_result.pos[idx, 0].tolist(),
                "y": mod4_result.pos[idx, 1].tolist(),
                "z": mod4_result.pos[idx, 2].tolist(),
            },
            "ai_ekf": {
                "x": mod5_result.pos[idx, 0].tolist(),
                "y": mod5_result.pos[idx, 1].tolist(),
                "z": mod5_result.pos[idx, 2].tolist(),
            },
        },
        "errors_vs_time": {
            "classical_dr": err_mod2.tolist(),
            "zupt_dr": err_mod3.tolist(),
            "neural_dr": err_mod4.tolist(),
            "ai_ekf": err_mod5.tolist(),
        },
        "telemetry": {
            "speed_gt": np.linalg.norm(gt_vel[idx], axis=1).tolist(),
            "speed_ekf": np.linalg.norm(mod5_result.vel[idx], axis=1).tolist(),
            "roll_deg": mod5_result.euler_deg[idx, 0].tolist(),
            "pitch_deg": mod5_result.euler_deg[idx, 1].tolist(),
            "yaw_deg": mod5_result.euler_deg[idx, 2].tolist(),
            "acc_bias_x": mod5_result.acc_bias[idx, 0].tolist(),
            "acc_bias_y": mod5_result.acc_bias[idx, 1].tolist(),
            "acc_bias_z": mod5_result.acc_bias[idx, 2].tolist(),
            "pos_3sigma_x": mod5_result.pos_std_3sigma[idx, 0].tolist(),
            "pos_3sigma_y": mod5_result.pos_std_3sigma[idx, 1].tolist(),
            "pos_3sigma_z": mod5_result.pos_std_3sigma[idx, 2].tolist(),
            "gps_valid": (traj.gps_valid[idx].astype(int)).tolist() if traj.gps_valid is not None else [1] * len(t[idx]),
            "stance_flag": (mod5_result.stance_flags[idx].astype(int)).tolist(),
            "handoff_state": [s.value for s in [mod5_result.handoff_states[i] for i in range(0, N, step)]],
            "motion_context": [mod5_result.motion_contexts[i] for i in range(0, N, step)],
        }
    }
    return payload


import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler


# ============================================================================
# 2. HTTP & REST API SERVER
# ============================================================================

class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Multi-Threaded HTTP/1.1 Request Handler for NAVIS Dashboard."""

    protocol_version = "HTTP/1.1"
    STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    def _send_data(self, data_bytes: bytes, content_type: str = "application/json", status: int = 200):
        """Send complete HTTP response with exact Content-Length."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Origin, Accept")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(data_bytes)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Origin, Accept")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        try:
            parsed_path = self.path.split("?")[0].rstrip("/")
            if not parsed_path:
                parsed_path = "/"

            if parsed_path == "/api/profiles":
                profiles = [
                    {"id": "urban_driving", "name": "Urban Driving (Turns & Stop-and-Go)", "category": "vehicular"},
                    {"id": "figure_eight", "name": "Figure-Eight Pattern", "category": "pedestrian"},
                    {"id": "helical_3d", "name": "3D Aerial Helical Climb", "category": "aerial"},
                    {"id": "pedestrian", "name": "Pedestrian Stride Walking", "category": "pedestrian"},
                    {"id": "planetary_rover", "name": "Planetary Rover (ISRO Lunar Profile)", "category": "rover"},
                    {"id": "straight", "name": "Straight Line Acceleration", "category": "vehicular"},
                    {"id": "stationary", "name": "Stationary Standstill (Calibration)", "category": "stationary"},
                ]
                imu_grades = [
                    {"id": "consumer", "name": "Consumer MEMS (Smartphone / Drone)"},
                    {"id": "industrial", "name": "Industrial MEMS (Robotics / Automotive)"},
                    {"id": "tactical", "name": "Tactical / Space Grade (ISRO High-Precision)"},
                ]
                resp = json.dumps({"profiles": profiles, "imu_grades": imu_grades}).encode("utf-8")
                self._send_data(resp, "application/json", 200)

            elif parsed_path == "/api/sample":
                payload = generate_dashboard_data(
                    profile="urban_driving",
                    duration=30.0,
                    dt=0.02,
                    imu_grade="consumer",
                    gps_outage=(10.0, 22.0),
                    seed=42
                )
                resp = json.dumps(payload).encode("utf-8")
                self._send_data(resp, "application/json", 200)

            else:
                req_file = "index.html" if parsed_path in ("/", "") else parsed_path.lstrip("/")
                file_path = os.path.join(self.STATIC_DIR, req_file)

                if os.path.exists(file_path) and not os.path.isdir(file_path):
                    mime, _ = mimetypes.guess_type(file_path)
                    mime = mime or "application/octet-stream"
                    with open(file_path, "rb") as f:
                        content = f.read()
                    self._send_data(content, mime, 200)
                else:
                    self._send_data(b"404 Not Found", "text/plain", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            err_resp = json.dumps({"error": str(e)}).encode("utf-8")
            self._send_data(err_resp, "application/json", 500)

    def do_POST(self):
        try:
            parsed_path = self.path.split("?")[0].rstrip("/")
            if parsed_path == "/api/simulate":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                params = json.loads(body) if body else {}

                profile = params.get("profile", "urban_driving")
                duration = float(params.get("duration", 40.0))
                imu_grade = params.get("imu_grade", "consumer")
                dt = float(params.get("dt", 0.02))
                seed = int(params.get("seed", 42))

                outage_raw = params.get("gps_outage", [15.0, 30.0])
                outage = (float(outage_raw[0]), float(outage_raw[1])) if outage_raw and len(outage_raw) == 2 else None

                payload = generate_dashboard_data(
                    profile=profile,
                    duration=duration,
                    dt=dt,
                    imu_grade=imu_grade,
                    gps_outage=outage,
                    seed=seed
                )
                resp = json.dumps(payload).encode("utf-8")
                self._send_data(resp, "application/json", 200)
            else:
                self._send_data(b"404 Not Found", "text/plain", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            err_resp = json.dumps({"error": str(e)}).encode("utf-8")
            self._send_data(err_resp, "application/json", 500)

    def log_message(self, format, *args):
        # Clean formatted logging
        print(f"[NAVIS Server] {args[0]} {args[1]}")


def launch_dashboard(host: str = "127.0.0.1", port: int = 8000):
    """Launch the NAVIS Interactive Dashboard HTTP Server."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    print(f"\n=======================================================")
    print(f"  NAVIS Interactive 3D Trajectory Dashboard Server     ")
    print(f"=======================================================")
    print(f"[*] Serving live dashboard at: http://{host}:{port}/")
    print(f"[*] Press Ctrl+C in terminal to stop server.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Launch NAVIS 3D Navigation Dashboard Server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    args = parser.parse_args()

    launch_dashboard(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
