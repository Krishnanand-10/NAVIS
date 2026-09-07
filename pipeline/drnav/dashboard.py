"""
NAVIS SIH26168 Interactive 3D Trajectory & Live Phone IMU Dashboard Server

Serves an interactive web dashboard on http://0.0.0.0:8080 displaying:
- Interactive 3D WebGL Multi-Path Trajectory Engine (East [m], North [m], Up [m])
- Pure Cartesian Coordinate System matching sih26168-dead-reckoning-main
- Multi-Angle Camera Toggles (3D Isometric, 2D Top XY, 2D Profile Altitude)
- Mission Playback, Scrubber & Vehicle Pose Tracking
- Live Phone IMU Hardware Telemetry (Pitch, Roll, Yaw, Accel, Gyro, ZUPT)
- Bi-directional Phone-to-PC Telemetry Sync via /api/telemetry
- Position Error (Log Scale) with 10 m threshold
- IMU Bias & Observability Metrics
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav import metrics, plotting
from drnav.align import coarse_align
from drnav.baseline_ins import run_baseline
from drnav.replay import run_eskf, truth_outages
from drnav.synth import SCENARIOS, simulate

OUT_DIR = Path(__file__).resolve().parents[1] / "out"


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


LOCAL_IP = get_local_ip()

# Global in-memory storage for live phone telemetry stream
latest_telemetry: Dict[str, Any] = {
    "active": False,
    "last_seen": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "yaw": 0.0,
    "ax": 0.0,
    "ay": 0.0,
    "az": 9.81,
    "amag": 9.81,
    "gx": 0.0,
    "gy": 0.0,
    "gz": 0.0,
    "still": True,
    "source": "None",
}

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NAVIS: SIH26168 3D Dead Reckoning & IMU Dashboard</title>
    <!-- Plotly.js 3D WebGL -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #080c14;
            --card-bg: #0f172a;
            --card-border: #1e293b;
            --accent-blue: #2f7ec4;
            --accent-cyan: #38bdf8;
            --accent-orange: #dd8452;
            --accent-red: #c44e52;
            --accent-gold: #f2c14e;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}

        header {{
            background: linear-gradient(90deg, #0b0f19, #1e1b4b);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }}

        .logo-area h1 {{
            font-size: 1.35rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .logo-area p {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .controls {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}

        select, button {{
            background-color: #1e293b;
            color: var(--text-main);
            border: 1px solid #334155;
            padding: 0.45rem 0.9rem;
            border-radius: 0.375rem;
            font-size: 0.85rem;
            cursor: pointer;
            outline: none;
            transition: all 0.2s;
        }}
        select:hover, button:hover {{
            background-color: #334155;
        }}
        .btn-run {{
            background-color: #2563eb;
            border-color: #3b82f6;
            font-weight: 600;
        }}
        .btn-run:hover {{
            background-color: #1d4ed8;
        }}
        .btn-play {{
            background-color: #059669;
            border-color: #10b981;
            font-weight: 600;
        }}
        .btn-play:hover {{
            background-color: #10b981;
        }}

        .main-container {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.25rem;
            padding: 1.25rem;
            max-width: 1650px;
            margin: 0 auto;
            width: 100%;
        }}

        .ip-banner {{
            background: #1e1b4b;
            border: 1px solid #4338ca;
            padding: 0.65rem 1.25rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            color: #e0e7ff;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .ip-banner code {{
            background: #0f172a;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            color: #38bdf8;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}

        .metric-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 1rem 1.25rem;
            display: flex;
            flex-direction: column;
        }}

        .metric-card .title {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }}

        .metric-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .metric-card .subtitle {{
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        .visualizer-section {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}

        .visualizer-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .visualizer-header h2 {{
            font-size: 1.05rem;
            font-weight: 600;
        }}

        .view-btns {{
            display: flex;
            gap: 0.4rem;
        }}

        .view-btns button.active {{
            background-color: #38bdf8;
            color: #080c14;
            font-weight: 600;
            border-color: #38bdf8;
        }}

        #plotly3d {{
            width: 100%;
            height: 560px;
            border-radius: 0.375rem;
            background-color: #050811;
        }}

        /* Playback Scrubber Bar */
        .playback-bar {{
            display: flex;
            align-items: center;
            gap: 1rem;
            background: #090e1a;
            border: 1px solid var(--card-border);
            padding: 0.6rem 1rem;
            border-radius: 0.375rem;
            flex-wrap: wrap;
        }}

        .scrubber {{
            flex: 1;
            min-width: 200px;
            accent-color: #38bdf8;
            cursor: pointer;
        }}

        .time-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #38bdf8;
            min-width: 90px;
        }}

        .charts-container {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.25rem;
        }}

        @media (min-width: 1024px) {{
            .charts-container {{
                grid-template-columns: 1fr 1fr;
            }}
        }}

        .chart-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            height: 440px;
        }}

        .chart-card h2 {{
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}

        .plot-area {{
            flex: 1;
            width: 100%;
            height: 100%;
        }}

        footer {{
            text-align: center;
            padding: 1.25rem;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: auto;
            border-top: 1px solid var(--card-border);
        }}

        .legend-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.8rem;
            padding: 0.2rem 0.5rem;
            background: #1e293b;
            border-radius: 0.25rem;
            color: var(--text-main);
        }}
        .dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <h1>NAVIS 3D Trajectory & Dead Reckoning Engine</h1>
            <p>ISRO SIH26168 Multi-Path Inertial Fusion & Real-Time Sensor Suite</p>
        </div>
        <div class="controls">
            <label for="scenarioSelect" style="font-size: 0.85rem; color: var(--text-muted);">Trajectory Profile:</label>
            <select id="scenarioSelect" onchange="loadScenario()">
                <option value="tunnel_drive">Tunnel Drive (120s GNSS Outage + Traffic Stop)</option>
                <option value="parking_ramp">Parking Ramp (150s Spiral Ascent & Parking)</option>
                <option value="urban_canyon">Urban Canyon (Severe Multipath Distortion)</option>
                <option value="pedestrian_mall">Pedestrian Mall (180s Indoor Gait & Stairs)</option>
            </select>
            <button class="btn-run" onclick="loadScenario()">Run Estimator</button>
            <button id="animBtn" class="btn-play" onclick="toggleAnimation()">▶ Play Trajectory</button>
        </div>
    </header>

    <div class="main-container">
        <!-- Wi-Fi Connect Banner -->
        <div class="ip-banner">
            <span>📱 <strong>Live Phone IMU Stream:</strong> Open this URL in Chrome on your phone to stream hardware sensors:</span>
            <code>http://{LOCAL_IP}:8080</code>
            <span id="syncStatus" style="font-size: 0.8rem; color: #a78bfa; margin-left: auto;">● Ready</span>
        </div>

        <!-- Metric Summary Cards -->
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="title">15-State ESKF RMSE</span>
                <span class="value" id="eskfRmse" style="color: #38bdf8;">-- m</span>
                <span class="subtitle" style="color: #4ade80;">Error-State Kalman Filter</span>
            </div>
            <div class="metric-card">
                <span class="title">Classical DR RMSE</span>
                <span class="value" id="baseRmse" style="color: var(--accent-orange);">-- m</span>
                <span class="subtitle" style="color: #c44e52;">Uncorrected Strapdown</span>
            </div>
            <div class="metric-card">
                <span class="title">Drift Reduction</span>
                <span class="value" id="driftReduction" style="color: #4ade80;">-- %</span>
                <span class="subtitle">Accuracy Gain vs Classical</span>
            </div>
            <div class="metric-card">
                <span class="title">Reacquisition Jump</span>
                <span class="value" id="reacqJump" style="color: #f2c14e;">0.1 m</span>
                <span class="subtitle">Seamless Handover Tau</span>
            </div>
        </div>

        <!-- 3D TRAJECTORY VISUALIZER (CARTESIAN ENU METRIC FRAME) -->
        <div class="visualizer-section">
            <div class="visualizer-header">
                <div>
                    <h2>📐 Interactive 3D Trajectory Engine (Cartesian ENU Frame)</h2>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">Exact metric coordinates (East [m], North [m], Up [m]) matching drnav.synth simulator</span>
                </div>
                <div class="view-btns">
                    <button class="active" id="btn3D" onclick="setViewMode('3d')">3D Isometric</button>
                    <button id="btnTop" onclick="setViewMode('top')">2D Top (East vs North)</button>
                    <button id="btnSide" onclick="setViewMode('side')">2D Profile (East vs Altitude)</button>
                    <button onclick="resetCamera()">Reset View</button>
                </div>
            </div>

            <!-- Trace Visibility Legend -->
            <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.25rem;">
                <span class="legend-badge"><span class="dot" style="background: #ffffff;"></span> Ground Truth</span>
                <span class="legend-badge"><span class="dot" style="background: #ef4444;"></span> GNSS Fixes</span>
                <span class="legend-badge"><span class="dot" style="background: #f97316;"></span> Classical Strapdown DR</span>
                <span class="legend-badge"><span class="dot" style="background: #38bdf8;"></span> NAVIS 15-State ESKF</span>
                <span class="legend-badge"><span class="dot" style="background: #10b981;"></span> Active Vehicle Pose</span>
            </div>

            <!-- 3D Plotly WebGL Canvas -->
            <div id="plotly3d"></div>

            <!-- Playback Scrubber Bar -->
            <div class="playback-bar">
                <button id="btnPlayPause" class="btn-play" onclick="toggleAnimation()" style="padding: 0.35rem 0.8rem;">▶ Play</button>
                <button onclick="rewindAnimation()" style="padding: 0.35rem 0.8rem;">⏮ Rewind</button>
                <span class="time-badge" id="scrubberTime">0.0s / 0.0s</span>
                <input type="range" id="timeSlider" class="scrubber" min="0" max="100" value="0" step="0.1" oninput="onScrubberInput(this.value)">
                <div style="display: flex; gap: 0.3rem;">
                    <button onclick="setSpeed(1)" class="speed-btn" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">1x</button>
                    <button onclick="setSpeed(2)" class="speed-btn" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">2x</button>
                    <button onclick="setSpeed(5)" class="speed-btn" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">5x</button>
                </div>
            </div>
        </div>

        <!-- 📱 LIVE PHONE IMU & MOTION TELEMETRY CARD -->
        <div class="visualizer-section" style="border: 1px solid #3b82f6;">
            <div class="visualizer-header">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <h2>📱 Live Phone IMU & Motion Telemetry (Hardware Sensors)</h2>
                    <span id="phoneStatusBadge" class="legend-badge" style="background: #1e293b; color: #94a3b8;">
                        <span id="phoneStatusDot" class="dot" style="background: #64748b;"></span>
                        <span id="phoneStatusText">Waiting for Phone...</span>
                    </span>
                </div>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <button id="sensorPermBtn" onclick="requestSensorPermission()" style="padding: 0.35rem 0.8rem; font-size: 0.85rem; background: #2563eb; color: white;">📱 Connect Phone Sensors</button>
                    <button onclick="simulatePhoneMovement()" style="padding: 0.35rem 0.8rem; font-size: 0.85rem; background: #334155; color: #94a3b8;">Test Shake/Tilt</button>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 1rem; margin-top: 0.25rem;">
                <!-- 1. Attitude Indicator Canvas -->
                <div style="background: #050811; border-radius: 0.5rem; padding: 1rem; border: 1px solid var(--card-border); display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem; width: 100%; display: flex; justify-content: space-between;">
                        <span>Artificial Horizon (Attitude)</span>
                        <span id="motionStatusBadge" style="color: #4ade80; font-weight: 600;">STATIONARY (ZUPT)</span>
                    </div>
                    <canvas id="horizonCanvas" width="220" height="160" style="border-radius: 0.5rem; background: #000; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);"></canvas>
                    <div style="display: flex; justify-content: space-around; width: 100%; margin-top: 0.75rem; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
                        <div>P: <strong id="valPitch" style="color: #38bdf8;">0.0°</strong></div>
                        <div>R: <strong id="valRoll" style="color: #38bdf8;">0.0°</strong></div>
                        <div>HDG: <strong id="valYaw" style="color: #f2c14e;">0.0°</strong></div>
                    </div>
                </div>

                <!-- 2. Accelerometer 3-Axis -->
                <div style="background: #050811; border-radius: 0.5rem; padding: 1rem; border: 1px solid var(--card-border); display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                        <span>Accelerometer (m/s²)</span>
                        <span>Total: <strong id="valAmag" style="color: #38bdf8; font-family: 'JetBrains Mono', monospace;">9.81</strong> m/s²</span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.6rem;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>X (Lateral)</span><strong id="valAx" style="font-family: 'JetBrains Mono', monospace;">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barAx" style="background: #38bdf8; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>Y (Longitudinal)</span><strong id="valAy" style="font-family: 'JetBrains Mono', monospace;">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barAy" style="background: #38bdf8; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>Z (Vertical)</span><strong id="valAz" style="font-family: 'JetBrains Mono', monospace;">9.81</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barAz" style="background: #38bdf8; height: 100%; width: 80%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.5rem;">
                        Flat phone = ~9.81 m/s² vertical gravity vector.
                    </div>
                </div>

                <!-- 3. Gyroscope 3-Axis -->
                <div style="background: #050811; border-radius: 0.5rem; padding: 1rem; border: 1px solid var(--card-border); display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;">
                        <span>Gyroscope Rotation Rate (deg/s)</span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.6rem;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>ωX (Roll Rate)</span><strong id="valGx" style="font-family: 'JetBrains Mono', monospace;">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barGx" style="background: #a78bfa; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>ωY (Pitch Rate)</span><strong id="valGy" style="font-family: 'JetBrains Mono', monospace;">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barGy" style="background: #a78bfa; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>ωZ (Yaw Rate)</span><strong id="valGz" style="font-family: 'JetBrains Mono', monospace;">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barGz" style="background: #a78bfa; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.5rem;">
                        Angular velocity around phone body axes.
                    </div>
                </div>
            </div>
        </div>

        <!-- 2D Error Charts & Plots -->
        <div class="charts-container">
            <div class="chart-card">
                <h2>Position Error Over Time (Log Scale, 10m Threshold)</h2>
                <div id="errPlot" class="plot-area"></div>
            </div>
            <div class="chart-card">
                <h2>IMU Accelerometer & Gyroscope Bias Convergence</h2>
                <div id="biasPlot" class="plot-area"></div>
            </div>
        </div>
    </div>

    <footer>
        ISRO SIH26168 Autonomous Dead Reckoning Navigation Suite
    </footer>

    <script>
        let currentData = null;
        let animTimer = null;
        let animIndex = 0;
        let playbackSpeed = 1.0;
        let viewMode = '3d';

        // Telemetry State
        let isPhoneBroadcasting = false;
        let lastPostTimestamp = 0;
        let localTelemetry = {{
            pitch: 0.0, roll: 0.0, yaw: 0.0,
            ax: 0.0, ay: 0.0, az: 9.81, amag: 9.81,
            gx: 0.0, gy: 0.0, gz: 0.0,
            still: true
        }};

        // ==========================================
        // 3D PLOTLY TRAJECTORY ENGINE
        // ==========================================

        function render3DTrajectory(data) {{
            const truth = data.truth_pos;
            const gnss = data.gnss_pos;
            const base = data.base_pos;
            const eskf = data.eskf_pos;

            const traces = [
                {{
                    type: 'scatter3d',
                    mode: 'lines',
                    name: 'Ground Truth',
                    x: truth.map(p => p[0]),
                    y: truth.map(p => p[1]),
                    z: truth.map(p => p[2]),
                    line: {{ color: '#ffffff', width: 4 }}
                }},
                {{
                    type: 'scatter3d',
                    mode: 'markers',
                    name: 'GNSS Fixes',
                    x: gnss.map(p => p[0]),
                    y: gnss.map(p => p[1]),
                    z: gnss.map(p => p[2]),
                    marker: {{ color: '#ef4444', size: 3, symbol: 'circle' }}
                }},
                {{
                    type: 'scatter3d',
                    mode: 'lines',
                    name: 'Classical Strapdown DR',
                    x: base.map(p => p[0]),
                    y: base.map(p => p[1]),
                    z: base.map(p => p[2]),
                    line: {{ color: '#f97316', width: 3, dash: 'dash' }}
                }},
                {{
                    type: 'scatter3d',
                    mode: 'lines',
                    name: 'NAVIS 15-State ESKF',
                    x: eskf.map(p => p[0]),
                    y: eskf.map(p => p[1]),
                    z: eskf.map(p => p[2]),
                    line: {{ color: '#38bdf8', width: 5 }}
                }},
                {{
                    type: 'scatter3d',
                    mode: 'markers',
                    name: 'Vehicle Pose',
                    x: [eskf[0][0]],
                    y: [eskf[0][1]],
                    z: [eskf[0][2]],
                    marker: {{ color: '#10b981', size: 9, symbol: 'diamond' }}
                }}
            ];

            const layout = {{
                autosize: true,
                paper_bgcolor: '#050811',
                plot_bgcolor: '#050811',
                margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                showlegend: false,
                scene: getCameraScene(viewMode)
            }};

            Plotly.newPlot('plotly3d', traces, layout, {{ responsive: true, displayModeBar: false }});
        }}

        function getCameraScene(mode) {{
            let camera;
            if (mode === 'top') {{
                camera = {{ eye: {{ x: 0.001, y: 0.001, z: 2.5 }}, up: {{ x: 0, y: 1, z: 0 }} }};
            }} else if (mode === 'side') {{
                camera = {{ eye: {{ x: 0.001, y: -2.5, z: 0.001 }}, up: {{ x: 0, y: 0, z: 1 }} }};
            }} else {{
                camera = {{ eye: {{ x: 1.5, y: 1.5, z: 1.2 }}, up: {{ x: 0, y: 0, z: 1 }} }};
            }}

            return {{
                xaxis: {{ title: 'East (X) [m]', color: '#94a3b8', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                yaxis: {{ title: 'North (Y) [m]', color: '#94a3b8', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                zaxis: {{ title: 'Up (Z) [m]', color: '#94a3b8', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
                aspectmode: 'data',
                camera: camera
            }};
        }}

        function setViewMode(mode) {{
            viewMode = mode;
            document.getElementById('btn3D').classList.toggle('active', mode === '3d');
            document.getElementById('btnTop').classList.toggle('active', mode === 'top');
            document.getElementById('btnSide').classList.toggle('active', mode === 'side');

            Plotly.relayout('plotly3d', {{ 'scene.camera': getCameraScene(mode).camera }});
        }}

        function resetCamera() {{
            setViewMode('3d');
        }}

        // ==========================================
        // 2D ERROR & BIAS CHARTS
        // ==========================================

        function renderErrorCharts(data) {{
            const t = data.t;

            // Position Error Log Plot
            const tracesErr = [
                {{
                    x: t,
                    y: data.base_err,
                    mode: 'lines',
                    name: 'Classical DR Error',
                    line: {{ color: '#f97316', width: 1.5, dash: 'dash' }}
                }},
                {{
                    x: t,
                    y: data.eskf_err,
                    mode: 'lines',
                    name: 'ESKF Error',
                    line: {{ color: '#38bdf8', width: 2.5 }}
                }}
            ];

            const layoutErr = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#050811',
                margin: {{ l: 50, r: 20, t: 20, b: 45 }},
                xaxis: {{ title: 'Time (s)', gridcolor: '#1e293b', color: '#94a3b8' }},
                yaxis: {{ title: '2D Position Error (m)', type: 'log', gridcolor: '#1e293b', color: '#94a3b8' }},
                shapes: [
                    {{
                        type: 'line',
                        x0: t[0],
                        x1: t[t.length - 1],
                        y0: 10,
                        y1: 10,
                        line: {{ color: '#f2c14e', width: 1.5, dash: 'dashdot' }}
                    }}
                ],
                annotations: [
                    {{
                        x: t[Math.floor(t.length * 0.8)],
                        y: 1.05,
                        yref: 'paper',
                        text: '10 m Threshold',
                        showarrow: false,
                        font: {{ color: '#f2c14e', size: 11 }}
                    }}
                ],
                legend: {{ x: 0.02, y: 0.98, bgcolor: 'rgba(15, 23, 42, 0.8)', font: {{ color: '#f8fafc' }} }}
            }};

            Plotly.newPlot('errPlot', tracesErr, layoutErr, {{ responsive: true, displayModeBar: false }});

            // Gyro & Accel Bias Plot
            const tracesBias = [
                {{ x: t, y: data.gyro_bias_x, mode: 'lines', name: 'bg_x (deg/s)', line: {{ color: '#ef4444', width: 1.5 }} }},
                {{ x: t, y: data.gyro_bias_y, mode: 'lines', name: 'bg_y (deg/s)', line: {{ color: '#10b981', width: 1.5 }} }},
                {{ x: t, y: data.gyro_bias_z, mode: 'lines', name: 'bg_z (deg/s)', line: {{ color: '#38bdf8', width: 1.5 }} }}
            ];

            const layoutBias = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#050811',
                margin: {{ l: 50, r: 20, t: 20, b: 45 }},
                xaxis: {{ title: 'Time (s)', gridcolor: '#1e293b', color: '#94a3b8' }},
                yaxis: {{ title: 'Estimated Gyro Bias (°/s)', gridcolor: '#1e293b', color: '#94a3b8' }},
                legend: {{ x: 0.02, y: 0.98, bgcolor: 'rgba(15, 23, 42, 0.8)', font: {{ color: '#f8fafc' }} }}
            }};

            Plotly.newPlot('biasPlot', tracesBias, layoutBias, {{ responsive: true, displayModeBar: false }});
        }}

        // ==========================================
        // MISSION PLAYBACK & SCRUBBER
        // ==========================================

        function toggleAnimation() {{
            if (animTimer) {{
                pauseAnimation();
            }} else {{
                startAnimation();
            }}
        }}

        function startAnimation() {{
            if (!currentData || !currentData.eskf_pos) return;
            const btn = document.getElementById('btnPlayPause');
            const topBtn = document.getElementById('animBtn');
            btn.innerText = "⏸ Pause";
            topBtn.innerText = "⏸ Pause";
            btn.style.backgroundColor = "#dc2626";

            const eskf = currentData.eskf_pos;
            const totalFrames = eskf.length;
            const totalDuration = currentData.t[currentData.t.length - 1];

            animTimer = setInterval(() => {{
                if (animIndex < totalFrames) {{
                    const p = eskf[animIndex];
                    Plotly.restyle('plotly3d', {{ x: [[p[0]]], y: [[p[1]]], z: [[p[2]]] }}, [4]);

                    const currentTime = currentData.t[animIndex];
                    document.getElementById('scrubberTime').innerText = `${{currentTime.toFixed(1)}}s / ${{totalDuration.toFixed(1)}}s`;
                    document.getElementById('timeSlider').value = (animIndex / totalFrames) * 100;

                    animIndex += Math.max(1, Math.floor(playbackSpeed * 3));
                }} else {{
                    pauseAnimation();
                }}
            }}, 40);
        }}

        function pauseAnimation() {{
            if (animTimer) {{
                clearInterval(animTimer);
                animTimer = null;
            }}
            const btn = document.getElementById('btnPlayPause');
            const topBtn = document.getElementById('animBtn');
            if (btn) btn.innerText = "▶ Play";
            if (topBtn) topBtn.innerText = "▶ Play Trajectory";
            if (btn) btn.style.backgroundColor = "#059669";
        }}

        function rewindAnimation() {{
            pauseAnimation();
            animIndex = 0;
            if (currentData && currentData.eskf_pos) {{
                const p = currentData.eskf_pos[0];
                Plotly.restyle('plotly3d', {{ x: [[p[0]]], y: [[p[1]]], z: [[p[2]]] }}, [4]);
                document.getElementById('timeSlider').value = 0;
                document.getElementById('scrubberTime').innerText = `0.0s / ${{currentData.t[currentData.t.length - 1].toFixed(1)}}s`;
            }}
        }}

        function onScrubberInput(val) {{
            if (!currentData || !currentData.eskf_pos) return;
            const totalFrames = currentData.eskf_pos.length;
            animIndex = Math.min(totalFrames - 1, Math.floor((val / 100) * totalFrames));
            const p = currentData.eskf_pos[animIndex];
            Plotly.restyle('plotly3d', {{ x: [[p[0]]], y: [[p[1]]], z: [[p[2]]] }}, [4]);
            const currentTime = currentData.t[animIndex];
            const totalDuration = currentData.t[currentData.t.length - 1];
            document.getElementById('scrubberTime').innerText = `${{currentTime.toFixed(1)}}s / ${{totalDuration.toFixed(1)}}s`;
        }}

        function setSpeed(s) {{
            playbackSpeed = s;
        }}

        // ==========================================
        // 📱 LIVE PHONE IMU & SENSORS
        // ==========================================

        function drawHorizon(pitch, roll, yaw) {{
            const canvas = document.getElementById('horizonCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;

            ctx.save();
            ctx.clearRect(0, 0, w, h);

            ctx.translate(cx, cy);
            const rollRad = (roll * Math.PI) / 180;
            ctx.rotate(rollRad);

            const pitchOffset = Math.max(-cy, Math.min(cy, pitch * 1.5));

            // Sky
            ctx.fillStyle = "#1e3a8a";
            ctx.fillRect(-w, -h * 2, w * 2, h * 2 + pitchOffset);

            // Ground
            ctx.fillStyle = "#334155";
            ctx.fillRect(-w, pitchOffset, w * 2, h * 2);

            // Horizon Dividing Line
            ctx.strokeStyle = "#f8fafc";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(-w, pitchOffset);
            ctx.lineTo(w, pitchOffset);
            ctx.stroke();

            // Pitch Marks
            ctx.strokeStyle = "rgba(248, 250, 252, 0.5)";
            ctx.fillStyle = "rgba(248, 250, 252, 0.8)";
            ctx.font = "9px Inter, sans-serif";
            [-30, -20, -10, 10, 20, 30].forEach(deg => {{
                const y = pitchOffset - (deg * 1.5);
                if (y > -cy + 12 && y < cy - 12) {{
                    const lineLen = deg % 20 === 0 ? 24 : 14;
                    ctx.beginPath();
                    ctx.moveTo(-lineLen, y);
                    ctx.lineTo(lineLen, y);
                    ctx.stroke();
                    ctx.fillText(Math.abs(deg) + "°", lineLen + 4, y + 3);
                }}
            }});

            ctx.restore();

            // Center Fixed Reticle
            ctx.strokeStyle = "#f2c14e";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(cx - 24, cy);
            ctx.lineTo(cx - 8, cy);
            ctx.lineTo(cx, cy + 6);
            ctx.lineTo(cx + 8, cy);
            ctx.lineTo(cx + 24, cy);
            ctx.stroke();

            ctx.fillStyle = "#f2c14e";
            ctx.beginPath();
            ctx.arc(cx, cy, 3, 0, Math.PI * 2);
            ctx.fill();
        }}

        function updateTelemetryUI(t) {{
            document.getElementById('valPitch').innerText = t.pitch.toFixed(1) + "°";
            document.getElementById('valRoll').innerText = t.roll.toFixed(1) + "°";
            document.getElementById('valYaw').innerText = t.yaw.toFixed(1) + "°";

            document.getElementById('valAx').innerText = t.ax.toFixed(2);
            document.getElementById('valAy').innerText = t.ay.toFixed(2);
            document.getElementById('valAz').innerText = t.az.toFixed(2);
            document.getElementById('valAmag').innerText = t.amag.toFixed(2);

            document.getElementById('valGx').innerText = t.gx.toFixed(1);
            document.getElementById('valGy').innerText = t.gy.toFixed(1);
            document.getElementById('valGz').innerText = t.gz.toFixed(1);

            const normBar = (val, maxVal) => Math.max(0, Math.min(100, ((val + maxVal) / (2 * maxVal)) * 100)) + "%";
            document.getElementById('barAx').style.width = normBar(t.ax, 15);
            document.getElementById('barAy').style.width = normBar(t.ay, 15);
            document.getElementById('barAz').style.width = normBar(t.az, 15);

            document.getElementById('barGx').style.width = normBar(t.gx, 180);
            document.getElementById('barGy').style.width = normBar(t.gy, 180);
            document.getElementById('barGz').style.width = normBar(t.gz, 180);

            const statusBadge = document.getElementById('motionStatusBadge');
            if (t.still) {{
                statusBadge.innerText = "STATIONARY (ZUPT)";
                statusBadge.style.color = "#4ade80";
            }} else {{
                statusBadge.innerText = "IN MOTION (DYNAMIC)";
                statusBadge.style.color = "#38bdf8";
            }}

            drawHorizon(t.pitch, t.roll, t.yaw);
        }}

        function handleOrientation(e) {{
            localTelemetry.yaw = e.alpha || 0;
            localTelemetry.pitch = e.beta || 0;
            localTelemetry.roll = e.gamma || 0;
            isPhoneBroadcasting = true;
            onSensorStream();
        }}

        function handleMotion(e) {{
            const acc = e.accelerationIncludingGravity || e.acceleration;
            if (acc) {{
                localTelemetry.ax = acc.x || 0;
                localTelemetry.ay = acc.y || 0;
                localTelemetry.az = acc.z || 0;
                localTelemetry.amag = Math.sqrt(localTelemetry.ax*localTelemetry.ax + localTelemetry.ay*localTelemetry.ay + localTelemetry.az*localTelemetry.az);
            }}
            const rot = e.rotationRate;
            if (rot) {{
                localTelemetry.gx = rot.alpha || 0;
                localTelemetry.gy = rot.beta || 0;
                localTelemetry.gz = rot.gamma || 0;
            }}
            const gyroMag = Math.sqrt(localTelemetry.gx*localTelemetry.gx + localTelemetry.gy*localTelemetry.gy + localTelemetry.gz*localTelemetry.gz);
            const accDiff = Math.abs(localTelemetry.amag - 9.81);
            localTelemetry.still = (gyroMag < 4.0 && accDiff < 0.6);

            isPhoneBroadcasting = true;
            onSensorStream();
        }}

        function onSensorStream() {{
            updateTelemetryUI(localTelemetry);
            document.getElementById('phoneStatusDot').style.background = "#10b981";
            document.getElementById('phoneStatusText').innerText = "Streaming Local Phone Sensors";
            document.getElementById('phoneStatusBadge').style.color = "#10b981";

            const now = performance.now();
            if (now - lastPostTimestamp > 90) {{
                lastPostTimestamp = now;
                fetch('/api/telemetry', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(localTelemetry)
                }}).catch(() => {{}});
            }}
        }}

        function requestSensorPermission() {{
            if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {{
                DeviceOrientationEvent.requestPermission()
                    .then(res => {{
                        if (res === 'granted') {{
                            window.addEventListener('deviceorientation', handleOrientation, true);
                            window.addEventListener('devicemotion', handleMotion, true);
                            document.getElementById('sensorPermBtn').innerText = "✅ Sensors Connected";
                            document.getElementById('sensorPermBtn').style.background = "#10b981";
                        }}
                    }})
                    .catch(console.error);
            }} else {{
                window.addEventListener('deviceorientation', handleOrientation, true);
                window.addEventListener('devicemotion', handleMotion, true);
                document.getElementById('sensorPermBtn').innerText = "✅ Sensors Connected";
                document.getElementById('sensorPermBtn').style.background = "#10b981";
            }}
        }}

        function simulatePhoneMovement() {{
            let angle = 0;
            const simInterval = setInterval(() => {{
                angle += 0.1;
                localTelemetry.pitch = Math.sin(angle) * 22;
                localTelemetry.roll = Math.cos(angle * 0.8) * 35;
                localTelemetry.yaw = (localTelemetry.yaw + 2) % 360;
                localTelemetry.ax = Math.sin(angle) * 3.5;
                localTelemetry.ay = Math.cos(angle) * 2.8;
                localTelemetry.az = 9.81 + Math.sin(angle * 2) * 1.5;
                localTelemetry.amag = Math.sqrt(localTelemetry.ax**2 + localTelemetry.ay**2 + localTelemetry.az**2);
                localTelemetry.gx = Math.cos(angle) * 25;
                localTelemetry.gy = Math.sin(angle) * 15;
                localTelemetry.gz = 10;
                localTelemetry.still = false;
                onSensorStream();
                if (angle > 6) {{
                    clearInterval(simInterval);
                    localTelemetry.still = true;
                    onSensorStream();
                }}
            }}, 50);
        }}

        // PC Background Poller for Phone Telemetry
        setInterval(async () => {{
            if (isPhoneBroadcasting) return;
            try {{
                const res = await fetch('/api/telemetry');
                if (res.ok) {{
                    const data = await res.json();
                    if (data.active) {{
                        document.getElementById('phoneStatusDot').style.background = "#38bdf8";
                        document.getElementById('phoneStatusText').innerText = "Receiving Phone Stream (Wi-Fi)";
                        document.getElementById('phoneStatusBadge').style.color = "#38bdf8";
                        updateTelemetryUI(data);
                    }} else {{
                        document.getElementById('phoneStatusDot').style.background = "#64748b";
                        document.getElementById('phoneStatusText').innerText = "Waiting for Phone... (Open http://{LOCAL_IP}:8080 on mobile)";
                        document.getElementById('phoneStatusBadge').style.color = "#94a3b8";
                    }}
                }}
            }} catch(e) {{}}
        }}, 120);

        // ==========================================
        // DATA LOADER
        // ==========================================

        async function loadScenario() {{
            rewindAnimation();
            const scenario = document.getElementById('scenarioSelect').value;
            const statusText = document.getElementById('syncStatus');
            statusText.innerText = "Simulating " + scenario + "...";
            statusText.style.color = "#f2c14e";

            try {{
                const res = await fetch(`/api/run?scenario=${{scenario}}`);
                currentData = await res.json();

                document.getElementById('eskfRmse').innerText = `${{currentData.eskf_rmse.toFixed(2)}} m`;
                document.getElementById('baseRmse').innerText = `${{currentData.base_rmse.toFixed(2)}} m`;
                const gain = Math.max(0, ((currentData.base_rmse - currentData.eskf_rmse) / currentData.base_rmse) * 100);
                document.getElementById('driftReduction').innerText = `${{gain.toFixed(1)}} %`;
                document.getElementById('reacqJump').innerText = `${{currentData.max_jump.toFixed(2)}} m`;

                render3DTrajectory(currentData);
                renderErrorCharts(currentData);

                statusText.innerText = "● System Online";
                statusText.style.color = "#4ade80";
            }} catch (err) {{
                console.error(err);
                statusText.innerText = "● Simulation Error";
                statusText.style.color = "#c44e52";
            }}
        }}

        window.onload = () => {{
            drawHorizon(0, 0, 0);
            loadScenario();
        }};
    </script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if self.path == "/api/telemetry":
            now = time.time()
            is_active = (now - latest_telemetry.get("last_seen", 0.0)) < 3.0
            resp_data = dict(latest_telemetry)
            resp_data["active"] = is_active
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        if self.path.startswith("/api/run"):
            self.handle_run_api()
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/telemetry":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                global latest_telemetry
                latest_telemetry.update(data)
                latest_telemetry["active"] = True
                latest_telemetry["last_seen"] = time.time()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception:
                self.send_response(400)
                self.end_headers()
            return

        self.send_error(404, "Not Found")

    def handle_run_api(self) -> None:
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(self.path).query)
        scenario_name = query.get("scenario", ["tunnel_drive"])[0]

        if scenario_name not in SCENARIOS:
            scenario_name = "tunnel_drive"

        sc_fn = SCENARIOS[scenario_name]
        scenario = sc_fn()
        rec = simulate(scenario, seed=0)

        align = coarse_align(rec)
        outages, labels = truth_outages(rec)
        base = run_baseline(rec, align)
        est = run_eskf(rec, align)

        s_base = metrics.evaluate(rec.t, base["pos"], rec.truth_pos, outages, labels)
        s_eskf = metrics.evaluate(rec.t, est["pos"], rec.truth_pos, outages, labels)

        base_err = np.linalg.norm(base["pos"][:, :2] - rec.truth_pos[:, :2], axis=1)
        eskf_err = np.linalg.norm(est["pos"][:, :2] - rec.truth_pos[:, :2], axis=1)

        # Gyro bias in deg/s
        gyro_bias_deg = np.rad2deg(est["bias"][:, 3:6])

        def clean_arr(arr: np.ndarray) -> list:
            clean = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            return clean.tolist()

        response_data: Dict[str, Any] = {
            "t": clean_arr(rec.t),
            "truth_pos": clean_arr(rec.truth_pos),
            "gnss_pos": clean_arr(rec.gnss[:, 1:4]),
            "base_pos": clean_arr(base["pos"]),
            "eskf_pos": clean_arr(est["pos"]),
            "base_err": clean_arr(base_err),
            "eskf_err": clean_arr(eskf_err),
            "gyro_bias_x": clean_arr(gyro_bias_deg[:, 0]),
            "gyro_bias_y": clean_arr(gyro_bias_deg[:, 1]),
            "gyro_bias_z": clean_arr(gyro_bias_deg[:, 2]),
            "base_rmse": float(np.nan_to_num(s_base.ate_rmse, nan=0.0)),
            "eskf_rmse": float(np.nan_to_num(s_eskf.ate_rmse, nan=0.0)),
            "max_jump": float(np.nan_to_num(s_eskf.max_jump, nan=0.0)),
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        pass


def start_dashboard(port: int = 8080, open_browser: bool = True) -> None:
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    url_local = f"http://localhost:{port}"
    url_network = f"http://{LOCAL_IP}:{port}"
    print(f"[NAVIS] Dashboard running on PC: {url_local}")
    print(f"[NAVIS] Dashboard running on Phone (Wi-Fi): {url_network}")

    if open_browser:
        try:
            webbrowser.open(url_local)
        except Exception:
            pass
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAVIS Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve dashboard on")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()
    start_dashboard(args.port, open_browser=not args.no_browser)
