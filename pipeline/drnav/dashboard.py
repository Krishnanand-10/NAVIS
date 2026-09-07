"""
NAVIS SIH26168 Interactive Map Dashboard Server

Serves an interactive web dashboard on http://0.0.0.0:8080 displaying:
- Real-Time HTML5 Browser GPS Geolocation Auto-Detection
- Interactive Leaflet OpenStreetMap View (OpenStreetMap / Dark Mode / Satellite)
- Real-World Geodetic Trajectories anchored at User's Exact Physical Location
- Live Vehicle Movement / Navigation Animation
- Live Phone IMU & Motion Telemetry (Pitch, Roll, Yaw, Accel, Gyro, ZUPT)
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


def enu_to_geodetic(e: np.ndarray, n: np.ndarray, lat0: float = 12.9716, lon0: float = 77.5946) -> np.ndarray:
    """Convert relative ENU East/North (meters) to WGS84 Latitude/Longitude degrees."""
    r_earth = 6378137.0
    d_lat = (n / r_earth) * (180.0 / np.pi)
    d_lon = (e / (r_earth * np.cos(np.radians(lat0)))) * (180.0 / np.pi)
    lat = lat0 + d_lat
    lon = lon0 + d_lon
    return np.column_stack([lat, lon])


HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NAVIS: SIH26168 Interactive Map & Live IMU Dashboard</title>
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #131b2e;
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
            background: linear-gradient(90deg, #0f172a, #1e1b4b);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }}

        .logo-area h1 {{
            font-size: 1.4rem;
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
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            font-size: 0.9rem;
            cursor: pointer;
            outline: none;
            transition: all 0.2s;
        }}
        select:hover, button:hover {{
            background-color: #334155;
        }}
        .btn-play {{
            background-color: #059669;
            border-color: #10b981;
        }}
        .btn-play:hover {{
            background-color: #10b981;
        }}
        .btn-gps {{
            background-color: #8b5cf6;
            border-color: #a78bfa;
        }}
        .btn-gps:hover {{
            background-color: #7c3aed;
        }}

        .main-container {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.25rem;
            padding: 1.25rem;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }}

        .ip-banner {{
            background: #1e1b4b;
            border: 1px solid #4338ca;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 0.9rem;
            color: #e0e7ff;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .ip-banner code {{
            background: #0f172a;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            color: #38bdf8;
            font-weight: 600;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }}

        .metric-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
        }}

        .metric-card .title {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .metric-card .value {{
            font-size: 1.6rem;
            font-weight: 700;
        }}

        .metric-card .subtitle {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}

        .map-section {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}

        .map-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .map-header h2 {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        #map {{
            width: 100%;
            height: 520px;
            border-radius: 0.375rem;
            background-color: #0f172a;
            position: relative;
            z-index: 1;
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
            height: 460px;
        }}

        .chart-card h2 {{
            font-size: 1rem;
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
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: auto;
            border-top: 1px solid var(--card-border);
        }}

        .legend-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
            padding: 0.25rem 0.6rem;
            background: #1e293b;
            border-radius: 0.25rem;
            color: var(--text-main);
        }}
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <h1>ISRO SIH26168 Interactive Map Dashboard</h1>
            <p>Live OpenStreetMap Dead Reckoning & Real-Time Phone IMU Suite</p>
        </div>
        <div class="controls">
            <button class="btn-gps" onclick="detectGPSLocation()">📍 Use My GPS Location</button>
            <label for="scenarioSelect" style="font-size: 0.9rem; color: var(--text-muted);">Scenario:</label>
            <select id="scenarioSelect">
                <option value="tunnel_drive">Tunnel Drive (GNSS Outage + Ramp)</option>
                <option value="urban_canyon">Urban Canyon (Multipath Noise)</option>
                <option value="parking_ramp">Parking Ramp (Spiral Elevation)</option>
                <option value="pedestrian_mall">Pedestrian Mall (Variable Gait)</option>
            </select>
            <button id="runBtn" onclick="loadScenario()">Run Filter</button>
            <button id="animBtn" class="btn-play" onclick="toggleAnimation()">▶ Play Live Drive</button>
        </div>
    </header>

    <div class="main-container">
        <!-- Wi-Fi Connect Banner -->
        <div class="ip-banner">
            <span>📱 <strong>Mobile Access:</strong> Open this URL in Chrome on your phone (same Wi-Fi):</span>
            <code>http://{LOCAL_IP}:8080</code>
            <span id="gpsStatus" style="font-size: 0.85rem; color: #a78bfa; margin-left: auto;">📍 Detecting your GPS location...</span>
        </div>

        <!-- Metric Summary Cards -->
        <div class="metrics-grid">
            <div class="metric-card">
                <span class="title">ESKF ATE RMSE</span>
                <span class="value" id="eskfRmse" style="color: #38bdf8;">-- m</span>
                <span class="subtitle" style="color: #4ade80;">Error-State Kalman Filter</span>
            </div>
            <div class="metric-card">
                <span class="title">Classical DR RMSE</span>
                <span class="value" id="baseRmse" style="color: var(--accent-orange);">-- m</span>
                <span class="subtitle" style="color: #c44e52;">Uncorrected Strapdown</span>
            </div>
            <div class="metric-card">
                <span class="title">Origin Coordinates</span>
                <span class="value" id="originCoords" style="font-size: 1.1rem; color: #a78bfa;">12.9716, 77.5946</span>
                <span class="subtitle" id="coordsSub">Reference Lat / Lon</span>
            </div>
            <div class="metric-card">
                <span class="title">Status</span>
                <span class="value" id="statusText" style="color: #38bdf8;">Ready</span>
                <span class="subtitle" id="statusSub">Local Execution Engine</span>
            </div>
        </div>

        <!-- 📱 LIVE PHONE IMU & MOTION TELEMETRY CARD -->
        <div class="map-section" style="border: 1px solid #3b82f6;">
            <div class="map-header">
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
                <div style="background: #0f172a; border-radius: 0.5rem; padding: 1rem; border: 1px solid var(--card-border); display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem; width: 100%; display: flex; justify-content: space-between;">
                        <span>Artificial Horizon (Attitude)</span>
                        <span id="motionStatusBadge" style="color: #4ade80; font-weight: 600;">STATIONARY (ZUPT)</span>
                    </div>
                    <canvas id="horizonCanvas" width="220" height="170" style="border-radius: 0.5rem; background: #000; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);"></canvas>
                    <div style="display: flex; justify-content: space-around; width: 100%; margin-top: 0.75rem; font-size: 0.85rem;">
                        <div>Pitch: <strong id="valPitch" style="color: #38bdf8;">0.0°</strong></div>
                        <div>Roll: <strong id="valRoll" style="color: #38bdf8;">0.0°</strong></div>
                        <div>Compass: <strong id="valYaw" style="color: #f2c14e;">0.0°</strong></div>
                    </div>
                </div>

                <!-- 2. Accelerometer 3-Axis -->
                <div style="background: #0f172a; border-radius: 0.5rem; padding: 1rem; border: 1px solid var(--card-border); display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                        <span>Accelerometer (m/s²)</span>
                        <span>Total: <strong id="valAmag" style="color: #38bdf8;">9.81</strong> m/s²</span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.6rem;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>X (Lateral)</span><strong id="valAx">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barAx" style="background: #38bdf8; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>Y (Longitudinal)</span><strong id="valAy">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barAy" style="background: #38bdf8; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>Z (Vertical)</span><strong id="valAz">9.81</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barAz" style="background: #38bdf8; height: 100%; width: 80%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                        Level phone = ~9.81 m/s² gravity vector on Z.
                    </div>
                </div>

                <!-- 3. Gyroscope 3-Axis -->
                <div style="background: #0f172a; border-radius: 0.5rem; padding: 1rem; border: 1px solid var(--card-border); display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;">
                        <span>Gyroscope Rotation Rate (deg/s)</span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.6rem;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>ωX (Roll Rate)</span><strong id="valGx">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barGx" style="background: #a78bfa; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>ωY (Pitch Rate)</span><strong id="valGy">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barGy" style="background: #a78bfa; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                <span>ωZ (Yaw Rate)</span><strong id="valGz">0.00</strong>
                            </div>
                            <div style="background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden;">
                                <div id="barGz" style="background: #a78bfa; height: 100%; width: 50%; transition: width 0.05s;"></div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                        Angular velocity around phone body axes.
                    </div>
                </div>
            </div>
        </div>

        <!-- OpenStreetMap View -->
        <div class="map-section">
            <div class="map-header">
                <h2>🗺️ OpenStreetMap Navigation View (Live Trajectory Overlay)</h2>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <span class="legend-badge"><span class="dot" style="background: #000000;"></span> Ground Truth</span>
                    <span class="legend-badge"><span class="dot" style="background: #ef4444;"></span> GNSS Fixes</span>
                    <span class="legend-badge"><span class="dot" style="background: #f97316;"></span> Classical DR</span>
                    <span class="legend-badge"><span class="dot" style="background: #2563eb;"></span> ESKF Filter</span>
                    <span class="legend-badge"><span class="dot" style="background: #10b981;"></span> My GPS Location</span>
                </div>
            </div>
            <div id="map"></div>
        </div>

        <!-- Plotly Charts -->
        <div class="charts-container">
            <div class="chart-card">
                <h2>2D Position Trajectory (East vs North - Equal Aspect)</h2>
                <div id="trajPlot" class="plot-area"></div>
            </div>
            <div class="chart-card">
                <h2>Position Error Over Time (Log Scale, 10m Threshold)</h2>
                <div id="errPlot" class="plot-area"></div>
            </div>
        </div>

        <!-- Matplotlib Figures -->
        <div class="charts-container" style="margin-top: 1rem;">
            <div class="chart-card" style="grid-column: 1 / -1; height: 580px;">
                <h2>Matplotlib Benchmark Figures (drnav.plotting PNG exports)</h2>
                <div style="display: flex; gap: 1rem; height: 100%; width: 100%; justify-content: center; align-items: center; overflow: hidden;">
                    <img id="matplotImg" src="" style="max-height: 100%; max-width: 50%; object-fit: contain; border-radius: 0.375rem; border: 1px solid var(--card-border);" alt="Trajectory Plot" />
                    <img id="biasImg" src="" style="max-height: 100%; max-width: 50%; object-fit: contain; border-radius: 0.375rem; border: 1px solid var(--card-border);" alt="IMU Bias Plot" />
                </div>
            </div>
        </div>
    </div>

    <footer>
        ISRO SIH26168 Navigation Baseline & Error-State Kalman Filter Suite
    </footer>

    <script>
        let map, truthLayer, gnssLayer, baseLayer, eskfLayer, vehicleMarker, userGpsMarker;
        let animationTimer = null;
        let animIndex = 0;
        let currentData = null;
        let userLat = 12.9716;
        let userLon = 77.5946;

        // Telemetry State
        let isPhoneBroadcasting = false;
        let lastPostTimestamp = 0;
        let localTelemetry = {{
            pitch: 0.0, roll: 0.0, yaw: 0.0,
            ax: 0.0, ay: 0.0, az: 9.81, amag: 9.81,
            gx: 0.0, gy: 0.0, gz: 0.0,
            still: true
        }};

        function initMap() {{
            map = L.map('map').setView([userLat, userLon], 15);

            const osmTiles = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap contributors'
            }}).addTo(map);

            const darkTiles = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap &copy; CARTO'
            }});

            const satTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                maxZoom: 19,
                attribution: '&copy; Esri World Imagery'
            }});

            L.control.layers({{
                "OpenStreetMap": osmTiles,
                "Dark Mode": darkTiles,
                "Satellite": satTiles
            }}).addTo(map);

            truthLayer = L.polyline([], {{color: '#000000', weight: 4, opacity: 0.9}}).addTo(map);
            baseLayer = L.polyline([], {{color: '#f97316', weight: 3, dashArray: '6, 6', opacity: 0.8}}).addTo(map);
            eskfLayer = L.polyline([], {{color: '#2563eb', weight: 4, opacity: 0.95}}).addTo(map);
            gnssLayer = L.layerGroup().addTo(map);

            const vehicleIcon = L.divIcon({{
                className: 'custom-vehicle-icon',
                html: '<div style="background:#0284c7; width:18px; height:18px; border-radius:50%; border:3px solid #ffffff; box-shadow:0 0 10px #0284c7;"></div>',
                iconSize: [18, 18],
                iconAnchor: [9, 9]
            }});
            vehicleMarker = L.marker([userLat, userLon], {{icon: vehicleIcon}}).addTo(map);

            const userIcon = L.divIcon({{
                className: 'custom-user-icon',
                html: '<div style="background:#10b981; width:22px; height:22px; border-radius:50%; border:3px solid #ffffff; box-shadow:0 0 12px #10b981; display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:bold;">📍</div>',
                iconSize: [22, 22],
                iconAnchor: [11, 11]
            }});
            userGpsMarker = L.marker([userLat, userLon], {{icon: userIcon}}).addTo(map).bindPopup("<b>Your Current Location</b>");

            setTimeout(() => {{
                map.invalidateSize();
            }}, 200);
        }}

        // ==========================================
        // 📱 LIVE PHONE IMU & SENSORS IMPLEMENTATION
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

            // Rotate canvas for Roll
            ctx.translate(cx, cy);
            const rollRad = (roll * Math.PI) / 180;
            ctx.rotate(rollRad);

            // Vertical pitch offset (1.5 px per degree)
            const pitchOffset = Math.max(-cy, Math.min(cy, pitch * 1.5));

            // Sky (Blue)
            ctx.fillStyle = "#1e3a8a";
            ctx.fillRect(-w, -h * 2, w * 2, h * 2 + pitchOffset);

            // Ground (Slate / Earth)
            ctx.fillStyle = "#334155";
            ctx.fillRect(-w, pitchOffset, w * 2, h * 2);

            // White Horizon Dividing Line
            ctx.strokeStyle = "#f8fafc";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(-w, pitchOffset);
            ctx.lineTo(w, pitchOffset);
            ctx.stroke();

            // Pitch Ladder Marks (-30° to +30°)
            ctx.strokeStyle = "rgba(248, 250, 252, 0.5)";
            ctx.fillStyle = "rgba(248, 250, 252, 0.8)";
            ctx.font = "9px Inter, sans-serif";
            [-30, -20, -10, 10, 20, 30].forEach(deg => {{
                const y = pitchOffset - (deg * 1.5);
                if (y > -cy + 12 && y < cy - 12) {{
                    const lineLen = deg % 20 === 0 ? 26 : 14;
                    ctx.beginPath();
                    ctx.moveTo(-lineLen, y);
                    ctx.lineTo(lineLen, y);
                    ctx.stroke();
                    ctx.fillText(Math.abs(deg) + "°", lineLen + 4, y + 3);
                }}
            }});

            ctx.restore();

            // Fixed Reticle (Aircraft symbol)
            ctx.strokeStyle = "#f2c14e";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(cx - 25, cy);
            ctx.lineTo(cx - 8, cy);
            ctx.lineTo(cx, cy + 6);
            ctx.lineTo(cx + 8, cy);
            ctx.lineTo(cx + 25, cy);
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
        // GPS DETECTION & SCENARIO RUNNER
        // ==========================================

        function detectGPSLocation() {{
            const gpsStatus = document.getElementById('gpsStatus');
            if ("geolocation" in navigator) {{
                gpsStatus.innerText = "📍 Requesting GPS position...";
                navigator.geolocation.getCurrentPosition(
                    (pos) => {{
                        userLat = pos.coords.latitude;
                        userLon = pos.coords.longitude;
                        gpsStatus.innerText = `📍 GPS Fixed: ${{userLat.toFixed(4)}}, ${{userLon.toFixed(4)}}`;
                        document.getElementById('originCoords').innerText = `${{userLat.toFixed(4)}}, ${{userLon.toFixed(4)}}`;
                        document.getElementById('coordsSub').innerText = "Real-Time Device GPS Origin";
                        
                        userGpsMarker.setLatLng([userLat, userLon]);
                        userGpsMarker.getPopup().setContent(`<b>Your Real GPS Location</b><br>Lat: ${{userLat.toFixed(6)}}<br>Lon: ${{userLon.toFixed(6)}}`);
                        map.setView([userLat, userLon], 16);
                        
                        loadScenario();
                    }},
                    (err) => {{
                        gpsStatus.innerText = "📍 GPS Permission Denied (Using default reference origin)";
                        loadScenario();
                    }},
                    {{ enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }}
                );
            }} else {{
                gpsStatus.innerText = "📍 Browser Geolocation Not Supported";
                loadScenario();
            }}
        }}

        async function loadScenario() {{
            stopAnimation();
            const scenario = document.getElementById('scenarioSelect').value;
            const btn = document.getElementById('runBtn');
            const statusText = document.getElementById('statusText');

            btn.disabled = true;
            statusText.innerText = "Simulating...";
            statusText.style.color = "#f2c14e";

            try {{
                const res = await fetch(`/api/run?scenario=${{scenario}}&lat=${{userLat}}&lon=${{userLon}}`);
                currentData = await res.json();

                document.getElementById('eskfRmse').innerText = `${{currentData.eskf_rmse.toFixed(2)}} m`;
                document.getElementById('baseRmse').innerText = `${{currentData.base_rmse.toFixed(2)}} m`;
                document.getElementById('matplotImg').src = `data:image/png;base64,${{currentData.fig_png}}`;
                document.getElementById('biasImg').src = `data:image/png;base64,${{currentData.bias_png}}`;

                updateMap(currentData);
                renderPlotly(currentData);

                statusText.innerText = "Active";
                statusText.style.color = "#4ade80";
            }} catch (err) {{
                console.error(err);
                statusText.innerText = "Error";
                statusText.style.color = "#c44e52";
            }} finally {{
                btn.disabled = false;
            }}
        }}

        function updateMap(data) {{
            const truthLatLngs = data.truth_latlon;
            const baseLatLngs = data.base_latlon;
            const eskfLatLngs = data.eskf_latlon;
            const gnssLatLngs = data.gnss_latlon;

            truthLayer.setLatLngs(truthLatLngs);
            baseLayer.setLatLngs(baseLatLngs);
            eskfLayer.setLatLngs(eskfLatLngs);

            gnssLayer.clearLayers();
            gnssLatLngs.forEach(pt => {{
                L.circleMarker(pt, {{
                    radius: 3,
                    color: '#ef4444',
                    fillColor: '#ef4444',
                    fillOpacity: 0.8
                }}).addTo(gnssLayer);
            }});

            if (eskfLatLngs.length > 0) {{
                vehicleMarker.setLatLng(eskfLatLngs[0]);
                const bounds = L.latLngBounds(truthLatLngs);
                map.fitBounds(bounds, {{padding: [40, 40]}});
                map.invalidateSize();
            }}
        }}

        function toggleAnimation() {{
            if (animationTimer) {{
                stopAnimation();
            }} else {{
                startAnimation();
            }}
        }}

        function startAnimation() {{
            if (!currentData || !currentData.eskf_latlon) return;
            const btn = document.getElementById('animBtn');
            btn.innerText = "⏸ Pause Live Drive";
            btn.style.backgroundColor = "#dc2626";

            animIndex = 0;
            const latlngs = currentData.eskf_latlon;
            animationTimer = setInterval(() => {{
                if (animIndex < latlngs.length) {{
                    vehicleMarker.setLatLng(latlngs[animIndex]);
                    map.panTo(latlngs[animIndex], {{animate: true, duration: 0.05}});
                    animIndex += 5;
                }} else {{
                    stopAnimation();
                }}
            }}, 50);
        }}

        function stopAnimation() {{
            if (animationTimer) {{
                clearInterval(animationTimer);
                animationTimer = null;
            }}
            const btn = document.getElementById('animBtn');
            if (btn) {{
                btn.innerText = "▶ Play Live Drive";
                btn.style.backgroundColor = "#059669";
            }}
        }}

        function renderPlotly(data) {{
            const t = data.t;
            const truth = data.truth_pos;
            const gnss = data.gnss_pos;
            const base = data.base_pos;
            const eskf = data.eskf_pos;

            const tracesTraj = [
                {{
                    x: truth.map(p => p[0]),
                    y: truth.map(p => p[1]),
                    mode: 'lines',
                    name: 'Ground Truth',
                    line: {{ color: '#ffffff', width: 2.5 }}
                }},
                {{
                    x: gnss.map(p => p[0]),
                    y: gnss.map(p => p[1]),
                    mode: 'markers',
                    name: 'GNSS Fixes',
                    marker: {{ color: '#ef4444', size: 4, symbol: 'cross' }}
                }},
                {{
                    x: base.map(p => p[0]),
                    y: base.map(p => p[1]),
                    mode: 'lines',
                    name: 'Classical DR',
                    line: {{ color: '#f97316', width: 2, dash: 'dash' }}
                }},
                {{
                    x: eskf.map(p => p[0]),
                    y: eskf.map(p => p[1]),
                    mode: 'lines',
                    name: 'ESKF Filter',
                    line: {{ color: '#3b82f6', width: 3 }}
                }}
            ];

            const layoutTraj = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0f172a',
                margin: {{ l: 50, r: 20, t: 20, b: 50 }},
                xaxis: {{
                    title: 'East (m)',
                    gridcolor: '#1e293b',
                    zerolinecolor: '#334155',
                    color: '#94a3b8',
                    scaleanchor: 'y',
                    scaleratio: 1
                }},
                yaxis: {{
                    title: 'North (m)',
                    gridcolor: '#1e293b',
                    zerolinecolor: '#334155',
                    color: '#94a3b8'
                }},
                legend: {{
                    x: 0.02, y: 0.98,
                    bgcolor: 'rgba(19, 27, 46, 0.8)',
                    font: {{ color: '#f8fafc' }}
                }}
            }};

            Plotly.newPlot('trajPlot', tracesTraj, layoutTraj, {{responsive: true}});

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
                    line: {{ color: '#3b82f6', width: 2.5 }}
                }}
            ];

            const layoutErr = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: '#0f172a',
                margin: {{ l: 50, r: 20, t: 20, b: 50 }},
                xaxis: {{
                    title: 'Time (s)',
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
                yaxis: {{
                    title: '2D Position Error (m)',
                    type: 'log',
                    gridcolor: '#1e293b',
                    color: '#94a3b8'
                }},
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
                legend: {{
                    x: 0.02, y: 0.98,
                    bgcolor: 'rgba(19, 27, 46, 0.8)',
                    font: {{ color: '#f8fafc' }}
                }}
            }};

            Plotly.newPlot('errPlot', tracesErr, layoutErr, {{responsive: true}});
        }}

        window.onload = () => {{
            initMap();
            drawHorizon(0, 0, 0);
            detectGPSLocation();
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

        try:
            lat0 = float(query.get("lat", [12.9716])[0])
            lon0 = float(query.get("lon", [77.5946])[0])
        except Exception:
            lat0 = 12.9716
            lon0 = 77.5946

        if scenario_name not in SCENARIOS:
            scenario_name = "tunnel_drive"

        sc_fn = SCENARIOS[scenario_name]
        scenario = sc_fn()
        rec = simulate(scenario, seed=0)

        align = coarse_align(rec)
        outages, labels = truth_outages(rec)
        base = run_baseline(rec, align)
        est = run_eskf(rec, align)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = OUT_DIR / f"{scenario_name}_dash.png"
        bias_path = OUT_DIR / f"{scenario_name}_bias_dash.png"

        s_base = metrics.evaluate(rec.t, base["pos"], rec.truth_pos, outages, labels)
        s_eskf = metrics.evaluate(rec.t, est["pos"], rec.truth_pos, outages, labels)

        plotting.plot_run(rec, base, est, s_base, s_eskf, outages,
                          f"{scenario_name} -- dead reckoning", fig_path)
        plotting.plot_bias(rec, est, bias_path)

        with open(fig_path, "rb") as f:
            fig_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(bias_path, "rb") as f:
            bias_b64 = base64.b64encode(f.read()).decode("utf-8")

        base_err = np.linalg.norm(base["pos"][:, :2] - rec.truth_pos[:, :2], axis=1)
        eskf_err = np.linalg.norm(est["pos"][:, :2] - rec.truth_pos[:, :2], axis=1)

        # Convert 2D ENU meters to Geodetic Lat/Lon using user's real GPS origin
        truth_latlon = enu_to_geodetic(rec.truth_pos[:, 0], rec.truth_pos[:, 1], lat0, lon0)
        base_latlon = enu_to_geodetic(base["pos"][:, 0], base["pos"][:, 1], lat0, lon0)
        eskf_latlon = enu_to_geodetic(est["pos"][:, 0], est["pos"][:, 1], lat0, lon0)
        gnss_latlon = enu_to_geodetic(rec.gnss[:, 1], rec.gnss[:, 2], lat0, lon0)

        def clean_arr(arr: np.ndarray) -> list:
            clean = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            return clean.tolist()

        response_data: Dict[str, Any] = {
            "t": clean_arr(rec.t),
            "truth_pos": clean_arr(rec.truth_pos),
            "gnss_pos": clean_arr(rec.gnss[:, 1:4]),
            "base_pos": clean_arr(base["pos"]),
            "eskf_pos": clean_arr(est["pos"]),
            "truth_latlon": clean_arr(truth_latlon),
            "base_latlon": clean_arr(base_latlon),
            "eskf_latlon": clean_arr(eskf_latlon),
            "gnss_latlon": clean_arr(gnss_latlon),
            "base_err": clean_arr(base_err),
            "eskf_err": clean_arr(eskf_err),
            "base_rmse": float(np.nan_to_num(s_base.ate_rmse, nan=0.0)),
            "eskf_rmse": float(np.nan_to_num(s_eskf.ate_rmse, nan=0.0)),
            "max_jump": float(np.nan_to_num(s_eskf.max_jump, nan=0.0)),
            "fig_png": fig_b64,
            "bias_png": bias_b64,
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
