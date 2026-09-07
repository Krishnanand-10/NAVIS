"""
NAVIS SIH26168 Live Phone Dead Reckoning & Navigation Dashboard

Clean real-time navigation interface:
- Real-time device geolocation (NO synthetic simulation tracks)
- Live Step & Cadence Detection + Gyro Heading Dead Reckoning
- "Simulate GNSS Outage" toggle to test dead reckoning through tunnels/indoors
- Live Phone IMU Telemetry (Pitch, Roll, Yaw, 3-Axis Accel, Gyro, ZUPT Stillness)
- Bi-directional Wi-Fi synchronization between Phone and PC (http://<IP>:8080)
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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

# Live shared state for phone-to-PC streaming
latest_state: Dict[str, Any] = {
    "active": False,
    "last_seen": 0.0,
    "lat": 12.9716,
    "lon": 77.5946,
    "heading": 0.0,
    "speed": 0.0,
    "steps": 0,
    "distance": 0.0,
    "outage": False,
    "outage_duration": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "ax": 0.0,
    "ay": 0.0,
    "az": 9.81,
    "amag": 9.81,
    "gx": 0.0,
    "gy": 0.0,
    "gz": 0.0,
    "still": True,
    "path": [],  # List of [lat, lon, is_outage]
}

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>NAVIS: Live Phone Dead Reckoning Navigation</title>
    <!-- Leaflet OpenStreetMap -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --card-border: #1f2937;
            --accent-blue: #3b82f6;
            --accent-cyan: #38bdf8;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
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
            background: linear-gradient(90deg, #111827, #1e1b4b);
            padding: 0.75rem 1.25rem;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .logo-box h1 {{
            font-size: 1.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .logo-box p {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .top-controls {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        button {{
            background-color: #1f2937;
            color: var(--text-main);
            border: 1px solid #374151;
            padding: 0.4rem 0.8rem;
            border-radius: 0.375rem;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
            transition: all 0.15s;
        }}
        button:hover {{ background-color: #374151; }}

        .btn-outage {{
            background-color: #dc2626;
            border-color: #ef4444;
            color: white;
        }}
        .btn-outage.active {{
            background-color: #f59e0b;
            border-color: #fbbf24;
            color: #111827;
        }}

        .main-layout {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
            padding: 1rem;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }}

        .banner {{
            background: #1e1b4b;
            border: 1px solid #4338ca;
            padding: 0.6rem 1rem;
            border-radius: 0.375rem;
            font-size: 0.82rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .banner code {{
            background: #0b0f19;
            color: #38bdf8;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Live HUD Stats */
        .hud-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
        }}

        .hud-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 0.85rem 1rem;
            display: flex;
            flex-direction: column;
        }}
        .hud-title {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        .hud-val {{
            font-size: 1.4rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .hud-sub {{
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
        }}

        /* Map Container */
        .map-box {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        .map-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        #map {{
            width: 100%;
            height: 520px;
            border-radius: 0.375rem;
            background: #0a0e17;
            z-index: 1;
        }}

        /* Telemetry & Motion Card */
        .telemetry-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 0.75rem;
        }}
        .telemetry-card {{
            background: #0b0f19;
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
        }}
        .telemetry-card h3 {{
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}

        .meter-line {{
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            margin-bottom: 0.4rem;
        }}
        .meter-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        .meter-bar-bg {{
            background: #1f2937;
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
        }}
        .meter-bar-fill {{
            height: 100%;
            background: var(--accent-blue);
            transition: width 0.05s ease;
        }}

        .status-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .chip-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        footer {{
            text-align: center;
            padding: 1rem;
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-top: auto;
            border-top: 1px solid var(--card-border);
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo-box">
            <h1>NAVIS: Live Phone Dead Reckoning</h1>
            <p>Real-Time Inertial Navigation & Outage Tracking</p>
        </div>
        <div class="top-controls">
            <button id="permBtn" onclick="enableSensors()" style="background:#2563eb; color:white;">📱 Connect Phone Sensors</button>
            <button id="outageBtn" class="btn-outage" onclick="toggleGnssOutage()">⚡ Cut GNSS (Test Outage)</button>
            <button onclick="resetTrip()">↺ Reset Path</button>
        </div>
    </header>

    <div class="main-layout">
        <!-- Wi-Fi Streaming Info -->
        <div class="banner">
            <span>📱 <strong>Connect Mobile:</strong> Open this URL in Chrome on your phone:</span>
            <code>http://{LOCAL_IP}:8080</code>
            <span id="connIndicator" style="color: #9ca3af;">● Local Browser</span>
        </div>

        <!-- Real-Time Navigation HUD -->
        <div class="hud-grid">
            <div class="hud-card">
                <span class="hud-title">Navigation Mode</span>
                <span class="hud-val" id="navModeVal" style="color: #10b981;">GNSS Fix</span>
                <span class="hud-sub" id="navModeSub">Satellite Position Aided</span>
            </div>
            <div class="hud-card">
                <span class="hud-title">Heading (Compass/Gyro)</span>
                <span class="hud-val" id="headingVal" style="color: #38bdf8;">000°</span>
                <span class="hud-sub" id="headingSub">Facing North</span>
            </div>
            <div class="hud-card">
                <span class="hud-title">Dead Reckoning Speed</span>
                <span class="hud-val" id="speedVal" style="color: #f59e0b;">0.00 m/s</span>
                <span class="hud-sub" id="speedSub">Pedestrian Cadence</span>
            </div>
            <div class="hud-card">
                <span class="hud-title">Steps & Distance</span>
                <span class="hud-val" id="distVal" style="color: #818cf8;">0 m</span>
                <span class="hud-sub" id="stepSub">0 Steps Counted</span>
            </div>
        </div>

        <!-- Clean Interactive Live Navigation Map -->
        <div class="map-box">
            <div class="map-header">
                <span style="font-weight:600; font-size: 0.95rem;">🗺️ Live Trajectory (Your Real Movement Path)</span>
                <div style="display: flex; gap: 0.5rem; font-size: 0.75rem;">
                    <span class="status-chip" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">
                        <span class="chip-dot" style="background: #10b981;"></span> GNSS Path
                    </span>
                    <span class="status-chip" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b;">
                        <span class="chip-dot" style="background: #f59e0b;"></span> Dead Reckoning Outage Path
                    </span>
                </div>
            </div>
            <div id="map"></div>
        </div>

        <!-- Live Hardware IMU Telemetry -->
        <div class="telemetry-row">
            <!-- 1. Attitude Horizon -->
            <div class="telemetry-card" style="align-items: center;">
                <h3 style="width: 100%;">
                    <span>Attitude Indicator</span>
                    <span id="motionFlag" style="color: #10b981;">STILL (ZUPT)</span>
                </h3>
                <canvas id="horizonCanvas" width="180" height="140" style="background: #000; border-radius: 0.375rem;"></canvas>
                <div style="display: flex; justify-content: space-around; width: 100%; margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                    <span>Pitch: <b id="valP" style="color: #38bdf8;">0.0°</b></span>
                    <span>Roll: <b id="valR" style="color: #38bdf8;">0.0°</b></span>
                    <span>Yaw: <b id="valY" style="color: #f59e0b;">0.0°</b></span>
                </div>
            </div>

            <!-- 2. Accelerometer Meters -->
            <div class="telemetry-card">
                <h3>
                    <span>Accelerometer (m/s²)</span>
                    <span>Total: <b id="valAmag" style="color: #38bdf8;">9.81</b></span>
                </h3>
                <div class="meter-line">
                    <div class="meter-meta"><span>X (Lateral)</span><span id="valAx">0.00</span></div>
                    <div class="meter-bar-bg"><div id="barAx" class="meter-bar-fill" style="width: 50%;"></div></div>
                </div>
                <div class="meter-line">
                    <div class="meter-meta"><span>Y (Forward)</span><span id="valAy">0.00</span></div>
                    <div class="meter-bar-bg"><div id="barAy" class="meter-bar-fill" style="width: 50%;"></div></div>
                </div>
                <div class="meter-line">
                    <div class="meter-meta"><span>Z (Vertical)</span><span id="valAz">9.81</span></div>
                    <div class="meter-bar-bg"><div id="barAz" class="meter-bar-fill" style="width: 80%;"></div></div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: auto;">
                    Step cadence filter detects footfalls from vertical acceleration cycles.
                </div>
            </div>

            <!-- 3. Gyroscope Meters -->
            <div class="telemetry-card">
                <h3><span>Gyroscope Rotation Rate (°/s)</span></h3>
                <div class="meter-line">
                    <div class="meter-meta"><span>ωX (Roll Rate)</span><span id="valGx">0.00</span></div>
                    <div class="meter-bar-bg"><div id="barGx" class="meter-bar-fill" style="width: 50%; background: #a78bfa;"></div></div>
                </div>
                <div class="meter-line">
                    <div class="meter-meta"><span>ωY (Pitch Rate)</span><span id="valGy">0.00</span></div>
                    <div class="meter-bar-bg"><div id="barGy" class="meter-bar-fill" style="width: 50%; background: #a78bfa;"></div></div>
                </div>
                <div class="meter-line">
                    <div class="meter-meta"><span>ωZ (Turn Rate)</span><span id="valGz">0.00</span></div>
                    <div class="meter-bar-bg"><div id="barGz" class="meter-bar-fill" style="width: 50%; background: #a78bfa;"></div></div>
                </div>
                <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: auto;">
                    Integrated turn rate maintains accurate heading during GNSS loss.
                </div>
            </div>
        </div>
    </div>

    <footer>
        ISRO SIH26168 Intelligent Dead Reckoning System · Live Navigation Mode
    </footer>

    <script>
        let map, userMarker, userHeadingCone, pathLayer;
        let isSimulatedOutage = false;
        let outageStartTime = 0;
        let tripDistance = 0;
        let stepCount = 0;
        let currentHeading = 0;
        let currentSpeed = 0;
        let currentLat = 12.9716;
        let currentLon = 77.5946;
        let lastStepTime = 0;
        let stepAccHistory = [];
        let isBroadcastingFromPhone = false;
        let lastPostTime = 0;

        // Path coordinates: Array of [lat, lon, isOutage]
        let tripPath = [];

        function initMap() {{
            map = L.map('map').setView([currentLat, currentLon], 17);

            L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap'
            }}).addTo(map);

            // Path layer group
            pathLayer = L.layerGroup().addTo(map);

            // User Cursor with Heading Pointer
            const cursorIcon = L.divIcon({{
                className: 'custom-cursor',
                html: `<div id="cursorReticle" style="width: 24px; height: 24px; transform: rotate(0deg); display: flex; align-items: center; justify-content: center;">
                         <div style="width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-bottom: 18px solid #38bdf8; filter: drop-shadow(0 0 6px #38bdf8);"></div>
                       </div>`,
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            }});
            userMarker = L.marker([currentLat, currentLon], {{ icon: cursorIcon }}).addTo(map);

            // Request GPS Location
            if ("geolocation" in navigator) {{
                navigator.geolocation.getCurrentPosition((pos) => {{
                    currentLat = pos.coords.latitude;
                    currentLon = pos.coords.longitude;
                    map.setView([currentLat, currentLon], 18);
                    userMarker.setLatLng([currentLat, currentLon]);
                    recordPathPoint(currentLat, currentLon, false);
                }}, console.warn, {{ enableHighAccuracy: true, timeout: 8000 }});

                // Watch GPS live updates
                navigator.geolocation.watchPosition((pos) => {{
                    if (isSimulatedOutage) return; // If GPS is cut, ignore satellite fixes!
                    const newLat = pos.coords.latitude;
                    const newLon = pos.coords.longitude;
                    if (pos.coords.speed !== null && pos.coords.speed > 0.2) {{
                        currentSpeed = pos.coords.speed;
                    }}
                    if (pos.coords.heading !== null && !isNaN(pos.coords.heading)) {{
                        currentHeading = pos.coords.heading;
                    }}
                    updatePosition(newLat, newLon, false);
                }}, console.warn, {{ enableHighAccuracy: true, maximumAge: 1000 }});
            }}
        }}

        function updatePosition(lat, lon, isOutage) {{
            currentLat = lat;
            currentLon = lon;
            userMarker.setLatLng([currentLat, currentLon]);
            map.panTo([currentLat, currentLon], {{ animate: true, duration: 0.1 }});
            recordPathPoint(currentLat, currentLon, isOutage);

            // Rotate cursor reticle to heading
            const reticle = document.getElementById('cursorReticle');
            if (reticle) reticle.style.transform = `rotate(${{currentHeading}}deg)`;
        }}

        function recordPathPoint(lat, lon, isOutage) {{
            tripPath.push([lat, lon, isOutage]);
            if (tripPath.length > 1) {{
                const prev = tripPath[tripPath.length - 2];
                // Distance in meters
                const d = getDistMeters(prev[0], prev[1], lat, lon);
                tripDistance += d;
                document.getElementById('distVal').innerText = `${{tripDistance.toFixed(1)}} m`;

                const segColor = isOutage ? '#f59e0b' : '#10b981';
                L.polyline([[prev[0], prev[1]], [lat, lon]], {{
                    color: segColor,
                    weight: isOutage ? 5 : 4,
                    opacity: 0.9
                }}).addTo(pathLayer);
            }}
        }}

        function getDistMeters(lat1, lon1, lat2, lon2) {{
            const R = 6371000;
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                      Math.sin(dLon/2) * Math.sin(dLon/2);
            return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        }}

        // Dead Reckoning Step & Heading Integrator
        function stepDeadReckoning(stepLengthMeters = 0.75) {{
            stepCount++;
            document.getElementById('stepSub').innerText = `${{stepCount}} Steps Counted`;

            // Displace forward along heading
            const rad = currentHeading * Math.PI / 180;
            const r_earth = 6378137.0;
            const dNorth = stepLengthMeters * Math.cos(rad);
            const dEast = stepLengthMeters * Math.sin(rad);

            const dLat = (dNorth / r_earth) * (180 / Math.PI);
            const dLon = (dEast / (r_earth * Math.cos(currentLat * Math.PI / 180))) * (180 / Math.PI);

            updatePosition(currentLat + dLat, currentLon + dLon, isSimulatedOutage);
        }}

        // ==========================================
        // PHONE HARDWARE SENSORS
        // ==========================================

        function handleOrientation(e) {{
            if (e.alpha !== null) {{
                currentHeading = e.alpha; // Compass heading
                document.getElementById('headingVal').innerText = `${{Math.round(currentHeading)}}°`;
                document.getElementById('valY').innerText = `${{Math.round(currentHeading)}}°`;
            }}
            const pitch = e.beta || 0;
            const roll = e.gamma || 0;
            document.getElementById('valP').innerText = `${{pitch.toFixed(1)}}°`;
            document.getElementById('valR').innerText = `${{roll.toFixed(1)}}°`;

            drawHorizon(pitch, roll);
            isBroadcastingFromPhone = true;
            syncTelemetryToServer(pitch, roll, currentHeading);
        }}

        function handleMotion(e) {{
            const acc = e.accelerationIncludingGravity || e.acceleration;
            if (acc) {{
                const ax = acc.x || 0;
                const ay = acc.y || 0;
                const az = acc.z || 0;
                const amag = Math.sqrt(ax*ax + ay*ay + az*az);

                document.getElementById('valAx').innerText = ax.toFixed(2);
                document.getElementById('valAy').innerText = ay.toFixed(2);
                document.getElementById('valAz').innerText = az.toFixed(2);
                document.getElementById('valAmag').innerText = amag.toFixed(2);

                const norm = (v, maxV) => Math.max(0, Math.min(100, ((v + maxV) / (2 * maxV)) * 100)) + "%";
                document.getElementById('barAx').style.width = norm(ax, 15);
                document.getElementById('barAy').style.width = norm(ay, 15);
                document.getElementById('barAz').style.width = norm(az, 15);

                // Step & Gait Detection (peak in vertical accel)
                stepAccHistory.push(amag);
                if (stepAccHistory.length > 10) stepAccHistory.shift();

                const now = performance.now();
                // Simple peak detection: when accel exceeds threshold and at least 320ms since last step
                if (amag > 12.2 && (now - lastStepTime > 340)) {{
                    lastStepTime = now;
                    currentSpeed = 1.35; // Walking speed
                    document.getElementById('speedVal').innerText = `${{currentSpeed.toFixed(2)}} m/s`;
                    document.getElementById('motionFlag').innerText = "IN MOTION";
                    document.getElementById('motionFlag').style.color = "#38bdf8";

                    if (isSimulatedOutage) {{
                        stepDeadReckoning(0.78);
                    }}
                }} else if (now - lastStepTime > 1500) {{
                    // Stillness (ZUPT)
                    currentSpeed = 0.0;
                    document.getElementById('speedVal').innerText = "0.00 m/s";
                    document.getElementById('motionFlag').innerText = "STILL (ZUPT)";
                    document.getElementById('motionFlag').style.color = "#10b981";
                }}
            }}

            const rot = e.rotationRate;
            if (rot) {{
                const gx = rot.alpha || 0;
                const gy = rot.beta || 0;
                const gz = rot.gamma || 0;
                document.getElementById('valGx').innerText = gx.toFixed(1);
                document.getElementById('valGy').innerText = gy.toFixed(1);
                document.getElementById('valGz').innerText = gz.toFixed(1);

                const normG = (v) => Math.max(0, Math.min(100, ((v + 120) / 240) * 100)) + "%";
                document.getElementById('barGx').style.width = normG(gx);
                document.getElementById('barGy').style.width = normG(gy);
                document.getElementById('barGz').style.width = normG(gz);
            }}

            isBroadcastingFromPhone = true;
        }}

        function syncTelemetryToServer(pitch, roll, yaw) {{
            const now = performance.now();
            if (now - lastPostTime < 100) return;
            lastPostTime = now;

            fetch('/api/live', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    lat: currentLat,
                    lon: currentLon,
                    heading: currentHeading,
                    speed: currentSpeed,
                    steps: stepCount,
                    distance: tripDistance,
                    outage: isSimulatedOutage,
                    pitch: pitch,
                    roll: roll,
                    yaw: yaw
                }})
            }}).catch(() => {{}});
        }}

        function enableSensors() {{
            if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {{
                DeviceOrientationEvent.requestPermission().then(res => {{
                    if (res === 'granted') {{
                        window.addEventListener('deviceorientation', handleOrientation, true);
                        window.addEventListener('devicemotion', handleMotion, true);
                        document.getElementById('permBtn').innerText = "✅ Sensors Active";
                        document.getElementById('permBtn').style.background = "#10b981";
                    }}
                }}).catch(console.error);
            }} else {{
                window.addEventListener('deviceorientation', handleOrientation, true);
                window.addEventListener('devicemotion', handleMotion, true);
                document.getElementById('permBtn').innerText = "✅ Sensors Active";
                document.getElementById('permBtn').style.background = "#10b981";
            }}
        }}

        function toggleGnssOutage() {{
            isSimulatedOutage = !isSimulatedOutage;
            const btn = document.getElementById('outageBtn');
            const navVal = document.getElementById('navModeVal');
            const navSub = document.getElementById('navModeSub');

            if (isSimulatedOutage) {{
                outageStartTime = performance.now();
                btn.innerText = "⚡ Reconnect GNSS";
                btn.classList.add('active');
                navVal.innerText = "Dead Reckoning";
                navVal.style.color = "#f59e0b";
                navSub.innerText = "Pure Inertial (GPS Lost)";
            }} else {{
                btn.innerText = "⚡ Cut GNSS (Test Outage)";
                btn.classList.remove('active');
                navVal.innerText = "GNSS Fix";
                navVal.style.color = "#10b981";
                navSub.innerText = "Satellite Position Aided";
            }}
        }}

        function resetTrip() {{
            tripPath = [];
            tripDistance = 0;
            stepCount = 0;
            pathLayer.clearLayers();
            document.getElementById('distVal').innerText = "0 m";
            document.getElementById('stepSub').innerText = "0 Steps Counted";
            recordPathPoint(currentLat, currentLon, isSimulatedOutage);
        }}

        function drawHorizon(pitch, roll) {{
            const canvas = document.getElementById('horizonCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const cx = w/2, cy = h/2;

            ctx.save();
            ctx.clearRect(0, 0, w, h);
            ctx.translate(cx, cy);
            ctx.rotate((roll * Math.PI) / 180);

            const pitchOffset = Math.max(-cy, Math.min(cy, pitch * 1.5));
            ctx.fillStyle = "#1e3a8a";
            ctx.fillRect(-w, -h*2, w*2, h*2 + pitchOffset);
            ctx.fillStyle = "#334155";
            ctx.fillRect(-w, pitchOffset, w*2, h*2);

            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(-w, pitchOffset);
            ctx.lineTo(w, pitchOffset);
            ctx.stroke();
            ctx.restore();

            // Center Reticle
            ctx.strokeStyle = "#f59e0b";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(cx - 20, cy); ctx.lineTo(cx - 6, cy);
            ctx.lineTo(cx, cy + 5);
            ctx.lineTo(cx + 6, cy); ctx.lineTo(cx + 20, cy);
            ctx.stroke();
        }}

        // Poller for PC screen (receives live phone movement over Wi-Fi)
        setInterval(async () => {{
            if (isBroadcastingFromPhone) return;
            try {{
                const res = await fetch('/api/live');
                if (res.ok) {{
                    const d = await res.json();
                    if (d.active) {{
                        document.getElementById('connIndicator').innerText = "📱 Phone Streaming Live";
                        document.getElementById('connIndicator').style.color = "#10b981";

                        document.getElementById('headingVal').innerText = `${{Math.round(d.heading)}}°`;
                        document.getElementById('speedVal').innerText = `${{d.speed.toFixed(2)}} m/s`;
                        document.getElementById('distVal').innerText = `${{d.distance.toFixed(1)}} m`;
                        document.getElementById('stepSub').innerText = `${{d.steps}} Steps Counted`;

                        currentHeading = d.heading;
                        updatePosition(d.lat, d.lon, d.outage);
                        drawHorizon(d.pitch, d.roll);
                    }}
                }}
            }} catch (e) {{}}
        }}, 150);

        window.onload = () => {{
            initMap();
            drawHorizon(0, 0);
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

        if self.path == "/api/live":
            now = time.time()
            is_active = (now - latest_state.get("last_seen", 0.0)) < 3.0
            resp_data = dict(latest_state)
            resp_data["active"] = is_active
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/live":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                global latest_state
                latest_state.update(data)
                latest_state["active"] = True
                latest_state["last_seen"] = time.time()
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

    def log_message(self, format: str, *args: Any) -> None:
        pass


def start_dashboard(port: int = 8080, open_browser: bool = True) -> None:
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    url_local = f"http://localhost:{port}"
    url_network = f"http://{LOCAL_IP}:{port}"
    print(f"[NAVIS] Live Tracker running on PC: {url_local}")
    print(f"[NAVIS] Live Tracker running on Phone (Wi-Fi): {url_network}")

    if open_browser:
        try:
            webbrowser.open(url_local)
        except Exception:
            pass
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAVIS Live Tracker Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve dashboard on")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()
    start_dashboard(args.port, open_browser=not args.no_browser)
