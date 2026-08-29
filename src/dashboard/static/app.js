/**
 * NAVIS 3D NAVIGATION DASHBOARD - CLIENT CONTROLLER & 3D WEBGL ENGINE
 */

(function () {
    'use strict';

    // Global Application State
    const AppState = {
        data: null,
        isPlaying: false,
        currentIndex: 0,
        playbackSpeed: 1.0,
        animationFrameId: null,
        lastFrameTime: 0,
        traceVisibility: {
            gt: true,
            mod2: true,
            mod3: true,
            mod4: true,
            mod5: true,
        }
    };

    // DOM Elements
    const elements = {
        profileSelect: document.getElementById('profile-select'),
        imuGradeSelect: document.getElementById('imu-grade-select'),
        durationInput: document.getElementById('duration-input'),
        seedInput: document.getElementById('seed-input'),
        gpsOutageToggle: document.getElementById('gps-outage-toggle'),
        outageStart: document.getElementById('outage-start'),
        outageEnd: document.getElementById('outage-end'),
        runSimBtn: document.getElementById('run-sim-btn'),
        loadingSpinner: document.getElementById('loading-spinner'),
        timeScrubber: document.getElementById('time-scrubber'),
        btnPlayPause: document.getElementById('btn-play-pause'),
        btnRewind: document.getElementById('btn-rewind'),
        speedBtns: document.querySelectorAll('.speed-btn'),
        telemetryTime: document.getElementById('telemetry-time'),
        gpsStatusChip: document.getElementById('gps-status-chip'),
        gpsStatusVal: document.getElementById('gps-status-val'),
        motionContextVal: document.getElementById('motion-context-val'),
        zuptVal: document.getElementById('zupt-val'),
        speedVal: document.getElementById('speed-val'),
        altVal: document.getElementById('alt-val'),
        rollVal: document.getElementById('roll-val'),
        pitchVal: document.getElementById('pitch-val'),
        yawVal: document.getElementById('yaw-val'),
        benchmarkTbody: document.getElementById('benchmark-tbody'),
        horizonCanvas: document.getElementById('horizon-canvas'),
        compassCanvas: document.getElementById('compass-canvas'),
    };

    // ========================================================================
    // 1. INITIALIZATION & DATA LOADING
    // ========================================================================

    async function init() {
        setupEventListeners();
        initCanvases();

        if (window.__PRELOADED_NAVIS_DATA__) {
            console.log('[NAVIS] Loading embedded preloaded dataset...');
            loadSimulationData(window.__PRELOADED_NAVIS_DATA__);
            return;
        }

        console.log('[NAVIS] Fetching initial sample simulation...');
        try {
            const response = await fetch('/api/sample', { cache: 'no-cache' });
            if (response.ok) {
                const data = await response.json();
                loadSimulationData(data);
            } else {
                elements.loadingSpinner.classList.add('hidden');
            }
        } catch (err) {
            console.warn('[NAVIS] Initial sample load waiting for server:', err);
            elements.loadingSpinner.classList.add('hidden');
        }
    }

    function setupEventListeners() {
        // Run Simulation
        elements.runSimBtn.addEventListener('click', () => runSimulation());

        // Playback Buttons
        elements.btnPlayPause.addEventListener('click', togglePlayPause);
        elements.btnRewind.addEventListener('click', rewind);

        // Scrubber
        elements.timeScrubber.addEventListener('input', (e) => {
            const pct = parseFloat(e.target.value);
            if (AppState.data) {
                const total = AppState.data.timestamps.length;
                const idx = Math.min(Math.floor((pct / 100) * total), total - 1);
                seekTo(idx);
            }
        });

        // Speed Buttons
        elements.speedBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.speedBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                AppState.playbackSpeed = parseFloat(btn.dataset.speed) || 1.0;
            });
        });

        // Trace Legend Checkboxes
        ['gt', 'mod2', 'mod3', 'mod4', 'mod5'].forEach(id => {
            const chk = document.getElementById(`trace-${id}`);
            if (chk) {
                chk.addEventListener('change', (e) => {
                    AppState.traceVisibility[id] = e.target.checked;
                    updateTraceVisibility();
                });
            }
        });

        // Camera Views
        document.getElementById('view-3d-btn').addEventListener('click', () => setCameraView('3d'));
        document.getElementById('view-top-btn').addEventListener('click', () => setCameraView('top'));
        document.getElementById('view-side-btn').addEventListener('click', () => setCameraView('side'));
        document.getElementById('reset-cam-btn').addEventListener('click', () => setCameraView('reset'));
    }

    // ========================================================================
    // 2. SIMULATION RUNNER & API CALL
    // ========================================================================

    async function runSimulation() {
        elements.loadingSpinner.classList.remove('hidden');
        elements.runSimBtn.disabled = true;

        const hasOutage = elements.gpsOutageToggle.checked;
        const outageStart = parseFloat(elements.outageStart.value) || 15.0;
        const outageEnd = parseFloat(elements.outageEnd.value) || 30.0;

        const params = {
            profile: elements.profileSelect.value,
            imu_grade: elements.imuGradeSelect.value,
            duration: parseFloat(elements.durationInput.value) || 40.0,
            seed: parseInt(elements.seedInput.value) || 42,
            gps_outage: hasOutage ? [outageStart, outageEnd] : null,
            dt: 0.02,
        };

        try {
            const res = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });

            if (res.ok) {
                const data = await res.json();
                loadSimulationData(data);
            } else {
                const errData = await res.json().catch(() => ({}));
                console.error('[NAVIS] Simulation backend error:', errData);
            }
        } catch (err) {
            console.error('[NAVIS] Simulation network error:', err);
        } finally {
            elements.loadingSpinner.classList.add('hidden');
            elements.runSimBtn.disabled = false;
        }
    }

    function loadSimulationData(data) {
        AppState.data = data;
        AppState.currentIndex = 0;
        pause();

        // Render Plots
        render3DTrajectory(data);
        renderDriftChart(data);
        renderBiasChart(data);
        renderBenchmarkTable(data.scorecards);

        // Update Gauges to start
        updateTelemetryGauges(0);
    }

    // ========================================================================
    // 3. 3D WEBGL TRAJECTORY PLOT (PLOTLY.JS)
    // ========================================================================

    function render3DTrajectory(data) {
        const tr = data.trajectories;

        const traces = [
            // 0: Ground Truth
            {
                type: 'scatter3d',
                mode: 'lines',
                name: 'Ground Truth',
                x: tr.ground_truth.x,
                y: tr.ground_truth.y,
                z: tr.ground_truth.z,
                line: { color: '#ffffff', width: 4, dash: 'dash' },
                visible: AppState.traceVisibility.gt,
            },
            // 1: Classical DR (Mod 2)
            {
                type: 'scatter3d',
                mode: 'lines',
                name: 'Classical DR (Mod 2)',
                x: tr.classical_dr.x,
                y: tr.classical_dr.y,
                z: tr.classical_dr.z,
                line: { color: '#ef4444', width: 3 },
                visible: AppState.traceVisibility.mod2,
            },
            // 2: AI-ZUPT (Mod 3)
            {
                type: 'scatter3d',
                mode: 'lines',
                name: 'AI-ZUPT (Mod 3)',
                x: tr.zupt_dr.x,
                y: tr.zupt_dr.y,
                z: tr.zupt_dr.z,
                line: { color: '#f59e0b', width: 3.5 },
                visible: AppState.traceVisibility.mod3,
            },
            // 3: Neural DR (Mod 4)
            {
                type: 'scatter3d',
                mode: 'lines',
                name: 'Neural DR (Mod 4)',
                x: tr.neural_dr.x,
                y: tr.neural_dr.y,
                z: tr.neural_dr.z,
                line: { color: '#06b6d4', width: 3.5 },
                visible: AppState.traceVisibility.mod4,
            },
            // 4: AI-EKF Fusion (Mod 5)
            {
                type: 'scatter3d',
                mode: 'lines',
                name: 'AI-EKF Fusion (Mod 5)',
                x: tr.ai_ekf.x,
                y: tr.ai_ekf.y,
                z: tr.ai_ekf.z,
                line: { color: '#10b981', width: 4.5 },
                visible: AppState.traceVisibility.mod5,
            },
            // 5: Current Position Vehicle Head Marker (EKF)
            {
                type: 'scatter3d',
                mode: 'markers',
                name: 'Vehicle State',
                x: [tr.ai_ekf.x[0]],
                y: [tr.ai_ekf.y[0]],
                z: [tr.ai_ekf.z[0]],
                marker: { color: '#00f0ff', size: 14, symbol: 'diamond', line: { color: '#ffffff', width: 2 } },
                showlegend: false,
            }
        ];

        const layout = {
            autosize: true,
            paper_bgcolor: '#000000',
            plot_bgcolor: '#000000',
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: false,
            scene: {
                xaxis: { title: 'East (X) [m]', color: '#94a3b8', gridcolor: '#1f2937', zerolinecolor: '#374151' },
                yaxis: { title: 'North (Y) [m]', color: '#94a3b8', gridcolor: '#1f2937', zerolinecolor: '#374151' },
                zaxis: { title: 'Up (Z) [m]', color: '#94a3b8', gridcolor: '#1f2937', zerolinecolor: '#374151' },
                aspectmode: 'data',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.2 }
                }
            }
        };

        Plotly.newPlot('plotly-3d-canvas', traces, layout, { responsive: true, displayModeBar: false });
    }

    function updateTraceVisibility() {
        if (!AppState.data) return;
        const vis = [
            AppState.traceVisibility.gt,
            AppState.traceVisibility.mod2,
            AppState.traceVisibility.mod3,
            AppState.traceVisibility.mod4,
            AppState.traceVisibility.mod5,
            true, // marker
        ];
        Plotly.restyle('plotly-3d-canvas', { visible: vis });
    }

    function setCameraView(viewType) {
        if (!AppState.data) return;
        let eye = { x: 1.5, y: 1.5, z: 1.2 };
        if (viewType === 'top') {
            eye = { x: 0.0, y: 0.001, z: 2.5 }; // Top-down
        } else if (viewType === 'side') {
            eye = { x: 2.5, y: 0.001, z: 0.2 }; // Side elevation
        }
        Plotly.relayout('plotly-3d-canvas', { 'scene.camera.eye': eye });
    }

    // ========================================================================
    // 4. PLAYBACK & ANIMATION CONTROLLER
    // ========================================================================

    function togglePlayPause() {
        if (AppState.isPlaying) {
            pause();
        } else {
            play();
        }
    }

    function play() {
        if (!AppState.data) return;
        const N = AppState.data.timestamps.length;
        if (AppState.currentIndex >= N - 1) {
            AppState.currentIndex = 0;
        }
        AppState.isPlaying = true;
        elements.btnPlayPause.innerHTML = '&#10074;&#10074;';
        AppState.lastFrameTime = performance.now();
        AppState.animationFrameId = requestAnimationFrame(animationLoop);
    }

    function pause() {
        AppState.isPlaying = false;
        elements.btnPlayPause.innerHTML = '&#9654;';
        if (AppState.animationFrameId) {
            cancelAnimationFrame(AppState.animationFrameId);
            AppState.animationFrameId = null;
        }
    }

    function rewind() {
        seekTo(0);
    }

    function seekTo(index) {
        if (!AppState.data) return;
        const N = AppState.data.timestamps.length;
        AppState.currentIndex = Math.max(0, Math.min(index, N - 1));

        elements.timeScrubber.value = (AppState.currentIndex / (N - 1)) * 100;
        updateTelemetryGauges(AppState.currentIndex);
        updateVehicleMarker(AppState.currentIndex);
    }

    function animationLoop(timestamp) {
        if (!AppState.isPlaying || !AppState.data) return;

        const delta = (timestamp - AppState.lastFrameTime) / 1000.0;
        AppState.lastFrameTime = timestamp;

        const d = AppState.data;
        const N = d.timestamps.length;
        const totalDuration = (d.timestamps[N - 1] - d.timestamps[0]) || 40.0;
        const effectiveDt = totalDuration / Math.max(1, N - 1);
        const stepsToAdvance = (delta * AppState.playbackSpeed) / effectiveDt;

        AppState.currentIndex += stepsToAdvance;

        if (AppState.currentIndex >= N - 1) {
            AppState.currentIndex = N - 1;
            seekTo(AppState.currentIndex);
            pause();
            return;
        }

        const idx = Math.floor(AppState.currentIndex);
        elements.timeScrubber.value = (idx / (N - 1)) * 100;
        updateTelemetryGauges(idx);
        updateVehicleMarker(idx);

        AppState.animationFrameId = requestAnimationFrame(animationLoop);
    }

    function updateVehicleMarker(idx) {
        if (!AppState.data) return;
        const tr = AppState.data.trajectories.ai_ekf;
        Plotly.restyle('plotly-3d-canvas', {
            x: [[tr.x[idx]]],
            y: [[tr.y[idx]]],
            z: [[tr.z[idx]]],
        }, [5]);
    }

    // ========================================================================
    // 5. INSTRUMENTS & TELEMETRY GAUGES
    // ========================================================================

    function updateTelemetryGauges(idx) {
        const d = AppState.data;
        if (!d) return;

        const t = d.timestamps[idx];
        const mins = Math.floor(t / 60);
        const secs = (t % 60).toFixed(2);
        elements.telemetryTime.textContent = `${mins.toString().padStart(2, '0')}:${secs.padStart(5, '0')}`;

        // Speed & Altitude
        const speed = d.telemetry.speed_ekf[idx];
        const alt = d.trajectories.ai_ekf.z[idx];
        elements.speedVal.innerHTML = `${speed.toFixed(2)} <small>m/s</small>`;
        elements.altVal.innerHTML = `${alt.toFixed(2)} <small>m</small>`;

        // Angles
        const roll = d.telemetry.roll_deg[idx];
        const pitch = d.telemetry.pitch_deg[idx];
        const yaw = (d.telemetry.yaw_deg[idx] + 360) % 360;

        elements.rollVal.textContent = `${roll >= 0 ? '+' : ''}${roll.toFixed(1)}°`;
        elements.pitchVal.textContent = `${pitch >= 0 ? '+' : ''}${pitch.toFixed(1)}°`;
        elements.yawVal.textContent = `${Math.round(yaw).toString().padStart(3, '0')}°`;

        // Chips
        const isGps = Boolean(d.telemetry.gps_valid[idx]);
        const handoffState = d.telemetry.handoff_state[idx];
        elements.gpsStatusVal.textContent = isGps ? 'LOCKED' : (handoffState === 'recovering_gps' ? 'RECOVERING' : 'DENIED (AI-DR)');
        elements.gpsStatusChip.querySelector('.pulse-dot').className = `pulse-dot ${isGps ? 'green' : (handoffState === 'recovering_gps' ? 'amber' : 'red')}`;

        elements.motionContextVal.textContent = d.telemetry.motion_context[idx].toUpperCase();
        const isStance = Boolean(d.telemetry.stance_flag[idx]);
        elements.zuptVal.textContent = isStance ? 'STANCE (ZUPT)' : 'MOVING';
        elements.zuptChip.querySelector('.pulse-dot').className = `pulse-dot ${isStance ? 'purple' : 'green'}`;

        // Draw Instrument Canvases
        drawHorizon(roll, pitch);
        drawCompass(yaw);
    }

    let ctxHorizon, ctxCompass;
    function initCanvases() {
        ctxHorizon = elements.horizonCanvas.getContext('2d');
        ctxCompass = elements.compassCanvas.getContext('2d');
        drawHorizon(0, 0);
        drawCompass(0);
    }

    function drawHorizon(rollDeg, pitchDeg) {
        if (!ctxHorizon) return;
        const w = elements.horizonCanvas.width;
        const h = elements.horizonCanvas.height;
        const cx = w / 2;
        const cy = h / 2;

        ctxHorizon.clearRect(0, 0, w, h);
        ctxHorizon.save();

        // Clip circular dial
        ctxHorizon.beginPath();
        ctxHorizon.arc(cx, cy, cx - 4, 0, Math.PI * 2);
        ctxHorizon.clip();

        // Rotate for roll
        ctxHorizon.translate(cx, cy);
        ctxHorizon.rotate((rollDeg * Math.PI) / 180);

        // Pitch displacement
        const pitchOffset = (pitchDeg / 90) * (h / 2);

        // Sky (blue/cyan)
        ctxHorizon.fillStyle = '#0284c7';
        ctxHorizon.fillRect(-w, -h + pitchOffset, w * 2, h);

        // Ground (brown/dark)
        ctxHorizon.fillStyle = '#78350f';
        ctxHorizon.fillRect(-w, pitchOffset, w * 2, h);

        // Horizon Line
        ctxHorizon.strokeStyle = '#ffffff';
        ctxHorizon.lineWidth = 2;
        ctxHorizon.beginPath();
        ctxHorizon.moveTo(-w, pitchOffset);
        ctxHorizon.lineTo(w, pitchOffset);
        ctxHorizon.stroke();

        ctxHorizon.restore();

        // Fixed Crosshair Aircraft Symbol
        ctxHorizon.strokeStyle = '#00f0ff';
        ctxHorizon.lineWidth = 2.5;
        ctxHorizon.beginPath();
        ctxHorizon.moveTo(cx - 25, cy);
        ctxHorizon.lineTo(cx - 8, cy);
        ctxHorizon.lineTo(cx - 8, cy + 6);
        ctxHorizon.moveTo(cx + 8, cy + 6);
        ctxHorizon.lineTo(cx + 8, cy);
        ctxHorizon.lineTo(cx + 25, cy);
        ctxHorizon.stroke();

        // Outer Bezel
        ctxHorizon.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctxHorizon.lineWidth = 3;
        ctxHorizon.beginPath();
        ctxHorizon.arc(cx, cy, cx - 2, 0, Math.PI * 2);
        ctxHorizon.stroke();
    }

    function drawCompass(yawDeg) {
        if (!ctxCompass) return;
        const w = elements.compassCanvas.width;
        const h = elements.compassCanvas.height;
        const cx = w / 2;
        const cy = h / 2;
        const r = cx - 8;

        ctxCompass.clearRect(0, 0, w, h);
        ctxCompass.save();

        ctxCompass.translate(cx, cy);
        ctxCompass.rotate((-yawDeg * Math.PI) / 180);

        // Dial Face
        ctxCompass.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctxCompass.lineWidth = 2;
        ctxCompass.beginPath();
        ctxCompass.arc(0, 0, r, 0, Math.PI * 2);
        ctxCompass.stroke();

        // Cardinal Tick Marks
        const cardinals = ['N', 'E', 'S', 'W'];
        for (let i = 0; i < 4; i++) {
            const angle = (i * Math.PI) / 2;
            const tx = (r - 16) * Math.sin(angle);
            const ty = -(r - 16) * Math.cos(angle);

            ctxCompass.fillStyle = (i === 0) ? '#ef4444' : '#94a3b8';
            ctxCompass.font = 'bold 12px Inter';
            ctxCompass.textAlign = 'center';
            ctxCompass.textBaseline = 'middle';
            ctxCompass.fillText(cardinals[i], tx, ty);
        }

        ctxCompass.restore();

        // Top Fixed Pointer (Neon Cyan)
        ctxCompass.fillStyle = '#00f0ff';
        ctxCompass.beginPath();
        ctxCompass.moveTo(cx, cy - r - 4);
        ctxCompass.lineTo(cx - 6, cy - r + 8);
        ctxCompass.lineTo(cx + 6, cy - r + 8);
        ctxCompass.closePath();
        ctxCompass.fill();
    }

    // ========================================================================
    // 6. BENCHMARK SCORECARDS & CHARTS
    // ========================================================================

    function renderBenchmarkTable(scorecards) {
        elements.benchmarkTbody.innerHTML = '';
        scorecards.forEach(sc => {
            const tr = document.createElement('tr');
            let pillClass = 'high';
            if (sc.reduction_pct === 0) pillClass = 'zero';
            else if (sc.reduction_pct < 50) pillClass = 'low';
            else if (sc.reduction_pct < 85) pillClass = 'med';

            tr.innerHTML = `
                <td>
                    <div class="benchmark-name">
                        <span class="dot" style="background: ${sc.color}"></span>
                        <span>${sc.name}</span>
                    </div>
                </td>
                <td>${sc.ate_rmse.toFixed(2)}m</td>
                <td>${sc.final_err.toFixed(2)}m</td>
                <td><span class="reduction-pill ${pillClass}">-${sc.reduction_pct.toFixed(1)}%</span></td>
            `;
            elements.benchmarkTbody.appendChild(tr);
        });
    }

    function renderDriftChart(data) {
        const t = data.timestamps;
        const err = data.errors_vs_time;

        const traces = [
            { x: t, y: err.classical_dr, mode: 'lines', name: 'Pure DR (Mod 2)', line: { color: '#ef4444', width: 2 } },
            { x: t, y: err.zupt_dr, mode: 'lines', name: 'AI-ZUPT (Mod 3)', line: { color: '#f59e0b', width: 2 } },
            { x: t, y: err.neural_dr, mode: 'lines', name: 'Neural DR (Mod 4)', line: { color: '#06b6d4', width: 2 } },
            { x: t, y: err.ai_ekf, mode: 'lines', name: 'AI-EKF (Mod 5)', line: { color: '#10b981', width: 2.5 } },
        ];

        const layout = {
            autosize: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 30, r: 10, b: 25, t: 10 },
            showlegend: false,
            xaxis: { color: '#64748b', gridcolor: '#1e293b' },
            yaxis: { color: '#64748b', gridcolor: '#1e293b' },
        };

        Plotly.newPlot('drift-chart', traces, layout, { responsive: true, displayModeBar: false });
    }

    function renderBiasChart(data) {
        const t = data.timestamps;
        const tel = data.telemetry;

        const traces = [
            { x: t, y: tel.acc_bias_x, mode: 'lines', name: 'b_ax', line: { color: '#ef4444', width: 1.5 } },
            { x: t, y: tel.acc_bias_y, mode: 'lines', name: 'b_ay', line: { color: '#10b981', width: 1.5 } },
            { x: t, y: tel.acc_bias_z, mode: 'lines', name: 'b_az', line: { color: '#3b82f6', width: 1.5 } },
        ];

        const layout = {
            autosize: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 30, r: 10, b: 25, t: 10 },
            showlegend: false,
            xaxis: { color: '#64748b', gridcolor: '#1e293b' },
            yaxis: { color: '#64748b', gridcolor: '#1e293b' },
        };

        Plotly.newPlot('bias-chart', traces, layout, { responsive: true, displayModeBar: false });
    }

    // Run on DOM loaded
    document.addEventListener('DOMContentLoaded', init);
})();
