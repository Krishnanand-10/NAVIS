# NAVIS: AI/ML-Based Intelligent Dead Reckoning System for Seamless Navigation

[![ISRO - SIH](https://img.shields.io/badge/Organisation-ISRO-orange.svg)](https://www.isro.gov.in/)
[![Problem Statement](https://img.shields.io/badge/PS%20Code-SIH26168-blue.svg)](#)
[![Theme](https://img.shields.io/badge/Theme-Smart%20Automation-green.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

---

## 📌 Problem Statement & Context
Satellite-based navigation systems (**GPS / GNSS / NavIC**) experience severe signal degradation, multipath errors, or complete outages in:
- **Tunnels & Underground Facilities**
- **Dense Urban Canyons & Deep Forest Canopies**
- **Indoors & Enclosed Structures**
- **Electronic Warfare / Jammed & Denied Environments**
- **Planetary / Lunar Exploration Trajectories** (relevant to ISRO's extraterrestrial mobility systems)

### 🎯 Objective
**NAVIS** is an end-to-end, AI/ML-augmented **Dead Reckoning (DR) & Sensor Fusion Framework** designed to deliver continuous, high-precision positioning and trajectory tracking using onboard inertial sensors (**IMU**: 3-axis Accelerometer, 3-axis Gyroscope, Magnetometer) and auxiliary signals (**Barometer, Wheel Odometry**), eliminating reliance on external satellite feeds.

---

## 🚀 Key Innovations & Architecture

Traditional dead reckoning suffers from **exponential drift errors** ($O(t^2)$ position divergence) due to compounded sensor noise, temperature variance, and bias instability. NAVIS mitigates this using a hybrid physics-AI pipeline:

```mermaid
graph LR
    subgraph Raw Sensor Streams
        IMU[6-DOF IMU: Accel + Gyro]
        MAG[Magnetometer]
        BARO[Barometer / Alt]
        ODO[Wheel Odometry / Stride]
    end

    subgraph Preprocessing & Motion Context
        PRE[Noise Filter & Calibration]
        CTX[Motion Context Classifier<br/>Walking / Vehicular / Aerial]
    end

    subgraph AI Drift & Bias Correction
        LSTM[Deep Bi-LSTM / TCN<br/>Sensor Bias & Velocity Estimator]
        ZUPT[AI Zero-Velocity Detection<br/>(AI-ZUPT)]
    end

    subgraph Sensor Fusion Engine
        EKF[Adaptive AI-Enhanced<br/>Extended Kalman Filter (AI-EKF)]
        HANDOFF[Seamless GPS-to-DR<br/>Handoff Controller]
    end

    subgraph Output
        TRAJ[Continuous High-Precision 3D Trajectory]
        DASH[Real-time Analytics Dashboard]
    end

    IMU & MAG & BARO & ODO --> PRE
    PRE --> CTX
    PRE --> LSTM
    PRE --> ZUPT
    CTX & LSTM & ZUPT --> EKF
    HANDOFF --> EKF
    EKF --> TRAJ
    TRAJ --> DASH
```

### ✨ Core Capabilities
1. **AI-Driven Bias & Velocity Learning**: Neural sequence models (Bi-LSTM / Temporal Convolutional Networks) learn high-dimensional dynamic sensor biases and compute pseudo-velocity vectors.
2. **Context-Aware Adaptive Fusion**: Automatically detects motion modes (pedestrian, terrestrial rover/vehicle, aerial drone/spacecraft) and adjusts Kalman Filter process covariance matrices ($Q_k, R_k$).
3. **AI-Assisted Zero-Velocity Update (ZUPT)**: Identifies stationary stance phases with sub-millisecond precision to reset accumulated velocity errors.
4. **Seamless GPS/NavIC Hand-off**: Smooth, bump-less state transitions when moving between open-sky GPS lock and GPS-denied zones.
5. **Interactive 2D/3D Evaluation Visualizer**: Live playback and benchmarking against ground truth, Standard DR, and Classical EKF.

---

## 📁 Repository Structure

```text
NAVIS/
├── data/                      # Dataset directories (synthetic, Oxford, KITTI)
│   ├── raw/
│   └── processed/
├── src/
│   ├── core/                  # Inertial kinematics, traditional DR baselines
│   │   ├── kinematics.py
│   │   └── traditional_dr.py
│   ├── models/                # Deep learning models for bias & velocity estimation
│   │   ├── motion_classifier.py
│   │   └── lstm_drift_estimator.py
│   ├── fusion/                # Extended Kalman Filter and GPS handoff logic
│   │   ├── ekf.py
│   │   ├── ai_ekf.py
│   │   └── handoff_manager.py
│   ├── utils/                 # Metrics, transformations, noise generators
│   │   ├── metrics.py
│   │   └── coordinate_transforms.py
│   └── simulation/            # Trajectory generators (straight, curves, loops)
│       └── trajectory_gen.py
├── dashboard/                 # Interactive visualizer & telemetry UI
├── notebooks/                 # Model training, ablation studies & evaluation
├── tests/                     # Unit and integration tests
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🛠️ Quick Start

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Krishnanand-10/NAVIS.git
cd NAVIS

# Create virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Synthetic Simulation & Benchmark
```bash
python -m src.simulation.trajectory_gen --mode complex_3d
python -m src.core.traditional_dr --dataset data/processed/sample_run.csv
```

---

## 📊 Benchmark & Evaluation Metrics
- **ATE (Absolute Trajectory Error)**: Root Mean Square Error ($RMSE$) against Ground Truth.
- **RPE (Relative Pose Error)**: Local drift per unit distance ($m/km$ or $\%$ of total distance travelled).
- **Cumulative Drift Reduction Percentage**: $\frac{Drift_{Standard} - Drift_{NAVIS}}{Drift_{Standard}} \times 100\%$.

---

## 👥 Contributors
Developed for **Smart India Hackathon (SIH)** under **ISRO Problem Statement SIH26168**.
- **Repository**: [Krishnanand-10/NAVIS](https://github.com/Krishnanand-10/NAVIS)
