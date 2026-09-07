# SIH26168 — AI-ML based Intelligent Dead Reckoning System for Seamless Navigation

**Organisation:** Indian Space Research Organisation (ISRO)
**Problem Statement ID:** SIH26168
**Theme:** Miscellaneous · **Category:** Software
**Brief prepared:** 4 September 2026

> ⚠️ **Before you build anything:** every public mirror of the SIH 2026 catalogue carries only the
> *title* of this problem statement. The full description, expected deliverables, and any stated
> accuracy targets sit behind the portal at **sih.gov.in**. Get the verbatim wording from your SPOC
> or the portal before locking down the architecture — a stated target such as "<10 m error after
> 5 minutes of outage" would change several decisions below. Everything in this document is decoded
> from the title plus domain knowledge of how this problem actually works.

---

## Table of contents

1. [What the problem statement is actually asking for](#1-what-the-problem-statement-is-actually-asking-for)
2. [Why an Android phone is the right device](#2-why-an-android-phone-is-the-right-device)
3. [Proposed architecture](#3-proposed-architecture)
4. [Training data strategy](#4-training-data-strategy)
5. [Metrics and demo plan](#5-metrics-and-demo-plan)
6. [The ISRO differentiator: NavIC](#6-the-isro-differentiator-navic)
7. [Risks to name before a judge does](#7-risks-to-name-before-a-judge-does)
8. [Suggested next steps](#8-suggested-next-steps)
9. [References](#9-references)

---

## 1. What the problem statement is actually asking for

### 1.1 Dead reckoning, defined

**Dead reckoning (DR)** is estimating where you are *now* from your last known position plus your
measured motion — heading, speed, and elapsed time. It requires no external signal. Ships did it for
centuries with a magnetic compass and a knotted rope trailing in the water.

### 1.2 The problem it solves

GNSS — GPS, and for ISRO specifically **NavIC / IRNSS** — fails or actively lies in exactly the
places where people most need navigation to work:

| Environment | Failure mode |
|---|---|
| Road tunnels, underpasses | Total signal loss |
| Urban canyons (Mumbai, Delhi CBD) | **Multipath** — signals bounce off glass towers; you get a *plausible but wrong* fix, sometimes 50–100 m off, placed on the wrong road |
| Multi-storey / basement parking | Total loss, plus you need floor level |
| Dense forest, deep valleys | Attenuation, few satellites, poor geometry (high HDOP) |
| Deliberate jamming / spoofing | Signal present but hostile |

So the ask is: **when the sky goes away, keep navigating accurately — and when it comes back,
rejoin without the map cursor teleporting.** That last clause is what the word "seamless" in the
title is doing.

### 1.3 Why "AI-ML"? Because classical DR fails fast

A phone-grade MEMS accelerometer carries a bias of roughly **10 mg (~0.1 m/s²)**. Position error
from double-integrating a constant bias grows as ½·a·t²:

| Elapsed time | Position error from bias alone |
|---|---|
| 10 s | ~5 m |
| 30 s | ~45 m |
| 60 s | ~180 m |
| 120 s | ~720 m |

Classical strapdown inertial navigation on cheap sensors is **useless within a minute**. Critically,
this error is not random noise you can average away — it is *structured*, temperature-dependent, and
motion-dependent.

That structure is precisely what a neural network can learn.

### 1.4 The modern approach: regress velocity, don't integrate acceleration

The key insight from the current literature (RoNIN, IONet, TLIO) is to **stop double-integrating
entirely**. Instead, feed a 1–2 second window of raw IMU data to a network and have it **regress
velocity directly**.

A 2-second IMU window contains enough signature — gait cadence when walking, vibration and turn
dynamics when driving — for a model to output *"you are moving at 1.4 m/s in this direction"*
without ever touching a double integral. There is no integration, therefore no cubic error growth.

This drops drift from hundreds of metres per minute to roughly **2–5% of distance travelled**.
Walk 200 m blind → land within ~5–10 m.

### 1.5 The key asymmetry: heading is easy, speed is hard

This single insight should shape your whole design.

- **Heading is easy.** A gyroscope integrates to good heading for minutes on end — drift is only a
  few degrees per minute.
- **Speed is hard.** Your entire problem reduces to estimating *how fast you are going along that
  heading.*

For **pedestrians**, ML solves speed well via step cadence — walking is periodic, and periodicity is
a strong learnable signal.

For **vehicles** it is genuinely harder, because there is no gait. You must lean instead on
last-known GNSS speed, longitudinal acceleration, road-map constraints, and (where available) wheel
odometry.

**Plan for these as two distinct modes with a mode classifier between them.**

---

## 2. Why an Android phone is the right device

### 2.1 The alternative, and why to avoid it

Building custom hardware means an Arduino/ESP32, an MPU-6050 IMU, a u-blox GNSS module, wiring,
power management, an enclosure, and a laptop to log to. That is **two weeks of your hackathon spent
on soldering and driver bugs** — and it produces something a judge cannot hold in their hand.

### 2.2 A modern Android phone is already the complete rig

| Capability | What you get |
|---|---|
| **IMU** | 3-axis accelerometer, gyroscope, magnetometer at 100–500 Hz via `SensorManager` |
| **Barometer** | Altitude and floor detection inside parking structures |
| **GNSS — the killer feature** | **Raw GNSS Measurements API** (`GnssMeasurementsEvent`, Android 7+): raw pseudoranges, carrier phase, Doppler, per-satellite C/N₀, and constellation ID — not just a cooked lat/long |
| **NavIC visibility** | `GnssStatus.CONSTELLATION_IRNSS` (Android 10+) on Qualcomm/MediaTek chipsets from ~2019 onward |
| **Camera** | ARCore provides 6-DoF visual-inertial odometry with no ML work from you |
| **NPU / GPU** | Run your model on-device in real time via TFLite + NNAPI |
| **Screen** | Your demo UI, in the judge's hand |
| **Data collection** | Gather training data by walking around campus and driving to a tunnel — no rig, no calibration lab |

### 2.3 The strategic argument

The phone **is** the product. *"Every Indian with a NavIC-capable phone gets uninterrupted
navigation in tunnels"* is a far stronger pitch to ISRO than *"we built a box."*

### 2.4 Honest trade-offs — raise these before a judge does

- Phone MEMS sensors are cheap, with worse bias instability than automotive-grade IMUs.
- Sensor sampling rates and timestamp quality vary across OEMs — **test on 3+ phone models.**
- **Device orientation is arbitrary** (pocket, cupholder, dashboard, hand). This is the classic
  killer of naive DR — and it is a strong argument *for* the ML approach, since RoNIN-style models
  are explicitly trained to be orientation-agnostic.
- **Magnetometers are badly distorted inside a steel car body.** Do not trust the compass while
  driving; trust the gyro.

---

## 3. Proposed architecture

Six layers. Build them in this order.

### ① Sensing & logging (Android, Kotlin)

Sample calibrated **and** uncalibrated accel/gyro at 100–200 Hz, plus rotation vector, barometer,
`GnssMeasurementsEvent`, `GnssStatus`, and `FusedLocationProvider` — all stamped with consistent
monotonic timestamps and written to a replayable log format.

> **Build this first, and build a desktop replay harness for the logs.**
> Recording a drive takes 30 minutes; you can then iterate on algorithms a hundred times without
> leaving the room. Teams that skip this lose the hackathon to logistics.

### ② GNSS health classifier

A small model (or well-tuned rules) taking satellite count, C/N₀ distribution, HDOP, and
position-residual jerk → **`GOOD` / `DEGRADED` / `LOST`**.

This is underrated. The hardest real case is *not* the tunnel where GPS obviously dies — it is the
urban canyon where GPS **confidently reports the wrong street**. Detecting that is a genuine
technical contribution and an excellent demo moment.

### ③ Learned inertial odometry — the core ML

A 1D CNN / temporal convolutional network (or a small LSTM) over a ~2 s gravity-aligned IMU window
(≈ 200 × 6 tensor) → **2D velocity + a predicted uncertainty**.

- That **uncertainty output matters** — it is what lets the filter in ④ trust the network
  appropriately rather than blindly.
- Add a lightweight activity classifier (stationary / walking / stairs / driving) to switch between
  a **pedestrian head** and a **vehicle head**.

### ④ Fusion — Error-State Kalman Filter (ESKF)

**State vector:** position, velocity, attitude, accelerometer bias, gyroscope bias.

Propagate on IMU; correct with GNSS when healthy, plus the ML velocity as a pseudo-measurement,
plus these free wins:

- **ZUPT (Zero-Velocity Update)** — detected stationarity resets velocity error to exactly zero.
  Enormous value at traffic lights.
- **Non-holonomic constraints (NHC)** — a car cannot move sideways or vertically relative to itself.
  Two free measurements every timestep.
- **Barometer** for vertical position and parking floor level.

### ⑤ Map matching (high payoff — do it if time allows)

A tunnel is **one-dimensional**. Snapping to the OpenStreetMap road graph with a decent speed
estimate gives near-exact position through a tunnel, because there is only one place you can be.
This is what commercial systems actually do, and it turns a good demo into an impressive one.

### ⑥ Seamless handover — do not teleport

When GNSS returns after 90 s of DR, the incoming fix will disagree with your estimate.
**Never snap.** Feed it as a measurement with proper covariance and blend the correction over
2–3 seconds so the cursor glides back.

This is literally the word "seamless" in your title — make sure it is a visible, explicitly
called-out feature in the demo.

---

## 4. Training data strategy

### 4.1 The core trick

You cannot collect enough tunnel drives. You do not have to.

> **Record with GNSS working perfectly, use that track as ground truth, then mask out the GNSS in
> software and train the model to reproduce the trajectory it can no longer see.**

Any open road becomes a synthetic tunnel. Augment with random device rotations, injected sensor
biases, and added noise.

### 4.2 Public datasets — all free

| Dataset | Use |
|---|---|
| **Google Smartphone Decimeter Challenge (2021 / 2022)** | Android raw GNSS + IMU with centimetre ground truth. Almost purpose-built for this PS. |
| **RoNIN, RIDI, OxIOD** | Pedestrian inertial odometry |
| **KITTI, comma2k19** | Vehicle trajectories |

---

## 5. Metrics and demo plan

### 5.1 Report these — vagueness loses points

- **Drift as % of distance travelled** during outage (target ≈ 2–5% pedestrian; < 5% vehicular with
  map matching)
- **Absolute position error at 30 s / 60 s / 300 s** of outage
- **Re-convergence time** and **max position discontinuity** on GNSS return (should be ~0)
- **Baseline comparison** — plot your system against classical strapdown INS on the *same* log.
  Classical will be ~200 m off within a minute; yours will not. **That single chart is your
  strongest slide.**

### 5.2 The demo

Drive through a real underpass or a multi-storey parking ramp. Show a split screen:

- **Left:** raw GPS dying and jumping onto adjacent buildings
- **Right:** your fused track staying on the road

Put a live source badge in the corner:

```
NavIC fix  →  DEAD RECKONING (42 s)  →  NavIC reacquired
```

---

## 6. The ISRO differentiator: NavIC

This is an **ISRO** problem statement, and most competing teams will build a generic GPS-denied
demo. Differentiate on the things ISRO's own engineers care about:

- Use the raw measurements API to **explicitly identify, log, and display NavIC / IRNSS
  satellites**, and show a **NavIC-only position solution** alongside GPS.
- Frame DR as **extending NavIC coverage**, not replacing it. NavIC is a regional constellation, so
  *continuity of service* is precisely the national-infrastructure story ISRO wants told.
- Mention the scale-up paths:
  - **L5 / S-band dual-frequency** for multipath rejection
  - The same fusion core running on an **automotive-grade IMU + CAN wheel odometry** for vehicles
  - **UAV deployment**, where GNSS denial is an operational reality rather than an inconvenience

---

## 7. Risks to name before a judge does

- MEMS bias drift with temperature
- Magnetic distortion inside vehicle bodies
- Android sensor rate and timestamp inconsistency across OEMs
- Thermal throttling and battery drain during sustained on-device inference
- ARCore VIO degrading in dark, textureless tunnels
- **Honest limitation:** unaided pure-inertial DR over 10+ minutes is *not* solved by anyone. This is
  exactly why map matching and opportunistic Wi-Fi/BLE fixes belong in the design. Saying this out
  loud builds credibility rather than costing you points.

---

## 8. Suggested next steps

1. **Pull the verbatim PS text off the SIH portal.** Confirm deliverables and any accuracy targets.
2. **Build layer ① (sensing & logging) this week.**
3. **Record a single drive through an underpass with a genuine GNSS dropout in it.**
   Once you have one real log, every other design decision becomes something you can argue about
   with evidence instead of opinion.
4. Build the desktop replay harness before writing any model code.
5. Split the team roughly: app/sensors · ML model · fusion filter · data collection + evaluation ·
   UI/map · docs & presentation.

---

## 9. References

- [SIH 2026 Problem Statements catalogue (SIH26168)](https://github.com/NoBugNinja/Smart-India-Hackathon-SIH-2026-Problem-Statements)
- [AI-IMU Dead-Reckoning (arXiv 1904.06064)](https://arxiv.org/pdf/1904.06064)
- [Lane Detection Aided Online Dead Reckoning for GNSS Denied Environments](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8538960/)
- [Dead Reckoning is Still Alive! — overview of the approach](https://medium.com/data-science/dead-reckoning-is-still-alive-8d8264f7bdee)
- [AI-Enhanced Dead Reckoning Techniques for Robust Position Estimation](https://www.researchgate.net/publication/396446290_AI-Enhanced_Dead_Reckoning_Techniques_for_Robust_Position_Estimation)
- Official portal — [sih.gov.in](https://sih.gov.in)
