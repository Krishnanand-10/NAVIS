# SIH26168 — AI-ML based Intelligent Dead Reckoning System

**Indian Space Research Organisation (ISRO) · Smart India Hackathon 2026**

Continuous navigation when GNSS disappears — tunnels, basement parking, urban
canyons — on an ordinary Android phone, with NavIC as the primary constellation.

See [`SIH26168_Dead_Reckoning_Brief.md`](SIH26168_Dead_Reckoning_Brief.md) for
the problem breakdown and the reasoning behind the approach, and
**[`PROJECT_STATUS.md`](PROJECT_STATUS.md) for what is done, what is left, and
the log of each working session** — start there if you are picking this up
cold, human or AI.

---

## What exists today

| Component | State |
|---|---|
| `pipeline/` — simulator, alignment, ESKF, metrics, plots | **working, 48 tests passing** |
| Learned speed head — dilated TCN with predicted uncertainty | **trained; see Results** |
| `android/` — raw-GNSS + IMU recorder with NavIC reporting | **written, needs Android Studio to build** |
| Map matching against OpenStreetMap | **not built** — see Roadmap |

The fusion stack was deliberately built against a **simulator first**, so every
algorithm decision could be tested against known ground truth before anyone
drove anywhere. Real recordings drop into the same `Recording` container and
replay through the identical code path.

---

## Quick start

```bash
cd pipeline
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt

# classical stack
../.venv/bin/python scripts/run_replay.py --scenario tunnel_drive
../.venv/bin/python scripts/run_replay.py --scenario all --save-logs
../.venv/bin/python -m pytest tests/ -q

# learned speed head
../.venv/bin/python scripts/train_speed.py --epochs 26 --rebuild
../.venv/bin/python scripts/calibrate_sigma.py        # honest uncertainty
../.venv/bin/python scripts/eval_generalisation.py    # across road surfaces
../.venv/bin/python scripts/eval_ml.py                # vs a constant-speed control
../.venv/bin/python scripts/export_model.py           # ONNX for the Android side
```

Figures land in `pipeline/out/`. `--save-logs` also writes CSV recordings to
`pipeline/out/logs/` in exactly the layout the Android app produces.

---

## Results

Mean horizontal RMSE over 5-6 seeds. The learned speed head is enabled for
pedestrians and **deliberately gated off for vehicles** — see finding 12.

| Scenario | Outage | Classical | ESKF | + learned head | vs classical |
|---|---:|---:|---:|---:|---:|
| `parking_ramp` — multi-storey spiral | 150 s | 575.6 m | **2.2 m** | gated off | 262× |
| `urban_canyon` — multipath, no outage | — | 19.5 m | **12.6 m** | gated off | 1.5× |
| `tunnel_drive` — tunnel, curve + jam inside | 120 s | 406.2 m | **59.0 m** | gated off | 6.9× |
| `pedestrian_mall` — indoors, varying pace | 150 s | 3015.3 m | 1110.8 m | **44.6 m** | 68× |

**Does the network earn its place?** Reported against a constant-speed control,
because an *untrained* network already appeared to improve pedestrian RMSE 19×
simply by bounding the drift (finding 11):

| pedestrian_mall | RMSE | median | exit err | drift |
|---|---:|---:|---:|---:|
| ESKF, no learned head | 1110.8 m | 1107.4 m | 2989 m | 1844 % |
| control: constant 1.24 m/s | 65.5 m | 69.9 m | 133 m | 82 % |
| **ESKF + learned head** | **44.6 m** | **36.8 m** | **90 m** | **56 %** |

**24.9× better than no head, and 1.5× better than the control** — the margin
over the control is the part attributable to learning.

Speed head in isolation: **0.083 m/s RMSE for pedestrians** (gait cadence is
highly learnable). Predicted uncertainty is calibrated to a mean z-score of
exactly 1.00 by `scripts/calibrate_sigma.py`, which matters more than raw
accuracy — a Kalman filter fed an over-confident measurement diverges.

Reacquisition is genuinely seamless: **median position jump 0.1 m**, worst 0.7 m,
against 144 m median without output smoothing.

## Architecture

```
Android phone                          Offline (identical code path)
─────────────                          ────────────────────────────
IMU 200 Hz  ──┐
barometer   ──┤                        synth.py ──► Recording ──┐
raw GNSS    ──┼──► LogWriter ──► CSV ──────────────────────────►┤
GnssStatus  ──┤    (ENU, NavIC)                                 │
NMEA (HDOP) ──┘                                                 ▼
                                                        align.coarse_align
                                                          (TRIAD: gravity
                                                           + accel event)
                                                                │
                                                                ▼
                          ┌──────── raw IMU ────────┐
                          │                         │
                   low-pass 5 Hz            gravity-levelled
                  (mechanisation)            2 s window, 100 Hz
                          │                         │
                          │                  ┌──────▼───────┐
                          │                  │  model.py    │
                          │                  │  dilated TCN │
                          │                  │  speed + σ   │
                          │                  └──────┬───────┘
                          ▼                         │
                    ┌───────────────────────────────▼───┐
                    │  eskf.py — 15-state ESKF          │
                    │  p v R ba bg                      │
                    │  ← GNSS pos/vel (health-gated)    │
                    │  ← ZUPT / ZARU  (guarded)         │
                    │  ← NHC, lateral only              │
                    │  ← learned speed + its own σ      │
                    │  → output smoothing (no teleport) │
                    └───────────────────────────────────┘
                                                                │
                                                    metrics.py ─┴─ plotting.py
```

---

## Engineering findings

These cost real debugging time and are worth knowing before you touch the
filter. Each one is measured, not asserted — the numbers are from the tunnel
scenario, and the reasoning is in the docstring of the function named.

**1. An IMU cannot distinguish rest from constant velocity.**
This is Galilean invariance, not a tuning problem. A stillness detector built
on accelerometer variance flags smooth 60 km/h cruising as stationary — it was
`True` 100 % of the time during cruise. Applying ZUPT on that alone braked the
solution to a halt mid-motorway, and ZUPT then kept confirming its own mistake.
Every zero-velocity claim needs an **independent** speed reference. `replay.run_eskf`

**2. ZUPT during a blind outage is the dangerous case.**
Corroborating with the filter's *own* speed is circular: once inertial drift
pulls the estimate below the threshold, it pins itself to zero. Mean RMSE went
from 18 m to 332 m when that fallback was allowed. ZUPT is now gated on GNSS
being both fresh and healthy — or on the ML head, once it exists. `replay.run_eskf`

**3. Repeated constraints at IMU rate make the filter delusional.**
200 assertions per second that "velocity is zero" are not 200 independent
measurements. σ_v collapsed to 0.002 m/s, after which a 1 Hz GNSS fix could not
argue the filter out of a wrong answer. Constraints are applied at 10 Hz. `EskfConfig.constraint_rate_hz`

**4. At rest, tilt error and horizontal accel bias are indistinguishable.**
Both produce a constant horizontal specific-force error. A tight ZUPT commits
to a split anyway, and commits confidently — it drove accel bias to 0.7 m/s²,
nine sigma past anything physical. `eskf.update_zupt`

**5. Vertical non-holonomic constraint is poison on a graded road.**
The mounting is estimated against a *level* vehicle frame, so a 2 % tunnel ramp
produces a real 0.58 m/s climb that reads as a constraint violation. The filter
can only explain it by tilting pitch, and 2° of pitch leaks 0.34 m/s² of gravity
into the horizontal channel. Outage-exit error: **30 m lateral-only, 1073 m with
vertical enabled.** `eskf.update_nhc`

**6. A chi-square gate without an escape hatch is a trap.**
Once the estimate is far enough wrong, every genuine fix looks like an outlier
and the filter never recovers. But the escape must **rebuild** the covariance,
not patch blocks of it — grafting fresh position variance onto stale
cross-correlations produced gains that drove accel bias to 1564 m/s². `eskf.reset_position`

**7. Multipath de-weighting buys continuity, not accuracy.**
Inflating the reported sigma ×4 for degraded fixes cut max position jump from
129 m to 24 m. It barely moved RMSE, because multipath is a *systematic* bias
common to every fix and no amount of filtering removes it. `gnss_health.inflation_factor`

**8. Reacquisition was teleporting, and nobody had measured it.**
Median position jump across seeds was 144 m and the worst was 416 m — in a
project whose title says *seamless*. The filter state should take a returning
GNSS correction immediately, but the *reported* position should not jump.
Corrections now accumulate into an offset that decays over ~2 s: **144 m → 0.1 m
median**, for about 7 % RMSE. `EskfConfig.output_smoothing_tau`

**9. Vehicle speed is not learnable from an idealised IMU at all.**
Cruising at 30 and at 90 km/h are kinematically identical. The first training
run correctly converged to predicting the dataset mean with maximum uncertainty
(RMSE 5.31 m/s, mean σ 5.16), because the simulator modelled no road vibration —
the only absolute speed cue a real vehicle IMU carries. Adding it took val RMSE
to 1.0 m/s in one epoch. `synth.ImuErrorModel.vib_rms_ref`

**10. The filter and the network want opposite things from the same log.**
Mechanising raw IMU integrates 0.4 m/s² of vibration the process-noise model
knows nothing about, so the mechanisation gets a *causally* low-passed stream
while the network keeps the raw one. Causal, not zero-phase: a zero-phase
filter cannot run on a phone, and using one offline would flatter these results
with performance the real system could never reproduce. `replay.prefilter_imu`

**11. An untrained network already "improved" pedestrian RMSE 19×.**
Random weights emitting a constant 0.66 m/s took pedestrian error from 549 m to
29 m, purely because any bounded speed claim stops unaided inertial integration
running away. So "adding ML helped 19×" is not evidence of learning. Every ML
result is reported against a **constant-speed control**.
`scripts/eval_ml.py::ConstantSpeedPrior`

**12. The learned head helps pedestrians and actively hurts vehicles.**
For a walker it reads speed off gait cadence to 0.083 m/s. For a vehicle it
infers speed from vibration *amplitude*, which is perfectly confounded with road
roughness: across held-out surfaces inside the trained range the bias sweeps
from **−8.0 m/s on smooth tarmac to +3.9 m/s on a rough street**, and on smooth
roads it is over-confident (calibration 16, rising to 270 outside the range).
Enabling it cost tunnel RMSE 59 m → 88 m. It is therefore off for vehicles by
default. The fix is not a constant: amplitude alone cannot separate speed from
surface, and real tyre vibration carries speed-dependent *spectral* structure
(wheel and tread harmonics scale with rotation rate) that the simulator does not
yet model. `EskfConfig.use_ml_speed_for_vehicle`

**13. A degenerate test scenario nearly buried the result.**
The original pedestrian scenario walked at a near-constant 1.3 m/s, which makes
a fixed-speed prior optimal by construction — the control beat the model 26.9 m
to 33.0 m, and the honest conclusion would have been "learning adds nothing".
Giving the walk a realistic pace range (0.6–1.7 m/s: crowds, queues, stairs)
reversed it to a 1.5× win. The model had not changed. `synth.pedestrian_mall`

---

## Known limitations

- **Vehicle speed is not solved.** The learned head is gated off for vehicles
  (finding 12) and the classical stack still drifts ~14 % through a tunnel that
  contains a stop. This is the largest open problem.
- **Urban canyon accuracy is bounded by the multipath bias itself** (~23 m).
  Closing it needs per-satellite exclusion from raw measurements, or map matching.
- **No map matching.** A tunnel is one-dimensional; snapping to the OSM road
  graph is the single biggest available accuracy win and is not implemented.
- **The filter has never seen real phone data.** Every number above is from the
  simulator. Sensor timestamp behaviour in particular varies across OEMs and
  must be checked per device — see the note in `ImuCollector`.

---

## Roadmap

1. **Record a real drive.** Build `android/` in Android Studio, log one underpass.
   Nothing else is worth tuning until a real log exists.
2. **Learned inertial odometry.** 1D CNN over a 2 s gravity-aligned IMU window →
   velocity + uncertainty. Train on masked-GNSS segments of real logs plus RoNIN
   and the Google Smartphone Decimeter Challenge datasets. The filter hook is
   already there: `replay.run_eskf(..., ml_speed=...)` and
   `eskf.update_ml_forward_speed`.
3. **Map matching** against OpenStreetMap during outages.
4. **Port the ESKF to Kotlin** for on-device real-time operation.
5. **NavIC-only position solution** from raw pseudoranges, shown beside GPS.

---

## Layout

```
SIH26168_Dead_Reckoning_Brief.md   problem statement analysis
PROJECT_STATUS.md                  state, decisions, session log — start here
pipeline/
  drnav/
    attitude.py       SO(3) helpers, ENU/body conventions
    log_format.py     CSV schema shared with the phone; geodetic → ENU
    synth.py          trajectory + sensor simulator inc. road vibration
    align.py          TRIAD initial alignment
    eskf.py           15-state error-state Kalman filter
    gnss_health.py    GOOD/DEGRADED/LOST classification
    baseline_ins.py   classical strapdown, the thing to beat
    replay.py         offline replay driver, IMU conditioning, smoothing
    metrics.py        drift %, reconvergence, discontinuity
    plotting.py       the comparison figures
    features.py       gravity-levelling, mount-invariant channels
    dataset.py        simulator-backed training windows
    model.py          dilated TCN, speed + predicted uncertainty
    ml_speed.py       batched inference wrapper for the filter hook
  scripts/
    run_replay.py         classical stack, figures and metrics
    train_speed.py        train the speed head
    calibrate_sigma.py    make the predicted uncertainty honest
    eval_ml.py            filter-level impact vs a constant-speed control
    eval_generalisation.py  does it survive a different road surface?
    export_model.py       ONNX + TorchScript for the Android side
  tests/test_core.py, tests/test_ml.py
android/
  app/src/main/java/com/sih26168/drnav/
    MainActivity.kt      Compose UI, live NavIC + DR-age badge
    RecordingService.kt  foreground recorder
    ImuCollector.kt      accel/gyro/barometer
    GnssCollector.kt     fixes, status, NMEA HDOP, raw measurements
    LogWriter.kt         CSV in the pipeline's schema
    Enu.kt               geodetic → ENU on device
```
