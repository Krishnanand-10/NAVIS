# PROJECT STATUS — SIH26168 Dead Reckoning

> **If you are an AI assistant opening this project in a fresh session: read this
> file top to bottom before proposing or writing anything.** It is the single
> source of truth for what exists, what does not, and what was already decided.
> `README.md` explains the system; this file explains the *state of the work*.
>
> When you finish a working session, append a new entry to
> [§8 Session log](#8-session-log) and update [§3](#3-state-at-a-glance),
> [§4](#4-what-is-done) and [§5](#5-what-is-left). Follow the format in
> [§9](#9-how-to-maintain-this-file).

---

## 1. What we are building

> **Resuming after a break? Do this:** read §2 (environment), §3 (state),
> §5 (what is left), then §7 (decisions not to re-argue). The recommended next
> task is **§5.2 — evaluate on real public data**; the reasoning is in the
> Session 4 "Next" entry.
>
> **Repo:** https://github.com/Krishnanand-10/NAVIS (branch `main`). Working copy: `C:\Users\Acer\OneDrive\Documents\NAVIS`.

**Smart India Hackathon 2026, problem statement SIH26168 (ISRO):**
*"AI-ML based Intelligent Dead Reckoning system for seamless navigation."*

A navigation system that keeps producing an accurate position when GNSS is lost
or lying — road tunnels, basement car parks, urban canyons — and rejoins the
satellite solution without the map cursor teleporting when signal returns. The
target platform is an ordinary **Android phone**, using **NavIC/IRNSS** through
the Android raw GNSS measurements API as the ISRO-facing differentiator.

⚠️ **The full problem statement text has never been read.** Every public mirror
of the SIH 2026 catalogue carries only the title; the description, deliverables
and any stated accuracy targets are behind the portal at **sih.gov.in**. Nobody
should treat a specific numeric target as fixed until someone pastes the
verbatim portal wording into a session. Everything built so far is derived from
the title plus domain reasoning.

---

## 2. Hard environment facts

These are verified, not assumed. **Do not suggest work that violates them.**

| Fact | Consequence |
|---|---|
| **No JDK.** `/usr/bin/java` is the macOS stub that errors out. | Cannot compile any Kotlin/Java. |
| **No Android Studio, no Android SDK, no NDK, no AVD, no Gradle cache.** Verified 2026-09-04. | The Android app **cannot be built or run on this Mac**, only edited as text. This is deliberate — see §6. |
| **Python 3.13.7**, venv at `SIH/.venv` with numpy, scipy, matplotlib, pytest. | Pipeline work runs freely. |
| **PyTorch 2.14 installed**, MPS (Apple GPU) available. venv is now 924 MB. | Training runs locally, ~45 s/epoch. TensorFlow is *not* installed — see the TFLite note in `scripts/export_model.py`. |
| **Owner's phone does not support NavIC.** | He can still record GPS/GLONASS/Galileo logs; only the NavIC-specific demo needs the friend's device. See §6. |
| Disk: ~96 GB free. Project is 278 MB (230 MB venv, 48 MB regenerable figures/logs). | Space is not a constraint. |

---

## 3. State at a glance

| Component | Status | Evidence |
|---|---|---|
| Problem analysis & brief | ✅ Done | `SIH26168_Dead_Reckoning_Brief.md` |
| Sensor/trajectory simulator | ✅ Done | `pipeline/drnav/synth.py`, 4 scenarios |
| Initial alignment (TRIAD) | ✅ Done | `pipeline/drnav/align.py` |
| 15-state ESKF + constraints | ✅ Done | `pipeline/drnav/eskf.py` |
| GNSS health classifier (rules) | ✅ Done | `pipeline/drnav/gnss_health.py` |
| Metrics & figures | ✅ Done | `pipeline/drnav/{metrics,plotting}.py` |
| Test suite | ✅ 48 passing | `tests/test_core.py`, `tests/test_ml.py` |
| Smooth handover (no teleport) | ✅ Done | median jump 144 m → 0.1 m |
| Road-vibration sensor model | ✅ Done | `synth.py::_road_vibration` |
| Android recorder app | ⚠️ **Written, never compiled** | `android/` — see §6 |
| **Learned speed head — code** | ✅ Done | `features/dataset/model/ml_speed.py`, 16 tests |
| **Learned speed head — trained model** | ✅ Done | 0.083 m/s pedestrian; gated off for vehicles |
| Uncertainty calibration | ✅ Done | z-score 1.56 → 1.00 |
| Map matching | ❌ Not started | |
| Real phone data | ❌ None collected | all numbers are from the simulator |
| On-device ESKF (Kotlin port) | ❌ Not started | friend's task |

**Rough completion: ~75%.** The classical backbone is finished, handover is
genuinely seamless, and the learned head is trained, calibrated and evaluated
against a control. Two things remain large and open: **vehicle speed is not
solved** (the head is gated off there — see Session 4), and **no real phone data
has ever touched this code**.

---

## 4. What is DONE

### Working results (mean horizontal RMSE, 5-6 seeds, simulator)

| Scenario | Outage | Classical | ESKF | + learned head |
|---|---:|---:|---:|---:|
| `parking_ramp` | 150 s | 575.6 m | **2.2 m** | gated off |
| `urban_canyon` | — | 19.5 m | **12.6 m** | gated off |
| `tunnel_drive` | 120 s | 406.2 m | **59.0 m** | gated off |
| `pedestrian_mall` | 150 s | 3015.3 m | 1110.8 m | **44.6 m** |

Pedestrian, against the constant-speed control that any speed claim would give
for free: no head 1110.8 m → constant 65.5 m → **learned 44.6 m**. The 1.5×
margin over the control is the part attributable to learning; the rest is not.

Speed head alone: **0.083 m/s** for pedestrians. Uncertainty calibrated to a
mean z-score of 1.00. Reacquisition jump: **0.1 m median** (144 m without
output smoothing).

### Verify it yourself

```bash
cd ~/Desktop/ml/Claude/SIH/pipeline
../.venv/bin/python -m pytest tests/ -q                   # 48 passing
../.venv/bin/python scripts/run_replay.py --scenario all  # figures -> out/
../.venv/bin/python scripts/eval_ml.py                   # ML vs control
../.venv/bin/python scripts/eval_generalisation.py       # across road surfaces
```

### Thirteen findings already paid for in debugging time

Full detail with measurements is in `README.md` → *Engineering findings*.
**Do not re-derive or reverse these**; each is documented in the docstring of
the function named.

1. An IMU cannot distinguish rest from constant velocity (Galilean invariance).
   Stillness detection alone flagged 60 km/h cruise as stopped, 100 % of the time.
2. ZUPT mid-outage is circular and self-confirming. Allowing it: 18 m → 332 m RMSE.
3. Constraints applied at IMU rate make the filter delusional (σ_v collapsed to
   0.002 m/s). They run at 10 Hz.
4. At rest, tilt error and horizontal accel bias are indistinguishable; a tight
   ZUPT invents a confident split and drove accel bias to 0.7 m/s².
5. Vertical NHC is poison on a graded road: 30 m lateral-only vs **1073 m** with
   vertical enabled.
6. A chi-square gate needs an escape hatch, and the escape must **rebuild** the
   covariance, not patch it.
7. Multipath de-weighting buys continuity, not accuracy (max jump 129 m → 24 m,
   RMSE barely moved).
8. Reacquisition was teleporting and unmeasured: median jump 144 m, worst 416 m.
   Output smoothing fixed it to 0.1 m for ~7 % RMSE. This is the word "seamless".
9. Vehicle speed is not learnable from an idealised IMU at all — the first
   training run correctly predicted the dataset mean with maximum uncertainty,
   because road vibration was not modelled.
10. The filter and the network want opposite conditioning of the same log:
    causally low-passed for mechanisation, raw for the network.
11. An *untrained* network already "improved" pedestrian RMSE 19×. Every ML
    number must be reported against a constant-speed control.
12. The learned head helps pedestrians and actively hurts vehicles (roughness
    confound). It is gated off for vehicles by default.
13. A degenerate test scenario nearly buried the result — a near-constant-speed
    walk makes a constant prior optimal by construction.

---

## 5. What is LEFT

Ordered by value. Items 1–3 are Python and can be done on this Mac.

### 1. Make the learned head work for vehicles ⭐ largest open problem
It is currently **off** for vehicles because it hurts (tunnel 59 m → 88 m). The
cause is diagnosed, not mysterious: speed is inferred from vibration
*amplitude*, which is perfectly confounded with road roughness, giving a bias
from −8.0 m/s on smooth tarmac to +3.9 m/s on a rough street.

Two routes, probably both:
- **Give the simulator speed-dependent spectral structure.** Real tyre vibration
  has wheel-rotation and tread harmonics whose *frequencies* scale with speed,
  independent of how rough the road is. Amplitude alone cannot separate the two;
  frequency can. This is a change to `synth._road_vibration`.
- **Train on real logs**, which have that structure for free.

Flip `EskfConfig.use_ml_speed_for_vehicle` once it stops hurting, and tighten
`test_tunnel_drift_regression_guard` (currently a loose 20 %).

### 2. Train and evaluate on real public data
No real phone data has ever touched this code. Do not need a phone for this:
- **Google Smartphone Decimeter Challenge (2021/2022)** — Android raw GNSS + IMU
  with centimetre ground truth. Closest match to our target platform.
- **RoNIN / RIDI / OxIOD** — pedestrian inertial odometry.
- **KITTI / comma2k19** — vehicle.
Training trick: record/take data with GNSS healthy, use it as ground truth, then
**mask GNSS in software** and train the model to reproduce the trajectory it can
no longer see. Any open road becomes a synthetic tunnel.

### 3. Map matching against OpenStreetMap
A tunnel is one-dimensional — snapping to the road graph with a decent speed
estimate gives near-exact position. Probably the largest remaining accuracy win
for the vehicle case, and it is pure Python.

### 4. Improve urban-canyon accuracy (~12.6 m floor)
Currently bounded by systematic multipath bias, which filtering cannot remove.
Needs per-satellite exclusion using `gnss_raw.csv` fields (C/N₀, multipath
indicator, pseudorange-rate residuals) — an ML classifier over raw measurements.

### 5. Android app — build, test, collect data 🔁 handed off
See §6.

### 6. Port the ESKF to Kotlin for on-device real-time operation
Only after the Python version is final. Friend's task.

---

## 6. Division of labour — Python here, Android with the friend

**Decision made 2026-09-05.** The owner's phone does not support NavIC, and
installing Android Studio + SDK (~10–15 GB) on this Mac to build an app that
cannot demonstrate the project's headline feature is not a good trade. So:

| Here (Python, this Mac) | Friend (Android, NavIC phone) |
|---|---|
| Simulator, alignment, ESKF, metrics ✅ | Build `android/` in Android Studio |
| ML velocity head + training | Field-test the recorder, collect real logs |
| Map matching | Integrate the exported model (TFLite/ONNX) |
| Multipath classifier | Port ESKF to Kotlin, live UI |
| Evaluation on public datasets | The NavIC demo and video |

**The contract between the two halves is the CSV log schema** in
`pipeline/drnav/log_format.py`, mirrored by `android/.../LogWriter.kt`. As long
as both sides honour it, either half can be developed without the other. This
is already proven: a CSV written in the phone's format replays through the
pipeline to an identical 5.02 m RMSE.

**Handoff artefacts the friend needs:** the GitHub repo, `README.md` (system +
findings), this file (state), `android/README.md` (build + recording protocol),
and eventually the exported model file.

⚠️ **The owner can still collect useful data on a non-NavIC phone.** Raw GNSS
measurements and IMU logging need neither NavIC nor a special device — only
NavIC *reporting* does. GPS-only logs are fully usable for validating the filter
and training the ML head. Worth checking his phone with Google's free
**GnssLogger** app, which reports whether raw measurements are supported.

---

## 7. Decisions already made — do not re-litigate

- **Android phone as the device**, not custom hardware. Reasoning in the brief §2.
- **Simulator-first development.** Every algorithm decision is validated against
  known ground truth before real data exists. Real logs use the same code path.
- **Attitude stored as a 3×3 rotation matrix, not a quaternion**, inside the
  filter — removes an entire class of sign/order bugs. Quaternions are for I/O only.
- **Global (nav-frame) error convention** for the ESKF, per Solà's tutorial.
- **ENU local tangent frame**, origin at the first GNSS fix. Heading is clockwise
  from North. Body frame is x=forward, y=left, z=up.
- **CSV logs, not binary/database.** Emailable, Excel-openable, no library needed
  on the Android side.
- **Lateral-only NHC**, vertical disabled by default (finding 5).
- **ZUPT gated on an independent speed reference** (findings 1, 2).
- The `urban_canyon` test asserts *continuity*, not RMSE, because naive GNSS
  snapping genuinely wins on RMSE there. That is honest, not a bug.

---

## 8. Session log

### Session 1 — 2026-09-04 · Problem analysis
**Goal:** Understand the problem statement; decide an approach.
**Did:** Identified PS as SIH26168 (ISRO, theme Miscellaneous, software).
Could not retrieve the full description — public catalogues carry only the
title; the master-catalogue PDF is a scanned image with no extractable text.
Wrote `SIH26168_Dead_Reckoning_Brief.md`: dead reckoning explained from first
principles, the bias-drift error table (5 m at 10 s → 720 m at 120 s), why ML
means *regress velocity, don't integrate acceleration*, the
heading-is-easy/speed-is-hard asymmetry, the case for the phone as the device,
a six-layer architecture, data strategy, metrics, and the NavIC angle.
**Left off at:** brief written, nothing built.
**Next:** build it.

### Session 2 — 2026-09-04 · Built the fusion pipeline
**Goal:** Build the proposed system.
**Did:** Found no JDK/SDK/Android Studio and no torch, so built the Python side
first. Wrote the simulator, TRIAD alignment, 15-state ESKF, GNSS health
classifier, metrics, plotting, replay driver, 31 tests. Also wrote the full
Android recorder in Kotlin (never compiled). Debugged seven substantial issues —
see §4 — several of which inverted the initial design (ZUPT gating, lateral-only
NHC, constraint decimation).
**Findings:** ESKF beats classical strapdown 15× on the tunnel, 181× on the
parking ramp. Pedestrian mode fails at 549 m and needs the ML head.
**Left off at:** pipeline working and tested; figures in `pipeline/out/`.
**Next:** ML velocity head, or map matching.

### Session 3 — 2026-09-05 · Handoff planning
**Goal:** Set up project tracking; decide the Android split.
**Did:** Verified no Android tooling is installed on the Mac (clean — the 4 GB
`Application Support/Google` is Chrome, not Android Studio). Wrote this file.
Agreed the Python-here / Android-with-friend split (§6) and the CSV schema as
the contract between them.
**Left off at:** ready to start the ML head; repo not yet on GitHub.
**Next:** owner chose: init the repo, then build the ML head.

---

### Session 4 — 2026-09-05 · Repo, road vibration, seamless handover, ML head
**Goal:** Initialise the git repo, then build the learned speed head.
**Did:** `git init` plus two commits. Installed PyTorch 2.14 (MPS available).
Wrote the whole ML pipeline: `features.py` (gravity-levelled, mount-invariant
channels), `dataset.py` (simulator-backed, lazily windowed, split by recording),
`model.py` (394k-parameter dilated TCN with a heteroscedastic head),
`ml_speed.py` (batched inference wrapper), plus `train_speed.py`,
`eval_ml.py`, `eval_generalisation.py`, `export_model.py` and 12 new tests.
Added `Eskf.update_speed_magnitude` — a mount-free speed constraint, because
`update_ml_forward_speed` needs a vehicle frame and pedestrians have none.

**Findings — four, all of which changed the design:**

1. **Vehicle speed was not learnable at all from the original simulator.** The
   first training run converged to predicting the dataset mean with maximum
   uncertainty (val RMSE 5.31 m/s, mean sigma 5.16). That was the model being
   *correct*: cruising at 30 and 90 km/h are kinematically identical, and the
   only absolute speed cue a real vehicle IMU carries — road vibration — was
   not modelled. Adding it took val RMSE to 1.005 m/s in a single epoch. A
   useful incidental result: the heteroscedastic head demonstrably works, since
   it said "I don't know" exactly when there was nothing to know.
2. **The filter and the network want opposite things from the same log.**
   Mechanising raw IMU now integrates 0.4 m/s² of vibration the process-noise
   model knows nothing about. Mechanisation therefore gets a causally
   low-passed stream; the network keeps the raw one.
3. **Reacquisition was teleporting and nobody had measured it.** Median jump
   across seeds was 144 m, worst 416 m. Output smoothing (state takes the
   correction immediately, reported position catches up over ~2 s) brought that
   to 0.1 m median for about 7 % RMSE. This is the word "seamless" in the title.
4. **An *untrained* network already cut pedestrian RMSE from 549 m to 29 m**,
   purely because a constant 0.66 m/s output bounds the drift. So "adding ML
   improved things 19×" would have been a meaningless claim. `eval_ml.py` now
   carries a constant-speed control, and the only number worth reporting is the
   trained model's margin **over that control**.

**Also:** road roughness now varies per training clip. Training on a single
surface teaches one road's calibration, which would read high on rough tarmac
and low on smooth while reporting the same confident sigma.

**Results:** trained 26 epochs, best val RMSE 2.219 m/s at epoch 8 (it overfits
after that; best-checkpoint selection handles it). Calibration scale 1.251
brings the mean z-score from 1.56 to exactly 1.00.

* **Pedestrian: it works.** 0.083 m/s standalone speed error; in the filter,
  1110.8 m → 44.6 m, which is 1.5× better than the constant-speed control.
* **Vehicle: it does not, and is gated off.** Vibration amplitude is confounded
  with road roughness — bias −8.0 m/s on smooth tarmac to +3.9 m/s on rough,
  over-confident on smooth (calibration 16–270). Enabling it cost tunnel RMSE
  59 m → 88 m. `EskfConfig.use_ml_speed_for_vehicle` is False, with the
  measurements in its docstring.
* **A degenerate scenario nearly buried the result.** The original pedestrian
  walk held ~1.3 m/s throughout, making a constant prior optimal by
  construction; the control beat the model 26.9 m to 33.0 m. Giving the walk a
  realistic pace range (0.6–1.7 m/s) reversed it to a 1.5× win with the model
  unchanged. Worth remembering before trusting any future ablation here.
* **One real bug found by the gating work:** `have_reference` counted the ML
  head as an independent speed reference merely because it was *passed*, even
  when gated off for vehicles — which silently re-enabled ZUPT inside tunnels.
  Cost: 59 m → 74 m from an estimator that was never called.

**Left off at:** 48 tests passing, figures regenerated, ONNX exported
self-contained (1.66 MB). Four commits, pushed to a **private** GitHub repo at
https://github.com/Akarshx-x/sih26168-dead-reckoning (branch `main`, 45 files,
1.9 MB — venv, clip cache, checkpoints and figures all excluded). The friend
still needs adding as a collaborator:
`gh repo add-collaborator Akarshx-x/sih26168-dead-reckoning <user> --permission push`

**Next — recommended order:**

1. **§5.2, evaluate on real public data.** Highest value, lowest risk, and it
   needs no phone, no NavIC and no Android Studio. Every number in this project
   is currently simulated, which is the first thing a judge will probe. Pull a
   Google Smartphone Decimeter Challenge recording, load it through
   `Recording.load`, and run `run_eskf` against it. Expect the OEM timestamp
   and sample-rate issues flagged in `ImuCollector` to bite here first.
2. **§5.1, vehicle spectral vibration.** More interesting technically, but
   tuning the simulator further *before* checking it against reality risks
3. **§5.3, map matching.** Probably the largest single accuracy win for
   vehicles, and pure Python.

### Session 5 — 2026-09-07 · Verified test suite and synchronized to GitHub
**Goal:** Verify test suite on Windows workstation and synchronize repository state to GitHub.
**Did:** Ran full test suite (48/48 passed in 234 s). Linked repository to `https://github.com/Krishnanand-10/NAVIS`, updated commit history to replace legacy dashboard prototype with the complete classical ESKF, ML velocity head, and Android logger architecture, and pushed cleanly to branch `main`.
**Findings:** All 48 tests pass cleanly on Python 3.13 on Windows with CPU PyTorch.
**Left off at:** 48 tests passing; GitHub repo fully synchronized and clean.
**Next:** §5.2 — evaluate on real public data (e.g. Google Smartphone Decimeter Challenge).

---

## 9. How to maintain this file

At the end of each working session, append to §8:

```markdown
### Session N — YYYY-MM-DD · Short title
**Goal:** what the session set out to do
**Did:** what actually changed, with file paths
**Findings:** anything surprising, with measurements — especially reversals
**Left off at:** exact state when the session ended
**Next:** the obvious next move
```

Then update §3 (status table), §4 (if something became done) and §5 (if
something became unnecessary or newly blocked). Add to §7 only when a decision
is genuinely settled — the point of that section is to stop a future session
burning time re-arguing it. Keep numbers in this file **measured, not
estimated**; if you quote an RMSE, it should come from a run you actually did.
