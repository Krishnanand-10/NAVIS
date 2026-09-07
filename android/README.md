# DR Logger — Android recorder

Records everything the offline pipeline needs from one drive: IMU at 200 Hz,
barometer, GNSS fixes, satellite status, NMEA-derived HDOP, and **raw GNSS
measurements including NavIC**.

## Build

Nothing here can be built from the command line without a JDK and the Android
SDK, neither of which is installed on this machine. The intended route is:

1. Install [Android Studio](https://developer.android.com/studio).
2. **Open** (not import) the `android/` directory. Gradle will fetch the wrapper
   and the SDK components on first sync.
3. Enable USB debugging on the phone, plug it in, press Run.

There is no `gradlew` checked in because Android Studio generates one on first
sync; run `gradle wrapper` yourself if you prefer the CLI.

## Requirements

- Android 8.0 (API 26) or newer.
- Raw GNSS measurements need API 24+ and hardware support. Almost every phone
  sold in India since 2019 has it; a handful of budget models report the
  capability and deliver nothing, so check the raw-measurement row on a new
  device before relying on it.
- NavIC reporting needs API 29+ *and* a NavIC-capable chipset (Qualcomm and
  MediaTek parts from roughly 2019 onward). Where it is absent the app still
  records normally, and the NavIC counter simply stays at zero.

## Output

One directory per recording under
`Android/data/com.sih26168.drnav/files/recordings/<mode>-<timestamp>/`:

| File | Contents |
|---|---|
| `meta.json` | mode, device, sample rates, ENU origin |
| `imu.csv` | `t, ax, ay, az, gx, gy, gz` |
| `gnss.csv` | `t, e, n, u, ve, vn, vu, nsat, cn0, hdop, acc` |
| `gnss_raw.csv` | per-satellite svid, constellation, C/N₀, pseudorange rate |
| `baro.csv` | `t, pressure_hpa` |

Positions are already in the local ENU frame, so a recording replays with no
conversion step:

```python
from drnav.log_format import Recording
from drnav.align import coarse_align
from drnav.replay import run_eskf

rec = Recording.load("vehicle-20260904-153000")
est = run_eskf(rec, coarse_align(rec))
```

## Recording a usable log

The alignment step needs both of these, or it will refuse to start:

1. **Stand still for 15–20 seconds at the start**, engine on, phone in its final
   position. This is the gravity reference for roll and pitch.
2. **One clear acceleration or braking event under open sky** before the first
   outage. This is what makes yaw observable — a phone can be in any pose, so
   the direction of travel cannot be assumed.

Then drive the interesting part. Do not reposition the phone mid-recording: the
mounting rotation is estimated once and assumed fixed.
