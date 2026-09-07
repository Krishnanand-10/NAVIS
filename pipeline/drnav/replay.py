"""Offline replay: run the full estimator over a recording.

This is the piece that makes the project tractable in a hackathon. Recording a
drive takes half an hour; replaying it takes a second. Every algorithm change
gets tested against the same logged reality instead of against another trip out
to the tunnel.

The same function runs on synthetic recordings and on real Android logs -- they
are both just :class:`~drnav.log_format.Recording` objects.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .align import Alignment, find_stationary
from .eskf import Eskf, EskfConfig
from .gnss_health import (GnssFeatures, GnssHealth, OutageTracker, classify,
                          inflation_factor)


def prefilter_imu(accel, gyro, rate: float, cutoff: float):
    """Low-pass the IMU for mechanisation only.

    Uses a **causal** filter rather than the zero-phase ``sosfiltfilt`` used
    elsewhere for analysis. Zero-phase filtering needs the whole signal and
    cannot run on a phone; using it offline would flatter these results with
    performance the real system could never reproduce. The ~40 ms group delay
    of a causal 5 Hz Butterworth is what the deployed filter would actually
    experience, so it is what gets measured here.
    """
    if not cutoff or cutoff <= 0:
        return accel, gyro
    nyq = 0.5 * rate
    sos = butter(2, min(cutoff / nyq, 0.99), btype="low", output="sos")
    return sosfilt(sos, accel, axis=0), sosfilt(sos, gyro, axis=0)


def run_eskf(rec, alignment: Alignment, cfg: EskfConfig | None = None,
             ml_speed=None) -> dict:
    """Replay a recording through the ESKF.

    ``ml_speed`` is an optional callable ``(t, imu_window) -> (speed, sigma)``
    standing in for the learned inertial-odometry head. Leave it ``None`` to
    run the classical-aided configuration; the plumbing is already here so the
    network can be dropped in without touching the filter.
    """
    cfg = cfg or EskfConfig()
    t, accel, gyro = rec.t, rec.accel, rec.gyro
    g = rec.gnss
    rate = rec.meta.imu_rate_hz
    is_vehicle = rec.meta.mode == "vehicle"

    # The mechanisation gets a de-vibrated stream; stationarity detection keeps
    # the raw one, because vibration is genuine evidence of motion -- a real
    # vehicle shakes when it rolls and goes quiet when it stops, which is a
    # partial escape from the rest-versus-constant-velocity ambiguity that an
    # ideal noiseless IMU cannot offer.
    accel_f, gyro_f = prefilter_imu(accel, gyro, rate, cfg.imu_lowpass_hz)
    still = find_stationary(accel, gyro, rate)

    i0 = int(np.searchsorted(t, alignment.t_start))
    k = int(np.searchsorted(g[:, 0], t[i0]))
    p0 = g[k, 1:4] if k < len(g) else np.zeros(3)
    v0 = g[k, 4:7] if k < len(g) else np.zeros(3)

    f = Eskf(cfg)
    f.initialise(p0, v0, alignment.R)
    f.x.ba = alignment.ba0.copy()
    f.x.bg = alignment.bg0.copy()

    tracker = OutageTracker()
    tracker.note_fix(float(t[i0]))
    rejected = 0
    gnss_speed = float(np.linalg.norm(v0[:2]))
    # Deferred correction carried by the *output*, not the state. See
    # EskfConfig.output_smoothing_tau.
    smooth_offset = np.zeros(3)
    decay = (np.exp(-1.0 / (rate * cfg.output_smoothing_tau))
             if cfg.output_smoothing_tau > 0 else 0.0)
    gnss_speed_good = False        # was the last speed from a healthy fix?

    # Constraints are applied at a decimated rate, not once per IMU sample.
    # Consecutive IMU samples are not independent observations: asserting
    # "velocity is zero" 200 times a second makes the filter believe it to
    # about 2 mm/s, after which a 1 Hz GNSS fix cannot argue it out of a wrong
    # answer. Decimating to ~10 Hz keeps the information without the delusion.
    constraint_stride = max(1, int(round(rate / cfg.constraint_rate_hz)))

    n = len(t)
    pos = np.full((n, 3), np.nan)
    vel = np.full((n, 3), np.nan)
    dr_age = np.zeros(n)
    health_log = []
    bias_log = np.full((n, 6), np.nan)

    for i in range(i0, n):
        dt = float(t[i] - t[i - 1]) if i > 0 else 1.0 / rate
        f.predict(accel_f[i], gyro_f[i], dt)

        # --- GNSS, if a fix has just arrived --------------------------------
        p_before = f.x.p.copy()
        while k < len(g) and g[k, 0] <= t[i]:
            feat = GnssFeatures(nsat=g[k, 7], cn0=g[k, 8],
                                hdop=g[k, 9], accuracy=g[k, 10])
            health = classify(feat)
            scale = inflation_factor(health)
            if np.isfinite(scale):
                sigma = max(float(g[k, 10]), 1.0) * scale
                accepted = f.update_gnss_position(g[k, 1:4], sigma=sigma)
                if accepted:
                    rejected = 0
                    tracker.note_fix(float(g[k, 0]))
                else:
                    # gated out as an outlier -- but a gate needs an escape
                    rejected += 1
                    if rejected >= cfg.zupt_reject_after:
                        f.reset_position(g[k, 1:4], g[k, 4:7],
                                         sigma_p=sigma, sigma_v=2.0)
                        rejected = 0
                        tracker.note_fix(float(g[k, 0]))
                        accepted = True
                if cfg.use_gnss_velocity and accepted:
                    f.update_gnss_velocity(
                        g[k, 4:7],
                        sigma=cfg.gnss_vel_sigma * scale)
                gnss_speed = float(np.linalg.norm(g[k, 4:6]))
                gnss_speed_good = health is GnssHealth.GOOD
                # learn the phone's mounting only from trustworthy motion
                sp = float(np.linalg.norm(g[k, 4:6]))
                if health is GnssHealth.GOOD and sp > 5.0 and is_vehicle:
                    f.observe_mounting(float(np.arctan2(g[k, 4], g[k, 5])))
            health_log.append((float(g[k, 0]), health.value))
            k += 1

        # --- constraint updates that need no external signal -----------------
        # An IMU cannot tell rest from constant velocity -- Galilean
        # invariance, not a tuning problem -- so a quiet IMU is necessary but
        # nowhere near sufficient evidence of a stop. Corroborate with an
        # independent speed: GNSS while we have it, the filter's own estimate
        # once we do not. Without this the solution gets braked to a halt on
        # the motorway and never recovers, because ZUPT then keeps confirming
        # its own mistake.
        age = t[i] - (tracker.last_fix_t if tracker.last_fix_t is not None else t[i])
        gnss_fresh = age < 3.0
        # A speed reference is "independent" only if it does not come from the
        # thing we are about to correct. GNSS qualifies; so does the learned
        # odometry head, which reads speed off the IMU pattern rather than
        # integrating it. The filter's own velocity does not.
        # A multipath-corrupted Doppler speed is not an independent reference
        # either -- it is the same bad sky that is already lying about position.
        #
        # The learned head also has to be *usable*, not merely supplied: it is
        # gated off for vehicles (see EskfConfig.use_ml_speed_for_vehicle), and
        # counting a disabled estimator as a reference silently re-enabled ZUPT
        # inside tunnels -- the precise failure documented above. Measured cost
        # of that bug: tunnel RMSE 59 m -> 74 m purely from passing an
        # estimator the vehicle path never actually calls.
        ml_allowed = (not is_vehicle) or cfg.use_ml_speed_for_vehicle
        ml_usable = ml_speed is not None and ml_allowed
        have_reference = (gnss_fresh and gnss_speed_good) or ml_usable
        speed_ref = gnss_speed if gnss_fresh else float(np.linalg.norm(f.x.v))

        # Falling back on the filter's own speed mid-outage is a trap with
        # teeth: once inertial drift pulls the estimate under the threshold,
        # ZUPT pins it to zero and then keeps confirming its own mistake, with
        # no external signal left to argue. Measured on the tunnel scenario,
        # allowing that fallback moved mean RMSE from 18 m to 332 m. So unless
        # an independent reference exists, we simply do not claim to be
        # stopped -- which is also the honest answer.
        at_rest = (bool(still[i]) and speed_ref < cfg.zupt_max_speed
                   and (have_reference or not cfg.zupt_requires_speed_reference))

        if i % constraint_stride == 0:
            if at_rest:
                if cfg.use_zupt:
                    f.update_zupt()
                if cfg.use_zaru:
                    f.update_zaru(gyro_f[i])
            elif cfg.use_nhc and is_vehicle:
                f.update_nhc()

        # The learned speed head runs at the constraint rate, not the IMU rate:
        # its input is a 2 s window, so consecutive samples carry almost the
        # same information, and treating them as independent measurements is
        # the same over-confidence trap documented above for ZUPT.
        if ml_usable and not at_rest and i % constraint_stride == 0:
            lo = max(0, i - int(2.0 * rate))
            out = ml_speed(t[i], rec.imu[lo:i + 1])
            if out is not None:
                f.update_speed_magnitude(out[0], sigma=out[1])

        # Everything a measurement just moved goes into the offset, which then
        # bleeds away exponentially. Propagation is untouched, so the reported
        # track still follows the vehicle's actual motion while catching up.
        smooth_offset += f.x.p - p_before
        smooth_offset *= decay

        dr_age[i] = tracker.step(float(t[i]))
        pos[i], vel[i] = f.x.p - smooth_offset, f.x.v
        bias_log[i] = np.concatenate([f.x.ba, f.x.bg])

    return {"t": t, "pos": pos, "vel": vel, "dr_age": dr_age,
            "health": health_log, "bias": bias_log,
            "outages": tracker.outages, "start_index": i0, "filter": f}


def truth_outages(rec) -> tuple[list, list]:
    """Outage windows and labels declared by a synthetic recording's metadata."""
    wins, labels = [], []
    for e in rec.meta.events:
        if e["kind"] == "outage":
            if wins and abs(wins[-1][1] - e["t_start"]) < 1e-6:
                wins[-1][1] = e["t_end"]          # merge adjacent outage segments
                labels[-1] = labels[-1].split(":")[0] + ": full outage"
            else:
                wins.append([e["t_start"], e["t_end"]])
                labels.append(e["name"])
    return wins, labels
