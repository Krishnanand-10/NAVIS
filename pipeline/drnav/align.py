"""Initial alignment: finding the phone's attitude before navigation can start.

Roll and pitch fall out of gravity the moment the device is still. Yaw does
not -- it is unobservable at rest, and a phone can be in any pose, so we cannot
assume it points along the direction of travel.

The fix is TRIAD: two non-parallel vector pairs, each known in both the body
and the nav frame, determine attitude uniquely.

* **Pair 1 -- gravity.** At rest the accelerometer reads specific force ``+g``
  along nav-up, so ``v_nav = [0, 0, 1]`` and ``v_body = mean(accel)``.
* **Pair 2 -- a horizontal acceleration event.** Differentiating GNSS velocity
  gives ``a_nav``; the accelerometer simultaneously reads ``a_nav - g`` in the
  body frame. Any decent braking or pull-away supplies this.

Both come free from the first minute of any recording that starts parked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import G_VEC_ENU


@dataclass
class Alignment:
    R: np.ndarray            # device->nav at t_start
    t_start: float           # when navigation may begin
    ba0: np.ndarray          # accelerometer bias seen during the still window
    bg0: np.ndarray          # gyro bias seen during the still window
    stationary: tuple[float, float]
    quality: float           # |sin| between the two TRIAD vectors; >0.3 is healthy


def triad(v1_b, v2_b, v1_n, v2_n) -> np.ndarray:
    """Body->nav rotation from two vector pairs (first pair trusted most)."""
    def frame(a, b):
        t1 = a / np.linalg.norm(a)
        c = np.cross(t1, b)
        t2 = c / np.linalg.norm(c)
        return np.column_stack([t1, t2, np.cross(t1, t2)])
    return frame(v1_n, v2_n) @ frame(v1_b, v2_b).T


def find_stationary(accel, gyro, rate, win_s=2.0, accel_std=0.35, gyro_norm=0.06):
    """Boolean mask of samples whose IMU signature is consistent with rest.

    Uses the *variability* of specific force rather than its magnitude: a phone
    in free fall or under sustained acceleration can still read 9.8 m/s^2.

    **This test is necessary but not sufficient.** An IMU cannot distinguish
    rest from constant velocity -- that is Galilean invariance, not a tuning
    problem -- so smooth cruising on a good road looks exactly like standing
    still. Callers must corroborate with an independent speed estimate before
    applying a zero-velocity update; :func:`drnav.replay.run_eskf` gates on the
    filter's own speed. Applying ZUPT on this mask alone will confidently
    brake the solution to a halt at 60 km/h.
    """
    w = max(4, int(win_s * rate))
    kern = np.ones(w) / w
    mag = np.linalg.norm(accel, axis=1)
    mean = np.convolve(mag, kern, mode="same")
    var = np.convolve(mag ** 2, kern, mode="same") - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    gmag = np.convolve(np.linalg.norm(gyro, axis=1), kern, mode="same")
    return (std < accel_std) & (gmag < gyro_norm)


def _longest_run(mask):
    """Start/stop indices of the longest True run, or None."""
    if not mask.any():
        return None
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k])


def coarse_align(rec, search_s: float = 120.0) -> Alignment:
    """Estimate initial attitude and biases from the opening of a recording."""
    rate = rec.meta.imu_rate_hz
    t, accel, gyro = rec.t, rec.accel, rec.gyro
    n_search = min(len(t), int(search_s * rate))

    still = find_stationary(accel[:n_search], gyro[:n_search], rate)

    # Corroborate with GNSS speed. A quiet IMU means "not accelerating", not
    # "not moving" -- steady cruising passes the IMU test easily, and picking
    # a cruise window here would take the gravity reference from a tilted,
    # accelerating platform and hand the yaw search an interval with no
    # manoeuvre left in it. This is the same ambiguity that governs ZUPT.
    g_all = rec.gnss
    if len(g_all):
        sp = np.linalg.norm(g_all[:, 4:7], axis=1)
        sp_at_imu = np.interp(t[:n_search], g_all[:, 0], sp,
                              left=sp[0], right=sp[-1])
        still = still & (sp_at_imu < 0.5)

    run = _longest_run(still)
    if run is None or (run[1] - run[0]) < int(1.0 * rate):
        raise RuntimeError(
            "no stationary window found in the first %.0f s -- initial "
            "alignment needs the device to sit still briefly at the start "
            "of a recording" % search_s)
    s0, s1 = run

    g_body = accel[s0:s1].mean(axis=0)
    bg0 = gyro[s0:s1].mean(axis=0)

    # --- find the strongest horizontal acceleration event from GNSS ---
    g = rec.gnss
    tg, vg = g[:, 0], g[:, 4:7]
    a_nav = np.gradient(vg, tg, axis=0)
    horiz = np.linalg.norm(a_nav[:, :2], axis=1)
    # any manoeuvre under usable GNSS will do, before or after the still window
    k = int(np.argmax(horiz))
    if horiz[k] < 0.4:
        raise RuntimeError(
            "no clear acceleration event found for yaw alignment -- the "
            "recording needs at least one pull-away or braking manoeuvre "
            "under good GNSS before the first outage")

    # average the accelerometer over a short window around the event
    i_ev = int(np.searchsorted(t, tg[k]))
    half = int(0.35 * rate)
    f_body = accel[max(0, i_ev - half):i_ev + half].mean(axis=0)
    f_nav = a_nav[k] - G_VEC_ENU

    R = triad(g_body, f_body, np.array([0.0, 0.0, 1.0]), f_nav)

    # how well-conditioned was the geometry? parallel vectors give no yaw info
    u1 = g_body / np.linalg.norm(g_body)
    u2 = f_body / np.linalg.norm(f_body)
    quality = float(np.linalg.norm(np.cross(u1, u2)))

    # residual specific force during the still window is accelerometer bias
    ba0 = g_body - R.T @ (-G_VEC_ENU)

    return Alignment(R=R, t_start=float(tg[k] + 1.0), ba0=ba0, bg0=bg0,
                     stationary=(float(t[s0]), float(t[s1])), quality=quality)
