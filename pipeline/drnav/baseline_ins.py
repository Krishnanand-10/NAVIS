"""The classical baseline: GNSS snapping plus unaided strapdown integration.

This is deliberately the naive system -- what you get without an estimator.
While GNSS is healthy it simply takes the fix; when GNSS disappears it
integrates the IMU with the biases it happened to have at that moment.

It exists to be beaten, and to be *plotted*. The single most persuasive slide
in the deck is this curve and the ESKF curve on the same axes: one leaves the
map inside a minute, the other stays on the road.
"""

from __future__ import annotations

import numpy as np

from .attitude import G_VEC_ENU, exp_so3, orthonormalise
from .gnss_health import GnssFeatures, GnssHealth, classify


def run_baseline(rec, alignment, use_health: bool = False,
                 lowpass_hz: float = 5.0) -> dict:
    """Integrate the recording with no error-state estimation.

    ``use_health=False`` reproduces the truly naive behaviour of trusting every
    fix, multipath included.
    """
    from .replay import prefilter_imu
    # Same IMU conditioning as the ESKF gets. Denying the baseline this would
    # make the comparison flattering rather than informative.
    accel, gyro = prefilter_imu(rec.accel, rec.gyro, rec.meta.imu_rate_hz, lowpass_hz)
    t = rec.t
    g = rec.gnss
    i0 = int(np.searchsorted(t, alignment.t_start))

    p = np.zeros(3)
    v = np.zeros(3)
    R = alignment.R.copy()
    ba, bg = alignment.ba0.copy(), alignment.bg0.copy()

    # seed from the first fix at or after alignment
    k = int(np.searchsorted(g[:, 0], t[i0]))
    if k < len(g):
        p, v = g[k, 1:4].copy(), g[k, 4:7].copy()

    n = len(t)
    out_p = np.full((n, 3), np.nan)
    out_v = np.full((n, 3), np.nan)

    for i in range(i0, n):
        dt = t[i] - t[i - 1] if i > 0 else 1.0 / rec.meta.imu_rate_hz
        f = accel[i] - ba
        w = gyro[i] - bg
        a_nav = R @ f + G_VEC_ENU
        p = p + v * dt + 0.5 * a_nav * dt * dt
        v = v + a_nav * dt
        R = R @ exp_so3(w * dt)
        if i % 200 == 0:
            R = orthonormalise(R)

        # snap to any fix that has just arrived
        while k < len(g) and g[k, 0] <= t[i]:
            usable = True
            if use_health:
                feat = GnssFeatures(nsat=g[k, 7], cn0=g[k, 8],
                                    hdop=g[k, 9], accuracy=g[k, 10])
                usable = classify(feat) is GnssHealth.GOOD
            if usable:
                p, v = g[k, 1:4].copy(), g[k, 4:7].copy()
            k += 1

        out_p[i], out_v[i] = p, v

    return {"t": t, "pos": out_p, "vel": out_v, "start_index": i0}
