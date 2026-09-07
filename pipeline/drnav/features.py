"""Turning raw IMU into something a network can learn speed from.

The problem this file solves is that a phone sits in an unknown pose. The same
drive recorded with the handset flat on a cradle and upside-down in a jacket
pocket produces completely different raw accelerometer traces, and a network
trained on one will not recognise the other.

The fix is to rotate every window into a **gravity-levelled frame** before the
network ever sees it:

* ``z`` points along the locally-estimated gravity direction, so "up" is always
  the same channel no matter how the phone is held;
* ``x`` and ``y`` span the horizontal plane, but their *azimuth is arbitrary* --
  there is nothing in a 2 s window of IMU that says which way is north.

That last point matters and is handled deliberately. We do not try to resolve
the azimuth here. Instead the training pipeline applies a **random yaw rotation
to every sample** (:func:`random_yaw`), so the network is forced to learn a
representation that does not depend on it. This is the RoNIN recipe, and it is
why the model generalises across mounting positions.

Gravity is estimated from the window's own low-passed accelerometer rather than
from the filter's attitude estimate. That keeps the network independent of the
thing it is meant to be correcting -- feeding a drifting attitude into the
model that corrects the drift is a feedback loop nobody wants to debug.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# The network's input channel layout, for anyone reading a tensor dump.
CHANNELS = ["ax_level", "ay_level", "az_level", "gx_level", "gy_level", "gz_level"]
N_CHANNELS = len(CHANNELS)


def gravity_direction(accel: np.ndarray, rate: float, cutoff: float = 0.4) -> np.ndarray:
    """Unit vector along gravity, in the body frame, per sample.

    A low-pass at 0.4 Hz keeps the sustained component (gravity plus any very
    slow manoeuvre) and rejects gait bounce, engine vibration and road noise.
    ``sosfiltfilt`` is zero-phase, so the estimate is not delayed relative to
    the samples it is levelling -- with a causal filter the levelling would lag
    the motion by a tenth of a second, which is enough to smear the very
    features the model keys on.
    """
    if len(accel) < 20:                       # too short to filter meaningfully
        g = accel.mean(axis=0, keepdims=True).repeat(len(accel), axis=0)
    else:
        nyq = 0.5 * rate
        wn = min(cutoff / nyq, 0.99)
        sos = butter(2, wn, btype="low", output="sos")
        # padlen must stay below the signal length for short windows
        padlen = min(3 * (sos.shape[0] * 2), len(accel) - 1)
        g = sosfiltfilt(sos, accel, axis=0, padlen=padlen)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return np.divide(g, n, out=np.tile([0.0, 0.0, 1.0], (len(accel), 1)), where=n > 1e-6)


def _level_frame(g_hat: np.ndarray) -> np.ndarray:
    """Body->level rotation whose third row is ``g_hat``.

    The horizontal axes are produced by Gram-Schmidt against whichever global
    axis is least parallel to gravity, which keeps the construction stable when
    the phone is lying flat *or* standing on edge. Their azimuth is arbitrary
    by construction; see the module docstring.
    """
    z = g_hat
    ref = np.tile([1.0, 0.0, 0.0], (len(z), 1))
    too_parallel = np.abs(z[:, 0]) > 0.9
    ref[too_parallel] = [0.0, 1.0, 0.0]

    x = ref - (np.sum(ref * z, axis=1, keepdims=True)) * z
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)        # (N, 3, 3), rows are the axes


def level_window(accel: np.ndarray, gyro: np.ndarray, rate: float) -> np.ndarray:
    """Rotate one window into the gravity-levelled frame.

    Returns ``(N, 6)``: levelled accelerometer then levelled gyroscope. The
    vertical accelerometer channel still contains gravity -- deliberately, as
    its exact value carries the accelerometer bias the model can learn to
    discount, and removing a nominal 9.81 would throw that away.
    """
    R = _level_frame(gravity_direction(accel, rate))
    a = np.einsum("nij,nj->ni", R, accel)
    w = np.einsum("nij,nj->ni", R, gyro)
    return np.concatenate([a, w], axis=1).astype(np.float32)


def random_yaw(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate a levelled window by a random angle about the vertical.

    The training-time counterpart to the arbitrary azimuth of the level frame.
    Applied to accelerometer and gyroscope horizontal pairs alike, and never at
    inference -- at inference the network simply has to cope, which is the
    whole point of having trained it this way.
    """
    c, s = np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)
    out = window.copy()
    for base in (0, 3):                       # accel block, then gyro block
        x, y = window[:, base], window[:, base + 1]
        out[:, base] = c * x - s * y
        out[:, base + 1] = s * x + c * y
    return out


def horizontal_speed(vel: np.ndarray) -> np.ndarray:
    """The regression target: ground speed, which is rotation-invariant.

    Speed rather than a velocity vector is a deliberate choice. This project's
    own analysis is that heading is easy and speed is hard -- a gyroscope holds
    heading to a few degrees a minute, while speed is what an unaided IMU loses
    almost immediately. So the network is pointed at the hard half, and its
    output drops straight into the filter's existing forward-speed update.
    """
    return np.linalg.norm(vel[..., :2], axis=-1)
