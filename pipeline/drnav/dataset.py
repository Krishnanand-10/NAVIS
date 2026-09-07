"""Training data for the learned speed head.

The simulator is the training set. That sounds like a shortcut and it is not:
it is the only way to get *labelled* velocity for an arbitrary phone pose
without a motion-capture rig, and it lets the model see far more variety of
mounting, bias and manoeuvre than a hackathon team could ever drive.

The domain gap to real phones is real and is not hidden anywhere in this
codebase -- see PROJECT_STATUS.md. The intended path is to pre-train here and
fine-tune on real logs using the masked-GNSS trick: record with GNSS healthy,
treat that track as ground truth, blank the GNSS in software, and train the
model to reproduce the trajectory it can no longer see. Any open road becomes a
synthetic tunnel.

Two details matter more than the architecture:

* **Windows are indexed lazily, not materialised.** Sixty recordings hold about
  70 MB as full trajectories; the same data cut into overlapping 2 s windows
  would be several gigabytes of almost entirely duplicated numbers.
* **Train/validation is split by recording, never by window.** Adjacent windows
  overlap by 95 %, so a random window-level split leaks the validation set into
  training and produces a beautiful, meaningless validation curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import horizontal_speed, level_window, random_yaw
from .synth import (OPEN_SKY, ImuErrorModel, Scenario, Segment, simulate)

NET_RATE = 100.0          # Hz the network sees; the sim runs at 200 and is halved
WINDOW_S = 2.0
WINDOW_N = int(NET_RATE * WINDOW_S)


# Road roughness varies enormously -- a resurfaced highway and a Bengaluru
# side street differ by several times in vibration amplitude at the same speed.
# Training on a single roughness teaches the model one road's calibration, and
# it will then over-read speed on rough tarmac and under-read on smooth. Set
# this to a single value only if you want to measure that upper bound
# deliberately; see PROJECT_STATUS.md for the two numbers side by side.
ROAD_ROUGHNESS = (0.20, 1.00)      # m/s^2 vertical vibration RMS at 60 km/h


def random_scenario(rng: np.random.Generator, mode: str) -> Scenario:
    """A random drive or walk, rich enough to cover the manoeuvre space.

    GNSS quality is irrelevant here -- the network only ever sees the IMU, so
    the training scenarios have no outages at all. What matters is variety of
    speed, turn rate, grade and stop-start behaviour.
    """
    pedestrian = mode == "pedestrian"
    v_max = rng.uniform(1.0, 2.1) if pedestrian else rng.uniform(8.0, 28.0)
    segments: list[Segment] = [Segment(rng.uniform(8, 20), 0.0, 0.0, OPEN_SKY,
                                       label="stationary")]
    for _ in range(rng.integers(6, 15)):
        stop = rng.random() < 0.18
        segments.append(Segment(
            duration=float(rng.uniform(8, 45)),
            speed=0.0 if stop else float(rng.uniform(0.15, 1.0) * v_max),
            turn_rate=float(rng.normal(0, 3.5 if pedestrian else 2.5)),
            gnss=OPEN_SKY,
            grade=float(rng.normal(0, 1.5)),
        ))
    # Vary the sensor quality too, so the model does not memorise one IMU.
    scale = float(rng.uniform(0.5, 2.0))
    roughness = float(rng.uniform(*ROAD_ROUGHNESS))
    errors = ImuErrorModel(
        accel_bias_sigma=0.08 * scale,
        accel_noise_density=0.0015 * float(rng.uniform(0.6, 1.8)),
        gyro_bias_sigma=0.010 * scale,
        gyro_noise_density=1.8e-4 * float(rng.uniform(0.6, 1.8)),
        vib_rms_ref=roughness,
        vib_gyro_rms_ref=0.025 * (roughness / 0.55),
    )
    return Scenario(
        name=f"train_{mode}",
        segments=segments,
        mode=mode,
        gait=pedestrian,
        accel_limit=0.8 if pedestrian else float(rng.uniform(1.5, 3.5)),
        imu_errors=errors,
    )


@dataclass
class Clip:
    """One recording, pre-levelled and decimated to the network's rate."""
    x: np.ndarray          # (T, 6) levelled accel + gyro
    speed: np.ndarray      # (T,) ground-truth horizontal speed
    mode: str


def build_clips(n_vehicle: int, n_pedestrian: int, seed: int = 0) -> list[Clip]:
    """Simulate recordings and reduce each to network-ready channels."""
    rng = np.random.default_rng(seed)
    clips: list[Clip] = []
    for mode, count in (("vehicle", n_vehicle), ("pedestrian", n_pedestrian)):
        for i in range(count):
            sc = random_scenario(rng, mode)
            rec = simulate(sc, seed=int(rng.integers(1 << 30)))
            step = int(round(rec.meta.imu_rate_hz / NET_RATE))
            x = level_window(rec.accel, rec.gyro, rec.meta.imu_rate_hz)[::step]
            speed = horizontal_speed(rec.truth_vel)[::step].astype(np.float32)
            clips.append(Clip(x=x, speed=speed, mode=mode))
    return clips


class WindowDataset:
    """Lazy windows over a list of clips, with yaw augmentation.

    Deliberately not a ``torch.utils.data.Dataset`` subclass -- keeping numpy
    here means the data pipeline is testable without importing torch, and the
    training script wraps it in a trivial adapter.
    """

    def __init__(self, clips: list[Clip], stride: int = 20,
                 augment: bool = True, seed: int = 0):
        self.clips = clips
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int]] = [
            (ci, s)
            for ci, c in enumerate(clips)
            for s in range(0, len(c.x) - WINDOW_N, stride)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.float32]:
        ci, s = self.index[i]
        c = self.clips[ci]
        w = c.x[s:s + WINDOW_N]
        if self.augment:
            w = random_yaw(w, self.rng)
        # The target is the speed at the *end* of the window: the filter asks
        # "how fast am I going now", not "how fast was I going on average".
        return w, c.speed[s + WINDOW_N - 1]

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialise everything. Only for validation sets, which are small."""
        xs = np.empty((len(self), WINDOW_N, 6), dtype=np.float32)
        ys = np.empty(len(self), dtype=np.float32)
        for i in range(len(self)):
            xs[i], ys[i] = self[i]
        return xs, ys


def split_by_clip(clips: list[Clip], val_fraction: float = 0.2, seed: int = 0):
    """Split *recordings*, not windows -- see the module docstring."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))
    n_val = max(1, int(len(clips) * val_fraction))
    val = [clips[i] for i in order[:n_val]]
    train = [clips[i] for i in order[n_val:]]
    return train, val
