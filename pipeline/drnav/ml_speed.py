"""Inference wrapper for the learned speed head.

Two things this file is careful about.

**Batching.** Replaying a five-minute recording sequentially through the network
means thousands of single-sample forward passes, which is dominated by
per-call overhead rather than arithmetic. :meth:`SpeedEstimator.precompute`
runs the whole recording in a handful of batches up front and the per-sample
call becomes a lookup. This is legitimate for *offline* replay -- the model
still only ever sees a causal 2 s window ending at the timestamp it is asked
about, so nothing from the future leaks in. On the phone it would run
sequentially at 10 Hz, which a 394k-parameter model manages comfortably.

**Honesty about uncertainty.** The sigma returned here is the network's own
predicted standard deviation, optionally inflated by ``sigma_scale``. If the
model turns out over-confident on real data, that scale is the dial to turn --
do not silently tighten it to make a demo look better, because the filter will
believe you.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import NET_RATE, WINDOW_N
from .features import level_window


class SpeedEstimator:
    """Callable matching the ``ml_speed`` hook in :func:`drnav.replay.run_eskf`."""

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 sigma_scale: float | None = None, sigma_floor: float = 0.05):
        import torch
        from .model import SpeedNet

        self.torch = torch
        self.sigma_floor = sigma_floor
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model = SpeedNet().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.train_metrics = ckpt.get("metrics", {})

        # Prefer the scale measured by scripts/calibrate_sigma.py on held-out
        # recordings. An explicit argument overrides it; falling back to 1.0
        # means shipping the network's raw, usually over-confident sigma.
        if sigma_scale is None:
            sigma_scale = float(ckpt.get("sigma_scale", 1.0))
        self.sigma_scale = sigma_scale

        self._t: np.ndarray | None = None
        self._speed: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    # -- offline path ------------------------------------------------------
    def precompute(self, rec, rate_hz: float = 10.0, batch: int = 512) -> None:
        """Evaluate every window in a recording ahead of the replay loop."""
        torch = self.torch
        step = max(1, int(round(rec.meta.imu_rate_hz / NET_RATE)))
        x = level_window(rec.accel, rec.gyro, rec.meta.imu_rate_hz)[::step]
        t = rec.t[::step]

        stride = max(1, int(round(NET_RATE / rate_hz)))
        starts = np.arange(0, len(x) - WINDOW_N, stride)
        if starts.size == 0:
            self._t = np.zeros(0); self._speed = np.zeros(0); self._sigma = np.zeros(0)
            return

        windows = np.stack([x[s:s + WINDOW_N] for s in starts]).astype(np.float32)
        speeds = np.empty(len(starts), dtype=np.float32)
        sigmas = np.empty(len(starts), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, len(windows), batch):
                chunk = torch.from_numpy(windows[i:i + batch]).to(self.device)
                s, lv = self.model(chunk)
                speeds[i:i + batch] = s.cpu().numpy()
                sigmas[i:i + batch] = np.exp(0.5 * lv.cpu().numpy())

        # each prediction describes the instant at the END of its window
        self._t = t[starts + WINDOW_N - 1]
        self._speed = speeds
        self._sigma = np.maximum(sigmas * self.sigma_scale, self.sigma_floor)

    # -- the hook ----------------------------------------------------------
    def __call__(self, t: float, imu_window: np.ndarray):
        """Return ``(speed, sigma)`` for time ``t``, or ``None`` if unavailable.

        ``imu_window`` is ignored when :meth:`precompute` has been run; it is
        kept in the signature so a live implementation can drop in unchanged.
        """
        if self._t is None or self._t.size == 0:
            return None
        i = int(np.searchsorted(self._t, t))
        if i >= len(self._t):
            i = len(self._t) - 1
        # refuse to answer with a stale prediction
        if abs(self._t[i] - t) > 0.5:
            return None
        return float(self._speed[i]), float(self._sigma[i])
