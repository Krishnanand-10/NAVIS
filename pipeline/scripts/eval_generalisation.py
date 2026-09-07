#!/usr/bin/env python3
"""Does the speed head generalise across road surfaces, or memorise one?

    python scripts/eval_generalisation.py

This is the honesty check on the vehicle half of the model. Vehicle speed is
inferred largely from road-induced vibration, and vibration amplitude depends
on the road as much as on the speed. A model trained on a single surface learns
that surface's calibration and will read high on rough tarmac and low on smooth
-- while reporting the same confident sigma, which is the dangerous part.

So the model is trained across a range of roughness (dataset.ROAD_ROUGHNESS)
and evaluated here on held-out recordings at fixed roughness levels spanning
and exceeding that range. What to look for:

* RMSE roughly flat across the trained range -> it generalises.
* RMSE rising sharply outside it -> expected, and the reason the predicted
  sigma matters; check that sigma rises too.
* Calibration near 1.0 at every level -> the uncertainty is honest, which is
  what the Kalman filter downstream actually depends on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav.dataset import (NET_RATE, ROAD_ROUGHNESS, WINDOW_N,   # noqa: E402
                           random_scenario)
from drnav.features import horizontal_speed, level_window        # noqa: E402
from drnav.model import SpeedNet                                 # noqa: E402
from drnav.synth import ImuErrorModel, simulate                  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


def clips_at_roughness(roughness: float, n: int, mode: str, seed: int):
    """Fresh recordings pinned to one road surface."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sc = random_scenario(rng, mode)
        sc.imu_errors = ImuErrorModel(
            vib_rms_ref=roughness,
            vib_gyro_rms_ref=0.025 * (roughness / 0.55),
        )
        rec = simulate(sc, seed=int(rng.integers(1 << 30)))
        step = int(round(rec.meta.imu_rate_hz / NET_RATE))
        out.append((level_window(rec.accel, rec.gyro, rec.meta.imu_rate_hz)[::step],
                    horizontal_speed(rec.truth_vel)[::step]))
    return out


@torch.no_grad()
def score(model, clips, device, stride=40):
    xs, ys = [], []
    for x, speed in clips:
        for s in range(0, len(x) - WINDOW_N, stride):
            xs.append(x[s:s + WINDOW_N]); ys.append(speed[s + WINDOW_N - 1])
    if not xs:
        return None
    X = torch.from_numpy(np.stack(xs).astype(np.float32)).to(device)
    y = np.array(ys, dtype=np.float32)
    mu, sg = [], []
    for i in range(0, len(X), 512):
        m, lv = model(X[i:i + 512])
        mu.append(m.cpu().numpy()); sg.append(np.exp(0.5 * lv.cpu().numpy()))
    mu = np.concatenate(mu); sg = np.concatenate(sg)
    err = mu - y
    moving = y > 0.5
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmse_moving": float(np.sqrt(np.mean(err[moving] ** 2))) if moving.any() else float("nan"),
        "bias": float(np.mean(err[moving])) if moving.any() else float("nan"),
        "sigma": float(np.mean(sg)),
        "calibration": float(np.mean((err / sg) ** 2)),
        "n": len(y),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT / "speednet.pt"))
    ap.add_argument("--clips", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SpeedNet().to(device); model.load_state_dict(ckpt["state_dict"]); model.eval()
    lo, hi = ROAD_ROUGHNESS
    print(f"trained roughness range: {lo}-{hi} m/s^2 vertical RMS at 60 km/h\n")

    print(f"{'surface':22s} {'roughness':>10s} {'RMSE':>8s} {'moving':>8s} "
          f"{'bias':>8s} {'sigma':>7s} {'calib':>7s}")
    print("-" * 76)
    levels = [("glass-smooth (untrained)", 0.10),
              ("smooth highway", 0.20),
              ("good road", 0.45),
              ("worn tarmac", 0.75),
              ("rough city street", 1.00),
              ("broken road (untrained)", 1.50)]
    for label, r in levels:
        m = score(model, clips_at_roughness(r, args.clips, "vehicle", seed=int(r * 1000)), device)
        inside = lo <= r <= hi
        mark = "" if inside else "   <- outside trained range"
        print(f"{label:22s} {r:10.2f} {m['rmse']:8.3f} {m['rmse_moving']:8.3f} "
              f"{m['bias']:+8.3f} {m['sigma']:7.3f} {m['calibration']:7.2f}{mark}")

    print()
    m = score(model, clips_at_roughness(0.55, args.clips, "pedestrian", seed=7), device)
    print(f"{'pedestrian (gait)':22s} {'n/a':>10s} {m['rmse']:8.3f} {m['rmse_moving']:8.3f} "
          f"{m['bias']:+8.3f} {m['sigma']:7.3f} {m['calibration']:7.2f}")


if __name__ == "__main__":
    main()
