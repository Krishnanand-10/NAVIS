#!/usr/bin/env python3
"""Fix the model's uncertainty scale after training.

    python scripts/calibrate_sigma.py

A heteroscedastic head trained with Gaussian NLL usually ends up somewhat
over-confident: the loss rewards shrinking sigma right up until the squared-error
term bites, and on data it has partly memorised the error term is small. An
over-confident measurement is the one thing a Kalman filter cannot tolerate --
it will weight the update far too heavily and drag the state off, which is
exactly the failure this project already documented for ZUPT.

The fix is a single scalar, computed on held-out recordings:

    scale = sqrt( mean( (error / sigma)^2 ) )

After multiplying every predicted sigma by that, the mean squared z-score is 1
by construction, which is what "calibrated" means. It is one number and it
cannot overfit anything, but it must be computed on clips the model did not
train on -- so this reuses the same by-recording split as training.

The scale is written back into the checkpoint and picked up automatically by
`drnav.ml_speed.SpeedEstimator`.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav.dataset import WindowDataset, split_by_clip     # noqa: E402
from drnav.model import SpeedNet                           # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


@torch.no_grad()
def collect(model, clips, device, stride=50):
    ds = WindowDataset(clips, stride=stride, augment=False, seed=11)
    xs, ys = ds.to_arrays()
    errs, sigmas = [], []
    for i in range(0, len(xs), 512):
        mu, lv = model(torch.from_numpy(xs[i:i + 512]).to(device))
        errs.append(mu.cpu().numpy() - ys[i:i + 512])
        sigmas.append(np.exp(0.5 * lv.cpu().numpy()))
    return np.concatenate(errs), np.concatenate(sigmas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT / "speednet.pt"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SpeedNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(OUT / "clips.pkl", "rb") as fh:
        clips = pickle.load(fh)
    _, val_clips = split_by_clip(clips, val_fraction=0.2, seed=1)   # same split as training

    err, sigma = collect(model, val_clips, device)
    before = float(np.mean((err / sigma) ** 2))
    scale = float(np.sqrt(before))
    after = float(np.mean((err / (sigma * scale)) ** 2))

    verdict = ("over-confident" if before > 1.3 else
               "under-confident" if before < 0.7 else "already well calibrated")
    print(f"validation windows      {len(err):,}")
    print(f"speed RMSE              {float(np.sqrt(np.mean(err ** 2))):.3f} m/s")
    print(f"mean predicted sigma    {float(np.mean(sigma)):.3f} m/s")
    print(f"calibration before      {before:6.2f}   ({verdict})")
    print(f"sigma scale factor      {scale:6.3f}")
    print(f"calibration after       {after:6.2f}   (1.00 is honest)")

    ckpt["sigma_scale"] = scale
    ckpt.setdefault("metrics", {})["calibration_raw"] = before
    torch.save(ckpt, args.checkpoint)
    print(f"\nwritten back to {args.checkpoint}")


if __name__ == "__main__":
    main()
