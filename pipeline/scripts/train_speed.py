#!/usr/bin/env python3
"""Train the learned speed head.

    python scripts/train_speed.py --epochs 25
    python scripts/train_speed.py --epochs 25 --rebuild   # regenerate clips

Writes `out/speednet.pt` (weights plus the config needed to rebuild the model)
and prints a calibration report, which matters as much as the RMSE: a Kalman
filter fed an over-confident measurement diverges, so a model whose predicted
sigma is honest is worth more than one that is marginally more accurate and
lies about it.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav.dataset import (WINDOW_N, build_clips, split_by_clip,  # noqa: E402
                           WindowDataset)
from drnav.model import (SpeedNet, count_parameters, gaussian_nll)  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
CLIPS_CACHE = OUT / "clips.pkl"
CHECKPOINT = OUT / "speednet.pt"


class TorchWindows(Dataset):
    """Thin adapter so the numpy WindowDataset can feed a DataLoader."""

    def __init__(self, inner: WindowDataset):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        x, y = self.inner[i]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clips(rebuild: bool, n_vehicle: int, n_pedestrian: int):
    if CLIPS_CACHE.exists() and not rebuild:
        with open(CLIPS_CACHE, "rb") as fh:
            return pickle.load(fh)
    print(f"simulating {n_vehicle} vehicle + {n_pedestrian} pedestrian recordings...")
    t0 = time.time()
    clips = build_clips(n_vehicle, n_pedestrian, seed=0)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CLIPS_CACHE, "wb") as fh:
        pickle.dump(clips, fh)
    print(f"  done in {time.time() - t0:.0f}s -> {CLIPS_CACHE}")
    return clips


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    errs, sigmas, targets = [], [], []
    for x, y in loader:
        s, lv = model(x.to(device))
        errs.append((s.cpu() - y).numpy())
        sigmas.append(torch.exp(0.5 * lv).cpu().numpy())
        targets.append(y.numpy())
    e = np.concatenate(errs); sg = np.concatenate(sigmas); tg = np.concatenate(targets)
    moving = tg > 0.5
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "rmse_moving": float(np.sqrt(np.mean(e[moving] ** 2))) if moving.any() else float("nan"),
        # A well-calibrated heteroscedastic model has mean (err/sigma)^2 ~ 1.
        # Below 1 means it is under-confident; above 1 means it is lying, which
        # is the failure mode that breaks the filter downstream.
        "calibration": float(np.mean((e / sg) ** 2)),
        "mean_sigma": float(np.mean(sg)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--vehicle", type=int, default=50)
    ap.add_argument("--pedestrian", type=int, default=34)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint, to keep runs side by side")
    args = ap.parse_args()

    global CHECKPOINT
    if args.tag:
        CHECKPOINT = OUT / f"speednet_{args.tag}.pt"
    device = pick_device()
    clips = load_clips(args.rebuild, args.vehicle, args.pedestrian)
    train_clips, val_clips = split_by_clip(clips, val_fraction=0.2, seed=1)
    print(f"clips: {len(train_clips)} train / {len(val_clips)} val   device: {device}")

    train_ds = TorchWindows(WindowDataset(train_clips, stride=20, augment=True, seed=2))
    val_ds = TorchWindows(WindowDataset(val_clips, stride=50, augment=False, seed=3))
    print(f"windows: {len(train_ds)} train / {len(val_ds)} val")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    model = SpeedNet().to(device)
    print(f"parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            s, lv = model(x)
            loss = gaussian_nll(s, lv, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item() * len(y); n += len(y)

        m = evaluate(model, val_dl, device)
        flag = ""
        if m["rmse"] < best:
            best = m["rmse"]
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "window_n": WINDOW_N,
                        "metrics": m}, CHECKPOINT)
            flag = "  *saved"
        print(f"epoch {epoch:3d}  loss {total / n:7.4f}   "
              f"val RMSE {m['rmse']:6.3f} m/s  (moving {m['rmse_moving']:6.3f})  "
              f"calib {m['calibration']:5.2f}  sigma {m['mean_sigma']:5.3f}  "
              f"{time.time() - t0:5.1f}s{flag}")

    print(f"\nbest val RMSE {best:.3f} m/s -> {CHECKPOINT}")


if __name__ == "__main__":
    main()
