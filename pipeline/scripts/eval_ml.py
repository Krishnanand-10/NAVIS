#!/usr/bin/env python3
"""Measure what the learned speed head actually buys, across every scenario.

    python scripts/eval_ml.py --seeds 6

Reports classical strapdown, the ESKF without the network, and the ESKF with
it, so the contribution of the ML component is separated from the contribution
of the filter around it. That separation matters: it is the difference between
"our AI system achieves X" and being able to say which part earned the X.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav import metrics, plotting                    # noqa: E402
from drnav.align import coarse_align                   # noqa: E402
from drnav.baseline_ins import run_baseline            # noqa: E402
from drnav.ml_speed import SpeedEstimator              # noqa: E402
from drnav.replay import run_eskf, truth_outages       # noqa: E402
from drnav.synth import SCENARIOS, simulate            # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


class ConstantSpeedPrior:
    """Control condition: assert a fixed speed, with no learning whatsoever.

    This exists because of a result that would otherwise be badly misread. An
    *untrained* network -- random weights, emitting a constant 0.66 m/s --
    dropped pedestrian RMSE from 549 m to 29 m, simply because any bounded
    speed claim stops unaided inertial integration running away.

    So "adding the ML head improved things 19x" is not evidence the model
    learned anything. The honest question is whether the trained network beats
    a constant of the same order, and that is what this class measures. Report
    both numbers or the headline is meaningless.
    """

    def __init__(self, speed: float, sigma: float = 1.0):
        self.speed = speed
        self.sigma = sigma

    def precompute(self, rec, **kw):
        pass

    def __call__(self, t, imu_window):
        return self.speed, self.sigma


def rmse(pos, truth):
    e = np.linalg.norm(pos[:, :2] - truth[:, :2], axis=1)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def exit_error(t, pos, truth, outages):
    if not outages:
        return float("nan")
    i = int(np.searchsorted(t, min(outages[-1][1], t[-1])))
    return float(np.linalg.norm(pos[i, :2] - truth[i, :2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--checkpoint", default=str(OUT / "speednet.pt"))
    ap.add_argument("--sigma-scale", type=float, default=None,
                    help="override the calibrated sigma scale stored in the "
                         "checkpoint by scripts/calibrate_sigma.py; leave unset "
                         "to use it (passing 1.0 ships the raw, over-confident "
                         "sigma and is only useful for measuring that cost)")
    args = ap.parse_args()

    est = SpeedEstimator(args.checkpoint, sigma_scale=args.sigma_scale)
    print(f"model val metrics at training time: {est.train_metrics}\n")

    header = (f"{'scenario':17s} {'estimator':24s} {'RMSE':>9s} {'median':>8s} "
              f"{'exit err':>9s} {'drift%':>8s}")
    print(header); print("-" * len(header))

    summary = {}
    for name in SCENARIOS:
        rows = {}
        # a fair constant for the control: the scenario's own mean moving speed
        probe = simulate(SCENARIOS[name](), seed=0)
        sp = np.linalg.norm(probe.truth_vel[:, :2], axis=1)
        const_speed = float(sp[sp > 0.3].mean()) if (sp > 0.3).any() else 1.0

        for label in ("classical strapdown", "ESKF (no ML)",
                      "control: constant speed", "ESKF + learned speed"):
            R, E, D = [], [], []
            for seed in range(args.seeds):
                rec = simulate(SCENARIOS[name](), seed=seed)
                al = coarse_align(rec)
                outages, _ = truth_outages(rec)
                t, truth = rec.t, rec.truth_pos

                if label == "classical strapdown":
                    res = run_baseline(rec, al)
                elif label == "ESKF (no ML)":
                    res = run_eskf(rec, al)
                elif label.startswith("control"):
                    res = run_eskf(rec, al, ml_speed=ConstantSpeedPrior(const_speed))
                else:
                    est.precompute(rec)
                    res = run_eskf(rec, al, ml_speed=est)

                R.append(rmse(res["pos"], truth))
                ee = exit_error(t, res["pos"], truth, outages)
                E.append(ee)
                if outages:
                    m = (t >= outages[0][0]) & (t <= outages[-1][1])
                    dist = float(np.sum(np.linalg.norm(np.diff(truth[m, :2], axis=0), axis=1)))
                    D.append(100 * ee / dist if dist > 1 else np.nan)
            rows[label] = (np.mean(R), np.median(R), np.nanmean(E), np.nanmean(D) if D else np.nan)
            print(f"{name:17s} {label:24s} {rows[label][0]:9.1f} {rows[label][1]:8.1f} "
                  f"{rows[label][2]:9.1f} {rows[label][3]:8.2f}")
        summary[name] = rows
        print()

    print("=" * 78)
    print("Does the network earn its place? (mean RMSE, m)")
    print(f"  {'scenario':17s} {'no ML':>10s} {'constant':>10s} {'learned':>10s}"
          f"   {'verdict':>22s}")
    for name, rows in summary.items():
        a = rows["ESKF (no ML)"][0]
        c = rows["control: constant speed"][0]
        b = rows["ESKF + learned speed"][0]
        if abs(a - b) < 0.05 and abs(a - c) < 0.05:
            # All three identical means no speed pseudo-measurement was applied
            # at all -- the head is gated off for vehicles. Reporting that as
            # "no better than a constant" would imply a comparison that never
            # happened. See EskfConfig.use_ml_speed_for_vehicle.
            verdict = "not applied (vehicle mode)"
        elif b < c * 0.85:
            verdict = f"learning helps ({c / b:.1f}x over control)"
        elif b < c * 1.15:
            verdict = "no better than a constant"
        else:
            verdict = f"WORSE than control ({b / c:.1f}x)"
        print(f"  {name:17s} {a:10.1f} {c:10.1f} {b:10.1f}   {verdict:>22s}")

    # figure for the scenario the network was built to rescue
    rec = simulate(SCENARIOS["pedestrian_mall"](), seed=0)
    al = coarse_align(rec)
    outages, _ = truth_outages(rec)
    est.precompute(rec)
    base = run_eskf(rec, al)                        # stands in as the "before"
    withml = run_eskf(rec, al, ml_speed=est)
    s_b = metrics.evaluate(rec.t, base["pos"], rec.truth_pos, outages)
    s_m = metrics.evaluate(rec.t, withml["pos"], rec.truth_pos, outages)
    path = plotting.plot_run(
        rec, base, withml, s_b, s_m, outages,
        "pedestrian indoors -- ESKF alone vs ESKF + learned speed head",
        OUT / "pedestrian_ml.png",
        base_label="ESKF, no learned head",
        est_label="ESKF + learned speed head")
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
