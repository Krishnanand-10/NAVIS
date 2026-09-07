#!/usr/bin/env python3
"""Generate a scenario, replay it through both estimators, report and plot.

    python scripts/run_replay.py --scenario tunnel_drive
    python scripts/run_replay.py --scenario all --seed 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav import metrics, plotting                     # noqa: E402
from drnav.align import coarse_align                    # noqa: E402
from drnav.baseline_ins import run_baseline             # noqa: E402
from drnav.replay import run_eskf, truth_outages        # noqa: E402
from drnav.synth import SCENARIOS, simulate             # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


def run_one(name: str, seed: int, save_logs: bool) -> dict:
    scenario = SCENARIOS[name]()
    rec = simulate(scenario, seed=seed)
    if save_logs:
        rec.save(OUT / "logs" / name)

    align = coarse_align(rec)
    outages, labels = truth_outages(rec)

    base = run_baseline(rec, align)
    est = run_eskf(rec, align)

    s_base = metrics.evaluate(rec.t, base["pos"], rec.truth_pos, outages, labels)
    s_eskf = metrics.evaluate(rec.t, est["pos"], rec.truth_pos, outages, labels)

    print(f"\n{'=' * 74}\n{name}   (seed {seed}, alignment quality "
          f"{align.quality:.2f}, nav starts t={align.t_start:.0f}s)\n{'=' * 74}")
    print(metrics.format_summary("classical strapdown DR", s_base))
    print()
    print(metrics.format_summary("ESKF (GNSS + ZUPT + ZARU + NHC)", s_eskf))

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plotting.plot_run(rec, base, est, s_base, s_eskf, outages,
                            f"{name} -- dead reckoning through GNSS outage",
                            OUT / f"{name}.png")
    plotting.plot_bias(rec, est, OUT / f"{name}_bias.png")
    print(f"\n  figure -> {fig}")

    return {"scenario": name, "seed": seed,
            "alignment_quality": align.quality,
            "baseline": s_base.to_dict(), "eskf": s_eskf.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="tunnel_drive",
                    choices=[*SCENARIOS, "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-logs", action="store_true",
                    help="write CSV recordings to out/logs/ for inspection")
    args = ap.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = [run_one(n, args.seed, args.save_logs) for n in names]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nmetrics -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
