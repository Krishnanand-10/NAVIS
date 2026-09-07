"""Evaluation metrics -- the numbers that go on the slide.

Judges reward specificity. Every figure here is defined so it can be quoted
without hedging, and every one is computable on synthetic data today and on
real logs the moment they exist.

The headline number for this problem statement is **drift as a percentage of
distance travelled during an outage**. Absolute error alone is unfair: 20 m
after 3 km is excellent, 20 m after 80 m is not.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def horizontal_error(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-sample horizontal position error, NaN where the estimate is absent."""
    return np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)


def path_length(pos: np.ndarray) -> float:
    """Total horizontal distance travelled along a trajectory."""
    d = np.diff(pos[:, :2], axis=0)
    return float(np.nansum(np.linalg.norm(d, axis=1)))


@dataclass
class OutageResult:
    label: str
    t_start: float
    t_end: float
    duration: float
    distance: float
    error_at_end: float
    max_error: float
    drift_pct: float
    reconverge_s: float | None


@dataclass
class Summary:
    ate_rmse: float           # RMSE of horizontal error over the whole run
    ate_p95: float
    max_error: float
    max_jump: float           # largest single-step position discontinuity, m
    outages: list

    def to_dict(self):
        d = asdict(self)
        d["outages"] = [asdict(o) if not isinstance(o, dict) else o
                        for o in self.outages]
        return d


def max_discontinuity(t: np.ndarray, pos: np.ndarray, max_speed=80.0) -> float:
    """Largest position jump that cannot be explained by motion.

    This is the "does the blue dot teleport?" metric. A system that snaps to a
    returning GNSS fix scores badly here even if its RMSE looks fine, which is
    exactly the distinction the word *seamless* in the problem statement is
    pointing at.
    """
    ok = np.isfinite(pos[:, 0])
    tt, pp = t[ok], pos[ok]
    if len(tt) < 2:
        return float("nan")
    dt = np.diff(tt)
    step = np.linalg.norm(np.diff(pp[:, :2], axis=0), axis=1)
    excess = step - max_speed * dt
    return float(np.max(np.maximum(excess, 0.0)))


def reconvergence_time(t, err, t_end, threshold=8.0, hold=3.0):
    """Seconds after an outage ends before error settles below ``threshold``."""
    after = t >= t_end
    if after.sum() < 2:
        return None            # outage runs to the end of the recording
    ta, ea = t[after], err[after]
    dt = np.median(np.diff(ta))
    if not np.isfinite(dt) or dt <= 0:
        return None
    good = ea < threshold
    n_hold = max(1, int(hold / dt))
    run = np.convolve(good.astype(int), np.ones(n_hold, dtype=int), mode="valid")
    hit = np.flatnonzero(run == n_hold)
    return float(ta[hit[0]] - t_end) if hit.size else None


def evaluate(t, est_pos, truth_pos, outages, labels=None) -> Summary:
    """Full metric bundle for one run.

    ``outages`` is a list of ``(t_start, t_end)`` pairs -- typically taken from
    the recording's own event list, or from :class:`~drnav.gnss_health.OutageTracker`.
    """
    err = horizontal_error(est_pos, truth_pos)
    valid = np.isfinite(err)

    results = []
    for i, (a, b) in enumerate(outages):
        m = (t >= a) & (t <= b) & valid
        if not m.any():
            continue
        seg_truth = truth_pos[m]
        e = err[m]
        dist = path_length(seg_truth)
        results.append(OutageResult(
            label=(labels[i] if labels and i < len(labels) else f"outage {i+1}"),
            t_start=float(a), t_end=float(b), duration=float(b - a),
            distance=dist,
            error_at_end=float(e[-1]),
            max_error=float(np.max(e)),
            drift_pct=float(100.0 * e[-1] / dist) if dist > 1 else float("nan"),
            reconverge_s=reconvergence_time(t[valid], err[valid], b),
        ))

    ev = err[valid]
    return Summary(
        ate_rmse=float(np.sqrt(np.mean(ev ** 2))) if ev.size else float("nan"),
        ate_p95=float(np.percentile(ev, 95)) if ev.size else float("nan"),
        max_error=float(np.max(ev)) if ev.size else float("nan"),
        max_jump=max_discontinuity(t, est_pos),
        outages=results,
    )


def format_summary(name: str, s: Summary) -> str:
    """Human-readable block for the console and the README."""
    lines = [f"{name}",
             f"  horizontal RMSE     {s.ate_rmse:8.2f} m",
             f"  95th percentile     {s.ate_p95:8.2f} m",
             f"  worst error         {s.max_error:8.2f} m",
             f"  max position jump   {s.max_jump:8.2f} m"]
    for o in s.outages:
        rc = "never" if o.reconverge_s is None else f"{o.reconverge_s:.1f}s"
        lines += [f"  - {o.label}  ({o.duration:.0f}s, {o.distance:.0f}m travelled)",
                  f"      error at exit {o.error_at_end:8.2f} m"
                  f"   worst {o.max_error:7.2f} m"
                  f"   drift {o.drift_pct:6.2f}%   reconverged {rc}"]
    return "\n".join(lines)
