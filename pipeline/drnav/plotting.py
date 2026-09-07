"""Figures for the deck.

The headline figure pairs the trajectory with the error-vs-time curve on a log
axis. The log axis matters: unaided strapdown and a properly aided filter differ
by two or three orders of magnitude, and a linear axis flattens the good curve
into the baseline of the plot where nobody can see it.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRUTH_C = "#111111"
GNSS_C = "#c44e52"
BASE_C = "#dd8452"
ESKF_C = "#2f7ec4"
OUTAGE_C = "#f2c14e"


def _shade(ax, windows, ymin=None, ymax=None, label="GNSS outage"):
    for i, (a, b) in enumerate(windows):
        ax.axvspan(a, b, color=OUTAGE_C, alpha=0.28, lw=0,
                   label=label if i == 0 else None)


def plot_run(rec, base, est, summ_base, summ_eskf, outages, title, path,
             base_label="classical strapdown DR", est_label="ESKF (this project)"):
    """Trajectory + error-over-time comparison, saved to ``path``.

    The two curve labels are parameters because this figure is reused for a
    second comparison -- ESKF with and without the learned speed head -- where
    calling the reference curve "classical strapdown" would be simply wrong.
    """
    truth = rec.truth_pos
    t = rec.t
    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[3, 1.5],
                          hspace=0.32, wspace=0.2)

    # ---- trajectory ----
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(truth[:, 0], truth[:, 1], color=TRUTH_C, lw=2.4, label="ground truth", zorder=5)
    ax.scatter(rec.gnss[:, 1], rec.gnss[:, 2], s=13, color=GNSS_C, alpha=0.65,
               label="GNSS fixes", zorder=4)
    ax.plot(base["pos"][:, 0], base["pos"][:, 1], color=BASE_C, lw=1.8, ls="--",
            label=base_label, zorder=6)
    ax.plot(est["pos"][:, 0], est["pos"][:, 1], color=ESKF_C, lw=2.2,
            label=est_label, zorder=7)

    for a, b in outages:
        m = (t >= a) & (t <= b)
        ax.plot(truth[m, 0], truth[m, 1], color=OUTAGE_C, lw=6.5, alpha=0.75,
                solid_capstyle="round", zorder=3)
    ax.plot([], [], color=OUTAGE_C, lw=6, alpha=0.75, label="GNSS unavailable")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.25)

    # ---- error over time ----
    ax2 = fig.add_subplot(gs[0, 1])
    eb = np.linalg.norm(base["pos"][:, :2] - truth[:, :2], axis=1)
    ee = np.linalg.norm(est["pos"][:, :2] - truth[:, :2], axis=1)
    ax2.semilogy(t, eb, color=BASE_C, lw=1.7, ls="--", label=base_label)
    ax2.semilogy(t, ee, color=ESKF_C, lw=2.0, label=est_label)
    _shade(ax2, outages)
    ax2.axhline(10, color="grey", lw=0.9, ls=":", zorder=1)
    ax2.text(t[-1], 10.5, "10 m", ha="right", va="bottom", fontsize=8, color="grey")
    ax2.set_ylabel("horizontal error (m)")
    ax2.set_title("Position error (log scale)", fontsize=11, loc="left")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.25, which="both")
    ax2.set_ylim(0.05, max(1e3, float(np.nanmax(eb)) * 1.5))

    # ---- dead-reckoning age, i.e. the UI badge ----
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.fill_between(t, 0, est["dr_age"], color=ESKF_C, alpha=0.35, lw=0)
    ax3.plot(t, est["dr_age"], color=ESKF_C, lw=1.4)
    _shade(ax3, outages, label=None)
    ax3.set_ylabel("DR age (s)")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Time since last usable fix", fontsize=10, loc="left")
    ax3.grid(alpha=0.25)

    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_bias(rec, est, path):
    """Bias observability check: are the IMU biases actually converging?"""
    t = rec.t
    b = est["bias"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for j, lbl in enumerate("xyz"):
        axes[0].plot(t, b[:, j], lw=1.3, label=f"accel bias {lbl}")
        axes[1].plot(t, np.degrees(b[:, 3 + j]), lw=1.3, label=f"gyro bias {lbl}")
    axes[0].set_ylabel("m/s$^2$")
    axes[1].set_ylabel("deg/s")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title("Estimated IMU biases", fontsize=11, loc="left")
    for a in axes:
        a.grid(alpha=0.25)
        a.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
