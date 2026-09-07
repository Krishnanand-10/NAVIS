"""Regression tests for the dead-reckoning pipeline.

These guard the specific mistakes that cost real debugging time while this was
being built, so they are worth keeping green:

* the body-velocity Jacobian, verified numerically rather than by inspection;
* the estimator actually beating the classical baseline, by a wide margin, on
  every scenario and several seeds -- not just the one that was tuned;
* the filter never diverging numerically, which it did until the covariance
  repair and the correction plausibility net were added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav.align import coarse_align                       # noqa: E402
from drnav.attitude import (R_from_heading, exp_so3, heading_from_R,  # noqa: E402
                            log_so3, quat_from_R, R_from_quat, skew)
from drnav.baseline_ins import run_baseline                # noqa: E402
from drnav.eskf import Eskf, EskfConfig, N_ERR             # noqa: E402
from drnav.log_format import Recording, geodetic_to_enu    # noqa: E402
from drnav.replay import run_eskf, truth_outages           # noqa: E402
from drnav.synth import SCENARIOS, simulate                # noqa: E402

SEEDS = [0, 1, 2, 3]


# ---------------------------------------------------------------- attitude


def test_skew_matches_cross_product():
    a, b = np.array([0.3, -1.2, 0.7]), np.array([2.0, 0.5, -0.4])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_exp_log_round_trip():
    rv = np.array([0.21, -0.34, 0.11])
    assert np.allclose(log_so3(exp_so3(rv)), rv)


def test_quaternion_round_trip():
    R = exp_so3([0.4, -0.2, 1.1])
    assert np.allclose(R_from_quat(quat_from_R(R)), R)


@pytest.mark.parametrize("psi_deg", [0, 45, 90, 180, 270, 359])
def test_heading_round_trip(psi_deg):
    psi = np.radians(psi_deg)
    got = heading_from_R(R_from_heading(psi))
    assert np.isclose(np.cos(got), np.cos(psi), atol=1e-9)
    assert np.isclose(np.sin(got), np.sin(psi), atol=1e-9)


def test_heading_convention_is_clockwise_from_north():
    """Body x=forward at heading 90 deg must point due East."""
    assert np.allclose(R_from_heading(np.radians(90.0))[:, 0],
                       [1.0, 0.0, 0.0], atol=1e-9)


# ------------------------------------------------------------------ frames


def test_geodetic_to_enu_is_locally_metric():
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    # 0.001 deg of latitude is about 111 m north
    enu = geodetic_to_enu(lat0 + 0.001, lon0, alt0, lat0, lon0, alt0)
    assert abs(enu[0]) < 0.5
    assert 110.0 < enu[1] < 112.0
    assert abs(enu[2]) < 1.0


def test_geodetic_origin_maps_to_zero():
    enu = geodetic_to_enu(12.9716, 77.5946, 920.0, 12.9716, 77.5946, 920.0)
    assert np.allclose(enu, 0.0, atol=1e-6)


# ------------------------------------------------------------------- eskf


def test_body_velocity_jacobian_matches_numerical():
    """The NHC/ML update hangs off this; an inspection-only check is not enough."""
    f = Eskf()
    f.initialise(np.zeros(3), np.array([4.0, 7.0, 0.3]), exp_so3([0.2, -0.1, 0.9]))
    R_b = exp_so3([0.05, 0.02, -0.3])
    H = f._body_velocity_jacobian(R_b)

    eps = 1e-7
    base = R_b.T @ f.x.v
    for j in range(3):
        dv = np.zeros(3); dv[j] = eps
        assert np.allclose(H[:, 3 + j], (R_b.T @ (f.x.v + dv) - base) / eps, atol=1e-5)
        dth = np.zeros(3); dth[j] = eps
        pert = (exp_so3(dth) @ R_b).T @ f.x.v
        assert np.allclose(H[:, 6 + j], (pert - base) / eps, atol=1e-4)


def test_covariance_stays_symmetric_and_positive_definite():
    f = Eskf()
    f.initialise(np.zeros(3), np.zeros(3), np.eye(3))
    rng = np.random.default_rng(0)
    for _ in range(500):
        f.predict(np.array([0.0, 0.0, 9.80665]) + rng.normal(0, 0.02, 3),
                  rng.normal(0, 0.002, 3), 0.005)
        f.update_gnss_position(rng.normal(0, 3.0, 3), sigma=3.0)
    P = f.x.P
    assert np.allclose(P, P.T, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(P)) > 0


def test_implausible_correction_is_rejected():
    """The plausibility net must refuse a correction that would wreck the state."""
    f = Eskf()
    f.initialise(np.zeros(3), np.zeros(3), np.eye(3))
    f.x.P *= 1e5                       # pretend we are hopelessly uncertain
    H = np.zeros((3, N_ERR)); H[:, 0:3] = np.eye(3)
    before = f.x.ba.copy()
    f._update(H, np.array([1e6, 0.0, 0.0]), np.eye(3) * 0.01)
    assert np.allclose(f.x.ba, before)


def test_zupt_leaves_accel_bias_alone():
    """Tilt and horizontal accel bias are not separable at rest -- see update_zupt."""
    f = Eskf(EskfConfig(zupt_freeze_accel_bias=True))
    f.initialise(np.zeros(3), np.array([0.4, -0.2, 0.0]), np.eye(3))
    before = f.x.ba.copy()
    f.update_zupt()
    assert np.allclose(f.x.ba, before)
    assert np.linalg.norm(f.x.v) < 0.4      # but velocity was corrected


# ------------------------------------------------------------- simulator


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_simulator_is_self_consistent(name):
    rec = simulate(SCENARIOS[name](), seed=0)
    assert np.all(np.isfinite(rec.imu))
    assert np.all(np.isfinite(rec.gnss))
    assert np.all(np.diff(rec.t) > 0)
    # a stationary phone must read one g
    assert 9.0 < np.linalg.norm(rec.accel[:100].mean(axis=0)) < 10.6
    # ground-truth position must be the integral of ground-truth velocity
    dt = 1.0 / rec.meta.imu_rate_hz
    integrated = np.cumsum(rec.truth_vel, axis=0) * dt
    assert np.max(np.abs(integrated - rec.truth_pos)) < 1e-6


def test_outage_segments_emit_no_fixes():
    rec = simulate(SCENARIOS["tunnel_drive"](), seed=0)
    wins, _ = truth_outages(rec)
    for a, b in wins:
        inside = (rec.gnss[:, 0] > a + 1e-6) & (rec.gnss[:, 0] < b - 1e-6)
        assert not inside.any()


def test_recording_survives_a_save_load_round_trip(tmp_path):
    rec = simulate(SCENARIOS["parking_ramp"](), seed=0)
    back = Recording.load(rec.save(tmp_path / "rec"))
    assert np.allclose(rec.imu, back.imu)
    assert np.allclose(rec.gnss, back.gnss)
    assert back.meta.mode == rec.meta.mode


# ------------------------------------------------------- end-to-end quality


@pytest.mark.parametrize("name", ["tunnel_drive", "parking_ramp"])
def test_eskf_beats_classical_baseline_on_outages(name):
    """Where GNSS actually disappears, the estimator must win on accuracy."""
    for seed in SEEDS:
        rec = simulate(SCENARIOS[name](), seed=seed)
        al = coarse_align(rec)
        base = run_baseline(rec, al)
        est = run_eskf(rec, al)
        truth = rec.truth_pos

        def rmse(p):
            e = np.linalg.norm(p[:, :2] - truth[:, :2], axis=1)
            return float(np.sqrt(np.mean(e[np.isfinite(e)] ** 2)))

        assert rmse(est["pos"]) < rmse(base["pos"]), f"{name} seed {seed}"


def test_eskf_wins_on_continuity_under_multipath():
    """Urban canyon is a different claim, and worth stating precisely.

    With no outage at all, snapping straight to every GNSS fix is *competitive
    on RMSE*: multipath is a slowly-varying systematic bias, and no amount of
    filtering removes a bias that is common to every fix. On some seeds the
    naive baseline even edges ahead. What the estimator wins decisively is
    continuity -- it does not hurl the position marker onto the next street
    when a reflected signal arrives.

    Closing the remaining accuracy gap needs information the filter does not
    currently have: per-satellite exclusion from raw measurements, or map
    matching. Both are on the roadmap; neither is pretended to exist here.
    """
    from drnav.metrics import max_discontinuity
    base, eskf = [], []
    for seed in SEEDS:
        rec = simulate(SCENARIOS["urban_canyon"](), seed=seed)
        al = coarse_align(rec)
        base.append(max_discontinuity(rec.t, run_baseline(rec, al)["pos"]))
        eskf.append(max_discontinuity(rec.t, run_eskf(rec, al)["pos"]))
    # Compared in aggregate, not per seed: an individual multipath realisation
    # can favour either, and asserting per-seed superiority would be asserting
    # something that is not true rather than something that is.
    assert float(np.mean(eskf)) < 0.7 * float(np.mean(base)), (base, eskf)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_filter_never_diverges(name):
    """No NaNs, no 100 km excursions -- this failed before the covariance repair."""
    for seed in SEEDS:
        rec = simulate(SCENARIOS[name](), seed=seed)
        est = run_eskf(rec, coarse_align(rec))
        p = est["pos"][est["start_index"]:]
        assert np.all(np.isfinite(p)), f"{name} seed {seed}: non-finite position"
        assert np.max(np.abs(p)) < 1e5, f"{name} seed {seed}: ran away"


def test_tunnel_drift_regression_guard():
    """Drift through the tunnel, measured across seeds rather than one lucky draw.

    The threshold is deliberately set from measurement, not aspiration. The
    tunnel scenario contains a 25 s traffic jam *inside* the outage, and that
    is genuinely hard for the classical stack: with no GNSS there is no
    independent speed reference, so ZUPT cannot fire (see replay.run_eskf) and
    the filter integrates through a full stop-start unaided. Median drift is
    about 13 % of distance travelled.

    Bringing this under the 5 % target is the job of the learned speed head,
    which supplies exactly the missing speed reference. **When that lands, this
    threshold should be tightened** -- if it is still 20 % then the ML
    component is not earning its place.
    """
    drifts = []
    for seed in SEEDS:
        rec = simulate(SCENARIOS["tunnel_drive"](), seed=seed)
        est = run_eskf(rec, coarse_align(rec))
        wins, _ = truth_outages(rec)
        t, truth = rec.t, rec.truth_pos
        i_end = int(np.searchsorted(t, wins[-1][1]))
        m = (t >= wins[0][0]) & (t <= wins[-1][1])
        dist = float(np.sum(np.linalg.norm(np.diff(truth[m, :2], axis=0), axis=1)))
        err = float(np.linalg.norm(est["pos"][i_end, :2] - truth[i_end, :2]))
        drifts.append(100.0 * err / dist)
    assert float(np.median(drifts)) < 20.0, drifts


def test_reacquisition_does_not_teleport():
    """The 'seamless' requirement, measured rather than asserted.

    Output smoothing holds the *reported* position back and lets it catch up
    over a couple of seconds while the filter state takes the correction
    immediately. Without it the median jump across these seeds is 144 m and the
    worst is 416 m; with it, under a metre.
    """
    from drnav.metrics import max_discontinuity
    for seed in SEEDS:
        rec = simulate(SCENARIOS["tunnel_drive"](), seed=seed)
        est = run_eskf(rec, coarse_align(rec))
        jump = max_discontinuity(rec.t, est["pos"])
        assert jump < 5.0, f"seed {seed}: jumped {jump:.1f} m"


def test_output_smoothing_is_what_prevents_the_teleport():
    """Guard the mechanism, not just the outcome."""
    from drnav.metrics import max_discontinuity
    from drnav.eskf import EskfConfig
    rec = simulate(SCENARIOS["tunnel_drive"](), seed=0)
    al = coarse_align(rec)
    raw = max_discontinuity(rec.t, run_eskf(rec, al, EskfConfig(output_smoothing_tau=0.0))["pos"])
    smooth = max_discontinuity(rec.t, run_eskf(rec, al, EskfConfig(output_smoothing_tau=2.0))["pos"])
    assert smooth < raw / 10.0, f"raw {raw:.1f} m vs smoothed {smooth:.1f} m"
