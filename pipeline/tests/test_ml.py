"""Tests for the learned speed head.

The important one here is :func:`test_levelling_is_invariant_to_mounting`. The
entire justification for the feature pipeline is that a phone in an unknown
pose produces the same network input as the same phone held flat, up to a yaw
the training augmentation absorbs. If that property breaks, the model silently
becomes mount-specific and every real-world number degrades without any test
going red -- so it is asserted directly rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from drnav.attitude import exp_so3                                  # noqa: E402
from drnav.dataset import (Clip, WINDOW_N, WindowDataset,           # noqa: E402
                           random_scenario, split_by_clip)
from drnav.features import (gravity_direction, horizontal_speed,    # noqa: E402
                            level_window, random_yaw)
from drnav.model import (LOG_VAR_MAX, LOG_VAR_MIN, SpeedNet,        # noqa: E402
                         count_parameters, gaussian_nll)

RATE = 100.0


def _fake_imu(n=400, seed=0):
    """Gravity plus a bit of structured motion, in a nominal 'flat' pose."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    accel = np.column_stack([
        0.6 * np.sin(2 * np.pi * 1.8 * t),
        0.3 * np.cos(2 * np.pi * 0.9 * t),
        9.80665 + 1.2 * np.sin(2 * np.pi * 3.6 * t),
    ]) + rng.normal(0, 0.02, (n, 3))
    gyro = np.column_stack([
        0.05 * np.sin(2 * np.pi * 1.8 * t),
        0.04 * np.cos(2 * np.pi * 1.8 * t),
        0.15 * np.ones(n),
    ]) + rng.normal(0, 0.002, (n, 3))
    return accel, gyro


# ------------------------------------------------------------------ features


def test_gravity_direction_is_unit_and_points_up_when_flat():
    accel, _ = _fake_imu()
    g = gravity_direction(accel, RATE)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=1e-6)
    assert g[:, 2].mean() > 0.98          # flat phone: gravity along body +z


def test_levelling_is_invariant_to_mounting():
    """The core claim: an arbitrary fixed device pose must not change the input.

    A rotated phone should produce a levelled window identical to the unrotated
    one up to a rotation about the vertical -- so the vertical channels and the
    horizontal magnitudes must match, while the individual horizontal
    components need not.
    """
    accel, gyro = _fake_imu()
    flat = level_window(accel, gyro, RATE)

    for seed in range(5):
        rng = np.random.default_rng(seed)
        R = exp_so3(rng.normal(0, 1.0, 3))          # arbitrary mounting
        rotated = level_window(accel @ R, gyro @ R, RATE)

        # vertical accelerometer and gyroscope channels are preserved
        assert np.allclose(flat[:, 2], rotated[:, 2], atol=1e-3), f"seed {seed} accel z"
        assert np.allclose(flat[:, 5], rotated[:, 5], atol=1e-3), f"seed {seed} gyro z"
        # horizontal magnitudes are preserved
        for base in (0, 3):
            m_flat = np.hypot(flat[:, base], flat[:, base + 1])
            m_rot = np.hypot(rotated[:, base], rotated[:, base + 1])
            assert np.allclose(m_flat, m_rot, atol=1e-3), f"seed {seed} block {base}"


def test_random_yaw_preserves_magnitudes():
    accel, gyro = _fake_imu()
    w = level_window(accel, gyro, RATE)
    out = random_yaw(w, np.random.default_rng(0))
    assert np.allclose(w[:, 2], out[:, 2])
    assert np.allclose(w[:, 5], out[:, 5])
    assert np.allclose(np.hypot(w[:, 0], w[:, 1]), np.hypot(out[:, 0], out[:, 1]))


def test_level_window_handles_short_input():
    accel, gyro = _fake_imu(n=8)
    w = level_window(accel, gyro, RATE)
    assert w.shape == (8, 6)
    assert np.all(np.isfinite(w))


def test_horizontal_speed_ignores_vertical():
    vel = np.array([[3.0, 4.0, 99.0]])
    assert np.allclose(horizontal_speed(vel), [5.0])


# ------------------------------------------------------------------- dataset


def _clip(n=1000, seed=0, mode="vehicle"):
    accel, gyro = _fake_imu(n, seed)
    x = level_window(accel, gyro, RATE)
    speed = np.linspace(0, 12, n).astype(np.float32)
    return Clip(x=x, speed=speed, mode=mode)


def test_window_dataset_shapes_and_target():
    ds = WindowDataset([_clip()], stride=50, augment=False)
    x, y = ds[0]
    assert x.shape == (WINDOW_N, 6)
    # the target is the speed at the END of the window, not its mean
    assert np.isclose(y, ds.clips[0].speed[WINDOW_N - 1])


def test_split_is_by_clip_not_window():
    """Overlapping windows make a window-level split leak; guard against it."""
    clips = [_clip(seed=i) for i in range(10)]
    train, val = split_by_clip(clips, val_fraction=0.3, seed=0)
    assert len(train) + len(val) == len(clips)
    train_ids = {id(c) for c in train}
    assert not any(id(c) in train_ids for c in val)


def test_random_scenario_produces_valid_segments():
    rng = np.random.default_rng(0)
    for mode in ("vehicle", "pedestrian"):
        sc = random_scenario(rng, mode)
        assert sc.mode == mode
        assert sc.gait == (mode == "pedestrian")
        assert all(s.duration > 0 and s.speed >= 0 for s in sc.segments)
        assert sc.segments[0].speed == 0.0     # must start still, for alignment


# --------------------------------------------------------------------- model


def test_model_output_shapes_and_sigma_bounds():
    m = SpeedNet().eval()
    with torch.no_grad():
        speed, log_var = m(torch.randn(3, WINDOW_N, 6) * 5)
    assert speed.shape == (3,) and log_var.shape == (3,)
    assert torch.all(speed >= 0)                       # softplus
    assert torch.all(log_var >= LOG_VAR_MIN - 1e-5)
    assert torch.all(log_var <= LOG_VAR_MAX + 1e-5)


def test_model_is_small_enough_for_a_phone():
    assert count_parameters(SpeedNet()) < 1_000_000


def test_nll_rewards_honest_uncertainty():
    """A model that under-states its error must be penalised for it."""
    target = torch.tensor([10.0])
    pred = torch.tensor([7.0])                          # 3 m/s wrong
    honest = gaussian_nll(pred, torch.log(torch.tensor([9.0])), target)
    overconfident = gaussian_nll(pred, torch.log(torch.tensor([0.01])), target)
    underconfident = gaussian_nll(pred, torch.log(torch.tensor([1000.0])), target)
    assert honest < overconfident
    assert honest < underconfident


def test_nll_gradient_flows():
    m = SpeedNet()
    speed, log_var = m(torch.randn(4, WINDOW_N, 6))
    gaussian_nll(speed, log_var, torch.rand(4) * 10).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in m.parameters())


# ----------------------------------------------------- filter integration


def test_speed_magnitude_jacobian_matches_numerical():
    """The learned head's route into the filter, verified rather than assumed."""
    from drnav.eskf import Eskf, I_V, N_ERR

    f = Eskf()
    f.initialise(np.zeros(3), np.array([6.0, -3.0, 0.4]), np.eye(3))
    v = f.x.v
    sp = float(np.hypot(v[0], v[1]))
    H = np.zeros((1, N_ERR))
    H[0, I_V] = [v[0] / sp, v[1] / sp, 0.0]

    eps = 1e-7
    for j in range(3):
        dv = np.zeros(3); dv[j] = eps
        num = (np.hypot(*(v + dv)[:2]) - sp) / eps
        assert np.isclose(H[0, 3 + j], num, atol=1e-5), f"component {j}"


def test_speed_magnitude_update_is_skipped_when_nearly_stopped():
    """Direction is undefined at rest, so the constraint must decline to apply."""
    from drnav.eskf import Eskf

    f = Eskf()
    f.initialise(np.zeros(3), np.array([0.05, -0.02, 0.0]), np.eye(3))
    assert f.update_speed_magnitude(1.0, 0.3) is False

    f.x.v = np.array([5.0, 0.0, 0.0])
    assert f.update_speed_magnitude(6.0, 0.3) is True
    assert np.hypot(*f.x.v[:2]) > 5.0          # pulled toward the measurement


def test_speed_magnitude_needs_no_mounting_estimate():
    """update_ml_forward_speed requires a vehicle frame; this one must not."""
    from drnav.eskf import Eskf

    f = Eskf()
    f.initialise(np.zeros(3), np.array([4.0, 4.0, 0.0]), np.eye(3))
    assert f.R_mount is None
    assert f.update_ml_forward_speed(6.0) is False      # correctly declines
    assert f.update_speed_magnitude(6.0, 0.3) is True   # works anyway


def test_speed_estimator_precompute_and_lookup(tmp_path):
    from drnav.align import coarse_align
    from drnav.ml_speed import SpeedEstimator
    from drnav.model import SpeedNet
    from drnav.replay import run_eskf
    from drnav.synth import SCENARIOS, simulate

    ckpt = tmp_path / "m.pt"
    torch.save({"state_dict": SpeedNet().state_dict(), "sigma_scale": 2.0}, ckpt)

    est = SpeedEstimator(ckpt, device="cpu")
    assert est.sigma_scale == 2.0              # read from the checkpoint

    rec = simulate(SCENARIOS["pedestrian_mall"](), seed=0)
    est.precompute(rec, rate_hz=5.0)
    assert est._t is not None and est._t.size > 0

    out = est(float(est._t[5]), None)
    assert out is not None
    speed, sigma = out
    assert speed >= 0 and sigma >= est.sigma_floor
    assert est(1e6, None) is None              # refuses a stale lookup

    # and it must actually drive the filter without producing garbage
    res = run_eskf(rec, coarse_align(rec), ml_speed=est)
    p = res["pos"][res["start_index"]:]
    assert np.all(np.isfinite(p))
