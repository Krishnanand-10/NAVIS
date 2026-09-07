"""Synthetic trajectory and sensor simulator.

Why this exists: the whole fusion stack can be built and validated *before* a
single phone recording exists. We generate a ground-truth trajectory
analytically, then work backwards to the raw sensor signals a phone strapped
into that trajectory would have produced -- complete with MEMS bias, noise,
an arbitrary mounting rotation, and GNSS that drops out in a tunnel.

Because the ground truth is known exactly, every metric in :mod:`drnav.metrics`
is computable on day one. When real logs arrive they slot into the same
:class:`~drnav.log_format.Recording` container and nothing downstream changes.

Sensor error magnitudes below are representative of a mid-range phone IMU
(roughly a Bosch BMI-series part): ~10 mg accelerometer bias and ~0.5 deg/s
gyro bias, which is what produces the "180 m of drift in 60 seconds" figure
that motivates the entire project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from scipy.signal import butter, sosfiltfilt

from .attitude import G_VEC_ENU
from .log_format import Meta, Recording

# --------------------------------------------------------------------------
# sensor error model
# --------------------------------------------------------------------------


@dataclass
class ImuErrorModel:
    """Phone-grade MEMS error parameters."""

    accel_bias_sigma: float = 0.08      # m/s^2, constant turn-on bias (~8 mg)
    accel_noise_density: float = 0.0015  # m/s^2/sqrt(Hz)
    accel_bias_walk: float = 0.0004     # m/s^2/sqrt(s)
    gyro_bias_sigma: float = 0.010      # rad/s (~0.57 deg/s)
    gyro_noise_density: float = 1.8e-4  # rad/s/sqrt(Hz)
    gyro_bias_walk: float = 2.0e-5      # rad/s/sqrt(s)

    # Road-induced vibration. This is not a nuisance term to be tuned away --
    # it is the *only* thing in a vehicle's IMU that carries absolute speed.
    # Kinematically, cruising at 30 km/h and at 90 km/h are indistinguishable:
    # zero acceleration either way. What differs is how hard the tyres and
    # suspension are being excited by the road surface. Omit it and a learned
    # speed head has nothing to learn from, which is exactly what happened on
    # the first training run -- the model converged to predicting the dataset
    # mean with maximum uncertainty, correctly reporting that the data held no
    # answer. Amplitude grows roughly as v^0.8 for constant road roughness.
    vib_rms_ref: float = 0.55           # m/s^2 vertical RMS at 60 km/h
    vib_gyro_rms_ref: float = 0.025     # rad/s RMS at 60 km/h
    vib_idle: float = 0.06              # m/s^2 engine vibration at a standstill
    vib_band: tuple = (6.0, 40.0)       # Hz, tyre/suspension excitation


@dataclass(frozen=True)
class GnssQuality:
    """GNSS behaviour for one stretch of the route."""

    pos_sigma: float = 3.0        # m, horizontal 1-sigma
    vel_sigma: float = 0.10       # m/s
    nsat: int = 14
    cn0: float = 38.0             # dB-Hz
    hdop: float = 0.9
    multipath_bias: float = 0.0   # m, slowly-varying systematic offset


OPEN_SKY = GnssQuality()
URBAN_CANYON = GnssQuality(pos_sigma=12.0, vel_sigma=0.6, nsat=6, cn0=24.0,
                           hdop=2.8, multipath_bias=22.0)
FOLIAGE = GnssQuality(pos_sigma=7.0, vel_sigma=0.35, nsat=8, cn0=29.0,
                      hdop=1.9, multipath_bias=6.0)


# --------------------------------------------------------------------------
# route description
# --------------------------------------------------------------------------


@dataclass
class Segment:
    """One stretch of driving/walking.

    ``turn_rate`` is degrees per second (positive = turning right, i.e. heading
    increasing clockwise from North). ``gnss`` is either a
    :class:`GnssQuality` or ``None`` to mean a total outage.
    """

    duration: float
    speed: float                      # m/s target speed
    turn_rate: float = 0.0            # deg/s
    gnss: GnssQuality | None = OPEN_SKY
    grade: float = 0.0                # road grade, degrees (positive = uphill)
    label: str = ""


@dataclass
class Scenario:
    name: str
    segments: list[Segment]
    mode: str = "vehicle"
    accel_limit: float = 2.5          # m/s^2, how fast speed can change
    gait: bool = False                # add walking bob to the IMU
    imu_rate: float = 200.0
    gnss_rate: float = 1.0
    imu_errors: ImuErrorModel = field(default_factory=ImuErrorModel)


# --------------------------------------------------------------------------
# the simulator
# --------------------------------------------------------------------------


def simulate(scenario: Scenario, seed: int = 0) -> Recording:
    """Turn a :class:`Scenario` into a full :class:`Recording`."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / scenario.imu_rate

    # ---- 1. build the commanded profile at IMU rate ----------------------
    total = sum(s.duration for s in scenario.segments)
    n = int(round(total / dt))
    t = np.arange(n) * dt

    target_speed = np.empty(n)
    turn_rate = np.empty(n)
    grade = np.empty(n)
    seg_index = np.empty(n, dtype=int)

    cursor = 0
    for i, seg in enumerate(scenario.segments):
        k = int(round(seg.duration / dt))
        sl = slice(cursor, min(cursor + k, n))
        target_speed[sl] = seg.speed
        turn_rate[sl] = np.radians(seg.turn_rate)
        grade[sl] = np.radians(seg.grade)
        seg_index[sl] = i
        cursor += k
    if cursor < n:  # rounding slack -> extend the final segment
        target_speed[cursor:] = scenario.segments[-1].speed
        turn_rate[cursor:] = np.radians(scenario.segments[-1].turn_rate)
        grade[cursor:] = np.radians(scenario.segments[-1].grade)
        seg_index[cursor:] = len(scenario.segments) - 1

    # rate-limit the speed so longitudinal acceleration stays physical
    speed = np.empty(n)
    v = float(target_speed[0])
    max_dv = scenario.accel_limit * dt
    for i in range(n):
        v += np.clip(target_speed[i] - v, -max_dv, max_dv)
        speed[i] = v

    # heading integrates the turn rate; only turn when actually moving
    heading = np.cumsum(np.where(speed > 0.2, turn_rate, 0.0)) * dt

    # ---- 2. ground-truth kinematics in ENU -------------------------------
    horiz = speed * np.cos(grade)
    vel = np.column_stack([
        horiz * np.sin(heading),   # East
        horiz * np.cos(heading),   # North
        speed * np.sin(grade),     # Up
    ])
    pos = np.cumsum(vel, axis=0) * dt

    # ---- 3. ground-truth attitude ----------------------------------------
    # Vehicle body: x=forward, y=left, z=up. Bank into turns proportionally to
    # lateral acceleration, and pitch with the road grade.
    lat_acc = speed * turn_rate
    roll = -np.clip(lat_acc / 9.81, -0.25, 0.25) * 0.6   # gentle, realistic bank
    pitch = grade
    if scenario.gait:
        # A walker bobs and sways; this is the cadence signal an ML velocity
        # regressor keys on, so it must be present in the synthetic data.
        f_step = 1.8 + 0.25 * (speed / 1.4)
        phase = 2 * np.pi * np.cumsum(f_step) * dt
        roll = roll + np.radians(3.0) * np.sin(phase) * (speed > 0.2)
        pitch = pitch + np.radians(2.0) * np.sin(2 * phase) * (speed > 0.2)

    R_veh = _heading_rpy_to_matrix(heading, roll, pitch)

    # The phone sits at an arbitrary but fixed rotation inside the vehicle --
    # in a pocket, a cupholder, a windscreen cradle. Orientation-agnosticism is
    # a hard requirement, so we bake a random mounting into every run.
    mount = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    R_dev = R_veh @ mount

    # ---- 4. work backwards to ideal sensor signals -----------------------
    acc_nav = np.gradient(vel, dt, axis=0)
    if scenario.gait:
        # vertical bounce of the torso, which dominates a pedestrian's accel
        f_step = 1.8 + 0.25 * (speed / 1.4)
        phase = 2 * np.pi * np.cumsum(f_step) * dt
        bob = 1.6 * np.sin(2 * phase) * np.clip(speed / 1.4, 0, 1.5)
        acc_nav = acc_nav + np.column_stack([np.zeros(n), np.zeros(n), bob])

    # specific force in the device frame
    f_body = np.einsum("nji,nj->ni", R_dev, acc_nav - G_VEC_ENU)
    omega_body = _angular_rates(R_dev, dt)

    # ---- 5. corrupt them ---------------------------------------------------
    e = scenario.imu_errors
    ba = rng.normal(0, e.accel_bias_sigma, 3)
    bg = rng.normal(0, e.gyro_bias_sigma, 3)
    ba_walk = np.cumsum(rng.normal(0, e.accel_bias_walk * np.sqrt(dt), (n, 3)), axis=0)
    bg_walk = np.cumsum(rng.normal(0, e.gyro_bias_walk * np.sqrt(dt), (n, 3)), axis=0)
    sigma_a = e.accel_noise_density * np.sqrt(scenario.imu_rate)
    sigma_g = e.gyro_noise_density * np.sqrt(scenario.imu_rate)

    accel_meas = f_body + ba + ba_walk + rng.normal(0, sigma_a, (n, 3))
    gyro_meas = omega_body + bg + bg_walk + rng.normal(0, sigma_g, (n, 3))

    if not scenario.gait:
        # Vehicle: add speed-dependent road vibration in the VEHICLE frame,
        # then rotate it into the device frame. It rides on the chassis, so it
        # is a body-frame effect and must not touch the true trajectory.
        vib_a, vib_w = _road_vibration(speed, scenario.imu_rate, e, rng)
        accel_meas = accel_meas + vib_a @ mount
        gyro_meas = gyro_meas + vib_w @ mount

    imu = np.column_stack([t, accel_meas, gyro_meas])
    truth = np.column_stack([t, pos, vel, heading])

    # ---- 6. GNSS ----------------------------------------------------------
    gnss = _simulate_gnss(scenario, t, pos, vel, seg_index, rng)

    events = []
    for i, seg in enumerate(scenario.segments):
        idx = np.flatnonzero(seg_index == i)
        if idx.size == 0:
            continue
        events.append({
            "name": seg.label or f"segment {i}",
            "t_start": float(t[idx[0]]),
            "t_end": float(t[idx[-1]]),
            "kind": "outage" if seg.gnss is None else (
                "degraded" if seg.gnss.pos_sigma > 5 else "open"),
        })

    meta = Meta(
        mode=scenario.mode,
        source="synthetic",
        imu_rate_hz=scenario.imu_rate,
        gnss_rate_hz=scenario.gnss_rate,
        device=f"simulator:{scenario.name}",
        notes=f"seed={seed}",
        # ground-truth device->vehicle mounting, for validating the estimator
        mount=mount.tolist(),
        events=events,
    )
    return Recording(meta=meta, imu=imu, gnss=gnss, truth=truth)


def _road_vibration(speed, rate, errors: ImuErrorModel, rng):
    """Band-limited vibration whose amplitude tracks speed.

    White noise shaped into the 6-40 Hz tyre/suspension band, then scaled by a
    per-sample envelope that grows as ``v**0.8``. The vertical axis dominates;
    the horizontal axes get roughly half, which is what an accelerometer bolted
    to a car body actually sees.
    """
    n = len(speed)
    lo, hi = errors.vib_band
    nyq = 0.5 * rate
    sos = butter(2, [lo / nyq, min(hi / nyq, 0.98)], btype="band", output="sos")

    envelope = errors.vib_idle + errors.vib_rms_ref * (speed / 16.67) ** 0.8
    axis_gain = np.array([0.5, 0.5, 1.0])          # vertical dominates

    raw_a = rng.normal(0, 1.0, (n, 3))
    shaped_a = sosfiltfilt(sos, raw_a, axis=0)
    # sosfiltfilt changes the variance, so renormalise before scaling
    shaped_a /= np.std(shaped_a, axis=0, keepdims=True) + 1e-9
    vib_a = shaped_a * envelope[:, None] * axis_gain

    w_env = errors.vib_idle * 0.1 + errors.vib_gyro_rms_ref * (speed / 16.67) ** 0.8
    shaped_w = sosfiltfilt(sos, rng.normal(0, 1.0, (n, 3)), axis=0)
    shaped_w /= np.std(shaped_w, axis=0, keepdims=True) + 1e-9
    vib_w = shaped_w * w_env[:, None]

    return vib_a, vib_w


def _heading_rpy_to_matrix(heading, roll, pitch):
    """Stacked body->nav matrices from heading (CW from North) plus roll/pitch."""
    n = heading.size
    c, s = np.cos(heading), np.sin(heading)
    R_h = np.zeros((n, 3, 3))
    R_h[:, 0, 0], R_h[:, 1, 0] = s, c            # forward axis in ENU
    R_h[:, 0, 1], R_h[:, 1, 1] = -c, s           # left axis
    R_h[:, 2, 2] = 1.0                           # up axis
    R_rp = Rotation.from_euler("xy", np.column_stack([roll, pitch])).as_matrix()
    return R_h @ R_rp


def _angular_rates(R, dt):
    """Body-frame angular velocity from a stack of attitude matrices."""
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])   # R_prev.T @ R_next
    rv = Rotation.from_matrix(rel).as_rotvec() / dt
    return np.vstack([rv, rv[-1]])                  # repeat last to keep length


def _simulate_gnss(scenario, t, pos, vel, seg_index, rng):
    """Emit fixes at the GNSS rate, skipping outage stretches entirely."""
    step = int(round(scenario.imu_rate / scenario.gnss_rate))
    rows = []
    # multipath is systematic, not white: model it as a slow random walk that is
    # re-drawn whenever we enter a new stretch of sky.
    mp = np.zeros(3)
    prev_seg = -1
    for i in range(0, len(t), step):
        seg = scenario.segments[seg_index[i]]
        q = seg.gnss
        if q is None:
            prev_seg = seg_index[i]
            continue                      # tunnel: no fix at all
        if seg_index[i] != prev_seg:
            mp = rng.normal(0, q.multipath_bias, 3) * np.array([1, 1, 0.5])
            prev_seg = seg_index[i]
        else:
            mp = mp + rng.normal(0, q.multipath_bias * 0.05, 3)

        p_meas = pos[i] + mp + rng.normal(0, q.pos_sigma, 3) * np.array([1, 1, 1.6])
        v_meas = vel[i] + rng.normal(0, q.vel_sigma, 3)
        rows.append([
            t[i], *p_meas, *v_meas,
            q.nsat + rng.integers(-1, 2),
            q.cn0 + rng.normal(0, 1.5),
            q.hdop * rng.uniform(0.9, 1.2),
            q.pos_sigma * rng.uniform(0.8, 1.3),
        ])
    return np.array(rows) if rows else np.zeros((0, 11))


# --------------------------------------------------------------------------
# preset scenarios
# --------------------------------------------------------------------------


def tunnel_drive() -> Scenario:
    """The headline demo: a 95-second tunnel with a curve inside it.

    A curve matters -- a straight tunnel is trivially handled by "keep going",
    so any honest evaluation must bend while the sky is gone.
    """
    kmh = 1 / 3.6
    return Scenario(
        name="tunnel_drive",
        segments=[
            Segment(20, 0, 0, OPEN_SKY, label="stationary alignment"),
            Segment(40, 50 * kmh, 0, OPEN_SKY, label="open road"),
            Segment(25, 50 * kmh, 4.0, OPEN_SKY, label="right bend"),
            Segment(15, 0, 0, OPEN_SKY, label="traffic light (ZUPT)"),
            Segment(30, 60 * kmh, 0, OPEN_SKY, label="approach"),
            Segment(35, 60 * kmh, 0, None, grade=-2.0, label="TUNNEL: entry ramp"),
            Segment(30, 60 * kmh, -3.5, None, label="TUNNEL: left curve"),
            # A stop with no sky is precisely where a zero-velocity update
            # earns its keep: nothing else can arrest the drift.
            Segment(25, 0, 0, None, label="TUNNEL: traffic jam"),
            Segment(30, 55 * kmh, 0, None, grade=2.0, label="TUNNEL: exit ramp"),
            Segment(60, 55 * kmh, 1.5, OPEN_SKY, label="reacquisition"),
        ],
    )


def urban_canyon() -> Scenario:
    """Multipath, not outage: GNSS keeps reporting, but it is lying."""
    kmh = 1 / 3.6
    return Scenario(
        name="urban_canyon",
        segments=[
            Segment(20, 0, 0, OPEN_SKY, label="stationary alignment"),
            Segment(40, 40 * kmh, 0, OPEN_SKY, label="open road"),
            Segment(50, 30 * kmh, 2.0, URBAN_CANYON, label="CANYON: glass towers"),
            Segment(20, 0, 0, URBAN_CANYON, label="CANYON: at a light"),
            Segment(45, 35 * kmh, -2.5, URBAN_CANYON, label="CANYON: turn"),
            Segment(50, 45 * kmh, 0, OPEN_SKY, label="back to open sky"),
        ],
    )


def parking_ramp() -> Scenario:
    """Multi-storey spiral: total outage plus a real vertical component."""
    kmh = 1 / 3.6
    return Scenario(
        name="parking_ramp",
        segments=[
            Segment(15, 0, 0, OPEN_SKY, label="stationary alignment"),
            Segment(25, 25 * kmh, 0, OPEN_SKY, label="approach"),
            Segment(120, 12 * kmh, 12.0, None, grade=6.0, label="RAMP: spiral up"),
            Segment(20, 8 * kmh, 0, None, label="RAMP: searching for a bay"),
            Segment(10, 0, 0, None, label="RAMP: parked"),
        ],
    )


def pedestrian_mall() -> Scenario:
    """Walking indoors, at a realistically varying pace.

    The pace variation is the point, and the first version of this scenario
    did not have it. Walking at a near-constant 1.3 m/s throughout makes a
    fixed-speed prior almost optimal by construction, so that scenario could
    not distinguish a learned speed head from a hard-coded constant no matter
    how good the network was -- and measured against it, the constant won:
    26.9 m RMSE against the model's 33.0 m. The test was degenerate, not the
    model, whose standalone speed error is 0.083 m/s.

    Real indoor walking is nothing like constant. People stride down an empty
    concourse, shuffle through a crowd, stop dead in a queue, and slow on
    stairs. Those speeds span more than a factor of two, and resolving them is
    what a cadence-reading network can do and a constant cannot.
    """
    return Scenario(
        name="pedestrian_mall",
        mode="pedestrian",
        gait=True,
        accel_limit=0.8,
        segments=[
            Segment(15, 0.0, 0, OPEN_SKY, label="standing outside"),
            Segment(35, 1.6, 0, OPEN_SKY, label="brisk walk outdoors"),
            Segment(30, 1.5, 2.0, None, label="INDOORS: open concourse"),
            Segment(35, 0.7, 3.0, None, label="INDOORS: dense crowd"),
            Segment(20, 0.0, 0, None, label="INDOORS: queueing (ZUPT)"),
            Segment(25, 0.9, -2.0, None, label="INDOORS: browsing pace"),
            Segment(30, 0.6, 0, None, grade=8.0, label="INDOORS: stairs"),
            Segment(35, 1.5, -3.0, None, label="INDOORS: upper concourse"),
            Segment(40, 1.7, 0, OPEN_SKY, label="hurrying back outside"),
        ],
    )


SCENARIOS = {
    "tunnel_drive": tunnel_drive,
    "urban_canyon": urban_canyon,
    "parking_ramp": parking_ramp,
    "pedestrian_mall": pedestrian_mall,
}
