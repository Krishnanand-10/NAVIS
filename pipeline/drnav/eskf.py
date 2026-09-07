"""Error-state Kalman filter for GNSS/INS dead reckoning.

Design notes worth knowing before you edit this file
----------------------------------------------------

*Error definition.* We use the **global** (nav-frame) error convention::

    p_true = p + dp        v_true = v + dv        R_true = Exp(dtheta) @ R
    ba_true = ba + dba     bg_true = bg + dbg

so the 15-element error state is ``[dp, dv, dtheta, dba, dbg]``. The global
convention keeps the attitude Jacobians free of the nominal rotation in the
places that matter and is the formulation in Sola's ESKF tutorial.

*Attitude storage.* The nominal attitude is a 3x3 matrix, not a quaternion --
see :mod:`drnav.attitude` for why.

*Device frame vs vehicle frame.* The filter tracks the attitude of the **phone**,
which sits at an arbitrary fixed rotation inside the vehicle. Non-holonomic
constraints ("a car cannot slide sideways") are only true in the **vehicle**
frame, so we estimate the mounting rotation during good-GNSS motion and rotate
into the vehicle frame before applying them. Skipping this step is the single
most common way a phone-based DR system produces confidently wrong output.

*Initial alignment.* Yaw is unobservable while stationary, so we align with
TRIAD from two vector pairs: gravity (from the stationary accelerometer) and a
horizontal acceleration event (from differentiated GNSS velocity). This is what
lets the filter start with a usable attitude on a phone in an unknown pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attitude import (G_VEC_ENU, exp_so3, heading_from_R, orthonormalise,
                       skew)

# error-state index blocks
I_P, I_V, I_TH, I_BA, I_BG = (slice(0, 3), slice(3, 6), slice(6, 9),
                              slice(9, 12), slice(12, 15))
N_ERR = 15


@dataclass
class EskfConfig:
    """Tuning. Defaults are for a phone IMU at 100-200 Hz."""

    # process noise (continuous densities, matched to synth.ImuErrorModel)
    accel_noise_density: float = 0.0025   # m/s^2/sqrt(Hz)
    gyro_noise_density: float = 2.5e-4    # rad/s/sqrt(Hz)
    accel_bias_walk: float = 6e-4         # m/s^2/sqrt(s)
    gyro_bias_walk: float = 3e-5          # rad/s/sqrt(s)

    # measurement noise
    gnss_pos_sigma: float = 3.0           # m (overridden per-fix by reported accuracy)
    gnss_vel_sigma: float = 0.15          # m/s
    zupt_sigma: float = 0.05              # m/s
    zupt_freeze_accel_bias: bool = True   # see the warning in update_zupt
    zupt_requires_speed_reference: bool = True   # see the note in replay.run_eskf
    zaru_sigma: float = 0.002             # rad/s
    nhc_sigma: float = 0.10               # m/s, lateral slip allowance
    ml_speed_sigma: float = 0.35          # m/s, learned-odometry pseudo-measurement

    # initial uncertainty
    init_pos_sigma: float = 5.0
    init_vel_sigma: float = 1.0
    init_att_sigma: float = np.radians(8.0)
    init_ba_sigma: float = 0.15
    init_bg_sigma: float = 0.02

    # plausibility limits -- a MEMS part that exceeds these is broken, and a
    # filter that asks for more than these has diverged
    max_accel_bias: float = 2.0           # m/s^2
    max_gyro_bias: float = 0.35           # rad/s (~20 deg/s)
    max_step_rotation: float = 0.35       # rad, per single update

    # Road vibration lives at 6-40 Hz; vehicle dynamics live below about 2 Hz.
    # Mechanising the raw signal integrates 0.4 m/s^2 of vibration the process
    # noise model knows nothing about, and the filter diverges. Low-passing the
    # IMU before mechanisation removes it almost entirely. The learned speed
    # head deliberately does NOT get this filtered stream -- vibration is the
    # only absolute speed cue a vehicle IMU carries, so the two consumers of
    # the same log want opposite things.
    imu_lowpass_hz: float = 5.0           # 0 disables

    # Seamless handover. When GNSS returns after a long outage the incoming fix
    # disagrees with the inertial estimate, sometimes by hundreds of metres,
    # and applying that correction in one step teleports the position marker
    # across the map. The filter state should absolutely take the correction
    # immediately -- it is the better estimate -- but the *reported* position
    # is held back and allowed to catch up over a couple of seconds, so the
    # displayed track stays continuous. This is the difference between a
    # correct system and a "seamless" one, which is the word in the title.
    output_smoothing_tau: float = 2.0     # seconds; 0 reports the raw state

    # behaviour switches
    zupt_max_speed: float = 1.5           # m/s -- see note in replay.run_eskf
    constraint_rate_hz: float = 10.0      # ZUPT/ZARU/NHC application rate
    zupt_reject_after: int = 5            # consecutive gated fixes before reset

    use_zupt: bool = True
    use_zaru: bool = True
    use_nhc: bool = True
    nhc_vertical: bool = False            # see the warning in update_nhc

    # The learned speed head is enabled for pedestrians and OFF for vehicles,
    # and that asymmetry is a measurement, not a hunch.
    #
    # For a walker the network reads speed off gait cadence to 0.083 m/s and
    # rescues a mode that is otherwise unusable: 538 m -> 32 m RMSE indoors.
    #
    # For a vehicle it infers speed from road vibration amplitude, which is
    # perfectly confounded with road roughness. Held-out surfaces inside the
    # trained range give a bias sweeping from -8.0 m/s on smooth tarmac to
    # +3.9 m/s on a rough street, and on smooth roads it is over-confident
    # (calibration 16, rising to 270 outside the range) -- confidently wrong is
    # the one thing a Kalman filter cannot absorb. Switching it on costs
    # accuracy: tunnel RMSE 59 m -> 88 m, parking ramp 2.2 m -> 5.3 m.
    #
    # The fix is not a tuning constant. Amplitude alone cannot separate speed
    # from surface; real tyre vibration also carries speed-dependent spectral
    # structure (wheel and tread harmonics scale with rotation rate) which the
    # simulator does not yet model. Set this True once trained on data that
    # has it -- ideally real logs.
    use_ml_speed_for_vehicle: bool = False
    use_gnss_velocity: bool = True
    gnss_gate_chi2: float = 16.27         # 3-dof, 99.9% -- reject wild fixes


@dataclass
class EskfState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(N_ERR))

    def copy(self) -> "EskfState":
        return EskfState(self.p.copy(), self.v.copy(), self.R.copy(),
                         self.ba.copy(), self.bg.copy(), self.P.copy())


class Eskf:
    """15-state error-state KF. Call :meth:`predict` at IMU rate, ``update_*``
    whenever a measurement is available."""

    def __init__(self, cfg: EskfConfig | None = None):
        self.cfg = cfg or EskfConfig()
        self.x = EskfState()
        self._steps = 0
        # device->vehicle mounting rotation, learned during good GNSS motion
        self.R_mount: np.ndarray | None = None

    # -- initialisation ----------------------------------------------------
    def initialise(self, p, v, R, cfg: EskfConfig | None = None) -> None:
        cfg = cfg or self.cfg
        self.x = EskfState(p=np.asarray(p, float).copy(),
                           v=np.asarray(v, float).copy(),
                           R=np.asarray(R, float).copy())
        P = np.zeros((N_ERR, N_ERR))
        P[I_P, I_P] = np.eye(3) * cfg.init_pos_sigma ** 2
        P[I_V, I_V] = np.eye(3) * cfg.init_vel_sigma ** 2
        P[I_TH, I_TH] = np.eye(3) * cfg.init_att_sigma ** 2
        P[I_BA, I_BA] = np.eye(3) * cfg.init_ba_sigma ** 2
        P[I_BG, I_BG] = np.eye(3) * cfg.init_bg_sigma ** 2
        self.x.P = P

    def reset_position(self, p, v, sigma_p: float, sigma_v: float) -> None:
        """Hard re-seed after the filter has lost the plot.

        A chi-square gate protects against multipath, but a gate with no escape
        hatch is a trap: once the estimate is far enough wrong, every genuine
        fix looks like an outlier and the filter never recovers. If enough
        consecutive fixes are rejected we accept that the filter, not the
        receiver, is the broken one.
        """
        x = self.x
        x.p = np.asarray(p, float).copy()
        x.v = np.asarray(v, float).copy()
        # Rebuild the covariance from scratch rather than patching blocks of
        # it. By the time we get here the existing P has usually stopped being
        # a meaningful description of the error -- its cross-correlations are
        # what produced the bad gain in the first place -- and grafting fresh
        # position variance onto stale correlations produces gains large enough
        # to drive the accelerometer bias into the hundreds of m/s^2.
        c = self.cfg
        P = np.zeros((N_ERR, N_ERR))
        P[I_P, I_P] = np.eye(3) * sigma_p ** 2
        P[I_V, I_V] = np.eye(3) * sigma_v ** 2
        P[I_TH, I_TH] = np.eye(3) * c.init_att_sigma ** 2
        P[I_BA, I_BA] = np.eye(3) * c.init_ba_sigma ** 2
        P[I_BG, I_BG] = np.eye(3) * c.init_bg_sigma ** 2
        x.P = P
        # biases that grew implausible came from the same bad gain; drop them
        x.ba = np.clip(x.ba, -c.max_accel_bias, c.max_accel_bias)
        x.bg = np.clip(x.bg, -c.max_gyro_bias, c.max_gyro_bias)

    # -- propagation -------------------------------------------------------
    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """Mechanise one IMU sample and propagate the covariance."""
        c, x = self.cfg, self.x
        f = accel - x.ba                       # specific force, body
        w = gyro - x.bg                        # angular rate, body
        R = x.R
        a_nav = R @ f + G_VEC_ENU

        # nominal state
        x.p = x.p + x.v * dt + 0.5 * a_nav * dt * dt
        x.v = x.v + a_nav * dt
        x.R = R @ exp_so3(w * dt)
        self._steps += 1
        if self._steps % 200 == 0:
            x.R = orthonormalise(x.R)

        # error-state transition (first order is ample at >=100 Hz)
        F = np.zeros((N_ERR, N_ERR))
        F[I_P, I_V] = np.eye(3)
        F[I_V, I_TH] = -skew(R @ f)
        F[I_V, I_BA] = -R
        F[I_TH, I_BG] = -R
        Phi = np.eye(N_ERR) + F * dt

        Q = np.zeros((N_ERR, N_ERR))
        Q[I_V, I_V] = np.eye(3) * (c.accel_noise_density ** 2) * dt
        Q[I_TH, I_TH] = np.eye(3) * (c.gyro_noise_density ** 2) * dt
        Q[I_BA, I_BA] = np.eye(3) * (c.accel_bias_walk ** 2) * dt
        Q[I_BG, I_BG] = np.eye(3) * (c.gyro_bias_walk ** 2) * dt

        x.P = self._sanitise(Phi @ x.P @ Phi.T + Q)

    # -- generic correction ------------------------------------------------
    def _update(self, H: np.ndarray, r: np.ndarray, R_meas: np.ndarray,
                gate_chi2: float | None = None,
                freeze: tuple = ()) -> bool:
        """Joseph-form update. Returns False if the innovation was gated out.

        ``freeze`` names error-state blocks the update may not touch. Zeroing
        those rows of the Kalman gain makes the update suboptimal but still
        consistent (Joseph form tolerates any gain), and is the correct tool
        when a measurement cannot actually separate two states -- see
        :meth:`update_zupt`.
        """
        x = self.x
        S = H @ x.P @ H.T + R_meas
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        if gate_chi2 is not None and float(r @ Sinv @ r) > gate_chi2:
            return False

        K = x.P @ H.T @ Sinv
        for blk in freeze:
            K[blk, :] = 0.0
        dx = K @ r

        # Last line of defence. No single measurement should rotate the
        # platform by a third of a radian or move a MEMS bias by more than its
        # entire plausible range; if one wants to, the covariance is lying and
        # applying the correction would corrupt the state irrecoverably.
        c = self.cfg
        if (np.linalg.norm(dx[I_TH]) > c.max_step_rotation
                or np.linalg.norm(dx[I_BA]) > c.max_accel_bias
                or np.linalg.norm(dx[I_BG]) > c.max_gyro_bias
                or not np.all(np.isfinite(dx))):
            return False

        IKH = np.eye(N_ERR) - K @ H
        x.P = self._sanitise(IKH @ x.P @ IKH.T + K @ R_meas @ K.T)

        # inject the error into the nominal state, then reset
        x.p += dx[I_P]
        x.v += dx[I_V]
        x.R = exp_so3(dx[I_TH]) @ x.R          # global error -> left multiply
        x.ba += dx[I_BA]
        x.bg += dx[I_BG]
        return True

    _P_MAX = 1e6

    def _sanitise(self, P: np.ndarray) -> np.ndarray:
        """Keep the covariance symmetric, finite and bounded.

        Repeated Joseph updates with a frozen gain, or a badly conditioned
        innovation covariance, can nudge P off the positive-definite cone. Left
        alone that turns into a silent numerical blow-up thousands of samples
        later, which is far harder to diagnose than a clamp here.
        """
        P = 0.5 * (P + P.T)
        if not np.all(np.isfinite(P)):
            return np.eye(N_ERR)
        d = np.diag(P)
        if np.any(d <= 0) or np.any(d > self._P_MAX):
            # Clamping only the diagonal of a matrix that has left the
            # positive-definite cone makes it *less* consistent, so repair the
            # whole thing through its spectrum instead.
            w, V = np.linalg.eigh(P)
            w = np.clip(w, 1e-12, self._P_MAX)
            P = V @ np.diag(w) @ V.T
            P = 0.5 * (P + P.T)
        return P

    # -- specific measurements --------------------------------------------
    def update_gnss_position(self, z, sigma=None, gate=True) -> bool:
        c = self.cfg
        sigma = c.gnss_pos_sigma if sigma is None else sigma
        H = np.zeros((3, N_ERR))
        H[:, I_P] = np.eye(3)
        Rm = np.diag([sigma ** 2, sigma ** 2, (2.0 * sigma) ** 2])
        return self._update(H, np.asarray(z, float) - self.x.p, Rm,
                            c.gnss_gate_chi2 if gate else None)

    def update_gnss_velocity(self, z, sigma=None) -> bool:
        c = self.cfg
        sigma = c.gnss_vel_sigma if sigma is None else sigma
        H = np.zeros((3, N_ERR))
        H[:, I_V] = np.eye(3)
        return self._update(H, np.asarray(z, float) - self.x.v,
                            np.eye(3) * sigma ** 2, c.gnss_gate_chi2)

    def update_zupt(self) -> bool:
        """Zero-velocity update: while stopped, velocity is known to be zero.

        Exact, free, and available at every traffic light -- but it comes with
        a trap. **While stationary, a tilt error and a horizontal accelerometer
        bias are indistinguishable**: both produce a constant horizontal
        specific-force error, and no amount of standing still separates them.
        A tight ZUPT nonetheless forces the filter to commit to some split, and
        it commits *confidently*. On the tunnel scenario an unrestricted ZUPT
        at sigma=0.02 drove the accelerometer bias to 0.7 m/s^2 -- nine sigma
        past anything physical -- and the resulting error was 645 m at the
        first outage exit against 9 m with ZUPT disabled entirely.

        The fix is to let ZUPT correct velocity and attitude, which it can
        legitimately observe, while freezing the accelerometer bias it cannot.
        Bias stays observable through motion, where GNSS and NHC constrain it.
        """
        H = np.zeros((3, N_ERR))
        H[:, I_V] = np.eye(3)
        freeze = (I_BA,) if self.cfg.zupt_freeze_accel_bias else ()
        return self._update(H, -self.x.v, np.eye(3) * self.cfg.zupt_sigma ** 2,
                            freeze=freeze)

    def update_zaru(self, gyro: np.ndarray) -> bool:
        """While stationary the gyro reads its own bias -- observe it directly."""
        H = np.zeros((3, N_ERR))
        H[:, I_BG] = np.eye(3)
        return self._update(H, np.asarray(gyro, float) - self.x.bg,
                            np.eye(3) * self.cfg.zaru_sigma ** 2)

    def _body_velocity_jacobian(self, R_bn: np.ndarray) -> np.ndarray:
        """d(v_body)/d(error state) for a body frame given by ``R_bn`` (body->nav).

        v_b = R^T v  with  R_true = Exp(dtheta) R  gives
            v_b ~= R^T v + R^T dv + R^T [v]_x dtheta
        """
        H = np.zeros((3, N_ERR))
        H[:, I_V] = R_bn.T
        H[:, I_TH] = R_bn.T @ skew(self.x.v)
        return H

    def update_nhc(self) -> bool:
        """Non-holonomic constraint: a vehicle does not slide sideways.

        Applied in the **vehicle** frame via the estimated mounting rotation,
        and silently skipped until that mounting has been observed.

        **The vertical constraint is off by default, and should stay off.**
        The lateral one is nearly exact -- a car really cannot translate along
        its own y-axis. The vertical one is not: the mounting is estimated
        against a *level* vehicle frame, so on a graded road the true climb
        rate (``v * sin(grade)``, about 0.6 m/s at 60 km/h on a 2% tunnel ramp)
        shows up as a constraint violation. The filter has only one way to
        explain a persistent vertical velocity -- tilt the pitch -- and a 2 deg
        pitch error leaks ``g * sin(2 deg)`` = 0.34 m/s^2 of gravity into the
        horizontal channel, which integrates to roughly 200 m over a 35 s ramp.
        Measured on the tunnel scenario: enabling it moved the outage-exit
        error from 30 m to 1073 m. Estimate pitch properly before turning it on.
        """
        if self.R_mount is None:
            return False
        rows = [1, 2] if self.cfg.nhc_vertical else [1]
        R_veh = self.x.R @ self.R_mount.T       # vehicle->nav
        H = self._body_velocity_jacobian(R_veh)[rows]
        r = -(R_veh.T @ self.x.v)[rows]
        return self._update(H, r, np.eye(len(rows)) * self.cfg.nhc_sigma ** 2, 25.0)

    def update_ml_forward_speed(self, speed: float, sigma=None) -> bool:
        """Pseudo-measurement from the learned inertial-odometry head.

        The network regresses *velocity* rather than integrating acceleration,
        which is precisely why it does not inherit the cubic drift of a
        strapdown solution. Feed its predicted uncertainty in as ``sigma``.
        """
        if self.R_mount is None:
            return False
        c = self.cfg
        R_veh = self.x.R @ self.R_mount.T
        H = self._body_velocity_jacobian(R_veh)[0:1]
        r = np.array([speed - (R_veh.T @ self.x.v)[0]])
        sig = c.ml_speed_sigma if sigma is None else sigma
        return self._update(H, r, np.array([[sig ** 2]]), 25.0)

    def update_speed_magnitude(self, speed: float, sigma: float) -> bool:
        """Constrain horizontal *speed* without claiming to know the direction.

        This is the primary path for the learned odometry head, and it works in
        every mode. :meth:`update_ml_forward_speed` needs the device-to-vehicle
        mounting so it can talk about "forward", which only exists for a
        vehicle; a pedestrian has no such frame, and pedestrian mode is exactly
        where the learned head matters most.

        The measurement is nonlinear -- ``h(x) = |v_xy|`` -- so the Jacobian is
        the unit vector along the current horizontal velocity. It degenerates
        as speed approaches zero, where the direction is genuinely undefined,
        so the update is skipped there; ZUPT already covers being stopped.
        """
        v = self.x.v
        sp = float(np.hypot(v[0], v[1]))
        if sp < 0.3:
            return False
        H = np.zeros((1, N_ERR))
        H[0, I_V] = [v[0] / sp, v[1] / sp, 0.0]
        return self._update(H, np.array([speed - sp]),
                            np.array([[sigma ** 2]]), 25.0)

    # -- mounting estimation ----------------------------------------------
    def observe_mounting(self, course_rad: float, blend: float = 0.05) -> None:
        """Refine device->vehicle rotation from a trusted GNSS course.

        With the vehicle assumed level and travelling along its heading, the
        vehicle->nav rotation is known; combining it with the filter's device
        attitude yields the mounting. Blended slowly because any single course
        estimate is noisy.
        """
        from .attitude import R_from_heading
        R_veh_nav = R_from_heading(course_rad)
        R_new = R_veh_nav.T @ self.x.R          # device->vehicle
        if self.R_mount is None:
            self.R_mount = R_new
        else:
            # geodesic blend on SO(3)
            delta = self.R_mount.T @ R_new
            from .attitude import log_so3
            self.R_mount = orthonormalise(self.R_mount @ exp_so3(blend * log_so3(delta)))

    @property
    def heading(self) -> float:
        return heading_from_R(self.x.R)
