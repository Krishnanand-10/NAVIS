"""
Seamless GPS Handoff Controller & Multi-Sensor Integrity Manager

Provides robust fault-tolerant GPS handoff and outlier rejection:
1. Chi-Square (Chi^2) Innovation Gating:
   Detects and rejects spoofed GPS positions, multipath jumps, and degraded fixes.
2. Smooth Sigmoidal / Cosine Transition Blending:
   Eliminates discrete trajectory jumps and velocity spikes when entering or exiting
   GPS-denied zones (tunnels, indoor parking, urban canyons, electronic jamming).
3. Multi-Sensor Fallback Coordinator:
   Automatically switches and weights auxiliary updates (AI-ZUPT, Neural Velocity,
   Barometer, Magnetometer, Non-Holonomic Constraints) during GPS denial.
4. Outage Analytics & Navigation Health Metric:
   Real-time monitoring of filter consistency, covariance traces, and handoff state machine.
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List, Any
import numpy as np

from src.core.ekf import ErrorStateEKF, EKFState


class HandoffState(str, Enum):
    """Operational status of the GPS Handoff State Machine."""
    GPS_LOCKED = "gps_locked"              # Full high-accuracy GPS fix available
    ENTERING_OUTAGE = "entering_outage"    # GPS signal lost; transition into dead reckoning
    GPS_DENIED = "gps_denied"              # Pure dead reckoning (AI-ZUPT, Neural DR, Baro, NHC)
    RECOVERING_GPS = "recovering_gps"      # GPS regained; smoothly converging states without jumps


@dataclass
class HandoffConfig:
    """Configuration parameters for GPS Handoff Controller."""
    # Outage Detection Thresholds
    gps_timeout_sec: float = 2.0             # Time elapsed before declaring GPS outage [s]
    transition_duration_sec: float = 2.0     # Smooth sigmoidal blending window [s]

    # Chi-Square Innovation Gating (p = 0.001)
    chi2_gate_pos_3d: float = 16.27          # 3-DOF Chi-Square threshold
    chi2_gate_vel_3d: float = 16.27          # 3-DOF Chi-Square threshold
    consecutive_rejections_limit: int = 5    # Consecutive rejections before flagging spoofing

    # Multi-Sensor Measurement Covariances
    gps_pos_std_nominal: float = 2.5         # [m] Nominal consumer GPS accuracy
    gps_vel_std_nominal: float = 0.2         # [m/s] Nominal GPS Doppler velocity accuracy
    zupt_std_nominal: float = 0.01           # [m/s] Stance velocity accuracy
    neural_vel_std_scale: float = 1.0        # Multiplier on Module 4 neural uncertainty
    baro_alt_std_nominal: float = 1.2        # [m] Barometric altitude accuracy
    nhc_lateral_std: float = 0.1             # [m/s] Non-holonomic lateral velocity constraint
    nhc_vertical_std: float = 0.1            # [m/s] Non-holonomic vertical velocity constraint


@dataclass
class HandoffEvent:
    """Record of a GPS handoff or outage transition event."""
    timestamp: float
    event_type: str                          # 'outage_start', 'outage_end', 'spoofing_detected'
    state_before: HandoffState
    state_after: HandoffState
    duration_sec: float = 0.0
    pos_at_event: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drift_accumulated: float = 0.0


class GPSHandoffController:
    """
    Seamless GPS Handoff and Sensor Fusion Integrity Controller.
    """

    def __init__(self,
                 ekf: ErrorStateEKF,
                 config: Optional[HandoffConfig] = None):
        self.ekf = ekf
        self.config = config or HandoffConfig()

        self.state: HandoffState = HandoffState.GPS_DENIED
        self.last_valid_gps_time: float = -1.0
        self.transition_start_time: float = -1.0
        self.outage_start_time: float = -1.0

        # Integrity & Rejection statistics
        self.consecutive_gps_rejections: int = 0
        self.total_gps_updates: int = 0
        self.rejected_gps_updates: int = 0
        self.events: List[HandoffEvent] = []

    @property
    def is_gps_available(self) -> bool:
        """True if currently operating in GPS_LOCKED or RECOVERING_GPS state."""
        return self.state in (HandoffState.GPS_LOCKED, HandoffState.RECOVERING_GPS)

    # ========================================================================
    # 1. STATE MACHINE & SMOOTH SIGMOIDAL BLENDING
    # ========================================================================

    def _update_state_machine(self, current_time: float, gps_received: bool):
        """Update handoff state machine and manage smooth transitions."""
        if gps_received:
            self.consecutive_gps_rejections = 0
            if self.state in (HandoffState.GPS_DENIED, HandoffState.ENTERING_OUTAGE):
                # Outage ended -> begin smooth recovery
                outage_duration = current_time - self.outage_start_time if self.outage_start_time > 0 else 0.0
                event = HandoffEvent(
                    timestamp=current_time,
                    event_type='outage_end',
                    state_before=self.state,
                    state_after=HandoffState.RECOVERING_GPS,
                    duration_sec=outage_duration,
                    pos_at_event=self.ekf.pos.copy()
                )
                self.events.append(event)
                self.state = HandoffState.RECOVERING_GPS
                self.transition_start_time = current_time

            elif self.state == HandoffState.RECOVERING_GPS:
                # Check if recovery window has completed
                elapsed = current_time - self.transition_start_time
                if elapsed >= self.config.transition_duration_sec:
                    self.state = HandoffState.GPS_LOCKED

            elif self.state == HandoffState.GPS_LOCKED:
                pass  # Steady-state locked

            self.last_valid_gps_time = current_time

        else:  # No valid GPS at this step
            if self.last_valid_gps_time > 0:
                time_since_gps = current_time - self.last_valid_gps_time
                if time_since_gps >= self.config.gps_timeout_sec and self.state in (HandoffState.GPS_LOCKED, HandoffState.RECOVERING_GPS):
                    # Enter GPS outage
                    self.outage_start_time = current_time
                    event = HandoffEvent(
                        timestamp=current_time,
                        event_type='outage_start',
                        state_before=self.state,
                        state_after=HandoffState.GPS_DENIED,
                        pos_at_event=self.ekf.pos.copy()
                    )
                    self.events.append(event)
                    self.state = HandoffState.GPS_DENIED

    def _compute_recovery_blend_factor(self, current_time: float) -> float:
        """
        Compute smooth blending factor alpha in [0.0, 1.0] using cosine ramp.
        alpha = 0.0 -> full conservative noise (no jump)
        alpha = 1.0 -> full nominal GPS gain
        """
        if self.state == HandoffState.GPS_LOCKED:
            return 1.0
        elif self.state == HandoffState.RECOVERING_GPS:
            elapsed = current_time - self.transition_start_time
            t_ratio = float(np.clip(elapsed / max(self.config.transition_duration_sec, 1e-4), 0.0, 1.0))
            # Smooth S-curve (Raised cosine)
            return 0.5 * (1.0 - np.cos(np.pi * t_ratio))
        return 0.0

    # ========================================================================
    # 2. SENSOR UPDATE DISPATCHERS
    # ========================================================================

    def process_gps_update(self,
                          timestamp: float,
                          pos_meas: Optional[np.ndarray],
                          vel_meas: Optional[np.ndarray] = None,
                          pos_cov: Optional[Union[float, np.ndarray]] = None,
                          vel_cov: Optional[Union[float, np.ndarray]] = None,
                          is_valid: bool = True) -> Tuple[bool, float]:
        """
        Process GPS Position and Velocity measurement with Chi^2 gating and smooth blending.
        """
        if not is_valid or pos_meas is None or np.any(np.isnan(pos_meas)):
            self._update_state_machine(timestamp, gps_received=False)
            return False, 0.0

        self.total_gps_updates += 1

        # Determine effective GPS measurement covariance with smooth blending
        blend_factor = self._compute_recovery_blend_factor(timestamp)
        # When recovering (blend_factor < 1.0), inflate R to prevent step discontinuities
        noise_inflation = 1.0 if blend_factor >= 0.99 else 1.0 / max(blend_factor, 0.05)

        base_pos_cov = pos_cov if pos_cov is not None else (self.config.gps_pos_std_nominal ** 2)
        effective_pos_cov = np.asarray(base_pos_cov) * noise_inflation

        # 1. Update Position with Chi-Square Gating
        accepted_pos, d2_pos = self.ekf.update_position(
            pos_meas=pos_meas,
            pos_cov=effective_pos_cov,
            chi2_threshold=self.config.chi2_gate_pos_3d
        )

        if not accepted_pos:
            self.rejected_gps_updates += 1
            self.consecutive_gps_rejections += 1
            if self.consecutive_gps_rejections >= self.config.consecutive_rejections_limit:
                # Suspect spoofing or multipath failure
                event = HandoffEvent(
                    timestamp=timestamp,
                    event_type='spoofing_detected',
                    state_before=self.state,
                    state_after=HandoffState.GPS_DENIED,
                    pos_at_event=self.ekf.pos.copy()
                )
                self.events.append(event)
                self.state = HandoffState.GPS_DENIED
            return False, d2_pos

        # 2. Update Velocity (if available)
        if vel_meas is not None and not np.any(np.isnan(vel_meas)):
            base_vel_cov = vel_cov if vel_cov is not None else (self.config.gps_vel_std_nominal ** 2)
            effective_vel_cov = np.asarray(base_vel_cov) * noise_inflation
            self.ekf.update_velocity(
                vel_meas=vel_meas,
                vel_cov=effective_vel_cov,
                chi2_threshold=self.config.chi2_gate_vel_3d
            )

        self._update_state_machine(timestamp, gps_received=True)
        return True, d2_pos

    def process_zupt_update(self,
                           is_stance: bool,
                           zupt_cov_sigma: Optional[float] = None) -> bool:
        """
        Process Zero-Velocity Update when stance / stationary phase is detected.
        """
        if not is_stance:
            return False

        sigma = zupt_cov_sigma or self.config.zupt_std_nominal
        accepted, _ = self.ekf.update_zupt(zupt_cov_sigma=sigma)
        return accepted

    def process_neural_velocity_update(self,
                                      v_neural: np.ndarray,
                                      v_neural_cov: Optional[np.ndarray] = None,
                                      enable_only_during_outage: bool = True) -> bool:
        """
        Fuse Module 4 Deep Neural Velocity prediction into the EKF.
        """
        if enable_only_during_outage and self.state == HandoffState.GPS_LOCKED:
            return False  # Prioritize direct GPS when locked

        if v_neural_cov is not None:
            R = np.asarray(v_neural_cov) * self.config.neural_vel_std_scale
        else:
            R = (0.2 ** 2) * np.eye(3)

        accepted, _ = self.ekf.update_velocity(
            vel_meas=v_neural,
            vel_cov=R,
            chi2_threshold=self.config.chi2_gate_vel_3d
        )
        return accepted

    def process_barometer_update(self,
                                alt_meas: float,
                                alt_cov_sigma: Optional[float] = None) -> bool:
        """Process 1D barometric altitude measurement."""
        sigma = alt_cov_sigma or self.config.baro_alt_std_nominal
        accepted, _ = self.ekf.update_barometer(alt_meas=alt_meas, alt_cov_sigma=sigma)
        return accepted

    def process_nhc_update(self, is_wheeled_vehicle: bool = True) -> bool:
        """Process Non-Holonomic Constraints (lateral/vertical velocity damping)."""
        if not is_wheeled_vehicle:
            return False
        accepted, _ = self.ekf.update_nhc(
            lateral_cov=self.config.nhc_lateral_std,
            vertical_cov=self.config.nhc_vertical_std
        )
        return accepted
