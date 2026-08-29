"""
NAVIS Core Module: Inertial Kinematics, Strapdown Mechanization & Dead Reckoning Baselines
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.kinematics import (
        integrate_attitude_quaternion_rk4,
        integrate_attitude_quaternion_midpoint,
        integrate_attitude_quaternion_euler,
        integrate_attitude_euler_angles,
        specific_force_to_world_acc,
        integrate_translation_euler,
        integrate_translation_trapezoidal,
        integrate_translation_verlet,
        GravityModel,
        STANDARD_GRAVITY_ENU,
    )
    from src.core.error_dynamics import (
        INSErrorState,
        TheoreticalDriftPredictor,
        build_ins_error_jacobian,
        propagate_error_covariance,
    )
    from src.core.traditional_dr import (
        DeadReckoningEngine,
        DeadReckoningResult,
        run_dead_reckoning,
    )
    from src.core.zupt_dr import (
        ZUPTDeadReckoningEngine,
        ZUPTDeadReckoningResult,
        run_zupt_dead_reckoning,
    )
    from src.core.neural_dr import (
        NeuralDeadReckoningEngine,
        NeuralDeadReckoningResult,
        NeuralCorrectionMode,
        run_neural_dead_reckoning,
    )
    from src.core.ekf import (
        ErrorStateEKF,
        EKFConfig,
        EKFState,
    )
    from src.core.handoff_controller import (
        GPSHandoffController,
        HandoffConfig,
        HandoffState,
        HandoffEvent,
    )
    from src.core.ai_ekf_engine import (
        AIEKFNavigationEngine,
        AIEKFResult,
        run_ai_ekf_navigation,
    )


def __getattr__(name: str):
    if name in (
        "integrate_attitude_quaternion_rk4",
        "integrate_attitude_quaternion_midpoint",
        "integrate_attitude_quaternion_euler",
        "integrate_attitude_quaternion_exponential",
        "integrate_attitude_euler_angles",
        "specific_force_to_world_acc",
        "integrate_translation_euler",
        "integrate_translation_trapezoidal",
        "integrate_translation_verlet",
        "GravityModel",
        "STANDARD_GRAVITY_ENU",
    ):
        import src.core.kinematics as _k
        return getattr(_k, name)

    if name in (
        "INSErrorState",
        "TheoreticalDriftPredictor",
        "build_ins_error_jacobian",
        "propagate_error_covariance",
        "skew_symmetric",
    ):
        import src.core.error_dynamics as _e
        return getattr(_e, name)

    if name in (
        "DeadReckoningEngine",
        "DeadReckoningResult",
        "run_dead_reckoning",
    ):
        import src.core.traditional_dr as _t
        return getattr(_t, name)

    if name in (
        "ZUPTDeadReckoningEngine",
        "ZUPTDeadReckoningResult",
        "run_zupt_dead_reckoning",
    ):
        import src.core.zupt_dr as _z
        return getattr(_z, name)

    if name in (
        "NeuralDeadReckoningEngine",
        "NeuralDeadReckoningResult",
        "NeuralCorrectionMode",
        "run_neural_dead_reckoning",
    ):
        import src.core.neural_dr as _n
        return getattr(_n, name)

    if name in (
        "ErrorStateEKF",
        "EKFConfig",
        "EKFState",
    ):
        import src.core.ekf as _ekf
        return getattr(_ekf, name)

    if name in (
        "GPSHandoffController",
        "HandoffConfig",
        "HandoffState",
        "HandoffEvent",
    ):
        import src.core.handoff_controller as _hc
        return getattr(_hc, name)

    if name in (
        "AIEKFNavigationEngine",
        "AIEKFResult",
        "run_ai_ekf_navigation",
    ):
        import src.core.ai_ekf_engine as _aie
        return getattr(_aie, name)

    raise AttributeError(f"module 'src.core' has no attribute '{name}'")


__all__ = [
    # Module 1 & 2 Core Kinematics & Baseline DR
    "integrate_attitude_quaternion_rk4",
    "integrate_attitude_quaternion_midpoint",
    "integrate_attitude_quaternion_euler",
    "integrate_attitude_euler_angles",
    "specific_force_to_world_acc",
    "integrate_translation_euler",
    "integrate_translation_trapezoidal",
    "integrate_translation_verlet",
    "GravityModel",
    "STANDARD_GRAVITY_ENU",
    "INSErrorState",
    "TheoreticalDriftPredictor",
    "build_ins_error_jacobian",
    "propagate_error_covariance",
    "DeadReckoningEngine",
    "DeadReckoningResult",
    "run_dead_reckoning",
    # Module 3 ZUPT-DR
    "ZUPTDeadReckoningEngine",
    "ZUPTDeadReckoningResult",
    "run_zupt_dead_reckoning",
    # Module 4 Neural DR
    "NeuralDeadReckoningEngine",
    "NeuralDeadReckoningResult",
    "NeuralCorrectionMode",
    "run_neural_dead_reckoning",
    # Module 5 AI-EKF & GPS Handoff
    "ErrorStateEKF",
    "EKFConfig",
    "EKFState",
    "GPSHandoffController",
    "HandoffConfig",
    "HandoffState",
    "HandoffEvent",
    "AIEKFNavigationEngine",
    "AIEKFResult",
    "run_ai_ekf_navigation",
]
