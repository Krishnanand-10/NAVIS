"""
NAVIS AI Models: Zero-Velocity Detectors, Motion Context Classifiers, and Neural Drift & Bias Estimators
"""

from src.models.zupt_detector import (
    ZUPTDetector,
    AIZUPTDetector,
    AIZUPTNet,
    GLRTDetector,
    AREDetector,
    MAGDetector,
    detect_zero_velocity_statistical,
)
from src.models.motion_classifier import (
    MotionContext,
    MotionClassifier,
    MotionResNet1D,
    IMUFeatureExtractor,
    classify_motion_context,
)
from src.models.drift_estimator import (
    IMUScaler,
    IMUSequenceDataset,
    InertialDriftEstimator,
    NumPyDriftEstimator,
)
from src.models.train_drift_models import (
    train_drift_estimator,
    train_numpy_estimator,
    generate_multi_regime_training_data,
)

__all__ = [
    # Module 3 Models
    "ZUPTDetector",
    "AIZUPTDetector",
    "AIZUPTNet",
    "GLRTDetector",
    "AREDetector",
    "MAGDetector",
    "detect_zero_velocity_statistical",
    "MotionContext",
    "MotionClassifier",
    "MotionResNet1D",
    "IMUFeatureExtractor",
    "classify_motion_context",
    # Module 4 Models
    "IMUScaler",
    "IMUSequenceDataset",
    "InertialDriftEstimator",
    "NumPyDriftEstimator",
    "train_drift_estimator",
    "train_numpy_estimator",
    "generate_multi_regime_training_data",
]

