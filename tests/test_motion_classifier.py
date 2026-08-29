"""
Unit Tests for Motion Context Classifier & IMU Feature Extractor
"""

import unittest
import numpy as np

from src.models.motion_classifier import (
    MotionContext,
    IMUFeatureExtractor,
    classify_motion_context,
    MotionClassifier,
)
from src.simulation.trajectory_gen import TrajectoryGenerator


class TestMotionClassifier(unittest.TestCase):

    def test_motion_context_enum(self):
        """Test MotionContext enum parsing and string conversions."""
        self.assertEqual(MotionContext.from_str("pedestrian"), MotionContext.PEDESTRIAN)
        self.assertEqual(MotionContext.from_str("urban_driving"), MotionContext.VEHICULAR)
        self.assertEqual(MotionContext.from_str("drone"), MotionContext.AERIAL)
        self.assertEqual(MotionContext.from_str("rover"), MotionContext.PLANETARY_ROVER)
        self.assertEqual(MotionContext.from_str("stationary"), MotionContext.STATIONARY)

    def test_imu_feature_extractor(self):
        """Feature extractor produces 42-dimensional engineered feature vector."""
        acc_win = np.random.randn(100, 3) + np.array([0, 0, 9.81])
        gyro_win = np.random.randn(100, 3) * 0.1

        feats = IMUFeatureExtractor.extract_window_features(acc_win, gyro_win, dt=0.01)
        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(len(feats), 42)
        self.assertFalse(np.any(np.isnan(feats)))

    def test_classify_stationary_profile(self):
        """Rule-based classifier identifies stationary sensor."""
        gen = TrajectoryGenerator(profile="stationary", duration=5.0, dt=0.01, seed=42)
        df = gen.generate()
        acc = df[['acc_x', 'acc_y', 'acc_z']].values
        gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

        ctx, conf = classify_motion_context(acc[:100], gyro[:100], dt=0.01)
        self.assertEqual(ctx, MotionContext.STATIONARY)
        self.assertGreater(conf, 0.8)

    def test_classify_pedestrian_profile(self):
        """Classifier identifies cyclic vertical dynamics of pedestrian walking."""
        gen = TrajectoryGenerator(profile="pedestrian", duration=10.0, dt=0.01, seed=42)
        df = gen.generate()
        acc = df[['acc_x', 'acc_y', 'acc_z']].values
        gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

        ctx, conf = classify_motion_context(acc[:200], gyro[:200], dt=0.01)
        self.assertEqual(ctx, MotionContext.PEDESTRIAN)

    def test_motion_classifier_sequence_prediction(self):
        """Sliding-window sequence classifier returns per-sample and dominant contexts."""
        gen = TrajectoryGenerator(profile="urban_driving", duration=10.0, dt=0.01, seed=42)
        df = gen.generate()
        acc = df[['acc_x', 'acc_y', 'acc_z']].values
        gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

        classifier = MotionClassifier(window_size=100, step_size=25, dt=0.01)
        res = classifier.predict_sequence(acc, gyro, use_neural=False)

        self.assertEqual(len(res["contexts"]), len(df))
        self.assertEqual(res["probabilities"].shape, (len(df), 5))
        self.assertIn(res["dominant_context"], [MotionContext.VEHICULAR, MotionContext.STATIONARY])


if __name__ == "__main__":
    unittest.main()
