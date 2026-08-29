"""
Motion Context Classifier for Adaptive Inertial Navigation

Categorizes incoming 6-DOF IMU dynamics into operational motion regimes:
1. Stationary (standstill, traffic pause, stance)
2. Pedestrian (walking, jogging, stairs)
3. Vehicular / Ground Vehicle (terrestrial car, truck, stop-and-go, turns)
4. Aerial / UAV (drone hovering, cruise, high-frequency motor vibration)
5. Planetary Rover (slow traverse, rocky terrain crawling - ISRO lunar/planetary profile)

Provides:
- Statistical Multi-Domain Feature Extractor (Time + Frequency domain)
- Deep 1D Residual Convolutional Classifier (MotionResNet1D)
- Fast Heuristic Rule-Based Classifier for low-latency / edge deployments
- Sliding-Window Context Prediction Engine
"""

import os
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = object  # type: ignore


class MotionContext(IntEnum):
    """Motion Context Taxonomy for NAVIS."""
    STATIONARY = 0
    PEDESTRIAN = 1
    VEHICULAR = 2
    AERIAL = 3
    PLANETARY_ROVER = 4

    @classmethod
    def from_str(cls, name: str) -> "MotionContext":
        name = name.upper().strip()
        mapping = {
            "STATIONARY": cls.STATIONARY,
            "REST": cls.STATIONARY,
            "PEDESTRIAN": cls.PEDESTRIAN,
            "WALKING": cls.PEDESTRIAN,
            "VEHICULAR": cls.VEHICULAR,
            "DRIVING": cls.VEHICULAR,
            "URBAN_DRIVING": cls.VEHICULAR,
            "AERIAL": cls.AERIAL,
            "DRONE": cls.AERIAL,
            "HELICAL_3D": cls.AERIAL,
            "PLANETARY_ROVER": cls.PLANETARY_ROVER,
            "ROVER": cls.PLANETARY_ROVER,
        }
        return mapping.get(name, cls.STATIONARY)

    @property
    def label_name(self) -> str:
        return self.name.lower()


# ============================================================================
# 1. MULTI-DOMAIN FEATURE EXTRACTOR
# ============================================================================

class IMUFeatureExtractor:
    """
    Extracts comprehensive time-domain and frequency-domain kinematic features from IMU windows.
    """

    @staticmethod
    def extract_window_features(acc_window: np.ndarray,
                                gyro_window: np.ndarray,
                                dt: float = 0.01) -> np.ndarray:
        """
        Extract 42-dimensional engineered feature vector from single (W, 3) acc and gyro window.
        """
        acc = np.asarray(acc_window, dtype=np.float64)
        gyro = np.asarray(gyro_window, dtype=np.float64)
        fs = 1.0 / dt if dt > 0 else 100.0

        # Vector norms
        acc_mag = np.linalg.norm(acc, axis=-1)
        gyro_mag = np.linalg.norm(gyro, axis=-1)

        feats = []

        # 1. Time-domain statistics per axis (acc x,y,z and gyro x,y,z) -> 6 axes * 4 = 24 feats
        imu_all = np.hstack([acc, gyro])  # (W, 6)
        means = np.mean(imu_all, axis=0)
        stds = np.std(imu_all, axis=0)
        p2p = np.ptp(imu_all, axis=0)
        rms = np.sqrt(np.mean(imu_all ** 2, axis=0))
        feats.extend(means)
        feats.extend(stds)
        feats.extend(p2p)
        feats.extend(rms)

        # 2. Magnitude statistics (acc_mag, gyro_mag) -> 6 feats
        feats.extend([
            np.mean(acc_mag), np.std(acc_mag), np.ptp(acc_mag),
            np.mean(gyro_mag), np.std(gyro_mag), np.ptp(gyro_mag)
        ])

        # 3. Frequency domain (FFT peak frequency and spectral energy) -> 6 feats
        W = len(acc)
        if W >= 8:
            freqs = np.fft.rfftfreq(W, d=dt)
            fft_acc = np.abs(np.fft.rfft(acc - np.mean(acc, axis=0), axis=0))
            fft_gyro = np.abs(np.fft.rfft(gyro - np.mean(gyro, axis=0), axis=0))

            # Peak frequencies
            peak_freq_acc = freqs[np.argmax(np.mean(fft_acc, axis=-1))]
            peak_freq_gyro = freqs[np.argmax(np.mean(fft_gyro, axis=-1))]

            # Total spectral energy
            energy_acc = np.sum(fft_acc ** 2) / W
            energy_gyro = np.sum(fft_gyro ** 2) / W

            feats.extend([peak_freq_acc, peak_freq_gyro, energy_acc, energy_gyro])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        # 4. Zero-crossing rate on centered signals -> 6 feats
        zcr_acc = np.mean(np.diff(np.sign(acc - np.mean(acc, axis=0)), axis=0) != 0, axis=0)
        zcr_gyro = np.mean(np.diff(np.sign(gyro - np.mean(gyro, axis=0)), axis=0) != 0, axis=0)
        feats.extend(zcr_acc)
        feats.extend(zcr_gyro)

        # 5. Cross-axis correlations -> 2 feats
        corr_xy_acc = float(np.corrcoef(acc[:, 0], acc[:, 1])[0, 1]) if stds[0] > 1e-4 and stds[1] > 1e-4 else 0.0
        corr_xy_gyro = float(np.corrcoef(gyro[:, 0], gyro[:, 1])[0, 1]) if stds[3] > 1e-4 and stds[4] > 1e-4 else 0.0
        feats.extend([np.nan_to_num(corr_xy_acc), np.nan_to_num(corr_xy_gyro)])

        return np.array(feats, dtype=np.float32)


# ============================================================================
# 2. DEEP NEURAL 1D RESNET CLASSIFIER
# ============================================================================

if TORCH_AVAILABLE:
    class ResidualBlock1D(nn.Module):
        """1D Residual Convolutional Block with Skip Connection."""
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.relu(out)

    class MotionResNet1D(nn.Module):
        """
        1D Deep Residual Network for Motion Regime Classification.
        Input: (B, 6, WindowLength) -> Output: (B, 5) Softmax Probabilities.
        """
        def __init__(self, in_channels: int = 6, num_classes: int = 5):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True)
            )
            self.layer1 = ResidualBlock1D(32, 32, stride=1)
            self.layer2 = ResidualBlock1D(32, 64, stride=2)
            self.layer3 = ResidualBlock1D(64, 128, stride=2)

            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.stem(x)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.layer3(h)
            feat = self.global_pool(h).squeeze(-1)
            logits = self.fc(feat)
            return logits
else:
    class MotionResNet1D:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MotionResNet1D.")


# ============================================================================
# 3. RULE-BASED FAST BASELINE CLASSIFIER
# ============================================================================

def classify_motion_context(acc_window: np.ndarray,
                            gyro_window: np.ndarray,
                            dt: float = 0.01) -> Tuple[MotionContext, float]:
    """
    Heuristic rule-based motion context classifier based on physics domain kinematics.
    Returns: (MotionContext, Confidence)
    """
    acc = np.asarray(acc_window, dtype=np.float64)
    gyro = np.asarray(gyro_window, dtype=np.float64)

    acc_std = np.std(acc, axis=0)
    gyro_std = np.std(gyro, axis=0)
    total_acc_var = np.sum(acc_std ** 2)
    total_gyro_var = np.sum(gyro_std ** 2)

    # 1. Stationary Check
    if total_acc_var < 0.08 and total_gyro_var < 0.01:
        return MotionContext.STATIONARY, 0.95

    # 2. Pedestrian Check (Cyclic vertical oscillations, dominant frequency around 1.2 - 2.5 Hz)
    W = len(acc)
    if W >= 16:
        freqs = np.fft.rfftfreq(W, d=dt)
        fft_z = np.abs(np.fft.rfft(acc[:, 2] - np.mean(acc[:, 2])))
        peak_freq_z = freqs[np.argmax(fft_z)]
        # Pedestrian step frequency typically 1.0 - 3.0 Hz with prominent vertical variance
        if 0.8 <= peak_freq_z <= 3.5 and acc_std[2] > 0.8:
            return MotionContext.PEDESTRIAN, 0.90

    # 3. Aerial / UAV Check (High frequency engine vibrations, 3D angular rotation agility)
    if total_gyro_var > 0.8 and acc_std[2] > 1.5 and np.mean(np.abs(gyro[:, 2])) > 0.2:
        return MotionContext.AERIAL, 0.85

    # 4. Planetary Rover Check (Low speed, very low angular rates, moderate terrain bumpiness)
    if total_gyro_var < 0.08 and total_acc_var < 0.5:
        return MotionContext.PLANETARY_ROVER, 0.80

    # 5. Vehicular / Ground Vehicle Default
    return MotionContext.VEHICULAR, 0.85


# ============================================================================
# 4. UNIFIED MOTION CLASSIFIER ENGINE
# ============================================================================

class MotionClassifier:
    """
    Sliding-Window Motion Context Classifier.
    Predicts operational motion regimes across long sequences using neural or rule-based models.
    """

    def __init__(self,
                 weights_path: Optional[str] = None,
                 window_size: int = 100,
                 step_size: int = 25,
                 dt: float = 0.01,
                 device: Optional[str] = None):
        self.window_size = window_size
        self.step_size = step_size
        self.dt = dt

        if TORCH_AVAILABLE:
            self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = MotionResNet1D(in_channels=6, num_classes=5).to(self.device)
            self.model.eval()

            if weights_path and os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[MotionClassifier] Loaded neural weights from {weights_path}")
            self.is_neural = True
        else:
            self.is_neural = False

    def predict_sequence(self,
                         acc: np.ndarray,
                         gyro: np.ndarray,
                         use_neural: bool = True) -> Dict[str, Any]:
        """
        Classify an entire IMU sequence using overlapping sliding windows.
        Returns:
            - context_per_sample: (N,) MotionContext array for each timestep
            - context_probabilities: (N, 5) per-class confidence scores
            - dominant_context: Overall most frequent MotionContext in sequence
        """
        acc = np.asarray(acc, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        N = len(acc)
        W = self.window_size
        step = self.step_size

        sample_counts = np.zeros(N, dtype=np.float32)
        sample_probs = np.zeros((N, 5), dtype=np.float32)

        # Slide windows
        for start in range(0, N, step):
            end = min(N, start + W)
            w_acc = acc[start:end]
            w_gyro = gyro[start:end]
            win_len = len(w_acc)
            if win_len < 10:
                continue

            if use_neural and self.is_neural and win_len == W:
                # 1D CNN Inference
                feat = np.hstack([w_acc, w_gyro]).T[np.newaxis, ...]  # (1, 6, W)
                with torch.no_grad():
                    tensor_in = torch.from_numpy(feat).to(self.device)
                    logits = self.model(tensor_in)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            else:
                # Rule-based inference fallback
                ctx, conf = classify_motion_context(w_acc, w_gyro, dt=self.dt)
                probs = np.zeros(5, dtype=np.float32)
                probs[ctx.value] = conf
                # Distribute remaining prob
                rem = (1.0 - conf) / 4.0
                for c in range(5):
                    if c != ctx.value:
                        probs[c] = rem

            # Accumulate
            sample_probs[start:end] += probs
            sample_counts[start:end] += 1.0

        # Normalize
        sample_counts = np.where(sample_counts < 1.0, 1.0, sample_counts)
        sample_probs = sample_probs / sample_counts[:, np.newaxis]

        # Discrete labels
        sample_labels = np.argmax(sample_probs, axis=-1)
        contexts = np.array([MotionContext(l) for l in sample_labels])

        # Dominant mode
        dominant_label = int(np.argmax(np.bincount(sample_labels)))
        dominant_ctx = MotionContext(dominant_label)

        return {
            "contexts": contexts,
            "probabilities": sample_probs,
            "dominant_context": dominant_ctx,
            "dominant_label": dominant_ctx.label_name,
        }
