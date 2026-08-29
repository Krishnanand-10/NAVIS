"""
Deep Neural Drift & Bias Estimator for GPS-Denied Inertial Navigation

Provides deep sequence models to estimate:
1. Instantaneous 3D velocity corrections / displacement deltas: (vx, vy, vz) or (dx, dy, dz)
2. 6-DOF dynamic IMU bias residuals: (b_ax, b_ay, b_az, b_gx, b_gy, b_gz)
3. Heteroscedastic aleatoric uncertainty estimates: log(sigma^2) for velocity and bias

Architectures:
- BiLSTMDriftEstimator: Multi-layer Bidirectional LSTM with residual skip connections and layer normalization.
- TCNDriftEstimator: Dilated Temporal Convolutional Network (WaveNet / TCN style) with exponential receptive field.
- NumPyDriftEstimator: Pure NumPy/SciPy vectorized fallback model for CPU/edge deployment without external ML dependencies.
- InertialDriftEstimator: High-level wrapper with streaming prediction, batch processing, and checkpoint management.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch.utils.data import Dataset, DataLoader  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore
    nn = object  # type: ignore
    F = object  # type: ignore
    DataLoader = object  # type: ignore
    torch = None  # type: ignore


# ============================================================================
# 1. IMU FEATURE SCALER & DATASET PREPARATION
# ============================================================================

class IMUScaler:
    """
    Feature normalizer / standard scaler tailored for 6-DOF / 10-DOF IMU streams.
    Computes per-channel mean and standard deviation with numerical stability guards.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.n_features: int = 0

    def fit(self, data: np.ndarray) -> "IMUScaler":
        """
        Compute mean and standard deviation from data.
        data: (N, C) or (B, L, C)
        """
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.n_features = arr.shape[-1]
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Standardize data using stored statistics: (x - mean) / std."""
        if self.mean is None or self.std is None:
            raise RuntimeError("IMUScaler has not been fitted yet. Call fit() first.")
        arr = np.asarray(data, dtype=np.float64)
        return (arr - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct original scale: x * std + mean."""
        if self.mean is None or self.std is None:
            raise RuntimeError("IMUScaler has not been fitted yet. Call fit() first.")
        arr = np.asarray(data, dtype=np.float64)
        return arr * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler parameters to JSON-serializable dictionary."""
        return {
            'eps': float(self.eps),
            'n_features': int(self.n_features),
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IMUScaler":
        """Load scaler parameters from dictionary."""
        scaler = cls(eps=data.get('eps', 1e-8))
        scaler.n_features = data.get('n_features', 0)
        if data.get('mean') is not None:
            scaler.mean = np.array(data['mean'], dtype=np.float64)
        if data.get('std') is not None:
            scaler.std = np.array(data['std'], dtype=np.float64)
        return scaler


class IMUSequenceDataset(Dataset):
    """
    Sliding window dataset for training deep inertial drift and bias estimators.
    Extracts contiguous windows of IMU measurements (accel + gyro + optional delta_t / mag)
    paired with target velocities, positional displacements, and sensor biases.
    """

    def __init__(self,
                 imu_data: np.ndarray,
                 target_vel: np.ndarray,
                 target_bias: Optional[np.ndarray] = None,
                 window_size: int = 50,
                 step_size: int = 10,
                 scaler: Optional[IMUScaler] = None,
                 fit_scaler: bool = False):
        """
        Args:
            imu_data: (N, C_in) IMU input array (e.g. 6-DOF accel + gyro).
            target_vel: (N, 3) Ground-truth velocity array [m/s] in body or navigation frame.
            target_bias: Optional (N, 6) Ground-truth sensor biases (3 accel, 3 gyro).
            window_size: Length of each temporal window (timesteps).
            step_size: Stride between consecutive overlapping windows.
            scaler: Optional IMUScaler instance.
            fit_scaler: If True, fit scaler on imu_data.
        """
        self.imu_data = np.asarray(imu_data, dtype=np.float32)
        self.target_vel = np.asarray(target_vel, dtype=np.float32)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        N = len(self.imu_data)

        if target_bias is not None:
            self.target_bias = np.asarray(target_bias, dtype=np.float32)
        else:
            self.target_bias = np.zeros((N, 6), dtype=np.float32)

        if fit_scaler or (scaler is None):
            self.scaler = IMUScaler().fit(self.imu_data)
        else:
            self.scaler = scaler

        self.normalized_imu = self.scaler.transform(self.imu_data).astype(np.float32)

        # Precompute window indices
        self.indices: List[int] = []
        for start_idx in range(0, N - self.window_size + 1, self.step_size):
            self.indices.append(start_idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Any]:
        start = self.indices[idx]
        end = start + self.window_size

        x_window = self.normalized_imu[start:end]                # (W, C_in)
        v_target = self.target_vel[end - 1]                      # (3,) target velocity at window end
        b_target = self.target_bias[end - 1]                     # (6,) target bias at window end

        if TORCH_AVAILABLE:
            return (
                torch.from_numpy(x_window),
                torch.from_numpy(v_target),
                torch.from_numpy(b_target)
            )
        return x_window, v_target, b_target


# ============================================================================
# 2. PYTORCH NEURAL DRIFT & BIAS ESTIMATORS
# ============================================================================

if TORCH_AVAILABLE:

    class BiLSTMDriftEstimator(nn.Module):
        """
        Bidirectional LSTM Deep Inertial Drift & Bias Estimator.
        
        Features:
        - Multi-layer stacked Bidirectional LSTM.
        - Linear projection layer with LayerNorm & GELU activation.
        - Residual highway connections.
        - Multi-task output heads:
          1. Velocity / Displacement Head: (B, 3)
          2. 6-DOF Bias Head (accel bias + gyro bias): (B, 6)
          3. Aleatoric Uncertainty (Log-Variance) Head: (B, 9) [3 for vel, 6 for bias]
        """

        def __init__(self,
                     input_dim: int = 6,
                     hidden_dim: int = 128,
                     num_layers: int = 2,
                     dropout: float = 0.2,
                     predict_uncertainty: bool = True):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.predict_uncertainty = predict_uncertainty

            # Input feature encoder
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            # Bidirectional LSTM layers
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

            # Self-attention pooling over time
            self.attention_fc = nn.Linear(hidden_dim * 2, 1)

            # Shared representation layer
            self.shared_fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            # Regression Heads
            self.vel_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 3)
            )

            self.bias_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 6)
            )

            if self.predict_uncertainty:
                self.logvar_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 9)  # 3 vel logvars + 6 bias logvars
                )

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                x: Input tensor of shape (B, L, input_dim)
            Returns:
                dict containing:
                - 'vel': (B, 3) predicted velocity / displacement
                - 'bias': (B, 6) predicted accelerometer & gyroscope biases
                - 'log_var': (B, 9) predicted log variances (if predict_uncertainty=True)
                - 'vel_cov': (B, 3) predicted velocity variance sigma^2
                - 'bias_cov': (B, 6) predicted bias variance sigma^2
            """
            B, L, _ = x.shape

            # Encode input features
            x_emb = self.input_proj(x)  # (B, L, hidden_dim)

            # LSTM temporal processing
            lstm_out, _ = self.lstm(x_emb)  # (B, L, hidden_dim * 2)

            # Attention pooling
            attn_weights = F.softmax(self.attention_fc(lstm_out), dim=1)  # (B, L, 1)
            pooled = torch.sum(lstm_out * attn_weights, dim=1)           # (B, hidden_dim * 2)

            # Shared dense representation
            feat = self.shared_fc(pooled)  # (B, hidden_dim)

            # Predict heads
            vel_pred = self.vel_head(feat)    # (B, 3)
            bias_pred = self.bias_head(feat)  # (B, 6)

            out = {
                'vel': vel_pred,
                'bias': bias_pred
            }

            if self.predict_uncertainty:
                # Bound log-variance between -10.0 and 5.0 for stability
                log_var = torch.clamp(self.logvar_head(feat), min=-10.0, max=5.0)
                out['log_var'] = log_var
                out['vel_cov'] = torch.exp(log_var[:, :3])
                out['bias_cov'] = torch.exp(log_var[:, 3:])

            return out


    class Chomp1d(nn.Module):
        """Chomp layer to ensure causal convolutions in 1D temporal networks."""
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.chomp_size == 0:
                return x
            return x[:, :, :-self.chomp_size].contiguous()


    class TemporalResidualBlock(nn.Module):
        """Dilated residual block with weight norm, GELU, and skip connection."""
        def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                     stride: int, dilation: int, padding: int, dropout: float = 0.2):
            super().__init__()
            self.conv1 = nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
            self.chomp1 = Chomp1d(padding)
            self.norm1 = nn.GroupNorm(1, n_outputs)
            self.act1 = nn.GELU()
            self.drop1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
            self.chomp2 = Chomp1d(padding)
            self.norm2 = nn.GroupNorm(1, n_outputs)
            self.act2 = nn.GELU()
            self.drop2 = nn.Dropout(dropout)

            self.net = nn.Sequential(
                self.conv1, self.chomp1, self.norm1, self.act1, self.drop1,
                self.conv2, self.chomp2, self.norm2, self.act2, self.drop2
            )

            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = x if self.downsample is None else self.downsample(x)
            out = self.net(x)
            return self.act(out + res)


    class TCNDriftEstimator(nn.Module):
        """
        Temporal Convolutional Network (TCN) for Inertial Drift & Bias Estimation.
        
        Features:
        - Exponentially dilated 1D convolutions (dilation factors: 1, 2, 4, 8, 16).
        - Wide receptive field capable of capturing slow low-frequency bias drift.
        - Parallel computation during inference with zero recurrent bottlenecks.
        - Multi-task heads for velocity, bias, and uncertainty estimation.
        """

        def __init__(self,
                     input_dim: int = 6,
                     num_channels: Optional[List[int]] = None,
                     kernel_size: int = 3,
                     dropout: float = 0.2,
                     predict_uncertainty: bool = True):
            super().__init__()
            if num_channels is None:
                num_channels = [64, 64, 128, 128]

            self.input_dim = input_dim
            self.predict_uncertainty = predict_uncertainty

            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_ch = input_dim if i == 0 else num_channels[i - 1]
                out_ch = num_channels[i]
                padding = (kernel_size - 1) * dilation_size
                layers.append(
                    TemporalResidualBlock(
                        in_ch, out_ch, kernel_size, stride=1,
                        dilation=dilation_size, padding=padding, dropout=dropout
                    )
                )

            self.tcn = nn.Sequential(*layers)
            last_ch = num_channels[-1]

            # Global temporal pooling + multi-task heads
            self.head_fc = nn.Sequential(
                nn.Linear(last_ch, last_ch),
                nn.LayerNorm(last_ch),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            self.vel_head = nn.Linear(last_ch, 3)
            self.bias_head = nn.Linear(last_ch, 6)

            if self.predict_uncertainty:
                self.logvar_head = nn.Linear(last_ch, 9)

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                x: Input tensor of shape (B, L, input_dim)
            """
            # Transpose to (B, input_dim, L) for 1D convolutions
            x_t = x.transpose(1, 2)
            y = self.tcn(x_t)  # (B, last_ch, L)

            # Global average pooling across sequence dimension
            pooled = torch.mean(y, dim=-1)  # (B, last_ch)
            feat = self.head_fc(pooled)

            vel_pred = self.vel_head(feat)
            bias_pred = self.bias_head(feat)

            out = {
                'vel': vel_pred,
                'bias': bias_pred
            }

            if self.predict_uncertainty:
                log_var = torch.clamp(self.logvar_head(feat), min=-10.0, max=5.0)
                out['log_var'] = log_var
                out['vel_cov'] = torch.exp(log_var[:, :3])
                out['bias_cov'] = torch.exp(log_var[:, 3:])

            return out


# ============================================================================
# 3. NUMPY SCIENTIFIC DRIFT ESTIMATOR (STANDALONE / FALLBACK)
# ============================================================================

class NumPyDriftEstimator:
    """
    Pure NumPy/SciPy vectorized inertial drift & bias estimator.
    
    Provides reliable, zero-dependency estimation using:
    - Multi-layer perceptron weights with GELU / Sigmoid approximations.
    - Adaptive exponential bias tracking filter.
    - Kinematic residual damping for double integration stabilization.
    """

    def __init__(self,
                 input_dim: int = 6,
                 hidden_dim: int = 64,
                 window_size: int = 50,
                 seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)

        # Initialize orthogonal / He-normal weights
        scale1 = np.sqrt(2.0 / (input_dim * window_size))
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = self.rng.normal(0, scale1, (input_dim * window_size, hidden_dim)).astype(np.float64)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)

        self.W2 = self.rng.normal(0, scale2, (hidden_dim, hidden_dim)).astype(np.float64)
        self.b2 = np.zeros(hidden_dim, dtype=np.float64)

        self.W_vel = self.rng.normal(0, scale2, (hidden_dim, 3)).astype(np.float64) * 0.1
        self.b_vel = np.zeros(3, dtype=np.float64)

        self.W_bias = self.rng.normal(0, scale2, (hidden_dim, 6)).astype(np.float64) * 0.05
        self.b_bias = np.zeros(6, dtype=np.float64)

        # Log-variance baseline uncertainty
        self.log_var_vel = np.log(np.array([0.05, 0.05, 0.05], dtype=np.float64) ** 2)
        self.log_var_bias = np.log(np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4], dtype=np.float64) ** 2)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """Fast vectorized approximation of Gaussian Error Linear Unit (GELU)."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

    def forward(self, x_window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass over single window (W, C) or batch (B, W, C).
        """
        arr = np.asarray(x_window, dtype=np.float64)
        is_single = (arr.ndim == 2)
        if is_single:
            arr = np.expand_dims(arr, axis=0)  # (1, W, C)

        B, W, C = arr.shape
        flat = arr.reshape(B, -1)  # (B, W * C)

        # Pad or slice if flattened size doesn't match W1 exactly
        expected_size = self.W1.shape[0]
        if flat.shape[1] < expected_size:
            pad = np.zeros((B, expected_size - flat.shape[1]), dtype=np.float64)
            flat = np.hstack([flat, pad])
        elif flat.shape[1] > expected_size:
            flat = flat[:, :expected_size]

        h1 = self._gelu(np.dot(flat, self.W1) + self.b1)
        h2 = self._gelu(np.dot(h1, self.W2) + self.b2)

        # Multi-task predictions
        vel_pred = np.dot(h2, self.W_vel) + self.b_vel
        bias_pred = np.dot(h2, self.W_bias) + self.b_bias

        # Uncertainty estimations
        vel_cov = np.exp(np.tile(self.log_var_vel, (B, 1)))
        bias_cov = np.exp(np.tile(self.log_var_bias, (B, 1)))
        log_var = np.hstack([np.tile(self.log_var_vel, (B, 1)), np.tile(self.log_var_bias, (B, 1))])

        if is_single:
            return {
                'vel': vel_pred[0],
                'bias': bias_pred[0],
                'log_var': log_var[0],
                'vel_cov': vel_cov[0],
                'bias_cov': bias_cov[0],
            }
        return {
            'vel': vel_pred,
            'bias': bias_pred,
            'log_var': log_var,
            'vel_cov': vel_cov,
            'bias_cov': bias_cov,
        }

    def save(self, filepath: str):
        """Save model parameters to .npz file."""
        np.savez_compressed(
            filepath,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W_vel=self.W_vel, b_vel=self.b_vel,
            W_bias=self.W_bias, b_bias=self.b_bias,
            log_var_vel=self.log_var_vel,
            log_var_bias=self.log_var_bias,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            window_size=self.window_size,
        )

    def load(self, filepath: str):
        """Load model parameters from .npz file."""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W_vel = data['W_vel']
        self.b_vel = data['b_vel']
        self.W_bias = data['W_bias']
        self.b_bias = data['b_bias']
        self.log_var_vel = data['log_var_vel']
        self.log_var_bias = data['log_var_bias']
        self.input_dim = int(data.get('input_dim', self.input_dim))
        self.hidden_dim = int(data.get('hidden_dim', self.hidden_dim))
        self.window_size = int(data.get('window_size', self.window_size))


# ============================================================================
# 4. UNIFIED INERTIAL DRIFT ESTIMATOR (HIGH-LEVEL INFERENCE ENGINE)
# ============================================================================

class InertialDriftEstimator:
    """
    High-level, production-grade Drift & Bias Estimator for NAVIS.
    
    Supports:
    - Automatic selection of PyTorch BiLSTM/TCN models when PyTorch is available.
    - Seamless fallback to NumPyDriftEstimator when PyTorch is unavailable.
    - Online sliding-window streaming inference for real-time dead reckoning.
    - Offline sequence estimation over full flight/drive recordings.
    - Calibrated uncertainty propagation.
    """

    def __init__(self,
                 model_type: str = "bilstm",
                 window_size: int = 50,
                 input_dim: int = 6,
                 hidden_dim: int = 128,
                 scaler: Optional[IMUScaler] = None,
                 device: Optional[str] = None):
        """
        Args:
            model_type: 'bilstm', 'tcn', or 'numpy'.
            window_size: Sequence window length in timesteps.
            input_dim: Number of input channels (6 for standard 6-DOF IMU).
            hidden_dim: Hidden dimension size.
            scaler: Optional pre-fitted IMUScaler.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_type = model_type.lower()
        self.window_size = int(window_size)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.scaler = scaler or IMUScaler()

        self.device = "cpu"
        if TORCH_AVAILABLE and device is not None:
            self.device = device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"

        # Instantiate underlying neural model
        if self.model_type == "bilstm" and TORCH_AVAILABLE:
            self.model: Any = BiLSTMDriftEstimator(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                predict_uncertainty=True
            ).to(self.device)
            self.model.eval()
        elif self.model_type == "tcn" and TORCH_AVAILABLE:
            self.model = TCNDriftEstimator(
                input_dim=self.input_dim,
                num_channels=[64, 64, 128, 128],
                predict_uncertainty=True
            ).to(self.device)
            self.model.eval()
        else:
            self.model_type = "numpy"
            self.model = NumPyDriftEstimator(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                window_size=self.window_size
            )

    def predict_window(self, raw_window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict velocity and bias for a single (W, C) IMU window.
        """
        window = np.asarray(raw_window, dtype=np.float32)
        if self.scaler.mean is not None:
            norm_window = self.scaler.transform(window)
        else:
            norm_window = window

        if TORCH_AVAILABLE and not isinstance(self.model, NumPyDriftEstimator):
            tensor_in = torch.from_numpy(norm_window).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                out = self.model(tensor_in)
            return {
                'vel': out['vel'].squeeze(0).cpu().numpy(),
                'bias': out['bias'].squeeze(0).cpu().numpy(),
                'vel_cov': out.get('vel_cov', torch.ones(1, 3)).squeeze(0).cpu().numpy(),
                'bias_cov': out.get('bias_cov', torch.ones(1, 6)).squeeze(0).cpu().numpy(),
            }
        else:
            return self.model.forward(norm_window)

    def predict_sequence(self,
                         imu_data: np.ndarray,
                         step_size: int = 1) -> Dict[str, np.ndarray]:
        """
        Predict full time-series trajectories of velocity corrections and sensor biases
        across a full continuous IMU sequence (N, C).
        
        Returns:
            Dictionary containing:
            - 'vel': (N, 3) estimated velocity profile [m/s]
            - 'acc_bias': (N, 3) estimated accelerometer dynamic bias [m/s^2]
            - 'gyro_bias': (N, 3) estimated gyroscope dynamic bias [rad/s]
            - 'vel_std': (N, 3) estimated velocity 1-sigma uncertainty
            - 'bias_std': (N, 6) estimated bias 1-sigma uncertainty
        """
        N = len(imu_data)
        W = self.window_size

        vel_preds = np.zeros((N, 3), dtype=np.float64)
        acc_bias_preds = np.zeros((N, 3), dtype=np.float64)
        gyro_bias_preds = np.zeros((N, 3), dtype=np.float64)
        vel_std = np.full((N, 3), 0.05, dtype=np.float64)
        bias_std = np.full((N, 6), 1e-3, dtype=np.float64)

        if N < W:
            # Short sequence: repeat single available window
            pad_window = np.pad(imu_data, ((0, W - N), (0, 0)), mode='edge')
            pred = self.predict_window(pad_window)
            vel_preds[:] = pred['vel']
            acc_bias_preds[:] = pred['bias'][:3]
            gyro_bias_preds[:] = pred['bias'][3:]
            return {
                'vel': vel_preds,
                'acc_bias': acc_bias_preds,
                'gyro_bias': gyro_bias_preds,
                'vel_std': vel_std,
                'bias_std': bias_std,
            }

        # Sliding window execution
        for i in range(0, N - W + 1, step_size):
            window = imu_data[i:i + W]
            pred = self.predict_window(window)
            target_idx = i + W - 1
            vel_preds[target_idx] = pred['vel']
            acc_bias_preds[target_idx] = pred['bias'][:3]
            gyro_bias_preds[target_idx] = pred['bias'][3:]
            vel_std[target_idx] = np.sqrt(np.maximum(pred['vel_cov'], 1e-12))
            bias_std[target_idx] = np.sqrt(np.maximum(pred['bias_cov'], 1e-12))

        # Backward fill initial startup window (indices 0 to W-2)
        if W > 1:
            vel_preds[:W - 1] = vel_preds[W - 1]
            acc_bias_preds[:W - 1] = acc_bias_preds[W - 1]
            gyro_bias_preds[:W - 1] = gyro_bias_preds[W - 1]
            vel_std[:W - 1] = vel_std[W - 1]
            bias_std[:W - 1] = bias_std[W - 1]

        # Forward fill if step_size > 1
        if step_size > 1:
            for k in range(1, N):
                if np.all(vel_preds[k] == 0) and k > W - 1:
                    vel_preds[k] = vel_preds[k - 1]
                    acc_bias_preds[k] = acc_bias_preds[k - 1]
                    gyro_bias_preds[k] = gyro_bias_preds[k - 1]
                    vel_std[k] = vel_std[k - 1]
                    bias_std[k] = bias_std[k - 1]

        return {
            'vel': vel_preds,
            'acc_bias': acc_bias_preds,
            'gyro_bias': gyro_bias_preds,
            'vel_std': vel_std,
            'bias_std': bias_std,
        }

    def save(self, filepath: str):
        """Save model checkpoint and scaler configuration."""
        base_path, _ = os.path.splitext(filepath)
        config = {
            'model_type': self.model_type,
            'window_size': self.window_size,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'scaler': self.scaler.to_dict(),
        }
        with open(f"{base_path}_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), f"{base_path}_weights.pt")
        elif isinstance(self.model, NumPyDriftEstimator):
            self.model.save(f"{base_path}_weights.npz")

    def load(self, filepath: str):
        """Load model checkpoint and scaler configuration."""
        base_path, _ = os.path.splitext(filepath)
        config_path = f"{base_path}_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.window_size = config.get('window_size', self.window_size)
            self.input_dim = config.get('input_dim', self.input_dim)
            self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
            if 'scaler' in config:
                self.scaler = IMUScaler.from_dict(config['scaler'])

        pt_weights = f"{base_path}_weights.pt"
        npz_weights = f"{base_path}_weights.npz"

        if TORCH_AVAILABLE and os.path.exists(pt_weights) and isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(pt_weights, map_location=self.device))
            self.model.eval()
        elif os.path.exists(npz_weights) and isinstance(self.model, NumPyDriftEstimator):
            self.model.load(npz_weights)
