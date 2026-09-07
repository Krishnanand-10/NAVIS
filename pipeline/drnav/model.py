"""The learned inertial odometry network.

A dilated temporal convolutional network over a 2 s window of gravity-levelled
IMU, predicting ground speed **and its own uncertainty**.

Why regress speed rather than integrate acceleration: double integration turns
a constant accelerometer bias into an error growing as t squared, which is the
entire reason unaided dead reckoning fails inside a minute. A network reading
the *pattern* of a window -- gait cadence when walking, vibration and load
transfer when driving -- never integrates, so it never inherits that growth.
This is the RoNIN / IONet / TLIO family of approach.

Why predict uncertainty: the estimator this feeds is a Kalman filter, and a
measurement without a variance is useless to it. A heteroscedastic head lets
the model say "I am confident at a steady 60 km/h, and I have no idea what is
happening during this hard braking manoeuvre", and the filter weights it
accordingly. A single fixed sigma would force a compromise that is wrong in
both regimes.

Deliberately small -- 394k parameters, about 1.5 MB in float32 -- because this
has to run at 10 Hz on a mid-range phone alongside everything else.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import WINDOW_N
from .features import N_CHANNELS

# Predicted sigma is clamped to this band. The floor stops the network claiming
# certainty a MEMS sensor cannot support and driving the filter gain to
# infinity; the ceiling keeps a hopeless window from silently disabling the
# update instead of visibly failing.
LOG_VAR_MIN = 2.0 * torch.log(torch.tensor(0.05))   # sigma >= 0.05 m/s
LOG_VAR_MAX = 2.0 * torch.log(torch.tensor(8.0))    # sigma <= 8.0 m/s


class ResidualBlock(nn.Module):
    """Dilated conv pair with a residual connection."""

    def __init__(self, cin: int, cout: int, dilation: int, kernel: int = 5):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class SpeedNet(nn.Module):
    """(B, WINDOW_N, 6) -> speed in m/s and log-variance."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(N_CHANNELS, width, dilation=1),
            ResidualBlock(width, width, dilation=2),
            ResidualBlock(width, width * 2, dilation=4),
            ResidualBlock(width * 2, width * 2, dilation=8),
        )
        # Average pooling captures the sustained level, max pooling the peaks;
        # gait and load transfer live in the peaks, cruise lives in the average.
        self.head = nn.Sequential(
            nn.Linear(width * 4, 128), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(x.transpose(1, 2))            # (B, C, T)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        out = self.head(h)
        # softplus keeps speed non-negative without the dead gradient of a relu
        speed = F.softplus(out[:, 0])
        log_var = torch.clamp(out[:, 1], LOG_VAR_MIN.item(), LOG_VAR_MAX.item())
        return speed, log_var


def gaussian_nll(speed: torch.Tensor, log_var: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic negative log-likelihood.

    The ``log_var`` term is what stops the network escaping the loss by simply
    declaring maximum uncertainty everywhere: widening the predicted sigma is
    penalised directly, so it only pays where the error genuinely is large.
    """
    inv_var = torch.exp(-log_var)
    return (0.5 * (log_var + (target - speed) ** 2 * inv_var)).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
