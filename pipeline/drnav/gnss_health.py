"""GNSS health classification -- deciding when *not* to trust a fix.

The hard case in Indian cities is not the tunnel, where GNSS obviously
disappears. It is the urban canyon, where the receiver keeps producing fixes
that look perfectly healthy and are 30 metres onto the wrong road. A DR system
that blindly accepts those is worse than useless, because it drags a good
inertial solution off the road with it.

Two complementary signals are used:

1. **Signal-quality features** the receiver already reports -- satellite count,
   mean C/N0, HDOP, claimed accuracy. Cheap, available every fix, and enough to
   spot a degraded environment.
2. **Innovation consistency** against the filter's own prediction. A fix that
   disagrees with a well-converged inertial solution by many sigma is rejected
   regardless of how healthy it claims to be. This is what actually catches
   multipath.

``classify`` is deliberately a transparent rule set: it is the baseline an ML
classifier must beat, and it gives the team a working system on day one.
Swap in a learned model behind the same signature once labelled data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GnssHealth(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


@dataclass
class GnssFeatures:
    nsat: float
    cn0: float
    hdop: float
    accuracy: float          # receiver-reported horizontal accuracy, m


# thresholds tuned against the synthetic OPEN_SKY / URBAN_CANYON models and
# broadly consistent with what Android reports in Indian cities
GOOD_MIN_SATS = 9
GOOD_MIN_CN0 = 32.0
GOOD_MAX_HDOP = 1.6
GOOD_MAX_ACC = 8.0


def classify(f: GnssFeatures) -> GnssHealth:
    """Rule-based health from signal-quality features alone."""
    if f.nsat <= 3 or not np.isfinite(f.accuracy):
        return GnssHealth.LOST
    good = (f.nsat >= GOOD_MIN_SATS and f.cn0 >= GOOD_MIN_CN0
            and f.hdop <= GOOD_MAX_HDOP and f.accuracy <= GOOD_MAX_ACC)
    return GnssHealth.GOOD if good else GnssHealth.DEGRADED


def inflation_factor(health: GnssHealth) -> float:
    """How much to inflate the reported position sigma before using a fix.

    Degraded fixes are not discarded -- a 30 m fix still beats unbounded
    inertial drift -- but they are de-weighted hard so they nudge rather than
    yank the solution.
    """
    return {GnssHealth.GOOD: 1.0, GnssHealth.DEGRADED: 4.0,
            GnssHealth.LOST: np.inf}[health]


class OutageTracker:
    """Bookkeeping for how long we have been dead reckoning.

    Drives the UI badge ("DEAD RECKONING 42 s") and lets metrics attribute
    error to specific outages.
    """

    def __init__(self, lost_after: float = 3.0):
        self.lost_after = lost_after      # seconds without a usable fix
        self.last_fix_t: float | None = None
        self.outages: list[list[float]] = []
        self._open = False

    def note_fix(self, t: float) -> None:
        self.last_fix_t = t
        if self._open:
            self.outages[-1][1] = t
            self._open = False

    def step(self, t: float) -> float:
        """Return seconds since the last usable fix, opening an outage record."""
        if self.last_fix_t is None:
            return 0.0
        gap = t - self.last_fix_t
        if gap > self.lost_after and not self._open:
            self.outages.append([self.last_fix_t, t])
            self._open = True
        elif self._open:
            self.outages[-1][1] = t
        return gap
