"""The on-disk log schema shared by the Android recorder and this pipeline.

One recording is a directory containing plain CSV files. CSV is deliberate: it
survives being emailed around a hackathon team, opens in Excel for a sanity
check, and needs no library on the Android side.

    <recording>/
        meta.json      run metadata (device, rates, origin, mode)
        imu.csv        t_ns, ax, ay, az, gx, gy, gz            (~200 Hz)
        gnss.csv       t_ns, lat, lon, alt, speed, bearing, acc, nsat, cn0, hdop
        gnss_raw.csv   t_ns, svid, constellation, cn0, pseudorange_rate, ...
        baro.csv       t_ns, pressure_hpa
        truth.csv      t_ns, e, n, u, ve, vn, vu, heading      (synthetic only)

Positions in ``truth.csv`` and everything the filter produces live in the local
**ENU** tangent frame whose origin is recorded in ``meta.json``. Converting
GNSS lat/lon into that frame is the job of :func:`geodetic_to_enu`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# WGS-84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2 - _F)

IMU_COLUMNS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]
GNSS_COLUMNS = ["t", "e", "n", "u", "ve", "vn", "vu", "nsat", "cn0", "hdop", "acc"]
TRUTH_COLUMNS = ["t", "e", "n", "u", "ve", "vn", "vu", "heading"]


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert WGS-84 geodetic coordinates to a local ENU tangent plane.

    Accurate to well under a centimetre over the few-kilometre extent of any
    realistic recording, which is far below the errors we care about.
    """
    lat, lon, alt = np.asarray(lat), np.asarray(lon), np.asarray(alt)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat0_r, lon0_r = np.radians(lat0), np.radians(lon0)

    def _to_ecef(la, lo, h):
        s = np.sin(la)
        N = _A / np.sqrt(1 - _E2 * s * s)
        return (
            (N + h) * np.cos(la) * np.cos(lo),
            (N + h) * np.cos(la) * np.sin(lo),
            (N * (1 - _E2) + h) * s,
        )

    x, y, z = _to_ecef(lat_r, lon_r, alt)
    x0, y0, z0 = _to_ecef(lat0_r, lon0_r, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0

    sl, cl = np.sin(lat0_r), np.cos(lat0_r)
    so, co = np.sin(lon0_r), np.cos(lon0_r)
    east = -so * dx + co * dy
    north = -sl * co * dx - sl * so * dy + cl * dz
    up = cl * co * dx + cl * so * dy + sl * dz
    return np.stack([east, north, up], axis=-1)


@dataclass
class Meta:
    """Everything about a recording that is not a time series."""

    mode: str = "vehicle"           # "vehicle" | "pedestrian"
    source: str = "synthetic"       # "synthetic" | "android"
    imu_rate_hz: float = 200.0
    gnss_rate_hz: float = 1.0
    lat0: float = 12.9716           # default origin: Bengaluru
    lon0: float = 77.5946
    alt0: float = 920.0
    device: str = "simulator"
    notes: str = ""
    mount: list | None = None       # synthetic only: true device->vehicle rotation
    events: list = field(default_factory=list)  # [{name, t_start, t_end, kind}]


@dataclass
class Recording:
    """An in-memory recording: aligned time series plus metadata."""

    meta: Meta
    imu: np.ndarray                 # (N, 7)  t, ax..az, gx..gz
    gnss: np.ndarray                # (M, 11) see GNSS_COLUMNS
    truth: np.ndarray | None = None  # (N, 8)  see TRUTH_COLUMNS

    # -- convenient views -------------------------------------------------
    @property
    def t(self):
        return self.imu[:, 0]

    @property
    def accel(self):
        return self.imu[:, 1:4]

    @property
    def gyro(self):
        return self.imu[:, 4:7]

    @property
    def truth_pos(self):
        return None if self.truth is None else self.truth[:, 1:4]

    @property
    def truth_vel(self):
        return None if self.truth is None else self.truth[:, 4:7]

    # -- io ---------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))
        _write_csv(path / "imu.csv", IMU_COLUMNS, self.imu)
        _write_csv(path / "gnss.csv", GNSS_COLUMNS, self.gnss)
        if self.truth is not None:
            _write_csv(path / "truth.csv", TRUTH_COLUMNS, self.truth)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        path = Path(path)
        meta = Meta(**json.loads((path / "meta.json").read_text()))
        imu = _read_csv(path / "imu.csv")
        gnss = _read_csv(path / "gnss.csv")
        truth_file = path / "truth.csv"
        truth = _read_csv(truth_file) if truth_file.exists() else None
        return cls(meta=meta, imu=imu, gnss=gnss, truth=truth)


def _write_csv(path: Path, header: list[str], data: np.ndarray) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(np.asarray(data).tolist())


def _read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
