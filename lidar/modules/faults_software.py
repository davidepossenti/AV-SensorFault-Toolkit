"""
faults_software.py

LiDAR firmware and signal-processing faults: packet loss, wrong range gating
and silent failure of the acquisition chain.
Each function takes a clean PointCloud and returns a degraded copy.

Note: the temporal "temporal_misalignment" fault (frames delivered late) needs
the whole frame sequence, so lidar_injector.py handles it at the scheduling
level.
"""

from typing import Optional

import numpy as np

from common.lidar_loader import REFERENCE_RANGE_M, PointCloud, spherical_angles


# ---------------------------------------------------------------------------
# Data handling: packet loss
# ---------------------------------------------------------------------------

# Azimuth slice lost as a unit [deg]: assumed width of a data burst, sensor specific.
_SECTOR_WIDTH_DEG = 3.0

# Fraction of the revolution lost per severity.
_DATA_LOSS_FRACTION = {
    "mild": 0.02,
    "moderate": 0.05,
    "severe": 0.10,
    "extreme": 0.20,
}

# Delay of the delivered frame [frames] per severity (0.1 s to 0.8 s at 10 Hz).
_DELAY_FRAMES = {
    "mild": 1,
    "moderate": 2,
    "severe": 4,
    "extreme": 8,
}


def packet_loss(cloud: PointCloud, severity: str = "severe",
                seed: Optional[int] = None) -> PointCloud:
    """
    Simulate packet loss in the acquisition pipeline: randomly chosen azimuth
    slices of the revolution are missing for all beams. Transient: use a
    different seed for every frame.
    """
    severity = str(severity).lower().strip()
    if severity not in _DATA_LOSS_FRACTION:
        raise ValueError(f"packet_loss: unknown severity '{severity}'")
    fraction = _DATA_LOSS_FRACTION[severity]

    if cloud.n_points == 0:
        return cloud

    n_sectors = int(round(360.0 / _SECTOR_WIDTH_DEG))
    n_lost = int(round(fraction * n_sectors))
    if n_lost == 0:
        return cloud

    rng = np.random.default_rng(seed)
    width = 2.0 * np.pi / n_sectors
    offset = rng.uniform(0.0, width)
    lost = rng.choice(n_sectors, size=n_lost, replace=False)

    _, theta, _ = spherical_angles(cloud.xyz)
    sector = np.floor((theta + np.pi - offset) / width).astype(np.int64) % n_sectors
    return cloud.subset(~np.isin(sector, lost))


# ---------------------------------------------------------------------------
# Internal filtering: wrong range gate
# ---------------------------------------------------------------------------

# (gate_start_fraction, gate_width_fraction) of REFERENCE_RANGE_M per severity.
_RANGE_GATING_PRESETS = {
    "mild":     (0.10, 0.05),
    "moderate": (0.10, 0.10),
    "severe":   (0.10, 0.20),
    "extreme":  (0.10, 0.40),
}


def range_gating_fault(cloud: PointCloud, severity: str = "severe") -> PointCloud:
    """
    Simulate a misconfigured range gate in the internal filter: every return
    inside the gated range band is discarded, leaving an empty ring around the
    sensor. Absolute band (metres), independent of scene density.
    Deterministic given severity.
    """
    severity = str(severity).lower().strip()
    if severity not in _RANGE_GATING_PRESETS:
        raise ValueError(f"range_gating_fault: unknown severity '{severity}'")
    start_fraction, width_fraction = _RANGE_GATING_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    r_start = start_fraction * REFERENCE_RANGE_M
    r_end = r_start + width_fraction * REFERENCE_RANGE_M
    r = np.linalg.norm(cloud.xyz, axis=1)
    return cloud.subset(~((r >= r_start) & (r < r_end)))


# ---------------------------------------------------------------------------
# Silent failure
# ---------------------------------------------------------------------------

def silent_failure(cloud: PointCloud) -> PointCloud:
    """
    Simulate a crashed driver or communication interface: the frame is still
    delivered but without any point. Binary fault.
    """
    return cloud.subset(np.zeros(cloud.n_points, dtype=bool))


def temporal_delay_frames(severity: str) -> int:
    """Delay [frames] of the temporal_misalignment fault, used by lidar_injector.py."""
    severity = str(severity).lower().strip()
    if severity not in _DELAY_FRAMES:
        raise ValueError(f"temporal_misalignment: unknown severity '{severity}'")
    return _DELAY_FRAMES[severity]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_software(cloud: PointCloud, kind: str, **kwargs) -> PointCloud:
    kind = kind.lower().strip()

    dispatch = {
        "data_loss": lambda: packet_loss(cloud, **kwargs),
        "range_gating": lambda: range_gating_fault(cloud, **kwargs),
        "silent_failure": lambda: silent_failure(cloud),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown software fault: '{kind}'")
    return dispatch[kind]()