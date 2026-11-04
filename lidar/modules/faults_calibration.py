"""
faults_calibration.py

Fault injection functions for LiDAR calibration faults: extrinsic mounting
misalignment relative to the vehicle reference frame and thermal expansion
drift.
Each function takes a clean PointCloud and returns a degraded copy.
"""

from typing import Optional

import numpy as np

from common.lidar_loader import (
    PointCloud,
    rotation_matrix_yaw_pitch_roll,
    spherical_angles,
    spherical_to_cartesian,
)

# (max_yaw_deg, max_pitch_deg, max_roll_deg) per severity level.
_MOUNTING_MISALIGNMENT_PRESETS = {
    "mild":     (1.0, 0.5, 0.5),
    "moderate": (2.0, 1.0, 1.0),
    "severe":   (4.0, 2.0, 2.0),
    "extreme":  (6.0, 3.0, 3.0),
}

# (range_bias_m, yaw_offset_deg) per severity. Heuristic values.
_THERMAL_DRIFT_PRESETS = {
    "mild":     (0.02, 0.10),
    "moderate": (0.05, 0.25),
    "severe":   (0.10, 0.50),
    "extreme":  (0.20, 1.00),
}


def mounting_misalignment(cloud: PointCloud, severity: str = "severe",
                          seed: Optional[int] = None) -> PointCloud:
    """
    Simulate a persistent extrinsic misalignment (e.g. bracket loosening or
    vibration-induced drift): the entire point cloud is rigidly rotated
    relative to the vehicle reference frame. Point count is unaffected.
    """
    severity = str(severity).lower().strip()
    if severity not in _MOUNTING_MISALIGNMENT_PRESETS:
        raise ValueError(f"mounting_misalignment: unknown severity '{severity}'")
    max_yaw, max_pitch, max_roll = _MOUNTING_MISALIGNMENT_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    yaw = np.deg2rad(max_yaw * (0.5 + rng.random()) * rng.choice([-1, 1]))
    pitch = np.deg2rad(max_pitch * (0.5 + rng.random()) * rng.choice([-1, 1]))
    roll = np.deg2rad(max_roll * (0.5 + rng.random()) * rng.choice([-1, 1]))

    rotation = rotation_matrix_yaw_pitch_roll(yaw, pitch, roll)
    xyz_rotated = (rotation @ cloud.xyz.T).T.astype(np.float32)

    return PointCloud(xyz_rotated, cloud.intensity, cloud.extra)


def thermal_expansion_drift(cloud: PointCloud, severity: str = "severe",
                            seed: Optional[int] = None) -> PointCloud:
    """
    Simulate a slow thermal drift of the mounting and electronics: every range
    gains a constant bias and the cloud rotates in yaw (sign drawn from the
    seed). Use one seed per burst, since the drift persists, and
    --progressive-degradation to ramp it over the burst.
    """
    severity = str(severity).lower().strip()
    if severity not in _THERMAL_DRIFT_PRESETS:
        raise ValueError(f"thermal_expansion_drift: unknown severity '{severity}'")
    bias_m, yaw_deg = _THERMAL_DRIFT_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    sign = np.random.default_rng(seed).choice([-1.0, 1.0])
    r, theta, elevation = spherical_angles(cloud.xyz)
    xyz = spherical_to_cartesian(r + bias_m, theta + sign * np.deg2rad(yaw_deg), elevation)
    return PointCloud(xyz, cloud.intensity, cloud.extra)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_calibration(cloud: PointCloud, kind: str, **kwargs) -> PointCloud:
    kind = kind.lower().strip()

    dispatch = {
        "mounting": lambda: mounting_misalignment(cloud, **kwargs),
        "thermal_drift": lambda: thermal_expansion_drift(cloud, **kwargs),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown calibration fault: '{kind}'")
    return dispatch[kind]()