"""
faults_synergistic.py

LiDAR synergistic faults: two faults from other categories acting together.
A synergy is a declarative list of components applied in order, so adding one
needs no new fault function. Propagation faults (environment) come first and
sensor-side faults after, so the sensor acts on the already degraded signal.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from common.lidar_loader import PointCloud
from .faults_calibration import apply_fault_calibration
from .faults_environment import apply_fault_environment
from .faults_hardware import apply_fault_hardware
from .faults_software import apply_fault_software

_ROUTERS = {
    "hardware": apply_fault_hardware,
    "environment": apply_fault_environment,
    "calibration": apply_fault_calibration,
    "software": apply_fault_software,
}

# synergy -> [(family, kind, persistent)]. Persistent components model a
# standing condition (same pattern across a burst, seeded by `seed`); the
# others draw fresh randomness for every frame (seeded by `frame_seed`).
SYNERGIES: Dict[str, List[Tuple[str, str, bool]]] = {
    "scanning_fog": [("environment", "environmental_scatter", False), ("hardware", "scanning", True)],
    "mounting_fog": [("environment", "environmental_scatter", False), ("calibration", "mounting", True)],
    "drift_crosstalk": [("environment", "crosstalk", False), ("calibration", "thermal_drift", True)],
}

# Components whose functions take no seed (deterministic).
_SEEDLESS = {
    ("hardware", "laser"), ("hardware", "electrical"), ("software", "range_gating"),
    ("software", "silent_failure"), ("environment", "incomplete_echo"),
}


def synergistic_fault(cloud: PointCloud, kind: str, severity: str = "severe",
                      seed: Optional[int] = None, frame_seed: Optional[int] = None) -> PointCloud:
    """
    Apply the components of a synergy in order, all at the same severity.
    Component seeds are derived deterministically from `seed` (persistent
    components) and `frame_seed` (transient ones, falling back to `seed`).
    """
    components = SYNERGIES[kind]
    persistent_seeds = np.random.SeedSequence(seed).generate_state(len(components))
    transient_seeds = np.random.SeedSequence(
        seed if frame_seed is None else frame_seed).generate_state(len(components))

    for i, (family, component, persistent) in enumerate(components):
        kwargs = {"severity": severity}
        if (family, component) not in _SEEDLESS:
            kwargs["seed"] = int(persistent_seeds[i] if persistent else transient_seeds[i])
        cloud = _ROUTERS[family](cloud, component, **kwargs)
    return cloud


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_synergistic(cloud: PointCloud, kind: str, **kwargs) -> PointCloud:
    kind = kind.lower().strip()

    if kind not in SYNERGIES:
        raise ValueError(f"Unknown synergistic fault: '{kind}'")
    return synergistic_fault(cloud, kind, **kwargs)