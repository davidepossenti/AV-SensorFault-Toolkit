"""
faults_synergistic.py

Radar synergistic faults: two faults from other categories acting together.
A synergy is a declarative list of components applied in order, so adding one
needs no new fault function. The order is part of the definition: in
"thermal_multipath" the housing deformation acts first and the multipath ghosts
are generated from the already displaced echoes; in "temporal_multipath" the
ghosts are generated from the delayed frame.

The temporal component needs the frame sequence: radar_injector.py supplies the
delayed frame (see TEMPORAL_SYNERGIES) and the per-frame function skips it.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from common.radar_loader import RadarFrame
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

# Component handled by the injector instead of a per-frame function.
_TEMPORAL = ("software", "temporal_misalignment")

# synergy -> [(family, kind, persistent)]. Persistent components model a
# standing condition (same pattern across a burst, seeded by `seed`); the
# others draw fresh randomness for every frame (seeded by `frame_seed`).
SYNERGIES: Dict[str, List[Tuple[str, str, bool]]] = {
    "thermal_multipath": [("hardware", "thermal", True), ("environment", "multipath", False)],
    "temporal_multipath": [("software", "temporal_misalignment", False), ("environment", "multipath", False)],
    "drift_interference": [("calibration", "thermal_drift", True), ("environment", "interference", False)],
}

# Synergies whose input is the frame delivered `delay` frames earlier.
TEMPORAL_SYNERGIES = {
    kind for kind, components in SYNERGIES.items()
    if any((family, component) == _TEMPORAL for family, component, _ in components)
}


def synergistic_fault(frame: RadarFrame, kind: str, severity: str = "severe",
                      seed: Optional[int] = None, frame_seed: Optional[int] = None) -> RadarFrame:
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
        if (family, component) == _TEMPORAL:
            continue  # the injector already supplied the delayed frame
        component_seed = int(persistent_seeds[i] if persistent else transient_seeds[i])
        frame = _ROUTERS[family](frame, component, severity=severity, seed=component_seed)
    return frame


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_synergistic(frame: RadarFrame, kind: str, **kwargs) -> RadarFrame:
    kind = kind.lower().strip()

    if kind not in SYNERGIES:
        raise ValueError(f"Unknown synergistic fault: '{kind}'")
    return synergistic_fault(frame, kind, **kwargs)