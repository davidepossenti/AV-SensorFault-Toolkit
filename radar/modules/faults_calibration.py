"""
faults_calibration.py

Radar calibration faults: boresight misalignment, thermal expansion drift and
extrinsic misalignment.
Each function takes a RadarFrame and returns a degraded copy; the header
columns (including the reported azimuths) are never modified.
"""

from typing import Optional

import numpy as np

from common.radar_loader import RadarFrame, severity_preset, shift_rows

# Azimuth offset of the antenna boresight [deg] per severity.
_BORESIGHT_OFFSET_DEG = {
    "mild": 0.5,
    "moderate": 1.0,
    "severe": 2.0,
    "extreme": 4.0,
}

# (azimuth offset [deg], range bias [m]) of the thermal drift per severity.
# Heuristic values, not taken from a sensor specification.
_THERMAL_DRIFT_PRESETS = {
    "mild":     (0.25, 0.25),
    "moderate": (0.5, 0.5),
    "severe":   (1.0, 1.0),
    "extreme":  (2.0, 2.0),
}

# (yaw error [deg], translation error [m]) of the extrinsic misalignment per
# severity. Heuristic values, not taken from a sensor specification.
_EXTRINSIC_PRESETS = {
    "mild":     (0.5, 0.2),
    "moderate": (1.0, 0.4),
    "severe":   (2.0, 0.8),
    "extreme":  (4.0, 1.5),
}


def boresight_misalignment(frame: RadarFrame, severity: str = "severe",
                           seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate an azimuth offset between the antenna axis and its mounting: the
    echoes rotate by the offset while the reported azimuths stay put. The sign
    is drawn from the seed; use one seed per burst, since the offset persists.
    """
    offset_deg = severity_preset(_BORESIGHT_OFFSET_DEG, severity, "boresight_misalignment")
    sign = np.random.default_rng(seed).choice([-1.0, 1.0])

    shift = sign * np.deg2rad(offset_deg) / frame.azimuth_step_rad
    return frame.with_power(shift_rows(frame.power, shift))


def thermal_expansion_drift(frame: RadarFrame, severity: str = "severe",
                            seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate a slow thermal drift of the mounting: the echoes rotate by an
    azimuth offset (sign drawn from the seed) and appear farther by a range
    bias (expansion lengthens the path). Use one seed per burst, since the drift
    persists, and --progressive-degradation to ramp it over the burst.
    """
    offset_deg, bias_m = severity_preset(_THERMAL_DRIFT_PRESETS, severity, "thermal_expansion_drift")
    sign = np.random.default_rng(seed).choice([-1.0, 1.0])

    power = shift_rows(frame.power, sign * np.deg2rad(offset_deg) / frame.azimuth_step_rad)

    # Whole bins keep the noise statistics; the blind zone stays in place.
    bins = int(round(bias_m / frame.geometry.range_resolution_m))
    if bins > 0:
        shifted = power.copy()
        shifted[:, bins:] = power[:, :-bins]
        shifted[:, :frame.blind_bins] = power[:, :frame.blind_bins]
        power = shifted
    return frame.with_power(power)


def extrinsic_misalignment(frame: RadarFrame, severity: str = "severe",
                           seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate a rigid misplacement of the sensor frame (bracket loosening,
    vibration): a yaw error plus a translation of the origin, both in the sensor
    plane (pitch, roll and height do not exist in a 2D radar image). Every cell
    is resampled from its source position (nearest cell, which keeps the noise
    statistics). Directions are drawn from the seed; use one seed per burst.
    """
    yaw_deg, shift_m = severity_preset(_EXTRINSIC_PRESETS, severity, "extrinsic_misalignment")
    rng = np.random.default_rng(seed)
    yaw = np.deg2rad(yaw_deg) * rng.choice([-1.0, 1.0])
    direction = rng.uniform(0.0, 2.0 * np.pi)
    tx, ty = shift_m * np.cos(direction), shift_m * np.sin(direction)

    azimuth = frame.azimuth_rad.astype(np.float64)[:, None]
    radius = frame.range_m.astype(np.float64)[None, :]

    # An echo at true position p shows up at R(yaw) p + t: invert it to find,
    # for every output cell, the cell it comes from.
    px = radius * np.cos(azimuth) - tx
    py = radius * np.sin(azimuth) - ty
    sx = np.cos(yaw) * px + np.sin(yaw) * py
    sy = -np.sin(yaw) * px + np.cos(yaw) * py

    rows = np.rint((np.arctan2(sy, sx) - azimuth[0, 0]) / frame.azimuth_step_rad)
    rows = rows.astype(np.int64) % frame.n_rows
    cols = np.floor(np.hypot(sx, sy) / frame.geometry.range_resolution_m).astype(np.int64)

    floor, _ = frame.noise_floor()
    inside = cols < frame.n_bins
    power = np.where(inside, frame.power[rows, np.minimum(cols, frame.n_bins - 1)], floor)
    return frame.with_power(power)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_calibration(frame: RadarFrame, kind: str, **kwargs) -> RadarFrame:
    kind = kind.lower().strip()

    dispatch = {
        "boresight": lambda: boresight_misalignment(frame, **kwargs),
        "thermal_drift": lambda: thermal_expansion_drift(frame, **kwargs),
        "extrinsic": lambda: extrinsic_misalignment(frame, **kwargs),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown calibration fault: '{kind}'")
    return dispatch[kind]()