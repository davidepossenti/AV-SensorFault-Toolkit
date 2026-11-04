"""
faults_software.py

Radar firmware and signal-processing faults: wrong region-of-interest gating,
silent failure of the acquisition chain, packet loss and a failing interference
mitigation stage.
Each function takes a RadarFrame and returns a degraded copy; the header
columns are never modified.

Note: the temporal faults "inference_timing" (stale frames caused by scheduling
delays) and "temporal_misalignment" (frames delivered late) need the whole
frame sequence, so radar_injector.py handles them at the scheduling level.
"""

from typing import Optional

import numpy as np

from common.radar_loader import RadarFrame, sector_weight, severity_preset
from .faults_environment import add_noise

# Number of blanked bands per severity.
_BAND_COUNT = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "extreme": 4,
}

# Width of each band as a fraction of the Cartesian field of view (2 * max range).
_BAND_WIDTH_FRACTION = 0.10

# Azimuth rows lost together, as one acquisition packet (assumed, sensor specific).
_PACKET_ROWS = 4

# Fraction of the packets lost per severity.
_DATA_LOSS_FRACTION = {
    "mild": 0.02,
    "moderate": 0.05,
    "severe": 0.10,
    "extreme": 0.20,
}

# (azimuth sector width [deg], interference-to-noise ratio [dB]) per severity.
_MITIGATION_PRESETS = {
    "mild":     (10.0, 6.0),
    "moderate": (20.0, 10.0),
    "severe":   (40.0, 15.0),
    "extreme":  (70.0, 20.0),
}

# Delay of the delivered frame [frames] per severity (0.25 s to 2 s at 4 Hz).
_DELAY_FRAMES = {
    "mild": 1,
    "moderate": 2,
    "severe": 4,
    "extreme": 8,
}


def roi_filtering(frame: RadarFrame, severity: str = "severe",
                  seed: Optional[int] = None, orientation: str = "vertical") -> RadarFrame:
    """
    Simulate a wrong region-of-interest gate: `n` equal, equally spaced bands of
    the Cartesian field of view are blanked (detections discarded). Band width is
    relative to the maximum range, so the fault scales to any radar.
    Deterministic given severity.
    """
    count = severity_preset(_BAND_COUNT, severity, "roi_filtering")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(f"roi_filtering: unknown orientation '{orientation}'")

    x, y = frame.cartesian_xy()
    coord = x if orientation == "vertical" else y

    half_extent = frame.max_range_m
    width = _BAND_WIDTH_FRACTION * 2.0 * half_extent
    centers = -half_extent + (np.arange(count) + 0.5) / count * 2.0 * half_extent

    blanked = np.zeros(frame.power.shape, dtype=bool)
    for center in centers:
        blanked |= np.abs(coord - center) <= 0.5 * width
    return frame.with_power(np.where(blanked, 0.0, frame.power))


def silent_failure(frame: RadarFrame) -> RadarFrame:
    """
    Simulate a crashed driver or communication interface: the frame is still
    delivered, header included, but without any echo. Binary fault.
    """
    return frame.with_power(np.zeros_like(frame.power))


def packet_loss(frame: RadarFrame, severity: str = "severe",
                seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate packet loss in the acquisition pipeline: randomly chosen groups of
    azimuth rows arrive empty (header kept). Frame misordering is not modelled.
    Transient: use a different seed per frame.
    """
    fraction = severity_preset(_DATA_LOSS_FRACTION, severity, "packet_loss")

    n_groups = int(np.ceil(frame.n_rows / _PACKET_ROWS))
    n_lost = int(round(fraction * n_groups))
    if n_lost == 0:
        return frame

    lost = np.random.default_rng(seed).choice(n_groups, size=n_lost, replace=False)
    lost_rows = np.isin(np.arange(frame.n_rows) // _PACKET_ROWS, lost)

    power = frame.power.copy()
    power[lost_rows] = 0.0
    return frame.with_power(power)


def interference_mitigation_failure(frame: RadarFrame, severity: str = "severe",
                                    seed: Optional[int] = None,
                                    frame_seed: Optional[int] = None) -> RadarFrame:
    """
    Proxy for a failing interference mitigation stage: the interference that the
    DSP normally removes survives in a fixed azimuth sector as additive noise
    raising the floor to `inr_db`. Unlike radio_interference (random rows, fresh
    every frame) the sector persists: it comes from `seed` (one per burst) while
    the noise comes from `frame_seed` (a fresh one per frame, falling back to
    `seed`).
    """
    width_deg, inr_db = severity_preset(_MITIGATION_PRESETS, severity, "interference_mitigation_failure")
    center = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi)
    noise_rng = np.random.default_rng(seed if frame_seed is None else frame_seed)

    weight = sector_weight(frame.azimuth_rad, center, np.deg2rad(width_deg),
                           edge_rad=frame.azimuth_step_rad)
    return frame.with_power(add_noise(frame, frame.power, noise_rng, inr_db, row_weight=weight))


def temporal_delay_frames(severity: str) -> int:
    """Delay [frames] of the temporal_misalignment fault, used by radar_injector.py."""
    return severity_preset(_DELAY_FRAMES, severity, "temporal_misalignment")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_software(frame: RadarFrame, kind: str, **kwargs) -> RadarFrame:
    kind = kind.lower().strip()

    dispatch = {
        "filtering": lambda: roi_filtering(frame, **kwargs),
        "silent_failure": lambda: silent_failure(frame),
        "data_loss": lambda: packet_loss(frame, **kwargs),
        "interference_mitigation": lambda: interference_mitigation_failure(frame, **kwargs),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown software fault: '{kind}'")
    return dispatch[kind]()