"""
faults_hardware.py

Radar hardware faults: receiver front-end (LNA degradation), antenna element
failures and thermal deformation of the housing/radome.
Each function takes a RadarFrame and returns a degraded copy; the header
columns are never modified. Every function accepts `seed` for a uniform
interface, even when the fault is deterministic.
"""

from typing import Optional

import numpy as np

from common.radar_loader import (
    RadarFrame,
    attenuation_gain,
    severity_preset,
    smooth_rows,
)

# Receiver gain loss [dB] per severity.
_LNA_GAIN_LOSS_DB = {
    "mild": 2.0,
    "moderate": 4.0,
    "severe": 7.0,
    "extreme": 10.0,
}

# Fraction of failed antenna elements per severity.
_ANTENNA_FAILED_FRACTION = {
    "mild": 0.05,
    "moderate": 0.10,
    "severe": 0.20,
    "extreme": 0.30,
}

# Azimuth spread of the returns per unit of failed fraction [deg]. Heuristic
# stand-in for the sidelobe leakage of a damaged aperture.
_SPREAD_DEG_PER_FAILED_FRACTION = 6.0

# Peak beam deflection caused by the housing deformation [deg] per severity.
# Heuristic values, not taken from a sensor specification.
_THERMAL_DEFLECTION_DEG = {
    "mild": 0.5,
    "moderate": 1.0,
    "severe": 2.0,
    "extreme": 3.0,
}

# Cycles of the deflection pattern over one revolution (slow, smooth variation).
_THERMAL_HARMONICS = 2


def lna_degradation(frame: RadarFrame, severity: str = "severe",
                    seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate a degraded low-noise amplifier: echoes above the noise floor lose
    `loss_db` of gain while the noise floor stays unchanged, so weak targets
    sink into the noise. Deterministic given severity.
    """
    loss_db = severity_preset(_LNA_GAIN_LOSS_DB, severity, "lna_degradation")
    floor, _ = frame.noise_floor()
    gain = attenuation_gain(loss_db, frame.geometry)

    excess = np.maximum(frame.power - floor, 0.0)
    return frame.with_power(np.minimum(frame.power, floor) + gain * excess)


def antenna_failure(frame: RadarFrame, severity: str = "severe",
                    seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate failed antenna elements: coherent gain loss of 20*log10(1 - f) dB
    and azimuth spreading of the echoes (sidelobe leakage). Deterministic.
    """
    failed = severity_preset(_ANTENNA_FAILED_FRACTION, severity, "antenna_failure")
    floor, _ = frame.noise_floor()

    gain = attenuation_gain(-20.0 * np.log10(1.0 - failed), frame.geometry)
    sigma_rows = np.deg2rad(_SPREAD_DEG_PER_FAILED_FRACTION * failed) / frame.azimuth_step_rad

    excess = smooth_rows(np.maximum(frame.power - floor, 0.0), sigma_rows)
    return frame.with_power(np.minimum(frame.power, floor) + gain * excess)


def thermal_deformation(frame: RadarFrame, severity: str = "severe",
                        seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate thermal deformation of the housing/radome: the beam is deflected by
    an angle that varies smoothly with azimuth (a sinusoid), so echoes appear
    displaced in azimuth. The deflection is angular, hence the lateral
    displacement in metres grows with range. Whole rows are moved (nearest row),
    which keeps the noise statistics. The phase is drawn from the seed; use one
    seed per burst, since the deformation persists.
    """
    peak_deg = severity_preset(_THERMAL_DEFLECTION_DEG, severity, "thermal_deformation")
    phase = float(np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi))

    deflection = np.deg2rad(peak_deg) * np.sin(_THERMAL_HARMONICS * frame.azimuth_rad + phase)
    shift = np.rint(deflection / frame.azimuth_step_rad).astype(np.int64)
    source = (np.arange(frame.n_rows) + shift) % frame.n_rows
    return frame.with_power(frame.power[source])


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_hardware(frame: RadarFrame, kind: str, **kwargs) -> RadarFrame:
    kind = kind.lower().strip()

    dispatch = {
        "lna": lambda: lna_degradation(frame, **kwargs),
        "antenna": lambda: antenna_failure(frame, **kwargs),
        "thermal": lambda: thermal_deformation(frame, **kwargs),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown hardware fault: '{kind}'")
    return dispatch[kind]()