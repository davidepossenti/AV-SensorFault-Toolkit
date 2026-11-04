"""
faults_environment.py

Radar environmental faults: rain and snow attenuation, radome contamination and
condensation, radio interference, multipath ghosts, clutter, roadside foliage,
airborne scatter, and housing water ingress and corrosion (the last three are
proxies).
Each function takes a RadarFrame and returns a degraded copy; the header
columns are never modified. Every function accepts `seed` for a uniform
interface, even when the fault is deterministic. Level tables of the faults
added after multipath are heuristic unless noted.
"""

from typing import Optional

import numpy as np

from common.radar_loader import (
    RadarFrame,
    attenuation_gain,
    scale_signal,
    sector_weight,
    severity_preset,
    smooth_rows,
)

# ---------------------------------------------------------------------------
# Rain
# ---------------------------------------------------------------------------

# Rain rate [mm/h] per severity.
_RAIN_RATE_MM_H = {
    "mild": 2.5,
    "moderate": 10.0,
    "severe": 25.0,
    "extreme": 50.0,
}


def rain_attenuation(frame: RadarFrame, severity: str = "severe",
                     seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate rain: two-way path loss 2 * gamma * r with gamma = k * R^alpha
    (ITU-R P.838-3, coefficients in RadarGeometry). Echoes lose signal with
    range; the noise floor is unchanged. Volume clutter is not modelled.
    Deterministic given severity.
    """
    rate = severity_preset(_RAIN_RATE_MM_H, severity, "rain_attenuation")
    geometry = frame.geometry
    floor, _ = frame.noise_floor()

    gamma_db_per_km = geometry.rain_k * rate ** geometry.rain_alpha
    loss_db = 2.0 * gamma_db_per_km * frame.range_m / 1000.0
    gain = attenuation_gain(loss_db, geometry)[None, :]

    excess = np.maximum(frame.power - floor, 0.0)
    return frame.with_power(np.minimum(frame.power, floor) + gain * excess)


# ---------------------------------------------------------------------------
# Radome contamination
# ---------------------------------------------------------------------------

# (azimuth sector width [deg], two-way attenuation [dB]) per severity.
_RADOME_PRESETS = {
    "mild": (15.0, 1.5),
    "moderate": (30.0, 3.0),
    "severe": (60.0, 6.0),
    "extreme": (90.0, 10.0),
}


def radome_contamination(frame: RadarFrame, severity: str = "severe",
                         seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate mud, water or ice on the radome: echoes inside a randomly placed
    azimuth sector are attenuated. Use one seed per burst, since the dirt stays
    in place.
    """
    width_deg, loss_db = severity_preset(_RADOME_PRESETS, severity, "radome_contamination")
    rng = np.random.default_rng(seed)
    floor, _ = frame.noise_floor()

    center = rng.uniform(0.0, 2.0 * np.pi)
    weight = sector_weight(frame.azimuth_rad, center, np.deg2rad(width_deg),
                           edge_rad=frame.azimuth_step_rad)
    gain = attenuation_gain(loss_db * weight, frame.geometry)[:, None]

    excess = np.maximum(frame.power - floor, 0.0)
    return frame.with_power(np.minimum(frame.power, floor) + gain * excess)


# ---------------------------------------------------------------------------
# Radio interference
# ---------------------------------------------------------------------------

# (fraction of azimuth rows hit, interference-to-noise ratio [dB]) per severity.
_INTERFERENCE_PRESETS = {
    "mild": (0.05, 6.0),
    "moderate": (0.10, 10.0),
    "severe": (0.20, 15.0),
    "extreme": (0.35, 20.0),
}


def radio_interference(frame: RadarFrame, severity: str = "severe",
                       seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate an unsynchronised radar: randomly chosen azimuth rows get additive
    noise raising their floor to `inr_db` above the nominal level. Use a
    different seed per frame, since interference is intermittent.
    """
    row_fraction, inr_db = severity_preset(_INTERFERENCE_PRESETS, severity, "radio_interference")
    rng = np.random.default_rng(seed)
    floor, _ = frame.noise_floor()
    start = frame.blind_bins

    n_rows = max(1, int(round(row_fraction * frame.n_rows)))
    rows = rng.choice(frame.n_rows, size=n_rows, replace=False)

    ratio = 10.0 ** (inr_db / frame.geometry.db_factor)
    burst = rng.exponential(1.0, size=(n_rows, frame.n_bins - start)).astype(np.float32)

    power = frame.power.copy()
    power[rows, start:] += floor * (ratio - 1.0) * burst
    return frame.with_power(power)


# ---------------------------------------------------------------------------
# Multipath ghosts
# ---------------------------------------------------------------------------

# Ghost level below the direct echo [dB] per severity.
_GHOST_LEVEL_DB = {
    "mild": 12.0,
    "moderate": 9.0,
    "severe": 6.0,
    "extreme": 3.0,
}

# Only echoes this many robust sigmas above the floor produce ghosts.
_GHOST_THRESHOLD_SIGMA = 6.0


def multipath_ghosts(frame: RadarFrame, severity: str = "severe",
                     seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate multipath: every strong echo is repeated at twice its range (double
    bounce between sensor and target), attenuated by the ghost level.
    Deterministic given the scene.
    """
    ghost_db = severity_preset(_GHOST_LEVEL_DB, severity, "multipath_ghosts")
    floor, sigma = frame.noise_floor()

    signal = np.maximum(frame.power - floor, 0.0)
    signal[signal < _GHOST_THRESHOLD_SIGMA * sigma] = 0.0

    # Doubling the range axis maps source bin j onto bins 2j and 2j + 1.
    ghost = np.repeat(signal, 2, axis=1)[:, :frame.n_bins]
    ghost = attenuation_gain(ghost_db, frame.geometry) * ghost
    ghost[:, :frame.blind_bins] = 0.0
    return frame.with_power(frame.power + ghost)


# ---------------------------------------------------------------------------
# Shared helpers of the additive-noise faults
# ---------------------------------------------------------------------------

def add_noise(frame: RadarFrame, power: np.ndarray, rng: np.random.Generator, level_db: float,
              origin_m: Optional[float] = None, decay_m: Optional[float] = None,
              row_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Add exponentially distributed noise raising the local floor by `level_db`
    above the nominal one. The excess is flat in range or, with `decay_m`,
    starts at `origin_m` (default: the blind zone) and decays exponentially with
    range. `row_weight` (per azimuth row, in [0, 1]) restricts it to some rows.
    """
    floor, _ = frame.noise_floor()
    start = frame.blind_bins
    range_m = frame.range_m[start:].astype(np.float64)

    if decay_m is None:
        profile = np.ones_like(range_m)
    else:
        origin = frame.geometry.blind_zone_m if origin_m is None else origin_m
        profile = np.where(range_m >= origin, np.exp(-(range_m - origin) / decay_m), 0.0)

    ratio = 10.0 ** (level_db / frame.geometry.db_factor)
    amplitude = (floor * (ratio - 1.0) * profile).astype(np.float32)[None, :]
    noise = rng.exponential(1.0, size=(frame.n_rows, frame.n_bins - start)).astype(np.float32)
    if row_weight is not None:
        noise = noise * row_weight[:, None]

    out = power.copy()
    out[:, start:] += amplitude * noise
    return out


def _attenuate_with_clutter(frame: RadarFrame, gamma_db_per_km: float, cnr_db: float,
                            decay_m: float, seed: Optional[int]) -> RadarFrame:
    """Two-way path loss 2 * gamma * r on the echoes plus near-field backscatter."""
    rng = np.random.default_rng(seed)
    floor, _ = frame.noise_floor()

    loss_db = 2.0 * gamma_db_per_km * frame.range_m / 1000.0
    power = scale_signal(frame.power, floor, attenuation_gain(loss_db, frame.geometry)[None, :])
    return frame.with_power(add_noise(frame, power, rng, cnr_db, decay_m=decay_m))


def _housing_degradation(frame: RadarFrame, gain_loss_db: float, noise_rise_db: float,
                         seed: Optional[int]) -> RadarFrame:
    """Receiver gain loss on the echoes plus a higher noise floor everywhere."""
    rng = np.random.default_rng(seed)
    floor, _ = frame.noise_floor()

    power = scale_signal(frame.power, floor, attenuation_gain(gain_loss_db, frame.geometry))
    return frame.with_power(add_noise(frame, power, rng, noise_rise_db))


# ---------------------------------------------------------------------------
# Snow
# ---------------------------------------------------------------------------

# (specific attenuation [dB/km], clutter-to-noise ratio [dB]) per severity. The
# top attenuation matches the highest value measured at 77 GHz in dry snow
# (12 dB/km); the levels in between are heuristic.
_SNOW_PRESETS = {
    "mild":     (1.5, 3.0),
    "moderate": (3.0, 5.0),
    "severe":   (6.0, 8.0),
    "extreme":  (12.0, 12.0),
}

# Decay length of the backscatter of the falling flakes [m].
_SNOW_CLUTTER_DECAY_M = 40.0


def snow_attenuation(frame: RadarFrame, severity: str = "severe",
                     seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate snow: two-way path loss like rain, plus weak volume backscatter
    from the flakes near the sensor. Use a different seed per frame, since the
    flakes move. Ice on the radome is covered by `radome`.
    """
    gamma_db_per_km, cnr_db = severity_preset(_SNOW_PRESETS, severity, "snow_attenuation")
    return _attenuate_with_clutter(frame, gamma_db_per_km, cnr_db, _SNOW_CLUTTER_DECAY_M, seed)


# ---------------------------------------------------------------------------
# Condensation
# ---------------------------------------------------------------------------

# (two-way attenuation [dB], azimuth beam broadening [rows]) per severity.
# Conservative: measured losses of a 0.1-0.8 mm water film at 77 GHz are larger.
_CONDENSATION_PRESETS = {
    "mild":     (1.5, 0.5),
    "moderate": (3.0, 0.8),
    "severe":   (6.0, 1.2),
    "extreme":  (10.0, 1.8),
}


def condensation_film(frame: RadarFrame, severity: str = "severe",
                      seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate a moisture film over the whole radome: every echo loses the same
    two-way attenuation and the beam broadens in azimuth (detuning). Unlike
    `radome` it is not limited to a sector. Deterministic given severity.
    """
    loss_db, spread_rows = severity_preset(_CONDENSATION_PRESETS, severity, "condensation_film")
    floor, _ = frame.noise_floor()

    gain = attenuation_gain(loss_db, frame.geometry)
    excess = smooth_rows(np.maximum(frame.power - floor, 0.0), spread_rows)
    return frame.with_power(np.minimum(frame.power, floor) + gain * excess)


# ---------------------------------------------------------------------------
# Clutter
# ---------------------------------------------------------------------------

# Clutter-to-noise ratio at the edge of the blind zone [dB] per severity.
_CLUTTER_CNR_DB = {
    "mild": 6.0,
    "moderate": 10.0,
    "severe": 15.0,
    "extreme": 20.0,
}

# Decay length of the clutter with range [m].
_CLUTTER_DECAY_M = 30.0


def clutter_backscatter(frame: RadarFrame, severity: str = "severe",
                        seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate clutter: unwanted backscatter from the road surface or from rain
    that raises the floor near the sensor and decays exponentially with range.
    Use a different seed per frame.
    """
    cnr_db = severity_preset(_CLUTTER_CNR_DB, severity, "clutter_backscatter")
    rng = np.random.default_rng(seed)
    return frame.with_power(add_noise(frame, frame.power, rng, cnr_db, decay_m=_CLUTTER_DECAY_M))


# ---------------------------------------------------------------------------
# Foliage
# ---------------------------------------------------------------------------

# (azimuth sector width [deg], one-way foliage attenuation [dB], clutter-to-noise
# ratio [dB]) per severity. Foliage is nearly opaque at millimetre waves (about
# 20 dB per tree on average, from models valid below 60 GHz: an extrapolation).
_FOLIAGE_PRESETS = {
    "mild":     (20.0, 3.0, 3.0),
    "moderate": (40.0, 6.0, 6.0),
    "severe":   (60.0, 10.0, 9.0),
    "extreme":  (90.0, 15.0, 12.0),
}

_FOLIAGE_START_M = 10.0          # range where the vegetation starts
_FOLIAGE_DEPTH_M = 5.0           # depth over which the attenuation builds up
_FOLIAGE_CLUTTER_DECAY_M = 10.0  # decay of the clutter behind the vegetation edge


def foliage_interference(frame: RadarFrame, severity: str = "severe",
                         seed: Optional[int] = None,
                         frame_seed: Optional[int] = None) -> RadarFrame:
    """
    Simulate roadside vegetation: in a randomly placed azimuth sector the echoes
    beyond `_FOLIAGE_START_M` are attenuated (two-way, building up over
    `_FOLIAGE_DEPTH_M`) and the leaves add clutter. The sector comes from `seed`
    (one per burst, since the trees stay in place); the clutter noise from
    `frame_seed` (a fresh one per frame, falling back to `seed`).
    """
    width_deg, one_way_db, cnr_db = severity_preset(_FOLIAGE_PRESETS, severity, "foliage_interference")
    center = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi)
    noise_rng = np.random.default_rng(seed if frame_seed is None else frame_seed)
    floor, _ = frame.noise_floor()

    weight = sector_weight(frame.azimuth_rad, center, np.deg2rad(width_deg),
                           edge_rad=frame.azimuth_step_rad)
    depth = np.clip((frame.range_m - _FOLIAGE_START_M) / _FOLIAGE_DEPTH_M, 0.0, 1.0)
    loss_db = 2.0 * one_way_db * weight[:, None] * depth[None, :]
    power = scale_signal(frame.power, floor, attenuation_gain(loss_db, frame.geometry))

    return frame.with_power(add_noise(frame, power, noise_rng, cnr_db, origin_m=_FOLIAGE_START_M,
                                      decay_m=_FOLIAGE_CLUTTER_DECAY_M, row_weight=weight))


# ---------------------------------------------------------------------------
# External scatter (proxy)
# ---------------------------------------------------------------------------

# (specific attenuation [dB/km], clutter-to-noise ratio [dB]) per severity.
_SCATTER_PRESETS = {
    "mild":     (1.0, 3.0),
    "moderate": (2.0, 6.0),
    "severe":   (4.0, 9.0),
    "extreme":  (8.0, 12.0),
}

# Decay length of the backscatter of the airborne particles [m].
_SCATTER_CLUTTER_DECAY_M = 15.0


def external_scatter(frame: RadarFrame, severity: str = "severe",
                     seed: Optional[int] = None) -> RadarFrame:
    """
    Proxy for airborne dust, smoke or debris: mild path attenuation plus
    near-field backscatter. Use a different seed per frame.
    """
    gamma_db_per_km, cnr_db = severity_preset(_SCATTER_PRESETS, severity, "external_scatter")
    return _attenuate_with_clutter(frame, gamma_db_per_km, cnr_db, _SCATTER_CLUTTER_DECAY_M, seed)


# ---------------------------------------------------------------------------
# Water ingress and brackish corrosion (proxies)
# ---------------------------------------------------------------------------

# (receiver gain loss [dB], noise floor rise [dB]) per severity.
_WATER_INGRESS_PRESETS = {
    "mild":     (2.0, 3.0),
    "moderate": (4.0, 6.0),
    "severe":   (8.0, 10.0),
    "extreme":  (40.0, 15.0),  # near-total loss of the echoes
}

# Milder, earlier stage of the same degradation.
_CORROSION_PRESETS = {
    "mild":     (0.5, 1.0),
    "moderate": (1.0, 2.0),
    "severe":   (2.0, 3.0),
    "extreme":  (3.5, 5.0),
}


def water_ingress(frame: RadarFrame, severity: str = "severe",
                  seed: Optional[int] = None) -> RadarFrame:
    """
    Proxy for liquid penetrating the housing: receiver gain loss and a higher
    noise floor, up to a near-total loss of the echoes. Use a different seed per
    frame.
    """
    gain_loss_db, noise_rise_db = severity_preset(_WATER_INGRESS_PRESETS, severity, "water_ingress")
    return _housing_degradation(frame, gain_loss_db, noise_rise_db, seed)


def brackish_corrosion(frame: RadarFrame, severity: str = "severe",
                       seed: Optional[int] = None) -> RadarFrame:
    """
    Proxy for salt and humidity corroding the seals and connectors: a milder
    gain loss and noise rise than water ingress, which it eventually enables.
    Use a different seed per frame.
    """
    gain_loss_db, noise_rise_db = severity_preset(_CORROSION_PRESETS, severity, "brackish_corrosion")
    return _housing_degradation(frame, gain_loss_db, noise_rise_db, seed)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_environment(frame: RadarFrame, kind: str, **kwargs) -> RadarFrame:
    kind = kind.lower().strip()

    dispatch = {
        "rain": lambda: rain_attenuation(frame, **kwargs),
        "snow": lambda: snow_attenuation(frame, **kwargs),
        "radome": lambda: radome_contamination(frame, **kwargs),
        "condensation": lambda: condensation_film(frame, **kwargs),
        "interference": lambda: radio_interference(frame, **kwargs),
        "multipath": lambda: multipath_ghosts(frame, **kwargs),
        "clutter": lambda: clutter_backscatter(frame, **kwargs),
        "foliage": lambda: foliage_interference(frame, **kwargs),
        "scatter": lambda: external_scatter(frame, **kwargs),
        "water_ingress": lambda: water_ingress(frame, **kwargs),
        "corrosion": lambda: brackish_corrosion(frame, **kwargs),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown environmental fault: '{kind}'")
    return dispatch[kind]()