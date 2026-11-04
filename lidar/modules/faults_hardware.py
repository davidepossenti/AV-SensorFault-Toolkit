"""
faults_hardware.py

LiDAR hardware faults, grouped by subsystem: emitter, detector, scanner and
optics, cooling, interconnects, housing.
Each function takes a clean PointCloud and returns a degraded copy.

The temporal "microelectronics_freeze" fault (repeating an earlier frame)
needs the whole frame sequence, so lidar_injector.py handles it at the
scheduling level.
"""

from typing import Optional

import numpy as np

from common.lidar_loader import (
    LIDAR_VERTICAL_FOV_DEG,
    REFERENCE_RANGE_M,
    PointCloud,
    spherical_angles,
    spherical_to_cartesian,
)
from .faults_environment import near_field_start, spurious_points

# Native azimuth resolution of the sensor [deg]. Velodyne Alpha Prime (Boreas): 0.2.
# Set it to the value of your LiDAR (or move it next to LIDAR_VERTICAL_FOV_DEG).
LIDAR_AZIMUTH_RESOLUTION_DEG = 0.2

# Blind-zone estimate: low percentile of the frame ranges. The frame minimum
# is not used because a few stray returns sit below the real blind zone.
_BLIND_ZONE_PERCENTILE = 1.0


# ---------------------------------------------------------------------------
# Emitter: laser power loss
# ---------------------------------------------------------------------------

# Fraction of the nominal emitted power that remains, per severity.
_LASER_POWER_FRACTION = {
    "mild": 0.80,
    "moderate": 0.50,
    "severe": 0.25,
    "extreme": 0.10,
}


def max_range_after_power_loss(power_fraction: float) -> float:
    """Received power ~ P / R^2, so the maximum range scales as sqrt(P)."""
    return REFERENCE_RANGE_M * float(np.sqrt(power_fraction))


def laser_power_loss(cloud: PointCloud, severity: str = "severe") -> PointCloud:
    """
    Simulate degraded emitter power. Points beyond the reduced maximum range
    are dropped (absolute cutoff, independent of scene density) and the
    remaining intensities scale with the emitted power.
    Deterministic given severity (no stochastic component).
    """
    severity = str(severity).lower().strip()
    if severity not in _LASER_POWER_FRACTION:
        raise ValueError(f"laser_power_loss: unknown severity '{severity}'")
    power_fraction = _LASER_POWER_FRACTION[severity]

    if cloud.n_points == 0:
        return cloud

    r = np.linalg.norm(cloud.xyz, axis=1)
    kept = cloud.subset(r <= max_range_after_power_loss(power_fraction))
    intensity = (kept.intensity * power_fraction).astype(np.float32)
    return PointCloud(kept.xyz, intensity, kept.extra)


# ---------------------------------------------------------------------------
# Detector: photodetector sensitivity loss and false triggers
# ---------------------------------------------------------------------------

# (base_drop, range_intensity_gain, noise_fraction) per severity.
# Drop probability = base_drop + gain * r_norm * (1 - intensity_norm), so weak
# (distant, low-intensity) returns are lost first. noise_fraction is the share
# of spurious points added relative to the surviving point count.
_PHOTODETECTOR_PRESETS = {
    "mild":     (0.05, 0.2, 0.02),
    "moderate": (0.10, 0.4, 0.05),
    "severe":   (0.20, 0.7, 0.08),
    "extreme":  (0.30, 1.0, 0.12),
}


def photodetector_noise_and_dropout(cloud: PointCloud, severity: str = "severe",
                                    seed: Optional[int] = None) -> PointCloud:
    """
    Simulate reduced photodetector sensitivity: weak returns are dropped
    probabilistically and spurious points emulate dark-current false triggers.
    Synthetic points carry NaN extra fields so they stay identifiable.
    """
    severity = str(severity).lower().strip()
    if severity not in _PHOTODETECTOR_PRESETS:
        raise ValueError(f"photodetector_noise_and_dropout: unknown severity '{severity}'")
    base_drop, k_drop, noise_frac = _PHOTODETECTOR_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)

    r = np.linalg.norm(cloud.xyz, axis=1)
    r_norm = np.clip(r / REFERENCE_RANGE_M, 0.0, 1.0)

    q1, q99 = np.percentile(cloud.intensity, [1, 99])
    intensity_norm = np.clip((cloud.intensity - q1) / max(q99 - q1, 1e-6), 0.0, 1.0)

    p_drop = np.clip(base_drop + k_drop * r_norm * (1.0 - intensity_norm), 0.0, 1.0)
    kept = cloud.subset(rng.random(cloud.n_points) > p_drop)

    n_noise = int(noise_frac * kept.n_points)
    if n_noise <= 0:
        return kept

    # False triggers cannot occur inside the sensor's blind zone.
    r_lo = float(np.clip(np.percentile(r, _BLIND_ZONE_PERCENTILE),
                         0.5, REFERENCE_RANGE_M - 0.5))

    # Sample in the sensor's spherical frame so noise does not cluster at far corners.
    fov_min, fov_max = LIDAR_VERTICAL_FOV_DEG
    theta_noise = rng.uniform(-np.pi, np.pi, size=n_noise)
    elevation_noise = np.deg2rad(rng.uniform(fov_min, fov_max, size=n_noise))
    r_noise = rng.uniform(r_lo, REFERENCE_RANGE_M, size=n_noise)
    xyz_noise = spherical_to_cartesian(r_noise, theta_noise, elevation_noise)

    q10, q30 = np.percentile(kept.intensity, [10, 30])
    intensity_noise = rng.uniform(q10, q30, size=n_noise).astype(np.float32)

    noise_cloud = PointCloud(xyz_noise, intensity_noise, kept.synthesize_extra(n_noise))
    return kept.concat(noise_cloud)


# ---------------------------------------------------------------------------
# Scanner: scanning mechanism failure (worn rotor, stuck MEMS mirror)
# ---------------------------------------------------------------------------

# (azimuth_fov_fraction, azimuth_step_multiplier) per severity.
# The step multiplier is relative to the native azimuth resolution: 3.0 means
# only one azimuth column out of three is acquired.
_SCANNING_MECHANISM_PRESETS = {
    "mild":     (0.95, 1.5),
    "moderate": (0.85, 2.0),
    "severe":   (0.75, 3.0),
    "extreme":  (0.60, 5.0),
}


def scanning_mechanism_failure(cloud: PointCloud, severity: str = "severe",
                               seed: Optional[int] = None) -> PointCloud:
    """
    Simulate a degraded scanning mechanism: a blind azimuth sector (reduced
    field of view) and a coarser azimuth sampling, as if the scan speed
    dropped. Vertical sampling (the beams) is unaffected.
    The expected fraction of surviving points is about
    fov_fraction / step_multiplier.
    """
    severity = str(severity).lower().strip()
    if severity not in _SCANNING_MECHANISM_PRESETS:
        raise ValueError(f"scanning_mechanism_failure: unknown severity '{severity}'")
    fov_fraction, step_multiplier = _SCANNING_MECHANISM_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    _, theta, _ = spherical_angles(cloud.xyz)

    # Blind sector covering (1 - fov_fraction) of the azimuth extent, randomly placed.
    th_min, th_max = float(theta.min()), float(theta.max())
    blind_half = 0.5 * (1.0 - fov_fraction) * (th_max - th_min)
    blind_center = rng.uniform(th_min, th_max)
    az_distance = np.abs((theta - blind_center + np.pi) % (2.0 * np.pi) - np.pi)
    keep_fov = az_distance > blind_half

    # Coarser azimuth sampling: keep one native azimuth column every `step_multiplier`.
    column = np.floor(theta / np.deg2rad(LIDAR_AZIMUTH_RESOLUTION_DEG))
    keep_column = np.floor(column / step_multiplier) != np.floor((column - 1.0) / step_multiplier)

    return cloud.subset(keep_fov & keep_column)


# ---------------------------------------------------------------------------
# Optics: optical system degradation (mirror misalignment, lens contamination)
# ---------------------------------------------------------------------------

# (elevation_distortion_gain, angular_jitter_deg, vertical_edge_dropout_prob) per severity.
_OPTICAL_SYSTEM_PRESETS = {
    "mild":     (0.3, 0.2, 0.2),
    "moderate": (0.6, 0.4, 0.4),
    "severe":   (0.9, 0.8, 0.6),
    "extreme":  (1.2, 1.2, 0.8),
}

# A rotating sensor has no edges in azimuth, so azimuth jitter is small and uniform.
_AZIMUTH_JITTER_FRACTION = 0.25


def optical_system_fault(cloud: PointCloud, severity: str = "severe",
                         seed: Optional[int] = None) -> PointCloud:
    """
    Simulate a degraded optical path: beam deflection error that grows toward
    the vertical edges of the field of view (elevation distortion, elevation
    jitter and edge-biased dropout), plus a small uniform azimuth jitter.
    Azimuth is never remapped, so the scene is not folded onto itself.
    """
    severity = str(severity).lower().strip()
    if severity not in _OPTICAL_SYSTEM_PRESETS:
        raise ValueError(f"optical_system_fault: unknown severity '{severity}'")
    k_distort, jitter_deg, edge_dropout = _OPTICAL_SYSTEM_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    r, theta, elevation = spherical_angles(cloud.xyz)

    # Vertical field of view of the sensor (not a per-frame statistic).
    fov_min, fov_max = np.deg2rad(LIDAR_VERTICAL_FOV_DEG)
    el_center, el_half = 0.5 * (fov_min + fov_max), 0.5 * (fov_max - fov_min)

    d_elev = elevation - el_center
    d_norm = np.clip(np.abs(d_elev) / el_half, 0.0, 1.0)

    elevation2 = el_center + d_elev * (1.0 + k_distort * d_norm ** 2)

    jitter_rad = np.deg2rad(jitter_deg)
    elevation2 = elevation2 + jitter_rad * d_norm * rng.standard_normal(cloud.n_points)
    theta2 = theta + _AZIMUTH_JITTER_FRACTION * jitter_rad * rng.standard_normal(cloud.n_points)

    p_drop = np.clip(edge_dropout * d_norm ** 2, 0.0, 1.0)
    keep = rng.random(cloud.n_points) > p_drop
    if not keep.any():
        return cloud.subset(keep)

    xyz_new = spherical_to_cartesian(r[keep], theta2[keep], elevation2[keep])
    kept = cloud.subset(keep)
    return PointCloud(xyz_new, kept.intensity, kept.extra)


# ---------------------------------------------------------------------------
# Cooling: thermal management degradation (overheating, failing heat sink)
# ---------------------------------------------------------------------------

# (range_noise_base_m, range_noise_gain_m, residual_power_fraction) per severity.
_THERMAL_MANAGEMENT_PRESETS = {
    "mild":     (0.020, 0.08, 0.90),
    "moderate": (0.035, 0.12, 0.75),
    "severe":   (0.050, 0.15, 0.60),
    "extreme":  (0.080, 0.22, 0.40),
}


def thermal_management_fault(cloud: PointCloud, severity: str = "severe",
                             seed: Optional[int] = None) -> PointCloud:
    """
    Simulate degraded thermal management: the emitter throttles down, which
    reduces the maximum range and scales the intensities with the residual
    power, while range noise grows with distance.
    """
    severity = str(severity).lower().strip()
    if severity not in _THERMAL_MANAGEMENT_PRESETS:
        raise ValueError(f"thermal_management_fault: unknown severity '{severity}'")
    sigma_base, sigma_gain, power_fraction = _THERMAL_MANAGEMENT_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    r, theta, elevation = spherical_angles(cloud.xyz)

    r_norm = np.clip(r / REFERENCE_RANGE_M, 0.0, 1.0)
    sigma = sigma_base + sigma_gain * r_norm
    r_noisy = np.maximum(r + sigma * rng.standard_normal(cloud.n_points), 0.0)

    keep = r <= max_range_after_power_loss(power_fraction)
    xyz_new = spherical_to_cartesian(r_noisy[keep], theta[keep], elevation[keep])
    kept = cloud.subset(keep)
    intensity = (kept.intensity * power_fraction).astype(np.float32)
    return PointCloud(xyz_new, intensity, kept.extra)


# ---------------------------------------------------------------------------
# Interconnects: electrical connection failure (total signal loss)
# ---------------------------------------------------------------------------

def electrical_connection_blackout(cloud: PointCloud) -> PointCloud:
    """Total loss of signal from a power or data connection fault. Binary fault."""
    return cloud.subset(np.zeros(cloud.n_points, dtype=bool))


# ---------------------------------------------------------------------------
# Housing: enclosure seal degradation (dust and moisture ingress)
# ---------------------------------------------------------------------------

# (intensity_gain, spurious_point_fraction) per severity. Proxy: heuristic values.
_ENCLOSURE_PRESETS = {
    "mild":     (0.95, 0.02),
    "moderate": (0.85, 0.05),
    "severe":   (0.70, 0.10),
    "extreme":  (0.50, 0.20),
}

# Internal scattering shows up very close to the sensor: spurious band up to this range [m].
_ENCLOSURE_SCATTER_MAX_RANGE_M = 6.0


def enclosure_fault(cloud: PointCloud, severity: str = "severe",
                    seed: Optional[int] = None) -> PointCloud:
    """
    Proxy for a degraded enclosure seal: dust and moisture inside the housing
    scatter the beam before it leaves the sensor. Every return loses intensity
    (haze) and weak spurious points appear in the near field, uniform in
    azimuth. Use a different seed per frame.
    """
    severity = str(severity).lower().strip()
    if severity not in _ENCLOSURE_PRESETS:
        raise ValueError(f"enclosure_fault: unknown severity '{severity}'")
    intensity_gain, noise_frac = _ENCLOSURE_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    hazy = PointCloud(cloud.xyz, (cloud.intensity * intensity_gain).astype(np.float32), cloud.extra)

    n_noise = int(noise_frac * cloud.n_points)
    if n_noise <= 0:
        return hazy

    r_lo = near_field_start(cloud, _ENCLOSURE_SCATTER_MAX_RANGE_M)
    return hazy.concat(spurious_points(cloud, rng, n_noise, (r_lo, _ENCLOSURE_SCATTER_MAX_RANGE_M)))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_hardware(cloud: PointCloud, kind: str, **kwargs) -> PointCloud:
    kind = kind.lower().strip()

    dispatch = {
        "laser": lambda: laser_power_loss(cloud, **kwargs),
        "photodetector": lambda: photodetector_noise_and_dropout(cloud, **kwargs),
        "scanning": lambda: scanning_mechanism_failure(cloud, **kwargs),
        "optical": lambda: optical_system_fault(cloud, **kwargs),
        "thermal": lambda: thermal_management_fault(cloud, **kwargs),
        "electrical": lambda: electrical_connection_blackout(cloud),
        "enclosure": lambda: enclosure_fault(cloud, **kwargs),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown hardware fault: '{kind}'")
    return dispatch[kind]()