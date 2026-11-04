
"""
faults_environment.py

LiDAR environmental faults: atmospheric scattering (rain, fog), snow, strong
sunlight, crosstalk from other LiDARs, protective-window contamination,
housing water ingress and corrosion (proxies), wet or frozen ground, weak
echoes and mechanical instability.
Each function takes a clean PointCloud and returns a degraded copy. Level
tables of the faults after the window contamination are heuristic.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

from common.lidar_loader import (
    LIDAR_VERTICAL_FOV_DEG,
    REFERENCE_RANGE_M,
    PointCloud,
    spherical_angles,
    spherical_to_cartesian,
)


# ---------------------------------------------------------------------------
# Atmospheric scattering (rain / fog)
# ---------------------------------------------------------------------------

# (backscatter_point_fraction, extinction_coefficient_sigma [1/m]) per severity.
_ENVIRONMENTAL_SCATTER_PRESETS = {
    "mild":     (0.05, 0.03),
    "moderate": (0.10, 0.05),
    "severe":   (0.15, 0.08),
    "extreme":  (0.25, 0.12),
}

# Koschmieder relation: visibility = 3.912 / sigma, used as the maximum
# detection range in the scattering medium (sensor independent).
_KOSCHMIEDER_CONSTANT = 3.912

# Backscatter band: from the sensor blind zone up to this range [m].
_BACKSCATTER_MAX_RANGE_M = 8.0

# Blind-zone estimate: low percentile of the frame ranges. The frame minimum
# is not used because a few stray returns sit below the real blind zone.
_BLIND_ZONE_PERCENTILE = 1.0


def environmental_scatter(cloud: PointCloud, severity: str = "severe",
                          seed: Optional[int] = None) -> PointCloud:
    """
    Simulate rain/fog. Real returns beyond 3.912/sigma are dropped and the
    others are attenuated by exp(-2*sigma*r) (two-way path). Weak near-field
    backscatter points from airborne droplets are added.
    """
    severity = str(severity).lower().strip()
    if severity not in _ENVIRONMENTAL_SCATTER_PRESETS:
        raise ValueError(f"environmental_scatter: unknown severity '{severity}'")
    fog_frac, sigma = _ENVIRONMENTAL_SCATTER_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    r = np.linalg.norm(cloud.xyz, axis=1)

    keep = r <= _KOSCHMIEDER_CONSTANT / sigma
    kept = cloud.subset(keep)
    attenuated = (kept.intensity * np.exp(-2.0 * sigma * r[keep])).astype(np.float32)
    kept = PointCloud(kept.xyz, attenuated, kept.extra)

    n_fog = int(fog_frac * kept.n_points)
    if n_fog <= 0:
        return kept

    r_lo = float(np.clip(np.percentile(r, _BLIND_ZONE_PERCENTILE),
                         0.5, _BACKSCATTER_MAX_RANGE_M - 0.5))

    fov_min, fov_max = np.deg2rad(LIDAR_VERTICAL_FOV_DEG)
    theta_fog = rng.uniform(-np.pi, np.pi, size=n_fog)
    elevation_fog = rng.uniform(fov_min, fov_max, size=n_fog)
    r_fog = rng.uniform(r_lo, _BACKSCATTER_MAX_RANGE_M, size=n_fog)
    xyz_fog = spherical_to_cartesian(r_fog, theta_fog, elevation_fog)

    # Backscatter is weak: sample intensity from the frame's low percentiles.
    low, high = np.percentile(cloud.intensity, [1, 10])
    intensity_fog = rng.uniform(low, high, size=n_fog).astype(np.float32)

    fog_cloud = PointCloud(xyz_fog, intensity_fog, kept.synthesize_extra(n_fog))
    return kept.concat(fog_cloud)


# ---------------------------------------------------------------------------
# Protective window contamination (mud, dirt, ice, insects)
# ---------------------------------------------------------------------------

# (azimuth_width_deg, elevation_fraction_of_vertical_fov, opacity) per severity.
_WINDOW_CONTAMINATION_PRESETS = {
    "mild":     (15.0, 0.50, 0.60),
    "moderate": (30.0, 0.70, 0.80),
    "severe":   (60.0, 0.85, 0.90),
    "extreme":  (90.0, 1.00, 1.00),
}

# Extra drop probability reached at REFERENCE_RANGE_M: behind a partially
# opaque patch, weak long-range returns are lost first.
_WINDOW_RANGE_GAIN = 0.3


class WindowPatch(NamedTuple):
    """Angular footprint of the contaminated area, in the sensor frame."""
    az_center_deg: float
    az_width_deg: float
    elev_min_deg: float
    elev_max_deg: float


def _window_preset(severity: str):
    severity = str(severity).lower().strip()
    if severity not in _WINDOW_CONTAMINATION_PRESETS:
        raise ValueError(f"window_contamination: unknown severity '{severity}'")
    return _WINDOW_CONTAMINATION_PRESETS[severity]


def _sample_window_patch(az_width_deg: float, elev_fraction: float,
                         rng: np.random.Generator) -> WindowPatch:
    fov_min, fov_max = LIDAR_VERTICAL_FOV_DEG
    span = elev_fraction * (fov_max - fov_min)
    elev_start = rng.uniform(fov_min, fov_max - span)
    az_center = rng.uniform(-np.pi, np.pi)
    return WindowPatch(float(np.degrees(az_center)), az_width_deg,
                       float(elev_start), float(elev_start + span))


def window_patch_geometry(severity: str = "severe", seed: Optional[int] = None) -> WindowPatch:
    """Patch that window_contamination places for this severity and seed."""
    az_width_deg, elev_fraction, _ = _window_preset(severity)
    return _sample_window_patch(az_width_deg, elev_fraction, np.random.default_rng(seed))


def window_contamination(cloud: PointCloud, severity: str = "severe",
                         seed: Optional[int] = None) -> PointCloud:
    """
    Simulate a localized blind spot on the protective window: points inside a
    randomly placed azimuth/elevation patch are dropped with a probability
    given by the patch opacity, increasing with range.
    """
    az_width_deg, elev_fraction, opacity = _window_preset(severity)

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    patch = _sample_window_patch(az_width_deg, elev_fraction, rng)

    r, theta, elevation = spherical_angles(cloud.xyz)
    elevation_deg = np.degrees(elevation)

    # Angular distance to the patch center, wrapped to [0, pi].
    az_center = np.deg2rad(patch.az_center_deg)
    az_distance = np.abs((theta - az_center + np.pi) % (2.0 * np.pi) - np.pi)
    inside = (az_distance <= np.deg2rad(patch.az_width_deg) / 2.0) & \
             (elevation_deg >= patch.elev_min_deg) & (elevation_deg <= patch.elev_max_deg)

    p_drop = np.zeros(cloud.n_points, dtype=np.float32)
    range_term = np.clip(r[inside] / REFERENCE_RANGE_M, 0.0, 1.0)
    p_drop[inside] = np.clip(opacity + _WINDOW_RANGE_GAIN * range_term, 0.0, 1.0)

    keep = rng.random(cloud.n_points) > p_drop
    return cloud.subset(keep)


# ---------------------------------------------------------------------------
# Shared helpers of the faults that add or degrade points
# ---------------------------------------------------------------------------

def spurious_points(cloud: PointCloud, rng: np.random.Generator, n: int,
                    r_range: Tuple[float, float],
                    azimuth_sector: Optional[Tuple[float, float]] = None,
                    intensity_pct: Tuple[float, float] = (1.0, 10.0)) -> PointCloud:
    """
    `n` synthetic points uniform in range, in elevation (over the vertical FOV)
    and in azimuth (all around, or inside `azimuth_sector` = (center, width) in
    degrees). Intensities come from the given percentiles of `cloud`; extra
    fields are NaN so the points stay identifiable.
    """
    if azimuth_sector is None:
        theta = rng.uniform(-np.pi, np.pi, size=n)
    else:
        center, width = np.deg2rad(azimuth_sector)
        theta = center + rng.uniform(-0.5 * width, 0.5 * width, size=n)
    fov_min, fov_max = np.deg2rad(LIDAR_VERTICAL_FOV_DEG)
    elevation = rng.uniform(fov_min, fov_max, size=n)
    r = rng.uniform(r_range[0], r_range[1], size=n)

    low, high = np.percentile(cloud.intensity, intensity_pct)
    intensity = rng.uniform(low, high, size=n).astype(np.float32)
    return PointCloud(spherical_to_cartesian(r, theta, elevation), intensity, cloud.synthesize_extra(n))


def near_field_start(cloud: PointCloud, upper_m: float) -> float:
    """Blind-zone estimate, clipped so that the range band [start, upper_m] stays valid."""
    r = np.linalg.norm(cloud.xyz, axis=1)
    return float(np.clip(np.percentile(r, _BLIND_ZONE_PERCENTILE), 0.5, upper_m - 0.5))


# ---------------------------------------------------------------------------
# Snow
# ---------------------------------------------------------------------------

# (backscatter_point_fraction, extinction_coefficient_sigma [1/m]) per severity.
# More backscatter and less extinction than fog: flakes scatter strongly while
# the air stays clearer. Ice on the window is covered by window_contamination.
_SNOW_PRESETS = {
    "mild":     (0.10, 0.010),
    "moderate": (0.20, 0.020),
    "severe":   (0.35, 0.040),
    "extreme":  (0.50, 0.070),
}

# Backscatter band of the falling flakes: from the blind zone up to this range [m].
_SNOW_BACKSCATTER_MAX_RANGE_M = 25.0


def snow_attenuation(cloud: PointCloud, severity: str = "severe",
                     seed: Optional[int] = None) -> PointCloud:
    """
    Simulate snow: returns beyond 3.912/sigma are dropped, the others are
    attenuated by exp(-2*sigma*r), and flakes add denser, brighter near-field
    backscatter than fog. Use a different seed per frame, since the flakes move.
    """
    severity = str(severity).lower().strip()
    if severity not in _SNOW_PRESETS:
        raise ValueError(f"snow_attenuation: unknown severity '{severity}'")
    flake_frac, sigma = _SNOW_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    rng = np.random.default_rng(seed)
    r = np.linalg.norm(cloud.xyz, axis=1)

    keep = r <= _KOSCHMIEDER_CONSTANT / sigma
    kept = cloud.subset(keep)
    attenuated = (kept.intensity * np.exp(-2.0 * sigma * r[keep])).astype(np.float32)
    kept = PointCloud(kept.xyz, attenuated, kept.extra)

    n_flakes = int(flake_frac * kept.n_points)
    if n_flakes <= 0:
        return kept

    r_lo = near_field_start(cloud, _SNOW_BACKSCATTER_MAX_RANGE_M)
    flakes = spurious_points(cloud, rng, n_flakes, (r_lo, _SNOW_BACKSCATTER_MAX_RANGE_M),
                             intensity_pct=(5.0, 40.0))
    return kept.concat(flakes)


# ---------------------------------------------------------------------------
# Strong sunlight / flare
# ---------------------------------------------------------------------------

# (azimuth_sector_width_deg, drop_probability, noise_fraction) per severity.
# The noise fraction is relative to the clean point count.
_SUNLIGHT_PRESETS = {
    "mild":     (20.0, 0.30, 0.01),
    "moderate": (40.0, 0.50, 0.02),
    "severe":   (60.0, 0.70, 0.04),
    "extreme":  (90.0, 0.90, 0.08),
}


def sunlight_flare(cloud: PointCloud, severity: str = "severe",
                   seed: Optional[int] = None,
                   frame_seed: Optional[int] = None) -> PointCloud:
    """
    Simulate direct sunlight on the detector: inside an azimuth sector (the sun
    direction) returns are lost to saturation with probability `p_drop` and
    background light triggers spurious points at random ranges. The direction
    comes from `seed` (one per burst, since the sun stays put); the drops and
    the noise from `frame_seed` (a fresh one per frame, falling back to `seed`).
    """
    severity = str(severity).lower().strip()
    if severity not in _SUNLIGHT_PRESETS:
        raise ValueError(f"sunlight_flare: unknown severity '{severity}'")
    width_deg, p_drop, noise_frac = _SUNLIGHT_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    center_deg = float(np.degrees(np.random.default_rng(seed).uniform(-np.pi, np.pi)))
    rng = np.random.default_rng(seed if frame_seed is None else frame_seed)

    _, theta, _ = spherical_angles(cloud.xyz)
    distance = np.abs((theta - np.deg2rad(center_deg) + np.pi) % (2.0 * np.pi) - np.pi)
    in_sector = distance <= np.deg2rad(width_deg) / 2.0
    kept = cloud.subset(~(in_sector & (rng.random(cloud.n_points) < p_drop)))

    n_noise = int(noise_frac * cloud.n_points)
    if n_noise <= 0:
        return kept

    r_lo = near_field_start(cloud, REFERENCE_RANGE_M)
    noise = spurious_points(cloud, rng, n_noise, (r_lo, REFERENCE_RANGE_M),
                            azimuth_sector=(center_deg, width_deg), intensity_pct=(1.0, 30.0))
    return kept.concat(noise)


# ---------------------------------------------------------------------------
# Interference / crosstalk
# ---------------------------------------------------------------------------

# (number_of_stripes, noise_fraction) per severity. The noise fraction is
# relative to the clean point count.
_CROSSTALK_PRESETS = {
    "mild":     (2, 0.01),
    "moderate": (3, 0.02),
    "severe":   (4, 0.04),
    "extreme":  (6, 0.08),
}

# Azimuth width of each stripe of interfering pulses [deg].
_CROSSTALK_STRIPE_WIDTH_DEG = 2.0


def crosstalk_interference(cloud: PointCloud, severity: str = "severe",
                           seed: Optional[int] = None) -> PointCloud:
    """
    Simulate pulses from other LiDARs: structured noise made of a few narrow
    azimuth stripes of spurious points at random ranges and elevations. Real
    returns are untouched. Use a different seed per frame, since the other
    sensors are not synchronised.
    """
    severity = str(severity).lower().strip()
    if severity not in _CROSSTALK_PRESETS:
        raise ValueError(f"crosstalk_interference: unknown severity '{severity}'")
    n_stripes, noise_frac = _CROSSTALK_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    n_noise = int(noise_frac * cloud.n_points)
    if n_noise <= 0:
        return cloud

    rng = np.random.default_rng(seed)
    r_lo = near_field_start(cloud, REFERENCE_RANGE_M)
    per_stripe = max(1, n_noise // n_stripes)

    out = cloud
    for center in rng.uniform(-180.0, 180.0, size=n_stripes):
        out = out.concat(spurious_points(
            cloud, rng, per_stripe, (r_lo, REFERENCE_RANGE_M),
            azimuth_sector=(float(center), _CROSSTALK_STRIPE_WIDTH_DEG), intensity_pct=(1.0, 30.0)))
    return out


# ---------------------------------------------------------------------------
# Housing water ingress and brackish corrosion (proxies)
# ---------------------------------------------------------------------------

# (intensity_gain, max_range_m, noise_fraction) per severity.
_WATER_INGRESS_PRESETS = {
    "mild":     (0.90, 90.0, 0.02),
    "moderate": (0.70, 70.0, 0.05),
    "severe":   (0.50, 45.0, 0.10),
    "extreme":  (0.10, 15.0, 0.20),  # near-total loss
}

# Milder, earlier stage of the same degradation.
_CORROSION_PRESETS = {
    "mild":     (0.95, 95.0, 0.01),
    "moderate": (0.90, 85.0, 0.02),
    "severe":   (0.80, 70.0, 0.04),
    "extreme":  (0.70, 55.0, 0.07),
}

# Upper range of the near-field noise of the degraded housing [m].
_HOUSING_NOISE_MAX_RANGE_M = 10.0


def _housing_degradation(cloud: PointCloud, rng: np.random.Generator, intensity_gain: float,
                         max_range_m: float, noise_fraction: float) -> PointCloud:
    """Intensity loss, shorter maximum range and near-field spurious points."""
    r = np.linalg.norm(cloud.xyz, axis=1)
    kept = cloud.subset(r <= max_range_m)
    kept = PointCloud(kept.xyz, (kept.intensity * intensity_gain).astype(np.float32), kept.extra)

    n_noise = int(noise_fraction * kept.n_points)
    if n_noise <= 0:
        return kept

    upper = min(_HOUSING_NOISE_MAX_RANGE_M, max_range_m)
    return kept.concat(spurious_points(cloud, rng, n_noise, (near_field_start(cloud, upper), upper)))


def water_ingress(cloud: PointCloud, severity: str = "severe",
                  seed: Optional[int] = None) -> PointCloud:
    """
    Proxy for liquid penetrating the housing: fogged optics lower the
    intensities, shorten the maximum range and trigger near-field spurious
    points, up to a near-total loss. Use a different seed per frame.
    """
    severity = str(severity).lower().strip()
    if severity not in _WATER_INGRESS_PRESETS:
        raise ValueError(f"water_ingress: unknown severity '{severity}'")
    if cloud.n_points == 0:
        return cloud
    return _housing_degradation(cloud, np.random.default_rng(seed), *_WATER_INGRESS_PRESETS[severity])


def brackish_corrosion(cloud: PointCloud, severity: str = "severe",
                       seed: Optional[int] = None) -> PointCloud:
    """
    Proxy for salt and humidity corroding seals and connectors: a milder form of
    the water ingress degradation, which it eventually enables.
    Use a different seed per frame.
    """
    severity = str(severity).lower().strip()
    if severity not in _CORROSION_PRESETS:
        raise ValueError(f"brackish_corrosion: unknown severity '{severity}'")
    if cloud.n_points == 0:
        return cloud
    return _housing_degradation(cloud, np.random.default_rng(seed), *_CORROSION_PRESETS[severity])


# ---------------------------------------------------------------------------
# Wet or frozen ground
# ---------------------------------------------------------------------------

# (ground_drop_probability, ground_intensity_gain) per severity.
_WET_GROUND_PRESETS = {
    "mild":     (0.20, 0.90),
    "moderate": (0.40, 0.80),
    "severe":   (0.60, 0.65),
    "extreme":  (0.85, 0.50),
}

# Ground model: a plane fitted robustly (RANSAC) to the lower points within this
# horizontal range, so a sloped or tilted ground is handled. Points within the
# margin of the plane are ground. Planes steeper than about 25 degrees are rejected.
_GROUND_ESTIMATE_RANGE_M = 40.0
_GROUND_INLIER_TOL_M = 0.15
_GROUND_MARGIN_M = 0.3
_GROUND_RANSAC_ITERATIONS = 100
_GROUND_MIN_NORMAL_Z = 0.9

# Fallback when no plausible plane is found: flat ground at this percentile of z.
_GROUND_PERCENTILE = 5.0


def estimate_ground(cloud: PointCloud) -> np.ndarray:
    """
    Boolean mask of the ground points of a frame. A plane is fitted with RANSAC
    (fixed seed, so the estimate is a deterministic function of the frame) to the
    lower half of the points near the sensor and refined by least squares on its
    inliers. Falls back to a flat ground if no plausible plane is found.
    """
    xyz = cloud.xyz.astype(np.float64)
    near = np.flatnonzero(np.hypot(xyz[:, 0], xyz[:, 1]) <= _GROUND_ESTIMATE_RANGE_M)
    if len(near) < 3:
        return np.zeros(len(xyz), dtype=bool)

    z_near = xyz[near, 2]
    candidates = xyz[near][z_near <= np.median(z_near)]
    if len(candidates) < 3:
        return np.zeros(len(xyz), dtype=bool)

    rng = np.random.default_rng(0)
    best_inliers, best_count = None, 0
    for _ in range(_GROUND_RANSAC_ITERATIONS):
        a, b, c = candidates[rng.choice(len(candidates), size=3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9 or abs(normal[2]) / length < _GROUND_MIN_NORMAL_Z:
            continue
        inliers = np.abs((candidates - a) @ normal) / length < _GROUND_INLIER_TOL_M
        if inliers.sum() > best_count:
            best_inliers, best_count = inliers, int(inliers.sum())

    if best_inliers is None or best_count < 3:
        return xyz[:, 2] <= float(np.percentile(z_near, _GROUND_PERCENTILE)) + _GROUND_MARGIN_M

    points = candidates[best_inliers]
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    coef, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    plane_z = coef[0] * xyz[:, 0] + coef[1] * xyz[:, 1] + coef[2]
    return np.abs(xyz[:, 2] - plane_z) <= _GROUND_MARGIN_M


def wet_ground(cloud: PointCloud, severity: str = "severe",
               seed: Optional[int] = None) -> PointCloud:
    """
    Simulate water or ice on the asphalt absorbing the beam: ground returns
    (points near the ground plane, estimated from the frame) are lost with
    probability `p_drop` and the surviving ones are weaker. Assumes a locally
    planar ground. Use a different seed per frame.
    """
    severity = str(severity).lower().strip()
    if severity not in _WET_GROUND_PRESETS:
        raise ValueError(f"wet_ground: unknown severity '{severity}'")
    p_drop, intensity_gain = _WET_GROUND_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    ground = estimate_ground(cloud)
    if not ground.any():
        return cloud

    rng = np.random.default_rng(seed)
    lost = ground & (rng.random(cloud.n_points) < p_drop)
    intensity = np.where(ground, cloud.intensity * intensity_gain, cloud.intensity).astype(np.float32)
    return PointCloud(cloud.xyz, intensity, cloud.extra).subset(~lost)


# ---------------------------------------------------------------------------
# Incomplete echo
# ---------------------------------------------------------------------------

# Fraction of the weakest returns that never reach the detection threshold.
_INCOMPLETE_ECHO_FRACTION = {
    "mild": 0.05,
    "moderate": 0.10,
    "severe": 0.20,
    "extreme": 0.35,
}


def incomplete_echo(cloud: PointCloud, severity: str = "severe") -> PointCloud:
    """
    Simulate returns too weak to be detected (dark or absorptive surfaces): the
    weakest fraction of the returns is discarded, the farthest first among equal
    intensities (intensity is quantised, so ties are common). The fraction is
    relative to the frame because intensity has no absolute scale across sensors.
    Deterministic given severity.
    """
    severity = str(severity).lower().strip()
    if severity not in _INCOMPLETE_ECHO_FRACTION:
        raise ValueError(f"incomplete_echo: unknown severity '{severity}'")
    fraction = _INCOMPLETE_ECHO_FRACTION[severity]

    if cloud.n_points == 0:
        return cloud

    r = np.linalg.norm(cloud.xyz, axis=1)
    order = np.lexsort((-r, cloud.intensity))  # weakest first, then farthest first
    keep = np.ones(cloud.n_points, dtype=bool)
    keep[order[:int(round(fraction * cloud.n_points))]] = False
    return cloud.subset(keep)


# ---------------------------------------------------------------------------
# Mechanical instability
# ---------------------------------------------------------------------------

# (pitch amplitude [deg], roll amplitude [deg]) per severity.
_VIBRATION_PRESETS = {
    "mild":     (0.3, 0.3),
    "moderate": (0.6, 0.6),
    "severe":   (1.2, 1.2),
    "extreme":  (2.0, 2.0),
}

# Cycles of the pitch/roll oscillation over one revolution. A rotating sensor
# sweeps the azimuth over time, so azimuth stands in for the timestamp.
_VIBRATION_CYCLES = 2


def mechanical_instability(cloud: PointCloud, severity: str = "severe",
                           seed: Optional[int] = None) -> PointCloud:
    """
    Simulate vehicle vibration and a loose mounting: pitch and roll oscillate
    during the sweep, so every point is rotated by the angles of its own
    instant. Phases are drawn from the seed; use a different seed per frame.
    """
    severity = str(severity).lower().strip()
    if severity not in _VIBRATION_PRESETS:
        raise ValueError(f"mechanical_instability: unknown severity '{severity}'")
    pitch_deg, roll_deg = _VIBRATION_PRESETS[severity]

    if cloud.n_points == 0:
        return cloud

    phase_pitch, phase_roll = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=2)
    _, theta, _ = spherical_angles(cloud.xyz)
    pitch = np.deg2rad(pitch_deg) * np.sin(_VIBRATION_CYCLES * theta + phase_pitch)
    roll = np.deg2rad(roll_deg) * np.sin(_VIBRATION_CYCLES * theta + phase_roll)

    # Same convention as rotation_matrix_yaw_pitch_roll: roll first, then pitch.
    x, y, z = (cloud.xyz[:, i].astype(np.float64) for i in range(3))
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    y1, z1 = cr * y - sr * z, sr * y + cr * z
    xyz = np.stack([cp * x + sp * z1, y1, -sp * x + cp * z1], axis=1)
    return PointCloud(xyz, cloud.intensity, cloud.extra)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fault_environment(cloud: PointCloud, kind: str, **kwargs) -> PointCloud:
    kind = kind.lower().strip()

    dispatch = {
        "environmental_scatter": lambda: environmental_scatter(cloud, **kwargs),
        "window": lambda: window_contamination(cloud, **kwargs),
        "snow": lambda: snow_attenuation(cloud, **kwargs),
        "sunlight": lambda: sunlight_flare(cloud, **kwargs),
        "crosstalk": lambda: crosstalk_interference(cloud, **kwargs),
        "water_ingress": lambda: water_ingress(cloud, **kwargs),
        "corrosion": lambda: brackish_corrosion(cloud, **kwargs),
        "wet_ground": lambda: wet_ground(cloud, **kwargs),
        "incomplete_echo": lambda: incomplete_echo(cloud, **kwargs),
        "mechanical_instability": lambda: mechanical_instability(cloud, **kwargs),
    }

    if kind not in dispatch:
        raise ValueError(f"Unknown environmental fault: '{kind}'")
    return dispatch[kind]()