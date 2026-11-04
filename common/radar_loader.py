"""
radar_loader.py

Loading, saving and shared helpers for polar radar frames.

A frame is an 8-bit grayscale PNG with one row per azimuth and one column per
range bin, preceded by a few header columns (timestamp, azimuth encoder,
validity flag) that the faults never modify. Defaults follow the Boreas /
Navtech format; pass a RadarGeometry to adapt the toolkit to another radar.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Sensor geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadarGeometry:
    """Sensor constants. Defaults: Boreas (Navtech) polar frames."""
    metadata_columns: int = 11               # header columns before the range data
    azimuth_byte_offset: Optional[int] = 8   # first byte of the uint16 azimuth encoder (None: uniform azimuths)
    azimuth_counts: int = 5600               # encoder counts per revolution
    range_resolution_m: float = 0.0596       # metres per range bin
    blind_zone_m: float = 2.5                # near range zeroed by the sensor
    db_factor: float = 20.0                  # 20: stored values are amplitudes, 10: powers
    rain_k: float = 1.1395                   # rain attenuation k * R^alpha [dB/km], R in mm/h
    rain_alpha: float = 0.716                # ITU-R P.838-3 at 77.75 GHz, horizontal polarisation


BOREAS_GEOMETRY = RadarGeometry()

# Boreas sequences recorded after the sensor upgrade use a finer range resolution.
BOREAS_GEOMETRY_UPGRADED = replace(BOREAS_GEOMETRY, range_resolution_m=0.04381)


# ---------------------------------------------------------------------------
# Frame container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RadarFrame:
    """Header columns (kept as is) and range data in [0, 1], one row per azimuth."""
    meta: np.ndarray    # (rows, metadata_columns) uint8
    power: np.ndarray   # (rows, bins) float32
    geometry: RadarGeometry = BOREAS_GEOMETRY

    @property
    def n_rows(self) -> int:
        return self.power.shape[0]

    @property
    def n_bins(self) -> int:
        return self.power.shape[1]

    @property
    def azimuth_rad(self) -> np.ndarray:
        """Azimuth of each row [rad], decoded from the header when available."""
        offset = self.geometry.azimuth_byte_offset
        if offset is not None and self.meta.shape[1] >= offset + 2:
            counts = (self.meta[:, offset].astype(np.uint16)
                      | (self.meta[:, offset + 1].astype(np.uint16) << 8))
            return counts.astype(np.float32) / self.geometry.azimuth_counts * (2.0 * np.pi)
        return np.linspace(0.0, 2.0 * np.pi, self.n_rows, endpoint=False, dtype=np.float32)

    @property
    def azimuth_step_rad(self) -> float:
        return 2.0 * np.pi / self.n_rows

    @property
    def range_m(self) -> np.ndarray:
        """Range of each bin centre [m]."""
        return ((np.arange(self.n_bins, dtype=np.float32) + 0.5)
                * self.geometry.range_resolution_m)

    @property
    def max_range_m(self) -> float:
        return self.n_bins * self.geometry.range_resolution_m

    @property
    def blind_bins(self) -> int:
        return int(np.ceil(self.geometry.blind_zone_m / self.geometry.range_resolution_m))

    def cartesian_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Position of every cell in the sensor frame, in metres: (x, y) = r * (cos, sin)(azimuth)."""
        azimuth = self.azimuth_rad[:, None]
        radius = self.range_m[None, :]
        return radius * np.cos(azimuth), radius * np.sin(azimuth)

    def noise_floor(self) -> Tuple[float, float]:
        """Background level (median) and robust spread (scaled MAD) beyond the blind zone."""
        valid = self.power[:, self.blind_bins:]
        median = float(np.median(valid))
        sigma = 1.4826 * float(np.median(np.abs(valid - median)))
        eps = 1.0 / 255.0
        return max(median, eps), max(sigma, eps)

    def with_power(self, power: np.ndarray) -> "RadarFrame":
        """Copy with new range data (clipped to [0, 1]) and the same header."""
        return replace(self, power=np.clip(power, 0.0, 1.0).astype(np.float32))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_radar_png(path: str, geometry: RadarGeometry = BOREAS_GEOMETRY) -> RadarFrame:
    """Read a polar radar PNG (8-bit grayscale) into a RadarFrame."""
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"{path}: expected 8-bit grayscale, got mode '{image.mode}'")
        data = np.array(image)

    n_meta = geometry.metadata_columns
    if data.ndim != 2 or data.shape[1] <= n_meta:
        raise ValueError(f"{path}: unexpected shape {data.shape}")

    return RadarFrame(meta=data[:, :n_meta].copy(),
                      power=data[:, n_meta:].astype(np.float32) / 255.0,
                      geometry=geometry)


def save_radar_png(path: str, frame: RadarFrame) -> None:
    """Write a RadarFrame as an 8-bit PNG: header unchanged, data rounded to 1/255."""
    data = np.rint(np.clip(frame.power, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(np.hstack([frame.meta, data])).save(path)


# ---------------------------------------------------------------------------
# Shared operations
# ---------------------------------------------------------------------------

def severity_preset(table: Mapping[str, Any], severity: str, fault_name: str) -> Any:
    """Look up a severity level (case-insensitive) or raise a clear error."""
    key = str(severity).lower().strip()
    if key not in table:
        raise ValueError(f"{fault_name}: unknown severity '{severity}' "
                         f"(expected one of {', '.join(table)})")
    return table[key]


def attenuation_gain(loss_db, geometry: RadarGeometry) -> np.ndarray:
    """Linear gain on the stored values for a power loss given in dB."""
    loss = np.asarray(loss_db, dtype=np.float32)
    return np.power(10.0, -loss / geometry.db_factor).astype(np.float32)


def scale_signal(power: np.ndarray, floor: float, gain) -> np.ndarray:
    """Scale the part of `power` above `floor` by `gain`; the floor itself is unchanged."""
    return np.minimum(power, floor) + gain * np.maximum(power - floor, 0.0)


def smooth_rows(power: np.ndarray, sigma_rows: float) -> np.ndarray:
    """Circular Gaussian smoothing along the azimuth axis."""
    if sigma_rows < 1e-3:
        return power
    radius = max(1, int(np.ceil(3.0 * sigma_rows)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma_rows) ** 2)
    weights /= weights.sum()

    out = np.zeros_like(power)
    for offset, weight in zip(offsets, weights):
        out += weight * np.roll(power, int(offset), axis=0)
    return out


def shift_rows(power: np.ndarray, shift: float) -> np.ndarray:
    """
    Circular shift along the azimuth axis by a fractional number of rows.
    Integer shifts are exact rolls; fractional ones use a Fourier shift, which
    keeps the noise statistics (linear interpolation would smooth the noise).
    """
    if float(shift).is_integer():
        return np.roll(power, int(shift), axis=0)
    freq = np.fft.fftfreq(power.shape[0])[:, None]
    spectrum = np.fft.fft(power, axis=0) * np.exp(-2.0j * np.pi * freq * shift)
    return np.fft.ifft(spectrum, axis=0).real.astype(np.float32)


def sector_weight(azimuth_rad: np.ndarray, center_rad: float, width_rad: float,
                  edge_rad: float) -> np.ndarray:
    """Per-row weight in [0, 1]: 1 inside the sector, 0 outside, smooth over `edge_rad`."""
    distance = np.abs((azimuth_rad - center_rad + np.pi) % (2.0 * np.pi) - np.pi)
    t = np.clip((0.5 * width_rad + 0.5 * edge_rad - distance) / max(edge_rad, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)