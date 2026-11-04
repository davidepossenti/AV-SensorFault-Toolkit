"""
lidar_loader.py

Shared point cloud I/O utilities for LiDAR fault injection.

Design goals:
- Preserve the full per-point record (xyz, intensity, and any additional
  fields such as ring/laser id or timestamp) through the fault injection
  round-trip, instead of truncating to XYZI.
- Make the binary layout explicit. Flat float32 files are not
  self-describing, and inferring the stride from the file size is ambiguous
  (a 6-field file with an even point count is also divisible by 4).
- Never invent data: points synthesized by a fault carry NaN in every extra
  field, so they stay identifiable and no ring/timestamp is fabricated.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Supported per-point stride lengths (number of float32 values per point).
SUPPORTED_STRIDES = (4, 6)

# Default layout: (x, y, z, intensity, ring, time), as in Boreas.
DEFAULT_STRIDE = 6

# Sensor constants shared by LiDAR fault models (Velodyne Alpha Prime).
LIDAR_VERTICAL_FOV_DEG = (-25.0, 15.0)
REFERENCE_RANGE_M = 100.0  # fixed range used to normalize range-dependent effects


@dataclass
class PointCloud:
    """
    In-memory representation of a single LiDAR frame.

    xyz:       (N, 3) float32 point coordinates.
    intensity: (N,)   float32 per-point reflectance/intensity.
    extra:     (N, k) float32 additional per-point fields (e.g. ring, time).
               Shape is (N, 0) for plain XYZI data.
    """
    xyz: np.ndarray
    intensity: np.ndarray
    extra: np.ndarray

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=np.float32).reshape(-1, 3)
        self.intensity = np.asarray(self.intensity, dtype=np.float32).reshape(-1)
        self.extra = np.asarray(self.extra, dtype=np.float32)

        n = self.xyz.shape[0]
        if self.extra.ndim != 2 or self.extra.shape[0] != n or self.intensity.shape[0] != n:
            raise ValueError(
                f"Inconsistent PointCloud fields: xyz={self.xyz.shape}, "
                f"intensity={self.intensity.shape}, extra={self.extra.shape}"
            )

    @property
    def n_points(self) -> int:
        return self.xyz.shape[0]

    @property
    def stride(self) -> int:
        return 4 + self.extra.shape[1]

    def subset(self, mask: np.ndarray) -> "PointCloud":
        """Return a new PointCloud keeping the points selected by a boolean mask or index array."""
        return PointCloud(self.xyz[mask], self.intensity[mask], self.extra[mask])

    def concat(self, other: "PointCloud") -> "PointCloud":
        """Concatenate with another PointCloud sharing the same extra-field width."""
        if other.extra.shape[1] != self.extra.shape[1]:
            raise ValueError(
                f"Cannot concatenate point clouds with different extra-field "
                f"widths ({self.extra.shape[1]} vs {other.extra.shape[1]})"
            )
        return PointCloud(
            np.vstack([self.xyz, other.xyz]),
            np.concatenate([self.intensity, other.intensity]),
            np.vstack([self.extra, other.extra]),
        )

    def synthesize_extra(self, n_new: int) -> np.ndarray:
        """
        Extra-field block for points injected by a fault (e.g. backscatter or
        false triggers). Filled with NaN: no ring or timestamp is fabricated,
        and synthetic points remain identifiable in every extra field.
        """
        return np.full((n_new, self.extra.shape[1]), np.nan, dtype=np.float32)


def _empty_cloud(stride: int) -> PointCloud:
    return PointCloud(
        np.zeros((0, 3), np.float32),
        np.zeros(0, np.float32),
        np.zeros((0, stride - 4), np.float32),
    )


def load_point_cloud(path: str, stride: Optional[int] = None, allow_empty: bool = False) -> PointCloud:
    """
    Load a LiDAR frame from a flat binary float32 file.

    Args:
        path: Path to the .bin file.
        stride: Floats per point (4 = XYZI, 6 = XYZI + two extra fields).
            Defaults to DEFAULT_STRIDE.
        allow_empty: Return an empty cloud for a zero-byte file instead of
            raising. Needed to read frames produced by a total-loss fault.
    """
    stride = DEFAULT_STRIDE if stride is None else int(stride)
    if stride not in SUPPORTED_STRIDES:
        raise ValueError(f"Unsupported stride {stride}; expected one of {SUPPORTED_STRIDES}")

    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        if allow_empty:
            return _empty_cloud(stride)
        raise RuntimeError(f"Empty LiDAR file: {path}")

    if arr.size % stride != 0:
        raise RuntimeError(
            f"Cannot read {path} with stride {stride}: {arr.size} values is not a multiple. "
            f"If the file uses another layout, pass a different stride."
        )

    pts = arr.reshape(-1, stride)
    return PointCloud(pts[:, :3].copy(), pts[:, 3].copy(), pts[:, 4:].copy())


def save_point_cloud(path: str, cloud: PointCloud) -> None:
    """Write a PointCloud to a flat binary float32 file, preserving its per-point stride."""
    columns = [cloud.xyz, cloud.intensity.reshape(-1, 1)]
    if cloud.extra.shape[1] > 0:
        columns.append(cloud.extra)
    np.hstack(columns).astype(np.float32).tofile(path)


# ---------------------------------------------------------------------------
# Spherical geometry helpers, shared across LiDAR fault models that operate
# in the sensor's native (range, azimuth, elevation) frame.
# ---------------------------------------------------------------------------

def spherical_angles(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to (range, azimuth, elevation) in radians."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.linalg.norm(xyz, axis=1)
    r_xy = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    elevation = np.arctan2(z, np.maximum(r_xy, 1e-9))
    return r, theta, elevation


def spherical_to_cartesian(r: np.ndarray, theta: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Inverse of spherical_angles: (range, azimuth, elevation) -> (x, y, z)."""
    x = r * np.cos(elevation) * np.cos(theta)
    y = r * np.cos(elevation) * np.sin(theta)
    z = r * np.sin(elevation)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def rotation_matrix_yaw_pitch_roll(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build a 3x3 rotation matrix, used by faults simulating mounting misalignment."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return (rz @ ry @ rx).astype(np.float32)