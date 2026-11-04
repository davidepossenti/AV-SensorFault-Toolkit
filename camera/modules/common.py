"""Shared validation, image operations and deterministic random streams."""

from typing import Optional

import cv2
import numpy as np


SEVERITIES = ("mild", "moderate", "severe", "extreme")


def check_image(img: np.ndarray) -> np.ndarray:
    if not isinstance(img, np.ndarray):
        raise TypeError("Expected a NumPy array")
    if img.dtype != np.uint8:
        raise TypeError("Expected uint8 image")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HxWx3 BGR image")
    if min(img.shape[:2]) < 2:
        raise ValueError("Image dimensions must be at least 2x2")
    return img


def severity_index(severity: str) -> int:
    value = str(severity).lower().strip()
    if value not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity!r}")
    return SEVERITIES.index(value)


def preset(values, severity: str):
    return values[severity_index(severity)]


def to_float(img: np.ndarray) -> np.ndarray:
    return check_image(img).astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def random_stream(seed: Optional[int], stream: int = 0):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


def scaled_px(img: np.ndarray, value: float) -> float:
    return value * min(img.shape[:2]) / 1080.0


def affine(img, rotation=0.0, tx=0.0, ty=0.0, scale=1.0):
    check_image(img)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        ((w - 1) / 2.0, (h - 1) / 2.0), rotation, scale
    )
    matrix[:, 2] += (tx, ty)
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def soft_spots(img, count, radius_range, rng):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    radius_scale = min(h, w)

    for _ in range(int(count)):
        radius = max(
            1, int(round(rng.uniform(*radius_range) * radius_scale))
        )
        x = int(rng.integers(w))
        y = int(rng.integers(h))
        cv2.circle(mask, (x, y), radius, 1.0, -1)

    sigma = max(0.5, radius_range[0] * radius_scale * 0.4)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigma), 0.0, 1.0)


def atmospheric_veil(img, strength, airlight=0.92, seed=None):
    base = to_float(img)
    h, w = img.shape[:2]
    rng = random_stream(seed)

    coarse = rng.uniform(0.75, 1.0, (6, 8)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    transmission = np.clip(1.0 - strength * field, 0.0, 1.0)
    return to_uint8(
        base * transmission[..., None]
        + airlight * (1.0 - transmission[..., None])
    )


def bayer_masks(h, w, pattern="RGGB"):
    layouts = {
        "RGGB": (("R", "G"), ("G", "B")),
        "BGGR": (("B", "G"), ("G", "R")),
        "GRBG": (("G", "R"), ("B", "G")),
        "GBRG": (("G", "B"), ("R", "G")),
    }
    pattern = str(pattern).upper()
    if pattern not in layouts:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")

    yy, xx = np.indices((h, w))
    layout = np.array(layouts[pattern])
    colors = layout[yy % 2, xx % 2]
    return colors == "R", colors == "G", colors == "B"


def dispatch(functions, img, kind, kwargs):
    check_image(img)
    key = str(kind).lower().strip()
    if key not in functions:
        raise ValueError(f"Unknown fault kind: {kind!r}")
    out = functions[key](img, **kwargs)
    check_image(out)
    if out.shape != img.shape:
        raise ValueError(f"Fault changed image shape: {img.shape} -> {out.shape}")
    return out