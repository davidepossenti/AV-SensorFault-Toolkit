"""Calibration fault proxies for focus, color gains and image geometry."""

import cv2
import numpy as np

from .common import (
    affine,
    dispatch,
    preset,
    random_stream,
    scaled_px,
    to_float,
    to_uint8,
)


def blurred(img, severity="severe"):
    sigma = max(0.2, scaled_px(img, preset((1.2, 2.5, 4.5, 7.5), severity)))
    return cv2.GaussianBlur(
        img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE
    )


def white_balance(img, severity="severe", preset_name="warm"):
    gains = {
        "warm": (0.80, 1.00, 1.25),
        "cool": (1.18, 1.00, 0.88),
        "green": (0.95, 1.15, 0.95),
        "magenta": (1.10, 0.90, 1.10),
    }
    if preset_name not in gains:
        raise ValueError(f"Unknown white-balance preset: {preset_name}")
    strength = preset((0.35, 0.60, 1.00, 1.30), severity)
    gain = np.array(gains[preset_name], np.float32)
    gain /= gain.mean()
    gain = 1 + strength * (gain - 1)
    return to_uint8(to_float(img) * gain)


def lens_distortion(img, severity="severe"):
    k1, k2 = preset(
        ((-0.06, 0), (-0.12, 0), (-0.18, 0), (-0.28, 0.02)), severity
    )
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    scale = max(cx, cy)
    x, y = (xx - cx) / scale, (yy - cy) / scale
    radius2 = x * x + y * y
    radial = 1 + k1 * radius2 + k2 * radius2 ** 2
    return cv2.remap(
        img,
        (cx + scale * x * radial).astype(np.float32),
        (cy + scale * y * radial).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def mounting(img, severity="severe", seed=None):
    rotation, shift = preset(
        ((1, 4), (2, 8), (4, 14), (7, 22)), severity
    )
    signs = random_stream(seed).choice((-1, 1), size=3)
    translation = scaled_px(img, shift)
    return affine(
        img,
        rotation=rotation * signs[0],
        tx=translation * signs[1],
        ty=translation * signs[2],
    )


FUNCTIONS = {
    "blurred": blurred,
    "white_balance": white_balance,
    "lens_distortion": lens_distortion,
    "mounting": mounting,
}


def apply_fault_calibration(img, kind, **kwargs):
    return dispatch(FUNCTIONS, img, kind, kwargs)