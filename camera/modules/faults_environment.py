"""Environmental camera fault proxies in the processed-image domain."""

import cv2
import numpy as np

from .common import (
    affine,
    atmospheric_veil,
    dispatch,
    preset,
    random_stream,
    scaled_px,
    soft_spots,
    to_float,
    to_uint8,
)


def rain(img, severity="severe", seed=None):
    density, strength = preset(
        ((0.00008, 0.12), (0.00016, 0.20),
         (0.00030, 0.30), (0.00050, 0.42)), severity
    )
    rng = random_stream(seed)
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    count = max(1, round(h * w * density))
    length = max(2, round(scaled_px(img, 24)))

    for _ in range(count):
        x = int(rng.integers(w))
        y = int(rng.integers(h))
        cv2.line(layer, (x, y), (x - length // 4, y + length), 1.0, 1)

    layer = cv2.GaussianBlur(layer, (0, 0), 0.6)
    base = to_float(img)
    return to_uint8(base * (1 - 0.10 * strength) + strength * layer[..., None])


def snow(img, severity="severe", seed=None):
    count, strength = preset(
        ((80, 0.30), (160, 0.45), (300, 0.65), (500, 0.85)), severity
    )
    rng = random_stream(seed)
    mask = soft_spots(img, count, (0.001, 0.005), rng)
    alpha = np.clip(mask * strength, 0, 1)[..., None]
    base = to_float(img)
    return to_uint8(base * (1 - alpha) + 0.95 * alpha)


def ice(img, severity="severe", seed=None):
    count, strength = preset(
        ((8, 0.20), (16, 0.35), (24, 0.50), (40, 0.70)), severity
    )
    rng = random_stream(seed)
    mask = soft_spots(img, count, (0.025, 0.09), rng)
    base = to_float(img)
    blurred = to_float(cv2.GaussianBlur(
        img, (0, 0), max(0.5, scaled_px(img, 6))
    ))
    alpha = (mask * strength)[..., None]
    return to_uint8(base * (1 - alpha) + (0.6 * blurred + 0.4) * alpha)


def fog(img, severity="severe", seed=None):
    strength = preset((0.15, 0.30, 0.48, 0.68), severity)
    return atmospheric_veil(img, strength, seed=seed)


def external_scatter(img, severity="severe", seed=None):
    strength = preset((0.10, 0.20, 0.32, 0.48), severity)
    return atmospheric_veil(img, strength, airlight=0.68, seed=seed)


def window_contamination(img, severity="severe", seed=None):
    count, strength = preset(
        ((3, 0.30), (6, 0.45), (10, 0.65), (16, 0.85)), severity
    )
    rng = random_stream(seed)
    mask = soft_spots(img, count, (0.04, 0.12), rng)
    alpha = (mask * strength)[..., None]
    color = np.array([0.12, 0.16, 0.20], np.float32)
    return to_uint8(to_float(img) * (1 - alpha) + color * alpha)


def flare(img, severity="severe", seed=None):
    strength = preset((0.20, 0.40, 0.65, 0.90), severity)
    rng = random_stream(seed)
    h, w = img.shape[:2]
    sx = rng.uniform(0.05, 0.95) * w
    sy = rng.uniform(0.02, 0.30) * h
    yy, xx = np.indices((h, w), dtype=np.float32)
    sigma = max(1.0, 0.18 * min(h, w))
    halo = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sigma ** 2))
    out = to_float(img)
    tint = np.array([0.75, 0.90, 1.0], np.float32)
    out += strength * halo[..., None] * tint

    for index in range(1, 6):
        fraction = index / 3
        gx = sx + (w / 2 - sx) * fraction
        gy = sy + (h / 2 - sy) * fraction
        radius = max(1.0, sigma * (0.6 ** index))
        ghost = np.exp(
            -((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * radius ** 2)
        )
        out += strength * (0.65 ** index) * ghost[..., None] * tint

    return to_uint8(out)


def condensation(img, severity="severe", seed=None):
    strength, sigma = preset(
        ((0.15, 2), (0.28, 4), (0.42, 7), (0.60, 11)), severity
    )
    blurred = cv2.GaussianBlur(
        img, (0, 0), max(0.3, scaled_px(img, sigma))
    )
    return atmospheric_veil(blurred, strength, airlight=0.90, seed=seed)


def sensor_spots(img, severity="severe", seed=None):
    count, darkness = preset(
        ((10, 0.25), (20, 0.35), (30, 0.45), (45, 0.60)), severity
    )
    mask = soft_spots(
        img, count, (0.004, 0.015), random_stream(seed)
    )
    return to_uint8(to_float(img) * (1 - darkness * mask[..., None]))


def brackish_corrosion(img, severity="severe", seed=None):
    count, strength = preset(
        ((20, 0.12), (40, 0.22), (70, 0.35), (100, 0.50)), severity
    )
    rng = random_stream(seed)
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    edge_distance = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
    edge = np.clip(1 - edge_distance / max(1, 0.2 * min(h, w)), 0, 1)
    spots = soft_spots(img, count, (0.005, 0.025), rng)
    alpha = (strength * np.clip(edge + spots, 0, 1))[..., None]
    tint = np.array([0.65, 0.78, 0.85], np.float32)
    return to_uint8(to_float(img) * (1 - alpha) + tint * alpha)


def water_ingress(img, severity="severe", seed=None):
    count, haze = preset(
        ((8, 0.05), (15, 0.09), (24, 0.14), (38, 0.20)), severity
    )
    rng = random_stream(seed)
    mask = soft_spots(img, count, (0.01, 0.04), rng)
    base = to_float(img)
    blurred = to_float(cv2.GaussianBlur(
        img, (0, 0), max(0.5, scaled_px(img, 18))
    ))
    alpha = (0.85 * mask)[..., None]
    halo = (0.15 * cv2.GaussianBlur(
        mask, (0, 0), max(0.5, scaled_px(img, 14))
    ))[..., None]
    out = to_uint8(base * (1 - alpha) + blurred * alpha + halo)
    return atmospheric_veil(out, haze, airlight=0.90, seed=seed)


def water_ingress_blackout(img):
    return np.zeros_like(img)


def mechanical_instability(img, severity="severe", seed=None):
    rotation, translation = preset(
        ((0.3, 2), (0.6, 4), (1.2, 8), (2.0, 14)), severity
    )
    rng = random_stream(seed)
    shift = scaled_px(img, translation)
    return affine(
        img,
        rotation=rng.uniform(-rotation, rotation),
        tx=rng.uniform(-shift, shift),
        ty=rng.uniform(-shift, shift),
    )


FUNCTIONS = {
    "rain": rain,
    "snow": snow,
    "ice": ice,
    "fog": fog,
    "external_scatter": external_scatter,
    "window_contamination": window_contamination,
    "flare": flare,
    "condensation": condensation,
    "sensor_spots": sensor_spots,
    "brackish_corrosion": brackish_corrosion,
    "water_ingress": water_ingress,
    "water_ingress_blackout": water_ingress_blackout,
    "mechanical_instability": mechanical_instability,
}


def apply_fault_environment(img, kind, **kwargs):
    return dispatch(FUNCTIONS, img, kind, kwargs)