"""Hardware fault proxies for optics, electronics, CFA and sensor readout."""

import cv2
import numpy as np

from .common import (
    affine,
    bayer_masks,
    dispatch,
    preset,
    random_stream,
    scaled_px,
    to_float,
    to_uint8,
)


def brightness_blackout(img):
    return np.zeros_like(img)


def brightness_whiteout(img):
    return np.full_like(img, 255)


def brightness_exposure(img, severity="severe", direction="dark"):
    if direction not in {"dark", "bright"}:
        raise ValueError("direction must be 'dark' or 'bright'")
    ev = preset((0.5, 1.0, 2.0, 3.5), severity)
    ev *= -1 if direction == "dark" else 1
    return to_uint8(to_float(img) * (2.0 ** ev))


def broken_lens(img, severity="severe", seed=None):
    count, strength, displacement = preset(
        ((1, 0.18, 3), (2, 0.28, 6), (2, 0.40, 10), (3, 0.55, 16)),
        severity,
    )
    h, w = img.shape[:2]
    rng = random_stream(seed)
    cracks = np.zeros((h, w), np.float32)

    for _ in range(count):
        cx = int(rng.uniform(0.2, 0.8) * w)
        cy = int(rng.uniform(0.2, 0.8) * h)

        for angle in rng.uniform(0, 2 * np.pi, 18):
            x, y = cx, cy
            length = rng.uniform(0.25, 0.7) * np.hypot(h, w)

            for step in range(10):
                local_angle = angle + rng.normal(0, 0.12)
                nx = int(x + length / 10 * np.cos(local_angle))
                ny = int(y + length / 10 * np.sin(local_angle))
                width = max(1, round(scaled_px(img, 3 - step / 5)))
                cv2.line(cracks, (x, y), (nx, ny), 1.0, width)
                x, y = nx, ny

    smooth = cv2.GaussianBlur(cracks, (0, 0), max(0.6, scaled_px(img, 2)))
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    yy, xx = np.indices((h, w), dtype=np.float32)

    out = cv2.remap(
        to_float(img),
        xx + gx * scaled_px(img, displacement),
        yy + gy * scaled_px(img, displacement),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    out *= 1.0 - strength * smooth[..., None]
    out += 0.5 * strength * cracks[..., None]
    return to_uint8(out)


def broken_vr(img, severity="severe", seed=None):
    length = max(
        3, int(round(scaled_px(img, preset((11, 19, 31, 47), severity)))) | 1
    )
    angle = random_stream(seed).uniform(0, np.pi)
    kernel = np.zeros((length, length), np.float32)
    center = length // 2
    dx = int(round(center * np.cos(angle)))
    dy = int(round(center * np.sin(angle)))
    cv2.line(
        kernel, (center - dx, center - dy),
        (center + dx, center + dy), 1.0, 1,
    )
    kernel /= kernel.sum()
    return cv2.filter2D(
        img, -1, kernel, borderType=cv2.BORDER_REPLICATE
    )


def heat_failure_blur(img, severity="severe"):
    sigma, center_mix = preset(
        ((3, 0.25), (5, 0.35), (8, 0.45), (12, 0.60)), severity
    )
    blurred = to_float(cv2.GaussianBlur(
        img, (0, 0), max(0.2, scaled_px(img, sigma))
    ))
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    radius2 = (((xx - w / 2) / (w / 2)) ** 2
               + ((yy - h / 2) / (h / 2)) ** 2) / 2
    alpha = (center_mix + (1 - center_mix) * np.clip(radius2, 0, 1))[..., None]
    return to_uint8(to_float(img) * (1 - alpha) + blurred * alpha)


def heat_failure_zoom(img, severity="severe"):
    step = preset((0.03, 0.06, 0.08, 0.14), severity)
    return affine(img, scale=1.0 / (1.0 - 2.0 * step))


def electrical_overload_blackout(img):
    return np.zeros_like(img)


def electrical_overload_corrupt(img, severity="severe", seed=None):
    fraction, blocks = preset(
        ((0.03, 4), (0.06, 8), (0.10, 16), (0.18, 28)), severity
    )
    rng = random_stream(seed)
    out = img.copy()
    h, w = img.shape[:2]
    rows = rng.choice(h, size=max(1, int(h * fraction)), replace=False)
    out[rows] = rng.integers(
        0, 256, (len(rows), w, 3), dtype=np.uint8
    )

    for _ in range(blocks):
        bh = int(rng.integers(max(1, h // 30), max(2, h // 8 + 1)))
        bw = int(rng.integers(max(1, w // 30), max(2, w // 8 + 1)))
        y = int(rng.integers(0, h - bh + 1))
        x = int(rng.integers(0, w - bw + 1))
        block = out[y:y + bh, x:x + bw]
        if rng.random() < 0.5:
            block[:] = 255 - block
        else:
            block[:] = rng.integers(0, 256, block.shape, dtype=np.uint8)
    return out


def electrical_overload_partial(img, severity="severe", seed=None):
    gain, amplitude, density = preset(
        ((1.2, 0.08, 0.00005), (1.5, 0.15, 0.0001),
         (1.9, 0.25, 0.0002), (2.3, 0.35, 0.0005)),
        severity,
    )
    rng = random_stream(seed)
    base = to_float(img)
    h, w = img.shape[:2]
    phase = rng.uniform(0, 2 * np.pi)
    band = 1 + amplitude * np.sin(
        np.linspace(0, 12 * np.pi, h, dtype=np.float32) + phase
    )
    out = np.clip(base * gain * band[:, None, None], 0, 1)
    mask = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) > 0.8).astype(np.float32)
    glow = cv2.GaussianBlur(mask, (0, 0), max(0.5, scaled_px(img, 5)))
    out += 0.25 * glow[..., None]

    n = max(1, int(h * w * density))
    ys = rng.integers(h, size=n)
    xs = rng.integers(w, size=n)
    out[ys, xs] = 1.0
    return to_uint8(out)


def no_bayer(img, severity="severe", mode="scramble", pattern="RGGB"):
    if mode == "mono":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode != "scramble":
        raise ValueError("mode must be 'mono' or 'scramble'")

    cross, tint = preset(
        ((0.10, 0.08), (0.18, 0.14), (0.25, 0.20), (0.32, 0.32)),
        severity,
    )
    base = to_float(img)
    mixed = (1 - 3 * cross) * base + cross * base.sum(axis=2, keepdims=True)
    rmask, gmask, bmask = bayer_masks(*img.shape[:2], pattern)
    mixed[..., 2] *= 1 + tint * (bmask.astype(float) - rmask)
    mixed[..., 0] *= 1 + tint * (rmask.astype(float) - bmask)
    mixed[..., 1] *= np.where(gmask, 1 + 0.3 * tint, 1 - 0.15 * tint)
    return to_uint8(mixed)


def no_action_raw(img, pattern="RGGB"):
    masks = bayer_masks(*img.shape[:2], pattern)
    out = np.zeros_like(img)
    for channel, mask in zip((2, 1, 0), masks):
        out[..., channel][mask] = img[..., channel][mask]
    return out


def banding(img, severity="severe", orientation="horizontal", phase=0.0):
    period, amplitude = preset(
        ((20, 0.06), (18, 0.09), (16, 0.12), (12, 0.20)), severity
    )
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("Invalid banding orientation")
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    coordinate = yy if orientation == "horizontal" else xx
    period = max(2.0, scaled_px(img, period))
    wave = np.sin((coordinate + phase) * 2 * np.pi / period)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gain = 1 + amplitude * wave * (1 - 0.6 * gray)
    return to_uint8(to_float(img) * gain[..., None])


def bright_lines(img, severity="severe", seed=None):
    count, width, glow_strength = preset(
        ((1, 1, 0.05), (2, 2, 0.10), (3, 2, 0.15), (5, 3, 0.25)),
        severity,
    )
    rng = random_stream(seed)
    h, w = img.shape[:2]
    width = max(1, round(scaled_px(img, width)))
    mask = np.zeros((h, w), np.float32)
    for x in rng.choice(w, size=min(count, w), replace=False):
        start = max(0, int(x) - width // 2)
        mask[:, start:min(w, start + width)] = 1
    glow = cv2.GaussianBlur(mask, (0, 0), max(0.5, scaled_px(img, 3)))
    out = to_float(img) + glow_strength * glow[..., None]
    out[mask > 0] = 1
    return to_uint8(out)


def dead_pixels(img, severity="severe", hot_ratio=0.5, seed=None):
    if not 0 <= hot_ratio <= 1:
        raise ValueError("hot_ratio must be in [0, 1]")
    density, clusters, size = preset(
        ((1e-5, 20, 4), (2.5e-5, 35, 5), (4e-5, 60, 6), (8e-5, 100, 8)),
        severity,
    )
    h, w = img.shape[:2]
    rng = random_stream(seed)
    out = img.copy()

    # Isolated defective pixels.
    n = min(h * w, max(1, round(density * h * w)))
    positions = rng.choice(h * w, size=n, replace=False)
    out.reshape(-1, 3)[positions] = np.where(
        rng.random(n)[:, None] < hot_ratio, 255, 0
    )

    # Clustered defects, sized to stay visible in the full frame.
    side = max(2, round(scaled_px(img, size)))
    for _ in range(clusters):
        y = int(rng.integers(0, max(1, h - side + 1)))
        x = int(rng.integers(0, max(1, w - side + 1)))
        out[y:y + side, x:x + side] = 255 if rng.random() < hot_ratio else 0
    return out


FUNCTIONS = {
    "brightness_blackout": brightness_blackout,
    "brightness_whiteout": brightness_whiteout,
    "brightness_exposure": brightness_exposure,
    "broken_lens": broken_lens,
    "broken_vr": broken_vr,
    "heat_failure_blur": heat_failure_blur,
    "heat_failure_zoom": heat_failure_zoom,
    "electrical_overload_blackout": electrical_overload_blackout,
    "electrical_overload_corrupt": electrical_overload_corrupt,
    "electrical_overload_partial": electrical_overload_partial,
    "no_bayer": no_bayer,
    "no_action_raw": no_action_raw,
    "banding": banding,
    "bright_lines": bright_lines,
    "dead_pixels": dead_pixels,
}


def apply_fault_hardware(img, kind, **kwargs):
    return dispatch(FUNCTIONS, img, kind, kwargs)