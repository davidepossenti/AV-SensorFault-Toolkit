"""Software image faults and temporal parameter presets."""

import cv2
import numpy as np

from .common import (
    dispatch,
    preset,
    random_stream,
    to_uint8,
)


def temporal_delay_frames(severity="severe"):
    return preset((1, 2, 4, 8), severity)


def temporal_ghost_parameters(severity="severe"):
    return (
        temporal_delay_frames(severity),
        preset((0.10, 0.20, 0.35, 0.50), severity),
    )


def chromatic_aberration(img, severity="severe"):
    amount = preset((0.0010, 0.0018, 0.0025, 0.0040), severity)
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    x, y = xx - cx, yy - cy
    radius = np.hypot(x, y) / max(cx, cy)

    out = img.copy()
    for channel, sign in ((0, -1), (2, 1)):
        scale = 1 + sign * amount * radius
        out[..., channel] = cv2.remap(
            img[..., channel],
            (cx + x * scale).astype(np.float32),
            (cy + y * scale).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return out


def demosaic_incomplete(img, severity="severe"):
    block = preset((2, 3, 4, 6), severity)
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_NEAREST,
    )
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = out.astype(np.float32)
    out[..., 1] += preset((3, 6, 10, 16), severity)
    return np.clip(out, 0, 255).astype(np.uint8)


def noise(img, severity="severe", seed=None, noise_kind="mixed"):
    if noise_kind not in {"gaussian", "uniform", "impulse", "mixed"}:
        raise ValueError(f"Unknown noise kind: {noise_kind}")

    sigma, amplitude, probability = preset(
        ((4, 4, 0.003), (6, 6, 0.006),
         (8, 8, 0.010), (14, 14, 0.020)), severity
    )
    rng = random_stream(seed)
    out = img.astype(np.float32)

    if noise_kind in {"gaussian", "mixed"}:
        out += rng.normal(0, sigma, out.shape).astype(np.float32)
    if noise_kind in {"uniform", "mixed"}:
        out += rng.uniform(-amplitude, amplitude, out.shape).astype(np.float32)
    if noise_kind in {"impulse", "mixed"}:
        mask = rng.random(img.shape[:2]) < probability
        out[mask] = rng.choice(
            (0.0, 255.0), size=(int(mask.sum()), 1)
        )

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def data_loss(img, severity="severe", seed=None, mode="mixed", side="bottom"):
    if mode not in {"rows", "blocks", "partial", "mixed"}:
        raise ValueError(f"Unknown data-loss mode: {mode}")
    if side not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"Unknown partial-loss side: {side}")

    fraction = preset((0.05, 0.10, 0.20, 0.35), severity)
    coverage = preset((0.25, 0.40, 0.50, 0.75), severity)
    rng = random_stream(seed)
    h, w = img.shape[:2]
    out = img.copy()

    if mode == "mixed":
        mode = str(rng.choice(("rows", "blocks", "partial")))

    if mode == "rows":
        rows = rng.choice(h, size=max(1, round(h * fraction)), replace=False)
        out[rows] = 0

    elif mode == "blocks":
        count = 12
        side_fraction = np.sqrt(fraction / count)
        bh = max(1, min(h, round(h * side_fraction)))
        bw = max(1, min(w, round(w * side_fraction)))
        for _ in range(count):
            y = int(rng.integers(0, h - bh + 1))
            x = int(rng.integers(0, w - bw + 1))
            out[y:y + bh, x:x + bw] = 0

    else:
        if side in {"left", "right"}:
            size = max(1, round(w * coverage))
            if side == "left":
                out[:, :size] = 0
            else:
                out[:, w - size:] = 0
        else:
            size = max(1, round(h * coverage))
            if side == "top":
                out[:size] = 0
            else:
                out[h - size:] = 0

    return out


FUNCTIONS = {
    "chromatic_aberration": chromatic_aberration,
    "demosaic_incomplete": demosaic_incomplete,
    "noise": noise,
    "data_loss": data_loss,
}


def apply_fault_software(img, kind, **kwargs):
    return dispatch(FUNCTIONS, img, kind, kwargs)