"""Shared helpers of the camera verification and figure scripts."""

import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera.camera_injector import (
    BINARY_FAULTS,
    FAULT_REGISTRY,
    FRAME_SEED_FAULTS,
    ROUTERS,
    SEEDED_FAULTS,
    TEMPORAL_FAULTS,
    natural_key,
    severity_label,
)

DEFAULT_IN_DIR = os.environ.get("BOREAS_CAMERA_DIR", "data/boreas-2020-11-26-13-58/camera")
DEFAULT_IMAGE = os.path.join(DEFAULT_IN_DIR, "1606417097547313.png")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Non-default parameters exercised on top of the default call of each fault.
VARIANTS: List[Tuple[str, Dict]] = [
    ("hardware:brightness_exposure", {"direction": "bright"}),
    ("hardware:no_bayer", {"mode": "mono"}),
    ("hardware:banding", {"orientation": "vertical"}),
    ("calibration:white_balance", {"preset_name": "cool"}),
    ("calibration:white_balance", {"preset_name": "green"}),
    ("calibration:white_balance", {"preset_name": "magenta"}),
    ("software:noise", {"noise_kind": "gaussian"}),
    ("software:noise", {"noise_kind": "uniform"}),
    ("software:noise", {"noise_kind": "impulse"}),
    ("software:data_loss", {"mode": "rows"}),
    ("software:data_loss", {"mode": "blocks"}),
    ("software:data_loss", {"mode": "partial", "side": "bottom"}),
    ("software:data_loss", {"mode": "partial", "side": "top"}),
    ("software:data_loss", {"mode": "partial", "side": "left"}),
    ("software:data_loss", {"mode": "partial", "side": "right"}),
]


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def list_images(in_dir: str) -> List[str]:
    return sorted(
        (f for f in os.listdir(in_dir) if f.lower().endswith(IMAGE_EXTENSIONS)),
        key=natural_key,
    )


def per_frame_keys() -> List[str]:
    """Fault keys applied frame by frame, in registry order."""
    return [
        f"{family}:{kind}"
        for family, kinds in FAULT_REGISTRY.items()
        for kind in kinds
        if f"{family}:{kind}" not in TEMPORAL_FAULTS
    ]


def build_kwargs(key: str, severity: str = "severe", seed: int = 1,
                 frame_seed: int = 2, extra: Optional[Dict] = None) -> Dict:
    """Call kwargs resolved exactly as the injector does."""
    kwargs = dict(extra or {})
    label = severity_label(key, severity, kwargs)
    if label != "na":
        kwargs["severity"] = label
    if key in SEEDED_FAULTS:
        kwargs["seed"] = seed
    if key in FRAME_SEED_FAULTS:
        kwargs["frame_seed"] = frame_seed
    return kwargs


def apply_fault(img: np.ndarray, key: str, kwargs: Dict) -> np.ndarray:
    family, kind = key.split(":", 1)
    return ROUTERS[family](img, kind, **kwargs)


def tag(key: str, extra: Optional[Dict] = None) -> str:
    if not extra:
        return key
    return key + "[" + ",".join(f"{k}={v}" for k, v in sorted(extra.items())) + "]"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0 or max(img.shape[:2]) <= max_side:
        return img
    scale = max_side / max(img.shape[:2])
    size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def image_metrics(clean: np.ndarray, out: np.ndarray) -> Dict[str, float]:
    """Change statistics between a clean frame and a corrupted one."""
    diff = np.abs(clean.astype(np.int16) - out.astype(np.int16))
    lap_in = float(cv2.Laplacian(_gray(clean), cv2.CV_64F).var())
    lap_out = float(cv2.Laplacian(_gray(out), cv2.CV_64F).var())
    return {
        "changed_pct": 100.0 * float((diff.max(axis=2) > 0).mean()),
        "mad": float(diff.mean()),
        "luma_in": float(_gray(clean).mean()),
        "luma_out": float(_gray(out).mean()),
        "lap_ratio": lap_out / max(lap_in, 1e-9),
    }