"""Combined camera fault proxies."""

import numpy as np

from .common import (
    check_image,
    dispatch,
    random_stream,
    to_float,
    to_uint8,
)
from .faults_calibration import mounting
from .faults_environment import condensation, fog
from .faults_hardware import broken_lens


def component_seed(seed, stream):
    return int(random_stream(seed, stream).integers(0, 2**31))


def broken_lens_condensation(
    img, severity="severe", seed=None, frame_seed=None
):
    damaged = broken_lens(
        img, severity=severity, seed=component_seed(seed, 1)
    )
    return condensation(
        damaged, severity=severity, seed=component_seed(seed, 2)
    )


def mounting_fog(img, severity="severe", seed=None, frame_seed=None):
    shifted = mounting(
        img, severity=severity, seed=component_seed(seed, 1)
    )
    return fog(
        shifted,
        severity=severity,
        seed=seed if frame_seed is None else frame_seed,
    )


def temporal_ghosting(current, previous, alpha):
    check_image(current)
    check_image(previous)
    if current.shape != previous.shape:
        raise ValueError("Temporal frames must have identical shapes")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return to_uint8(
        (1 - alpha) * to_float(current) + alpha * to_float(previous)
    )


FUNCTIONS = {
    "broken_lens_condensation": broken_lens_condensation,
    "mounting_fog": mounting_fog,
}


def apply_fault_synergistic(img, kind, **kwargs):
    return dispatch(FUNCTIONS, img, kind, kwargs)