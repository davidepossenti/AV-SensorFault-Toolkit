#!/usr/bin/env python3
"""
make_camera_figures.py

Builds the paper figure: the ground-truth frame plus one fault per category,
saved as a composite (PNG and PDF), as individual full-resolution panels, and
with a JSON manifest of the exact parameters and seeds. With --catalog it
also builds one grid per category with every per-frame fault.

Usage (from the repository root):
    python make_camera_figures.py [--image PATH] [--out-dir DIR]
                                        [--severity LEVEL] [--catalog]
"""

import argparse
import json
import os
import re

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from camera.camera_injector import ALL_FAULT_KEYS, FAULT_REGISTRY, TEMPORAL_FAULTS
from camera.modules.common import SEVERITIES
from camera_check_utils import (
    DEFAULT_IMAGE, apply_fault, build_kwargs, read_bgr, resize_max_side, slug, tag,
)

# One representative fault per category (edit to change the figure).
SELECTION = [
    ("hardware:broken_lens", {}),
    ("environment:window_contamination", {}),
    ("calibration:lens_distortion", {}),
    ("software:data_loss", {"mode": "partial", "side": "bottom"}),
    ("synergistic:broken_lens_condensation", {}),
]


def display_name(key: str) -> str:
    family, kind = key.split(":", 1)
    return f"{family.capitalize()}: {kind.replace('_', ' ')}"


def render(clean, key, extra, severity, seed, frame_seed):
    kwargs = build_kwargs(key, severity, seed, frame_seed, extra)
    return apply_fault(clean.copy(), key, kwargs), kwargs


def plot_grid(panels, columns, path, dpi):
    rows = int(np.ceil(len(panels) / columns))
    height, width = panels[0][1].shape[:2]
    cell_w = 5.0
    cell_h = cell_w * height / width + 0.45
    fig, axes = plt.subplots(rows, columns, figsize=(cell_w * columns, cell_h * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(path + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def remove_stale_panels(out_dir: str) -> None:
    """Delete numbered panels of a previous run so they cannot be mistaken for new ones."""
    for name in os.listdir(out_dir):
        if re.fullmatch(r"\d{2}_.*\.png", name):
            os.remove(os.path.join(out_dir, name))


def main():
    parser = argparse.ArgumentParser(description="Paper figures of the camera faults")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--severity", default="severe", choices=SEVERITIES)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frame-seed", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-max-side", type=int, default=1200,
                        help="Longest side of the panels in the composite figure")
    parser.add_argument("--catalog", action="store_true",
                        help="Also build one grid per category with all per-frame faults")
    args = parser.parse_args()

    for key, _ in SELECTION:
        if key not in ALL_FAULT_KEYS:
            raise SystemExit(f"Unknown fault in SELECTION: {key}")
        if key in TEMPORAL_FAULTS:
            raise SystemExit(f"{key} needs a frame sequence and cannot be shown on a single frame")

    clean = read_bgr(args.image)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.image)).rstrip(os.sep) + "_figures"
    os.makedirs(out_dir, exist_ok=True)
    remove_stale_panels(out_dir)

    if not cv2.imwrite(os.path.join(out_dir, "00_ground_truth.png"), clean):
        raise SystemExit("Cannot write 00_ground_truth.png")
    panels = [("(a) Ground truth", resize_max_side(clean, args.figure_max_side))]
    manifest = [{"panel": "a", "fault": None, "file": "00_ground_truth.png"}]

    for number, (key, extra) in enumerate(SELECTION, start=1):
        out, kwargs = render(clean, key, extra, args.severity, args.seed, args.frame_seed)
        file_name = f"{number:02d}_{slug(tag(key, extra))}.png"
        if not cv2.imwrite(os.path.join(out_dir, file_name), out):
            raise SystemExit(f"Cannot write {file_name}")
        letter = chr(ord("a") + number)
        panels.append((f"({letter}) {display_name(key)}", resize_max_side(out, args.figure_max_side)))
        manifest.append({"panel": letter, "fault": key, "kwargs": kwargs, "file": file_name})
        print(f"rendered {key}")

    plot_grid(panels, 3, os.path.join(out_dir, "figure_camera_faults"), args.dpi)
    with open(os.path.join(out_dir, "figure_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"image": os.path.abspath(args.image), "severity": args.severity,
                   "panels": manifest}, handle, indent=2)

    if args.catalog:
        for family, kinds in FAULT_REGISTRY.items():
            keys = [f"{family}:{kind}" for kind in kinds if f"{family}:{kind}" not in TEMPORAL_FAULTS]
            catalog = [("Ground truth", resize_max_side(clean, args.figure_max_side))]
            for key in keys:
                out, _ = render(clean, key, {}, args.severity, args.seed, args.frame_seed)
                catalog.append((key.split(":", 1)[1].replace("_", " "),
                                resize_max_side(out, args.figure_max_side)))
            plot_grid(catalog, 4, os.path.join(out_dir, f"catalog_{family}"), args.dpi)
            print(f"catalog {family}: {len(keys)} faults")

    print(f"\nFigures saved in {out_dir}")


if __name__ == "__main__":
    main()