#!/usr/bin/env python3
"""
make_lidar_figures.py

Builds the LiDAR paper figures with the same layout as the camera ones: a
composite with the ground-truth frame plus one fault per category, and, with
--catalog, one grid per category with every per-frame fault. Each panel is a
top-down view colored by intensity (square-root scaling), rendered as a raster
below 1200 px so the PDFs stay light. Under every faulty cloud the clean frame
is drawn in light gray, so displaced points (calibration) and removed points
(blind sectors, gates) stay visible; gray and colored points are sampled
consistently. In the catalogs every faulty panel reports the change in the
number of points with respect to the clean frame.

Every fault is deterministic: seeds derive from the fault key, and the faults
whose damage depends on where the pattern lands (scanning, window and their
synergy) take the seed, among a fixed set of candidates, that hides the most
far-away points. Seeds, severity and point counts are saved in
figure_manifest.json.

Usage (from the repository root):
    python make_lidar_figures.py [--frame PATH] [--out-dir DIR]
                                 [--severity LEVEL] [--catalog]
"""

import argparse
import json
import os
import re
import sys
import zlib
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Make the repository root importable from the root or from a subfolder.
HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (HERE, os.path.dirname(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common.lidar_loader import (  # noqa: E402
    DEFAULT_STRIDE, SUPPORTED_STRIDES, PointCloud, load_point_cloud,
)
from lidar.lidar_injector import (  # noqa: E402
    FAULT_REGISTRY, FRAME_SEED_FAULTS, NO_SEED, ROUTERS, SEVERITY_AWARE,
    SEVERITY_LEVELS, TEMPORAL_FAULTS,
)

matplotlib.rcParams["pdf.fonttype"] = 42  # embed fonts as TrueType

DEFAULT_FRAME = os.environ.get("BOREAS_LIDAR_FRAME", "data/boreas-2020-11-26-13-58/lidar/1606417097502930.bin")

# One representative fault per category, in the order shown in the paper.
SELECTION = [
    ("hardware", "photodetector"),
    ("environment", "window"),
    ("calibration", "mounting"),
    ("software", "range_gating"),
    ("synergistic", "scanning_fog"),
]

# Faults whose visibility depends on the seed: it is searched, not drawn.
SEARCHED = {("hardware", "scanning"), ("environment", "window"), ("synergistic", "scanning_fog")}

# Seeds pinned for the default frame at severity "severe" (100 candidate seeds).
SEED_OVERRIDES = {
    "hardware:scanning": 63,
    "environment:window": 49,
    "synergistic:scanning_fog": 0,
}

INTENSITY_GAMMA = 0.5       # square-root scaling: dark returns stay distinguishable
FAR_RANGE_M = 15.0          # "far" points used to score candidate seeds
GHOST_COLOR = "#d0d0d0"     # clean frame drawn under the faulty one
SYNTHETIC_COLOR = "#c000c0"
GHOST_SIZE, REAL_SIZE, SYNTHETIC_SIZE = 0.5, 0.5, 0.7
SYNTHETIC_ALPHA = 0.55
RENDER_DPI = 200
BORDER_PX = 2
BORDER_RGB = (150, 150, 150)

Panel = Tuple[str, np.ndarray]  # title, RGB image


def stable_seed(label: str) -> int:
    """Deterministic seed derived from a label (unlike hash(), identical across runs)."""
    return zlib.crc32(label.encode()) % (2 ** 31)


def real_mask(cloud: PointCloud) -> np.ndarray:
    """Points added by a fault carry NaN in their extra fields."""
    if cloud.extra.shape[1] == 0:
        return np.ones(cloud.n_points, dtype=bool)
    return ~np.isnan(cloud.extra).any(axis=1)


def intensity_bounds(clean: PointCloud):
    """Robust intensity range of the clean frame, shared by all panels."""
    lo, hi = np.percentile(clean.intensity, [1, 99])
    return float(lo), float(max(hi, lo + 1e-6))


def format_change(before: int, after: int) -> str:
    """Change in the number of points, rounded to an integer percentage."""
    change = int(round(100.0 * (after - before) / before))
    return "0% pts" if change == 0 else f"{change:+d}% pts"


def apply_fault(cloud: PointCloud, fam: str, kind: str, severity: str,
                seed: Optional[int], frame_seed: int) -> PointCloud:
    """Call a fault the same way lidar_injector does."""
    kwargs = {}
    if (fam, kind) in SEVERITY_AWARE:
        kwargs["severity"] = severity
    if (fam, kind) not in NO_SEED:
        kwargs["seed"] = seed
    if (fam, kind) in FRAME_SEED_FAULTS:
        kwargs["frame_seed"] = frame_seed
    return ROUTERS[fam](cloud, kind, **kwargs)


def search_seed(cloud: PointCloud, fam: str, kind: str, severity: str,
                n_seeds: int, view_range: float) -> int:
    """Seed in [0, n_seeds) whose fault removes the most real points between FAR_RANGE_M and view_range."""
    frame_seed = stable_seed(f"{fam}:{kind}:frame")

    def far_real(c: PointCloud) -> int:
        r = np.linalg.norm(c.xyz, axis=1)
        return int((real_mask(c) & (r > FAR_RANGE_M) & (r < view_range)).sum())

    best_seed, best_count = 0, None
    for seed in range(n_seeds):
        count = far_real(apply_fault(cloud, fam, kind, severity, seed, frame_seed))
        if best_count is None or count < best_count:
            best_seed, best_count = seed, count
    return best_seed


def sampling_masks(clean: PointCloud, out: PointCloud, rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Boolean masks (clean_keep, out_keep) that pick the same physical points in
    both clouds, so the gray ground truth and the colored fault line up.
    Points sharing a (ring, time) key are sampled together; synthetic points are
    sampled on their own. Falls back to independent sampling when the clouds
    share no identity.
    """
    rng = np.random.default_rng(0)
    if rate >= 1.0:
        return np.ones(clean.n_points, dtype=bool), np.ones(out.n_points, dtype=bool)

    real = real_mask(out)
    if clean.extra.shape[1] >= 2 and out.extra.shape[1] >= 2:
        keys = np.concatenate([clean.extra[:, :2], out.extra[real, :2]], axis=0)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        clean_id, out_id = inverse[:clean.n_points], inverse[clean.n_points:]

        if len(out_id) == 0 or np.isin(out_id, clean_id).mean() >= 0.5:
            selected = rng.random(len(unique_keys)) < rate
            out_keep = np.zeros(out.n_points, dtype=bool)
            out_keep[real] = selected[out_id]
            out_keep[~real] = rng.random(int((~real).sum())) < rate
            return selected[clean_id], out_keep

    return rng.random(clean.n_points) < rate, rng.random(out.n_points) < rate


def draw_bev(ax, cloud: PointCloud, keep: np.ndarray, bounds, view_range: float,
             scale_bar: bool, ghost: Optional[PointCloud] = None,
             ghost_keep: Optional[np.ndarray] = None) -> None:
    """Top-down view (x right, y up): optional gray ghost, synthetic points, then real points on top."""
    def inside(c: PointCloud) -> np.ndarray:
        return (np.abs(c.xyz[:, 0]) <= view_range) & (np.abs(c.xyz[:, 1]) <= view_range)

    if ghost is not None and ghost_keep is not None:
        g = ghost_keep & inside(ghost)
        ax.scatter(ghost.xyz[g, 0], ghost.xyz[g, 1], c=GHOST_COLOR, s=GHOST_SIZE, marker="s",
                   linewidths=0, zorder=1)

    x, y = cloud.xyz[:, 0], cloud.xyz[:, 1]
    real = real_mask(cloud)
    lo, hi = bounds
    color = np.clip((cloud.intensity - lo) / (hi - lo), 0.0, 1.0) ** INTENSITY_GAMMA

    visible = keep & inside(cloud)
    syn = visible & ~real
    if syn.any():
        ax.scatter(x[syn], y[syn], c=SYNTHETIC_COLOR, s=SYNTHETIC_SIZE, marker="s",
                   linewidths=0, alpha=SYNTHETIC_ALPHA, zorder=2)
    sel = visible & real
    if sel.any():
        ax.scatter(x[sel], y[sel], c=color[sel], cmap="viridis", vmin=0, vmax=1,
                   s=REAL_SIZE, marker="s", linewidths=0, zorder=3)

    if scale_bar:
        x0 = y0 = -0.92 * view_range
        ax.plot([x0, x0 + 10.0], [y0, y0], color="black", linewidth=1.8,
                solid_capstyle="butt", zorder=4)
        ax.text(x0 + 5.0, y0 + 0.03 * view_range, "10 m", ha="center", va="bottom",
                fontsize=11, zorder=4)

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    ax.axis("off")


def render_panel(cloud: PointCloud, keep: np.ndarray, bounds, view_range: float,
                 scale_bar: bool, panel_px: int, ghost: Optional[PointCloud] = None,
                 ghost_keep: Optional[np.ndarray] = None) -> np.ndarray:
    """Render one top-down view to a square RGB array of panel_px pixels."""
    size = (panel_px + 0.01) / RENDER_DPI  # avoids a one-pixel floor from float rounding
    fig = Figure(figsize=(size, size), dpi=RENDER_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_bev(ax, cloud, keep, bounds, view_range, scale_bar, ghost, ghost_keep)
    canvas.draw()

    image = np.asarray(canvas.buffer_rgba())[:panel_px, :panel_px, :3].copy()
    if image.shape[:2] != (panel_px, panel_px):
        raise RuntimeError(f"Unexpected panel size {image.shape[:2]}")
    image[:BORDER_PX] = BORDER_RGB
    image[-BORDER_PX:] = BORDER_RGB
    image[:, :BORDER_PX] = BORDER_RGB
    image[:, -BORDER_PX:] = BORDER_RGB
    return image


def plot_grid(panels: List[Panel], columns: int, path: str, dpi: int,
              width_in: float = 7.2, font_size: float = 7) -> None:
    rows = int(np.ceil(len(panels) / columns))
    cell_w = width_in / columns
    cell_h = cell_w + 0.3
    fig, axes = plt.subplots(rows, columns, figsize=(width_in, cell_h * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=font_size)
    fig.tight_layout(pad=0.4)
    fig.savefig(path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(path + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def display_name(fam: str, kind: str) -> str:
    return f"{fam.capitalize()}: {kind.replace('_', ' ')}"


def remove_stale_panels(out_dir: str) -> None:
    """Delete numbered panels of a previous run so they cannot be mistaken for new ones."""
    for name in os.listdir(out_dir):
        if re.fullmatch(r"\d{2}_.*\.png", name):
            os.remove(os.path.join(out_dir, name))


def panel_px_type(value: str) -> int:
    px = int(value)
    if not 100 <= px < 1200:
        raise argparse.ArgumentTypeError("panel size must be between 100 and 1199 px")
    return px


def parse_args():
    p = argparse.ArgumentParser(description="Paper figures of the LiDAR faults")
    p.add_argument("--frame", default=DEFAULT_FRAME, help="Clean .bin frame")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, choices=SUPPORTED_STRIDES)
    p.add_argument("--severity", default="severe", choices=SEVERITY_LEVELS)
    p.add_argument("--view-range", type=float, default=60.0, help="Half-extent of the view [m]")
    p.add_argument("--max-points", type=int, default=60000,
                   help="Point budget per panel (same sampling rate for every panel; 0 = all)")
    p.add_argument("--search-seeds", type=int, default=100,
                   help="Candidate seeds for scene-dependent faults")
    p.add_argument("--panel-px", type=panel_px_type, default=800,
                   help="Side of each raster panel in pixels (below 1200)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-ghost", action="store_true",
                   help="Do not draw the clean frame in gray under the faults")
    p.add_argument("--no-scale-bar", action="store_true")
    p.add_argument("--catalog", action="store_true",
                   help="Also build one grid per category with all per-frame faults")
    return p.parse_args()


def main():
    args = parse_args()

    for fam, kind in SELECTION:
        if kind not in FAULT_REGISTRY.get(fam, []):
            raise SystemExit(f"Unknown fault in SELECTION: {fam}:{kind}")
        if (fam, kind) in TEMPORAL_FAULTS:
            raise SystemExit(f"{fam}:{kind} needs a frame sequence and cannot be shown on a single frame")

    clean = load_point_cloud(args.frame, stride=args.stride)
    bounds = intensity_bounds(clean)
    rate = min(1.0, args.max_points / clean.n_points) if args.max_points > 0 else 1.0
    scale_bar = not args.no_scale_bar
    ghost = None if args.no_ghost else clean
    use_overrides = (args.severity == "severe"
                     and os.path.abspath(args.frame) == os.path.abspath(DEFAULT_FRAME))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.frame)).rstrip(os.sep) + "_figures"
    os.makedirs(out_dir, exist_ok=True)
    remove_stale_panels(out_dir)

    clean_keep, _ = sampling_masks(clean, clean, rate)
    gt_image = render_panel(clean, clean_keep, bounds, args.view_range, scale_bar, args.panel_px)

    cache: Dict[str, Tuple[np.ndarray, dict]] = {}

    def panel_for(fam: str, kind: str) -> Tuple[np.ndarray, dict]:
        key = f"{fam}:{kind}"
        if key in cache:
            return cache[key]

        seed: Optional[int] = None
        frame_seed = stable_seed(f"{key}:frame")
        if (fam, kind) not in NO_SEED:
            if use_overrides and key in SEED_OVERRIDES:
                seed = SEED_OVERRIDES[key]
            elif (fam, kind) in SEARCHED:
                seed = search_seed(clean, fam, kind, args.severity, args.search_seeds, args.view_range)
            else:
                seed = stable_seed(key)

        out = apply_fault(clean, fam, kind, args.severity, seed, frame_seed)
        clean_keep_f, out_keep = sampling_masks(clean, out, rate)
        image = render_panel(out, out_keep, bounds, args.view_range, scale_bar, args.panel_px,
                             ghost, None if ghost is None else clean_keep_f)
        record = {
            "fault": key,
            "severity": args.severity if (fam, kind) in SEVERITY_AWARE else "na",
            "seed": None if seed is None else int(seed),
            "frame_seed": frame_seed if (fam, kind) in FRAME_SEED_FAULTS else None,
            "points_before": int(clean.n_points), "points_after": int(out.n_points),
        }
        print(f"{key:36s} severity={record['severity']:8s} seed={seed} "
              f"points {clean.n_points} -> {out.n_points}")
        cache[key] = (image, record)
        return cache[key]

    mpimg.imsave(os.path.join(out_dir, "00_ground_truth.png"), gt_image)
    panels: List[Panel] = [("(a) Ground truth", gt_image)]
    manifest = [{"panel": "a", "fault": None, "file": "00_ground_truth.png"}]

    for number, (fam, kind) in enumerate(SELECTION, start=1):
        image, record = panel_for(fam, kind)
        file_name = f"{number:02d}_{fam}_{kind}.png"
        mpimg.imsave(os.path.join(out_dir, file_name), image)
        letter = chr(ord("a") + number)
        panels.append((f"({letter}) {display_name(fam, kind)}", image))
        manifest.append({"panel": letter, "file": file_name, **record})

    plot_grid(panels, 3, os.path.join(out_dir, "figure_lidar_faults"), args.dpi)

    catalogs = {}
    if args.catalog:
        skipped = [f"{f}:{k}" for f, kinds in FAULT_REGISTRY.items() for k in kinds
                   if (f, k) in TEMPORAL_FAULTS]
        print(f"not shown (need a frame sequence): {', '.join(skipped)}")
        for fam, kinds in FAULT_REGISTRY.items():
            shown = [k for k in kinds if (fam, k) not in TEMPORAL_FAULTS]
            catalog: List[Panel] = [("Ground truth", gt_image)]
            for kind in shown:
                image, record = panel_for(fam, kind)
                change = format_change(record["points_before"], record["points_after"])
                catalog.append((f"{kind.replace('_', ' ')} ({change})", image))
            plot_grid(catalog, 4, os.path.join(out_dir, f"catalog_lidar_{fam}"), args.dpi,
                      font_size=6)
            catalogs[fam] = [f"{fam}:{k}" for k in shown]
            print(f"catalog {fam}: {len(shown)} faults")

    with open(os.path.join(out_dir, "figure_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"frame": os.path.abspath(args.frame), "severity": args.severity,
                   "view_range_m": args.view_range, "panel_px": args.panel_px,
                   "sampling_rate": rate, "ghost": not args.no_ghost,
                   "intensity_gamma": INTENSITY_GAMMA,
                   "seed_overrides_used": use_overrides,
                   "panels": manifest, "catalogs": catalogs}, handle, indent=2)

    print(f"\nFigures saved in {out_dir}")


if __name__ == "__main__":
    main()