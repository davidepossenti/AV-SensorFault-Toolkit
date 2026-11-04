#!/usr/bin/env python3
"""
make_radar_figures.py

Builds the radar paper figures with the same layout as the camera and LiDAR
ones: a composite with the ground-truth frame plus one fault per category,
and, with --catalog, one grid per category with every per-frame fault. Each
panel is a Cartesian top-down view of the polar frame (every polar cell is a
small polygon, averaged in blocks of range bins), rendered as a raster below
1200 px so the PDFs stay light. In the catalogs every faulty panel reports
the share of range cells whose 8-bit value changes with respect to the clean
frame.

Styles:
  composite  gray = unchanged, orange = clean frame only, blue = faulty frame
             only: attenuation turns orange, ghosts and added noise turn blue,
             displacements give orange/blue fringes. Unchanged echoes keep the
             gray of the ground truth panel. The palette is readable with the
             common forms of color-vision deficiency.
  plain      the faulty frame in gray, like the ground truth.

Every fault is deterministic: seeds derive from the fault key, and the faults
whose damage depends on where the pattern lands (radome sector, thermal
deformation phase) take the seed, among a fixed set of candidates, that
changes the image the most. Seeds, severity and the share of changed cells
are saved in figure_manifest.json.

Display: values are shown with a gamma and a scale taken from the clean
frame, identical for all panels (display only, the data is not modified).

Usage (from the repository root):
    python make_radar_figures.py [--frame PATH] [--out-dir DIR]
                                 [--severity LEVEL] [--style STYLE] [--catalog]
"""

import argparse
import json
import math
import os
import re
import sys
import zlib
from typing import Dict, List, NamedTuple, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Make the repository root importable from the root or from a subfolder.
HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (HERE, os.path.dirname(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common.radar_loader import RadarFrame, load_radar_png  # noqa: E402
from radar.radar_injector import (  # noqa: E402
    FAULT_REGISTRY, FRAME_SEED_FAULTS, GEOMETRIES, NO_SEED, ROUTERS, SEVERITY_AWARE,
    SEVERITY_LEVELS, TEMPORAL_FAULTS, resolve_geometry,
)

matplotlib.rcParams["pdf.fonttype"] = 42  # embed fonts as TrueType

DEFAULT_FRAME = os.environ.get("BOREAS_RADAR_FRAME", "data/boreas-2020-11-26-13-58/radar/1606417097528152.png")

# One representative fault per category, in the order shown in the paper.
SELECTION = [
    ("hardware", "lna"),
    ("environment", "radome"),
    ("calibration", "boresight"),
    ("software", "filtering"),
    ("synergistic", "thermal_multipath"),
]

# Faults whose visibility depends on the seed: it is searched, not drawn.
SEARCHED = {("environment", "radome"), ("synergistic", "thermal_multipath")}

# Pin seeds here (copied from figure_manifest.json) to freeze them for good.
# They apply to the default frame at severity "severe" only.
SEED_OVERRIDES: Dict[str, int] = {}

DISPLAY_GAMMA = 0.5
DISPLAY_PERCENTILE = 99.5
RENDER_DPI = 200
BORDER_PX = 2
BORDER_RGB = (150, 150, 150)

Panel = Tuple[str, np.ndarray]  # title, RGB image


class Grid(NamedTuple):
    verts: np.ndarray    # (K, 4, 2) polygons of the cells inside the view
    keep: np.ndarray     # (n_rows * n_blocks,) cells inside the view
    n_blocks: int        # range blocks per row
    block_bins: int      # range bins averaged in each block


def stable_seed(label: str) -> int:
    """Deterministic seed derived from a label (unlike hash(), identical across runs)."""
    return zlib.crc32(label.encode()) % (2 ** 31)


def apply_fault(frame: RadarFrame, fam: str, kind: str, severity: str,
                seed: Optional[int], frame_seed: int) -> RadarFrame:
    """Call a fault the same way radar_injector does."""
    kwargs = {}
    if (fam, kind) in SEVERITY_AWARE:
        kwargs["severity"] = severity
    if (fam, kind) not in NO_SEED:
        kwargs["seed"] = seed
    if (fam, kind) in FRAME_SEED_FAULTS:
        kwargs["frame_seed"] = frame_seed
    return ROUTERS[fam](frame, kind, **kwargs)


def search_seed(clean: RadarFrame, fam: str, kind: str, severity: str, n_seeds: int) -> int:
    """Seed in [0, n_seeds) whose fault changes the image the most (sum of absolute differences)."""
    frame_seed = stable_seed(f"{fam}:{kind}:frame")
    best_seed, best_score = 0, -1.0
    for seed in range(n_seeds):
        out = apply_fault(clean, fam, kind, severity, seed, frame_seed)
        score = float(np.abs(out.power - clean.power).sum())
        if score > best_score:
            best_seed, best_score = seed, score
    return best_seed


def changed_fraction(clean: RadarFrame, out: RadarFrame) -> float:
    """Share of range cells whose 8-bit value differs between the two frames."""
    return float((np.rint(np.abs(out.power - clean.power) * 255.0) >= 1).mean())


def format_changed(fraction: float) -> str:
    """Share of changed cells, rounded to an integer percentage."""
    percent = 100.0 * fraction
    if percent <= 0.0:
        return "0% cells"
    if percent < 1.0:
        return "<1% cells"
    return f"{int(round(percent))}% cells"


def build_grid(frame: RadarFrame, view_range: float, block_bins: int,
               rotate_deg: float, mirror: bool) -> Grid:
    """Polygons of the polar cells (blocks of `block_bins` range bins) that fall inside the square view."""
    step = block_bins * frame.geometry.range_resolution_m
    n_blocks = min(frame.n_bins // block_bins, int(math.ceil(view_range * math.sqrt(2.0) / step)))
    half = 0.5 * frame.azimuth_step_rad

    sign = -1.0 if mirror else 1.0
    azimuth = sign * frame.azimuth_rad.astype(np.float64)[:, None] + math.radians(rotate_deg)
    th_a, th_b = azimuth - half, azimuth + half
    r0 = (np.arange(n_blocks) * step)[None, :]
    r1 = r0 + step

    rr = np.stack(np.broadcast_arrays(r0, r1, r1, r0), axis=-1)            # (1, n_blocks, 4)
    tt = np.stack(np.broadcast_arrays(th_a, th_a, th_b, th_b), axis=-1)    # (rows, 1, 4)
    x, y = rr * np.cos(tt), rr * np.sin(tt)                                # (rows, n_blocks, 4)

    pad = 3.0 + step
    keep = ((np.abs(x.mean(-1)) <= view_range + pad) & (np.abs(y.mean(-1)) <= view_range + pad)).ravel()
    verts = np.stack([x, y], axis=-1).reshape(-1, 4, 2)[keep]
    return Grid(verts, keep, n_blocks, block_bins)


def cell_values(frame: RadarFrame, grid: Grid) -> np.ndarray:
    """Mean power of every (row, range block) cell."""
    used = grid.n_blocks * grid.block_bins
    return frame.power[:, :used].reshape(frame.n_rows, grid.n_blocks, grid.block_bins).mean(axis=2)


def to_display(values: np.ndarray, scale: float) -> np.ndarray:
    return np.power(np.clip(values / scale, 0.0, 1.0), DISPLAY_GAMMA)


def panel_colors(clean_vals: np.ndarray, out_vals: np.ndarray, scale: float,
                 style: str, grid: Grid) -> np.ndarray:
    """RGB color of every visible cell."""
    shown = to_display(out_vals, scale)
    if style == "composite":
        # Unchanged cells are gray (R = G = B); clean-only cells orange; faulty-only cells blue.
        clean = to_display(clean_vals, scale)
        rgb = np.stack([clean, 0.5 * (clean + shown), shown], axis=-1)
    else:
        rgb = np.stack([shown, shown, shown], axis=-1)
    return rgb.reshape(-1, 3)[grid.keep]


def draw_panel(ax, grid: Grid, rgb: np.ndarray, view_range: float, scale_bar_m: float) -> None:
    """Cartesian view (x right, y up): one polygon per polar cell, drawn on black."""
    ax.add_patch(Rectangle((-view_range, -view_range), 2 * view_range, 2 * view_range,
                           facecolor="black", edgecolor="none", zorder=0))
    # Edges share the face color so neighboring cells leave no seams.
    ax.add_collection(PolyCollection(grid.verts, facecolors=rgb, edgecolors=rgb,
                                     linewidths=0.2, zorder=1))
    if scale_bar_m > 0:
        x0 = y0 = -0.92 * view_range
        ax.plot([x0, x0 + scale_bar_m], [y0, y0], color="white", linewidth=1.8,
                solid_capstyle="butt", zorder=3)
        ax.text(x0 + 0.5 * scale_bar_m, y0 + 0.03 * view_range, f"{scale_bar_m:g} m",
                color="white", ha="center", va="bottom", fontsize=11, zorder=3)
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal")
    ax.axis("off")


def render_panel(grid: Grid, rgb: np.ndarray, view_range: float,
                 scale_bar_m: float, panel_px: int) -> np.ndarray:
    """Render one view to a square RGB array of panel_px pixels."""
    size = (panel_px + 0.01) / RENDER_DPI  # avoids a one-pixel floor from float rounding
    fig = Figure(figsize=(size, size), dpi=RENDER_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_panel(ax, grid, rgb, view_range, scale_bar_m)
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
    p = argparse.ArgumentParser(description="Paper figures of the radar faults")
    p.add_argument("--frame", default=DEFAULT_FRAME, help="Clean polar .png frame")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--geometry", default="boreas", choices=sorted(GEOMETRIES))
    p.add_argument("--range-resolution", type=float, default=None,
                   help="Override the metres per range bin of the chosen geometry")
    p.add_argument("--severity", default="severe", choices=SEVERITY_LEVELS)
    p.add_argument("--style", default="composite", choices=["composite", "plain"])
    p.add_argument("--view-range", type=float, default=100.0, help="Half-extent of the view [m]")
    p.add_argument("--block-bins", type=int, default=8,
                   help="Range bins averaged per cell (smaller = finer but slower)")
    p.add_argument("--rotate-deg", type=float, default=0.0,
                   help="Rotate the view (e.g. to match the orientation of the LiDAR figure)")
    p.add_argument("--mirror", action="store_true", help="Mirror the view (azimuth direction)")
    p.add_argument("--search-seeds", type=int, default=100,
                   help="Candidate seeds for scene-dependent faults")
    p.add_argument("--panel-px", type=panel_px_type, default=800,
                   help="Side of each raster panel in pixels (below 1200)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--scale-bar-m", type=float, default=20.0, help="Scale bar length [m]")
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

    geometry = resolve_geometry(args.geometry, args.range_resolution)
    clean = load_radar_png(args.frame, geometry)
    scale_bar_m = 0.0 if args.no_scale_bar else args.scale_bar_m

    grid = build_grid(clean, args.view_range, args.block_bins, args.rotate_deg, args.mirror)
    clean_vals = cell_values(clean, grid)
    scale = max(float(np.percentile(clean_vals, DISPLAY_PERCENTILE)), 1.0 / 255.0)
    mean_clean = float(clean.power.mean())
    use_overrides = (args.severity == "severe"
                     and os.path.abspath(args.frame) == os.path.abspath(DEFAULT_FRAME))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.frame)).rstrip(os.sep) + "_figures"
    os.makedirs(out_dir, exist_ok=True)
    remove_stale_panels(out_dir)

    gt_image = render_panel(grid, panel_colors(clean_vals, clean_vals, scale, "plain", grid),
                            args.view_range, scale_bar_m, args.panel_px)

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
                seed = search_seed(clean, fam, kind, args.severity, args.search_seeds)
            else:
                seed = stable_seed(key)

        out = apply_fault(clean, fam, kind, args.severity, seed, frame_seed)
        changed = changed_fraction(clean, out)
        rgb = panel_colors(clean_vals, cell_values(out, grid), scale, args.style, grid)
        image = render_panel(grid, rgb, args.view_range, scale_bar_m, args.panel_px)
        record = {
            "fault": key,
            "severity": args.severity if (fam, kind) in SEVERITY_AWARE else "na",
            "seed": None if seed is None else int(seed),
            "frame_seed": frame_seed if (fam, kind) in FRAME_SEED_FAULTS else None,
            "mean_power_before": mean_clean, "mean_power_after": float(out.power.mean()),
            "cells_changed": changed,
        }
        print(f"{key:36s} severity={record['severity']:8s} seed={seed} "
              f"cells changed {changed:.1%}")
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

    plot_grid(panels, 3, os.path.join(out_dir, "figure_radar_faults"), args.dpi)

    catalogs = {}
    if args.catalog:
        skipped = [f"{f}:{k}" for f, kinds in FAULT_REGISTRY.items() for k in kinds
                   if (f, k) in TEMPORAL_FAULTS]
        print(f"not shown (need a frame sequence): {', '.join(skipped) if skipped else 'none'}")
        for fam, kinds in FAULT_REGISTRY.items():
            shown = [k for k in kinds if (fam, k) not in TEMPORAL_FAULTS]
            catalog: List[Panel] = [("Ground truth", gt_image)]
            for kind in shown:
                image, record = panel_for(fam, kind)
                share = format_changed(record["cells_changed"])
                catalog.append((f"{kind.replace('_', ' ')} ({share})", image))
            plot_grid(catalog, 4, os.path.join(out_dir, f"catalog_radar_{fam}"), args.dpi,
                      font_size=6)
            catalogs[fam] = [f"{fam}:{k}" for k in shown]
            print(f"catalog {fam}: {len(shown)} faults")

    with open(os.path.join(out_dir, "figure_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"frame": os.path.abspath(args.frame), "geometry": args.geometry,
                   "severity": args.severity, "style": args.style,
                   "view_range_m": args.view_range, "block_bins": args.block_bins,
                   "rotate_deg": args.rotate_deg, "mirror": args.mirror,
                   "panel_px": args.panel_px, "display_gamma": DISPLAY_GAMMA,
                   "seed_overrides_used": use_overrides,
                   "panels": manifest, "catalogs": catalogs}, handle, indent=2)

    print(f"\nFigures saved in {out_dir}")


if __name__ == "__main__":
    main()