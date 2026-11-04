#!/usr/bin/env python3
"""
visualize_lidar_faults.py

Side-by-side visual comparison of clean vs fault-injected LiDAR frames.
Reads labels.csv produced by lidar_injector.py to select faulty frames and
annotate each figure with the injected fault and severity.

Each 2D figure contains:
  - Top row:    bird's-eye view (clean | corrupted | overlay)
  - Bottom row: side view (clean | corrupted) and range histogram
In the overlay, gray points are clean points that were removed or displaced
by the fault. Points injected synthetically (NaN extra fields) are magenta.

Both clouds are subsampled consistently when possible. Points are identified by
their first two extra fields (e.g. ring and time). These keys are not unique
(timestamps are quantized, so several points share one key), hence keys are
sampled as groups: every point carrying a selected key is kept in both clouds,
which gives the same sampling rate on both sides. When the corrupted points do
not come from the clean frame (e.g. a frozen frame), sampling is independent
and the overlay says so.
Intensity colors use the clean frame's range for both clouds.
"""

import argparse
import os
from typing import Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

from common.lidar_loader import DEFAULT_STRIDE, SUPPORTED_STRIDES, PointCloud, load_point_cloud

SYNTHETIC_COLOR = "magenta"


def _intensity_bounds(clean: PointCloud) -> Tuple[float, float]:
    """Robust intensity range of the clean frame, shared by both clouds."""
    if clean.n_points == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(clean.intensity, [1, 99])
    return float(lo), float(max(hi, lo + 1e-6))


def _intensity_colors(intensity: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((intensity - lo) / (hi - lo), 0.0, 1.0)


def _synthetic_mask(cloud: PointCloud) -> np.ndarray:
    """Points added by a fault carry NaN in their extra fields."""
    if cloud.extra.shape[1] == 0:
        return np.zeros(cloud.n_points, dtype=bool)
    return np.isnan(cloud.extra).any(axis=1)


def _subsample(cloud: PointCloud, max_points: int, rng: np.random.Generator) -> PointCloud:
    if max_points <= 0 or cloud.n_points <= max_points:
        return cloud
    return cloud.subset(rng.choice(cloud.n_points, size=max_points, replace=False))


def _paired_subsample(clean: PointCloud, corrupted: PointCloud, max_points: int,
                      rng: np.random.Generator) -> Tuple[PointCloud, PointCloud, bool]:
    """
    Subsample both clouds consistently. Returns (clean_sample, corrupted_sample, paired).

    Keys (first two extra fields) are sampled as groups, so the sampling rate is
    the same in both clouds even though several points share a key. Falls back to
    independent sampling (paired=False) if points carry no identity fields or if
    fewer than half of the corrupted real points match a clean key (e.g. a
    frozen frame).
    """
    if max_points <= 0 or max(clean.n_points, corrupted.n_points) <= max_points:
        return clean, corrupted, True
    if corrupted.n_points == 0:
        return _subsample(clean, max_points, rng), corrupted, True
    if clean.n_points == 0 or clean.extra.shape[1] < 2 or corrupted.extra.shape[1] < 2:
        return _subsample(clean, max_points, rng), _subsample(corrupted, max_points, rng), False

    synthetic = _synthetic_mask(corrupted)
    real = ~synthetic

    # One id per distinct key over both clouds.
    all_keys = np.concatenate([clean.extra[:, :2], corrupted.extra[real, :2]], axis=0)
    unique_keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    clean_id, corr_id = inverse[:clean.n_points], inverse[clean.n_points:]

    if len(corr_id) and np.isin(corr_id, clean_id).mean() < 0.5:
        return _subsample(clean, max_points, rng), _subsample(corrupted, max_points, rng), False

    rate = min(1.0, max_points / clean.n_points)
    selected = rng.random(len(unique_keys)) < rate

    corr_keep = np.zeros(corrupted.n_points, dtype=bool)
    corr_keep[real] = selected[corr_id]
    # Synthetic points have no clean counterpart: thin them at the same rate.
    corr_keep |= synthetic & (rng.random(corrupted.n_points) < rate)

    return clean.subset(selected[clean_id]), corrupted.subset(corr_keep), True


def _draw(ax, cloud: PointCloud, dims, bounds, synthetic: Optional[np.ndarray] = None):
    """Scatter two coordinate axes; real points by intensity, synthetic in magenta."""
    colors = _intensity_colors(cloud.intensity, bounds)
    real = ~synthetic if synthetic is not None else np.ones(cloud.n_points, dtype=bool)

    ax.scatter(cloud.xyz[real, dims[0]], cloud.xyz[real, dims[1]], c=colors[real],
               cmap="viridis", vmin=0, vmax=1, s=0.3, linewidths=0, rasterized=True)

    if synthetic is not None and synthetic.any():
        ax.scatter(cloud.xyz[synthetic, dims[0]], cloud.xyz[synthetic, dims[1]],
                   c=SYNTHETIC_COLOR, s=1.0, linewidths=0, rasterized=True, label="synthetic")
        ax.legend(loc="upper right", markerscale=6, fontsize=8)


def _draw_overlay(ax, clean: PointCloud, corrupted: PointCloud, synthetic: np.ndarray, dims):
    """Clean in gray underneath; corrupted real points in crimson; synthetic in magenta."""
    real = ~synthetic
    ax.scatter(clean.xyz[:, dims[0]], clean.xyz[:, dims[1]], c="lightgray", s=0.3,
               linewidths=0, rasterized=True, label="clean")
    ax.scatter(corrupted.xyz[real, dims[0]], corrupted.xyz[real, dims[1]], c="crimson", s=0.3,
               linewidths=0, rasterized=True, label="corrupted")
    if synthetic.any():
        ax.scatter(corrupted.xyz[synthetic, dims[0]], corrupted.xyz[synthetic, dims[1]],
                   c=SYNTHETIC_COLOR, s=1.0, linewidths=0, rasterized=True, label="synthetic")
    ax.legend(loc="upper right", markerscale=8, fontsize=8)


def plot_comparison(clean: PointCloud, corrupted: PointCloud, title: str, out_path: Optional[str],
                    view_range: float, z_limit: float, max_points: int, seed: int, show: bool):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    clean_s, corr_s, paired = _paired_subsample(clean, corrupted, max_points, rng)
    synth = _synthetic_mask(corr_s)
    bounds = _intensity_bounds(clean)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=13)

    # Top row: bird's-eye view (x forward, y left)
    _draw(axes[0, 0], clean_s, (0, 1), bounds)
    axes[0, 0].set_title(f"Clean BEV ({clean.n_points} pts)")
    _draw(axes[0, 1], corr_s, (0, 1), bounds, synth)
    axes[0, 1].set_title(f"Corrupted BEV ({corrupted.n_points} pts)")
    _draw_overlay(axes[0, 2], clean_s, corr_s, synth, (0, 1))
    axes[0, 2].set_title("Overlay BEV (visible gray = removed or displaced)" if paired
                         else "Overlay BEV (independent sampling: compare shapes only)")

    for ax in axes[0]:
        ax.set_xlim(-view_range, view_range)
        ax.set_ylim(-view_range, view_range)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    # Bottom row: side view (x forward, z up) and range histogram
    _draw(axes[1, 0], clean_s, (0, 2), bounds)
    axes[1, 0].set_title("Clean side view")
    _draw(axes[1, 1], corr_s, (0, 2), bounds, synth)
    axes[1, 1].set_title("Corrupted side view")
    for ax in axes[1, :2]:
        ax.set_xlim(-view_range, view_range)
        ax.set_ylim(-z_limit, z_limit)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")

    bins = np.linspace(0, view_range, 80)
    axes[1, 2].hist(np.linalg.norm(clean.xyz, axis=1), bins=bins, histtype="step",
                    color="gray", label="clean")
    if corrupted.n_points > 0:
        axes[1, 2].hist(np.linalg.norm(corrupted.xyz, axis=1), bins=bins, histtype="step",
                        color="crimson", label="corrupted")
    axes[1, 2].set_title("Range distribution")
    axes[1, 2].set_xlabel("range [m]")
    axes[1, 2].set_ylabel("points")
    axes[1, 2].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path:
        fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def plot_3d_overlay(clean: PointCloud, corrupted: PointCloud, title: str, out_path: Optional[str],
                    view_range: float, z_limit: float, max_points: int, seed: int, show: bool):
    """3D overlay (clean gray, corrupted crimson, synthetic magenta), rotatable when shown."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    clean_s, corr_s, _ = _paired_subsample(clean, corrupted, max_points, rng)
    synth = _synthetic_mask(corr_s)
    real = ~synth

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title, fontsize=11)

    ax.scatter(clean_s.xyz[:, 0], clean_s.xyz[:, 1], clean_s.xyz[:, 2],
               c="lightgray", s=0.5, linewidths=0, label="clean")
    ax.scatter(corr_s.xyz[real, 0], corr_s.xyz[real, 1], corr_s.xyz[real, 2],
               c="crimson", s=0.5, linewidths=0, label="corrupted")
    if synth.any():
        ax.scatter(corr_s.xyz[synth, 0], corr_s.xyz[synth, 1], corr_s.xyz[synth, 2],
                   c=SYNTHETIC_COLOR, s=1.5, linewidths=0, label="synthetic")

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_zlim(-z_limit, z_limit)
    ax.set_box_aspect((1, 1, 0.35))
    ax.view_init(elev=35, azim=-120)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right", markerscale=8, fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def select_rows(labels: pd.DataFrame, args) -> pd.DataFrame:
    """Pick one frame, or up to --max-figures faulty frames spread evenly over the sequence."""
    if args.frame is not None:
        return labels[labels["frame_idx"] == args.frame]

    rows = labels[labels["fault_label"] == 1]
    if args.fault:
        rows = rows[rows["fault_key"] == args.fault]
    if len(rows) > args.max_figures:
        picks = np.linspace(0, len(rows) - 1, args.max_figures).round().astype(int)
        rows = rows.iloc[np.unique(picks)]
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Visualize clean vs fault-injected LiDAR frames")
    p.add_argument("--clean-dir", required=True, help="Folder with the original .bin frames")
    p.add_argument("--corrupted-dir", required=True, help="Output folder of lidar_injector.py")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, choices=SUPPORTED_STRIDES,
                   help="Floats per point in the .bin files (6 = Boreas)")
    p.add_argument("--frame", type=int, default=None, help="Visualize a single frame index")
    p.add_argument("--fault", default=None, help="Only frames of this fault, e.g. 'environment:window'")
    p.add_argument("--max-figures", type=int, default=10, help="Max number of faulty frames to render")
    p.add_argument("--view-range", type=float, default=80.0, help="Half-extent of the plot area [m]")
    p.add_argument("--z-limit", type=float, default=20.0, help="Half-extent of the vertical axis [m]")
    p.add_argument("--max-points", type=int, default=60000, help="Subsample for 2D plots (0 = all)")
    p.add_argument("--max-points-3d", type=int, default=20000, help="Subsample for the 3D overlay")
    p.add_argument("--out-dir", default=None, help="Figure folder (default: <corrupted-dir>/visualizations)")
    p.add_argument("--view3d", action="store_true", help="Also render a 3D overlay per frame")
    p.add_argument("--show", action="store_true", help="Show figures interactively")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    labels = pd.read_csv(os.path.join(args.corrupted_dir, "labels.csv"), keep_default_na=False)
    out_dir = args.out_dir or os.path.join(args.corrupted_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    rows = select_rows(labels, args)
    if rows.empty:
        raise RuntimeError("No matching frames found in labels.csv")

    for _, row in rows.iterrows():
        name = row["lidar_name"]
        clean = load_point_cloud(os.path.join(args.clean_dir, name), stride=args.stride)
        corrupted = load_point_cloud(os.path.join(args.corrupted_dir, name),
                                     stride=args.stride, allow_empty=True)

        title = (f"frame {row['frame_idx']} | {row['fault_key']} | severity={row['severity']} | "
                 f"points {clean.n_points} -> {corrupted.n_points}")
        stem = f"frame_{int(row['frame_idx']):05d}_{row['fault_kind']}"

        out_2d = os.path.join(out_dir, f"{stem}.png")
        plot_comparison(clean, corrupted, title, out_2d, args.view_range, args.z_limit,
                        args.max_points, args.seed, args.show)
        print(f"Saved: {out_2d}")

        if args.view3d:
            out_3d = os.path.join(out_dir, f"{stem}_3d.png")
            plot_3d_overlay(clean, corrupted, title, out_3d, args.view_range, args.z_limit,
                            args.max_points_3d, args.seed, args.show)
            print(f"Saved: {out_3d}")


if __name__ == "__main__":
    main()