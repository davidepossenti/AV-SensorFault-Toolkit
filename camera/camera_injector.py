#!/usr/bin/env python3
"""Inject camera faults using JSON plans or non-overlapping random bursts."""

import argparse
import json
import math
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .modules.common import SEVERITIES, check_image, severity_index
from .modules.faults_hardware import apply_fault_hardware
from .modules.faults_environment import apply_fault_environment
from .modules.faults_calibration import apply_fault_calibration
from .modules.faults_software import (
    apply_fault_software,
    temporal_delay_frames,
    temporal_ghost_parameters,
)
from .modules.faults_synergistic import (
    apply_fault_synergistic,
    temporal_ghosting,
)


FAULT_REGISTRY = {
    "hardware": [
        "brightness_blackout",
        "brightness_whiteout",
        "brightness_exposure",
        "broken_lens",
        "broken_vr",
        "heat_failure_blur",
        "heat_failure_zoom",
        "electrical_overload_blackout",
        "electrical_overload_corrupt",
        "electrical_overload_partial",
        "no_bayer",
        "no_action_raw",
        "banding",
        "bright_lines",
        "dead_pixels",
    ],
    "environment": [
        "rain",
        "snow",
        "ice",
        "fog",
        "external_scatter",
        "window_contamination",
        "flare",
        "condensation",
        "sensor_spots",
        "brackish_corrosion",
        "water_ingress",
        "water_ingress_blackout",
        "mechanical_instability",
    ],
    "calibration": [
        "blurred",
        "white_balance",
        "lens_distortion",
        "mounting",
    ],
    "software": [
        "chromatic_aberration",
        "demosaic_incomplete",
        "noise",
        "data_loss",
        "silent_failure",
        "temporal_misalignment",
    ],
    "synergistic": [
        "broken_lens_condensation",
        "mounting_fog",
        "temporal_ghosting",
    ],
}

ALL_FAULT_KEYS = sorted(
    f"{family}:{kind}"
    for family, kinds in FAULT_REGISTRY.items()
    for kind in kinds
)

ROUTERS = {
    "hardware": apply_fault_hardware,
    "environment": apply_fault_environment,
    "calibration": apply_fault_calibration,
    "software": apply_fault_software,
    "synergistic": apply_fault_synergistic,
}

BINARY_FAULTS = {
    "hardware:brightness_blackout",
    "hardware:brightness_whiteout",
    "hardware:electrical_overload_blackout",
    "hardware:no_action_raw",
    "environment:water_ingress_blackout",
    "software:silent_failure",
}

TEMPORAL_FAULTS = {
    "software:silent_failure",
    "software:temporal_misalignment",
    "synergistic:temporal_ghosting",
}

SEEDED_FAULTS = {
    "hardware:broken_lens",
    "hardware:broken_vr",
    "hardware:electrical_overload_corrupt",
    "hardware:electrical_overload_partial",
    "hardware:bright_lines",
    "hardware:dead_pixels",
    "environment:rain",
    "environment:snow",
    "environment:ice",
    "environment:fog",
    "environment:external_scatter",
    "environment:window_contamination",
    "environment:flare",
    "environment:condensation",
    "environment:sensor_spots",
    "environment:brackish_corrosion",
    "environment:water_ingress",
    "environment:mechanical_instability",
    "calibration:mounting",
    "software:noise",
    "software:data_loss",
    "synergistic:broken_lens_condensation",
    "synergistic:mounting_fog",
}

BURST_SEED_FAULTS = {
    "hardware:broken_lens",
    "hardware:bright_lines",
    "hardware:dead_pixels",
    "environment:ice",
    "environment:window_contamination",
    "environment:flare",
    "environment:condensation",
    "environment:sensor_spots",
    "environment:brackish_corrosion",
    "environment:water_ingress",
    "calibration:mounting",
    "synergistic:broken_lens_condensation",
    "synergistic:mounting_fog",
}

FRAME_SEED_FAULTS = {"synergistic:mounting_fog"}

SUBSYSTEM = {
    "hardware:brightness_blackout": "lens",
    "hardware:brightness_whiteout": "lens",
    "hardware:brightness_exposure": "lens",
    "hardware:broken_lens": "lens",
    "hardware:broken_vr": "lens",
    "hardware:heat_failure_blur": "body",
    "hardware:heat_failure_zoom": "body",
    "hardware:electrical_overload_blackout": "body",
    "hardware:electrical_overload_corrupt": "body",
    "hardware:electrical_overload_partial": "body",
    "hardware:no_bayer": "bayer_filter",
    "hardware:no_action_raw": "isp",
    "hardware:banding": "image_sensor",
    "hardware:bright_lines": "image_sensor",
    "hardware:dead_pixels": "image_sensor",
    "environment:rain": "lens",
    "environment:snow": "lens",
    "environment:ice": "lens",
    "environment:fog": "optical_path",
    "environment:external_scatter": "optical_path",
    "environment:window_contamination": "lens",
    "environment:flare": "lens_detector",
    "environment:condensation": "lens",
    "environment:sensor_spots": "image_sensor",
    "environment:brackish_corrosion": "housing",
    "environment:water_ingress": "housing",
    "environment:water_ingress_blackout": "housing",
    "environment:mechanical_instability": "mounting",
    "calibration:blurred": "lens",
    "calibration:white_balance": "isp",
    "calibration:lens_distortion": "isp",
    "calibration:mounting": "calibration",
    "software:chromatic_aberration": "isp",
    "software:demosaic_incomplete": "isp",
    "software:noise": "isp",
    "software:data_loss": "firmware",
    "software:silent_failure": "comm_interface",
    "software:temporal_misalignment": "firmware",
    "synergistic:broken_lens_condensation": "lens",
    "synergistic:mounting_fog": "calib_path",
    "synergistic:temporal_ghosting": "sync_isp",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEED_LIMIT = 2**31 - 1


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def normalize_fault_key(value):
    key = str(value).lower().strip()
    if key not in ALL_FAULT_KEYS:
        raise ValueError(f"Unknown fault: {value!r}")
    return key


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return check_image(img)


def write_image(path, img):
    check_image(img)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Cannot write image: {path}")


def severity_label(key, severity, kwargs):
    if key in BINARY_FAULTS:
        return "na"
    if key == "hardware:no_bayer" and kwargs.get("mode", "scramble") == "mono":
        return "na"
    return SEVERITIES[severity_index(severity)]


def parse_weights(value):
    weights = [float(part.strip()) for part in value.split(",")]
    if (
        len(weights) != 4
        or not all(math.isfinite(weight) and weight >= 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("Severity weights require four finite non-negative values")
    return weights


def normalize_plan(raw_plan, default_severity, rng, start_ok, end_ok):
    if not isinstance(raw_plan, list):
        raise ValueError("Plan must be a JSON list")

    plan = []
    for number, raw in enumerate(raw_plan):
        if not isinstance(raw, dict):
            raise ValueError(f"Plan entry {number} must be an object")

        key = normalize_fault_key(raw["fault"])
        start = raw["start"]
        end = raw["end"]
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end <= start
        ):
            raise ValueError(f"Invalid interval in plan entry {number}")

        kwargs = dict(raw.get("kwargs") or {})
        severity = kwargs.pop(
            "severity", raw.get("severity", default_severity)
        )
        label = severity_label(key, severity, kwargs)

        if label != "na":
            kwargs["severity"] = label

        for field in ("seed", "frame_seed"):
            if field in kwargs:
                seed = kwargs[field]
                if (
                    isinstance(seed, bool)
                    or not isinstance(seed, int)
                    or not 0 <= seed <= SEED_LIMIT
                ):
                    raise ValueError(f"Invalid {field} in plan entry {number}")

        if "seed" in kwargs and key not in SEEDED_FAULTS:
            raise ValueError(f"{key} does not accept seed")
        if "frame_seed" in kwargs and key not in FRAME_SEED_FAULTS:
            raise ValueError(f"{key} does not accept frame_seed")

        if key in TEMPORAL_FAULTS:
            unexpected = set(kwargs) - {"severity"}
            if unexpected:
                raise ValueError(f"Unsupported temporal parameters: {unexpected}")

        start, end = max(start, start_ok), min(end, end_ok)
        if start >= end:
            continue

        if key == "software:silent_failure" and start == 0:
            raise ValueError("Silent failure requires a preceding clean frame")

        if key in BURST_SEED_FAULTS and "seed" not in kwargs:
            kwargs["seed"] = rng.randint(0, SEED_LIMIT)

        family, kind = key.split(":", 1)
        plan.append({
            "burst_id": number,
            "fault": key,
            "fault_family": family,
            "fault_kind": kind,
            "start": start,
            "end": end,
            "severity": label,
            "kwargs": kwargs,
        })

    schedule = {}
    for item in plan:
        for index in range(item["start"], item["end"]):
            if index in schedule:
                raise ValueError(f"Overlapping faults at frame {index}")
            schedule[index] = item

    return plan, schedule


def build_random_plan(args, start, end, rng):
    ratio = args.balance_ratio
    if ratio is None or not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("--balance-ratio must be in (0, 1)")
    if not args.fault:
        raise ValueError("Random mode requires at least one --fault")
    if args.n_bursts <= 0:
        raise ValueError("--n-bursts must be positive")
    if not 0 < args.burst_len_min <= args.burst_len_max:
        raise ValueError("Invalid burst lengths")

    weights = parse_weights(args.severity_weights)
    target = max(1, round((end - start) * ratio))
    occupied = set()
    plan = []

    for _ in range(max(100, args.n_bursts * 100)):
        if len(occupied) >= target or len(plan) >= args.n_bursts:
            break

        key = rng.choice(args.fault)
        length = rng.randint(args.burst_len_min, args.burst_len_max)
        lowest = max(start, 1) if key == "software:silent_failure" else start
        if lowest >= end:
            continue

        first = rng.randrange(lowest, end)
        last = first + length
        if last > end:
            continue
        if any(index in occupied for index in range(first, last)):
            continue

        if args.severity_mode == "fixed":
            severity = args.severity
        elif args.severity_mode == "cycle":
            severity = SEVERITIES[len(plan) % len(SEVERITIES)]
        else:
            severity = rng.choices(SEVERITIES, weights=weights, k=1)[0]

        plan.append({
            "fault": key,
            "start": first,
            "end": last,
            "severity": severity,
        })
        occupied.update(range(first, last))

    if len(occupied) < target:
        print(
            f"WARNING: requested {target} faulty frames; scheduled "
            f"{len(occupied)} within burst constraints"
        )
    return plan


def resolved_kwargs(spec, rng):
    key = spec["fault"]
    kwargs = dict(spec["kwargs"])
    if key in SEEDED_FAULTS and "seed" not in kwargs:
        kwargs["seed"] = rng.randint(0, SEED_LIMIT)
    if key in FRAME_SEED_FAULTS and "frame_seed" not in kwargs:
        kwargs["frame_seed"] = rng.randint(0, SEED_LIMIT)
    return kwargs


def apply_frame(img, spec, kwargs):
    return ROUTERS[spec["fault_family"]](
        img, spec["fault_kind"], **kwargs
    )


def preflight(plan, input_dir, files):
    for spec in plan:
        key = spec["fault"]
        if key in TEMPORAL_FAULTS:
            continue

        img = read_image(input_dir / files[spec["start"]])
        kwargs = dict(spec["kwargs"])
        if key in SEEDED_FAULTS:
            kwargs.setdefault("seed", 0)
        if key in FRAME_SEED_FAULTS:
            kwargs.setdefault("frame_seed", 1)
        apply_frame(img.copy(), spec, kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--keep-head", type=int, default=20)
    parser.add_argument("--keep-tail", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-json")
    parser.add_argument("--fault", action="append", choices=ALL_FAULT_KEYS)
    parser.add_argument("--balance-ratio", type=float)
    parser.add_argument("--n-bursts", type=int, default=4)
    parser.add_argument("--burst-len-min", type=int, default=10)
    parser.add_argument("--burst-len-max", type=int, default=25)
    parser.add_argument(
        "--severity-mode", choices=("fixed", "random", "cycle"), default="random"
    )
    parser.add_argument("--severity", choices=SEVERITIES, default="severe")
    parser.add_argument("--severity-weights", default="1,1,1,1")
    parser.add_argument("--write-ext", choices=("png", "bmp", "tif", "tiff"))
    parser.add_argument("--no-preflight", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.first_n, args.keep_head, args.keep_tail, args.seed) < 0:
        raise ValueError("Frame limits and seed must be non-negative")

    input_dir = Path(args.in_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        (
            entry.name for entry in input_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
    if args.first_n:
        files = files[:args.first_n]
    if not files:
        raise ValueError("No input images found")

    total = len(files)
    head = min(args.keep_head, total)
    tail = min(args.keep_tail, total - head)
    start, end = head, total - tail
    if start >= end:
        raise ValueError("Protected head and tail leave no injectable frames")

    scheduler = random.Random(args.seed)
    frame_rng = random.Random(args.seed ^ 0x5F3759DF)

    if args.plan_json:
        with open(args.plan_json, encoding="utf-8") as handle:
            raw_plan = json.load(handle)
    else:
        raw_plan = build_random_plan(args, start, end, scheduler)

    plan, schedule = normalize_plan(
        raw_plan, args.severity, scheduler, start, end
    )

    if not plan:
        print("WARNING: no faults scheduled")

    output_names = [
        f"{Path(name).stem}.{args.write_ext}" if args.write_ext else name
        for name in files
    ]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError("Output filenames collide after extension conversion")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(
        args.out_dir or input_dir.parent / f"CorruptedCamera_{stamp}"
    ).resolve()
    if output_dir == input_dir:
        raise ValueError("Input and output directories must differ")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")

    if not args.no_preflight:
        preflight(plan, input_dir, files)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "plan_used.json", "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    manifest = []
    failures = 0

    for index, (name, output_name) in enumerate(zip(files, output_names)):
        source = input_dir / name
        destination = output_dir / output_name
        spec = schedule.get(index)

        row = {
            "frame_idx": index,
            "input_name": name,
            "image_name": output_name,
            "fault_label": 0,
            "fault_key": "none",
            "fault_family": "none",
            "fault_kind": "none",
            "subsystem": "none",
            "severity": "none",
            "seed": None,
            "frame_seed": None,
            "source_frame_idx": index,
            "burst_id": None,
            "status": "ok",
            "error": "",
        }

        try:
            if spec is None:
                if args.write_ext is None:
                    shutil.copy2(source, destination)
                else:
                    write_image(destination, read_image(source))
            else:
                key = spec["fault"]
                kwargs = resolved_kwargs(spec, frame_rng)
                source_index = index

                if key == "software:silent_failure":
                    source_index = spec["start"] - 1
                    temporal_source = input_dir / files[source_index]
                    if args.write_ext is None and (
                        temporal_source.suffix.lower() == destination.suffix.lower()
                    ):
                        shutil.copy2(temporal_source, destination)
                    else:
                        write_image(destination, read_image(temporal_source))

                elif key == "software:temporal_misalignment":
                    source_index = max(
                        0, index - temporal_delay_frames(spec["severity"])
                    )
                    temporal_source = input_dir / files[source_index]
                    if args.write_ext is None and (
                        temporal_source.suffix.lower() == destination.suffix.lower()
                    ):
                        shutil.copy2(temporal_source, destination)
                    else:
                        write_image(destination, read_image(temporal_source))

                elif key == "synergistic:temporal_ghosting":
                    delay, alpha = temporal_ghost_parameters(spec["severity"])
                    source_index = max(0, index - delay)
                    out = temporal_ghosting(
                        read_image(source),
                        read_image(input_dir / files[source_index]),
                        alpha,
                    )
                    write_image(destination, out)

                else:
                    out = apply_frame(read_image(source), spec, kwargs)
                    write_image(destination, out)

                row.update({
                    "fault_label": 1,
                    "fault_key": key,
                    "fault_family": spec["fault_family"],
                    "fault_kind": spec["fault_kind"],
                    "subsystem": SUBSYSTEM[key],
                    "severity": spec["severity"],
                    "seed": kwargs.get("seed"),
                    "frame_seed": kwargs.get("frame_seed"),
                    "source_frame_idx": source_index,
                    "burst_id": spec["burst_id"],
                })

        except Exception as exc:
            failures += 1
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"[{index + 1}/{total}] {name}: {exc}")

            try:
                if args.write_ext is None:
                    shutil.copy2(source, destination)
                else:
                    write_image(destination, read_image(source))
                row["status"] = "fallback_clean"
            except Exception as fallback_exc:
                row["error"] += f"; fallback failed: {fallback_exc}"
                if destination.exists():
                    destination.unlink()

        manifest.append(row)

    table = pd.DataFrame(manifest)
    for column in ("seed", "frame_seed", "burst_id"):
        table[column] = pd.array(table[column], dtype="Int64")
    table.to_csv(output_dir / "labels.csv", index=False)

    summary = {
        "seed": args.seed,
        "total_frames": total,
        "faulty_frames": int(table["fault_label"].sum()),
        "failed_frames": failures,
        "scheduled_frames": len(schedule),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "write_extension": args.write_ext,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n===== Camera Injector Summary =====")
    print(f"Total frames:      {total}")
    print(f"Faulty frames:     {summary['faulty_frames']}")
    print(f"Failed frames:     {failures}")
    print(f"Output directory:  {output_dir}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
