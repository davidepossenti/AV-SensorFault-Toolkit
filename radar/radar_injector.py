#!/usr/bin/env python3
"""
radar_injector.py

Orchestrates fault injection over a sequence of polar radar frames (.png files)
across five fault categories: hardware, environment, calibration, software and
synergistic. Faults are scheduled either from a JSON plan or as random bursts
(contiguous frame ranges), each with an independently sampled severity by
default. Produces corrupted .png frames, a label manifest (labels.csv) and the
injection plan (plan_used.json). Clean, frozen and delayed frames are
byte-for-byte copies of the input files.

Manifest: for every frame, labels.csv records the fault label, taxonomy key,
family, kind, affected subsystem, severity, the burst seed (`seed`), the
per-frame seed (`frame_seed`), the source frame the output derives from
(`source_frame_idx`, equal to the frame itself unless the fault is temporal)
and the burst identifier (`burst_id`, the position of the burst in
plan_used.json; with --progressive-degradation every severity segment is a
burst of its own).

Seeding: faults modelling a persistent physical condition (radome dirt,
roadside foliage, boresight offset, thermal deformation and drift, extrinsic
misalignment, a failing interference mitigation stage) use one seed per burst,
so their pattern stays constant across the burst. Synergies containing such a
component inherit the burst seed. Foliage, the failing mitigation stage and the
drift + interference synergy also draw a fresh per-frame seed for their
transient noise. This holds for both random and JSON plans; a seed given
explicitly in a plan entry takes precedence. Faults made of random noise
(interference, snow, clutter, ...) draw a fresh seed for every frame.
Deterministic faults need no seed and record none. Everything is reproducible
via --seed.

Temporal exceptions: "inference_timing" repeats the last healthy frame before
the burst; "temporal_misalignment" delivers, for every frame of the burst, the
frame that is d frames older (d grows with severity). The synergy
"temporal_multipath" applies its per-frame component (multipath) to that
delayed frame. They read the clean input sequence, so they are handled here
rather than by a category module.

Faults that physically degrade gradually (e.g. LNA, antenna, thermal drift) can
be set to worsen within a burst via --progressive-degradation (random mode only).
"""

import argparse
import json
import os
import random
import re
import shutil
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Set, Tuple

import pandas as pd

from common.radar_loader import (
    BOREAS_GEOMETRY,
    BOREAS_GEOMETRY_UPGRADED,
    RadarFrame,
    RadarGeometry,
    load_radar_png,
    save_radar_png,
)
from .modules.faults_hardware import apply_fault_hardware
from .modules.faults_environment import apply_fault_environment
from .modules.faults_calibration import apply_fault_calibration
from .modules.faults_software import apply_fault_software, temporal_delay_frames
from .modules.faults_synergistic import TEMPORAL_SYNERGIES, apply_fault_synergistic


SEVERITY_LEVELS = ["mild", "moderate", "severe", "extreme"]

# Canonical fault keys per category. Must match the kinds accepted by each
# category's dispatcher, plus the temporal exceptions below.
FAULT_REGISTRY: Dict[str, List[str]] = {
    "hardware": ["lna", "antenna", "thermal"],
    "environment": ["rain", "snow", "radome", "condensation", "interference", "multipath",
                    "clutter", "foliage", "scatter", "water_ingress", "corrosion"],
    "calibration": ["boresight", "thermal_drift", "extrinsic"],
    "software": ["filtering", "silent_failure", "data_loss", "interference_mitigation",
                 "inference_timing", "temporal_misalignment"],
    "synergistic": ["thermal_multipath", "temporal_multipath", "drift_interference"],
}

ALL_FAULT_KEYS = sorted(f"{fam}:{kind}" for fam, kinds in FAULT_REGISTRY.items() for kind in kinds)

ROUTERS = {
    "hardware": apply_fault_hardware,
    "environment": apply_fault_environment,
    "calibration": apply_fault_calibration,
    "software": apply_fault_software,
    "synergistic": apply_fault_synergistic,
}

# Deterministic faults: no seed is drawn, passed or recorded.
NO_SEED = {
    ("hardware", "lna"), ("hardware", "antenna"),
    ("environment", "rain"), ("environment", "condensation"), ("environment", "multipath"),
    ("software", "filtering"), ("software", "silent_failure"),
    ("synergistic", "temporal_multipath"),
}

# Fault kinds handled as a temporal exception rather than a per-frame transform.
TEMPORAL_FAULTS = {("software", "inference_timing"), ("software", "temporal_misalignment")}

# Temporal fault that repeats the last healthy frame (the other one delays by d frames).
FREEZE_FAULT = ("software", "inference_timing")

# Fault kinds with a severity scale. Binary faults and the freeze have none.
SEVERITY_AWARE = {
    ("hardware", "lna"), ("hardware", "antenna"), ("hardware", "thermal"),
    ("environment", "rain"), ("environment", "snow"), ("environment", "radome"),
    ("environment", "condensation"), ("environment", "interference"),
    ("environment", "multipath"), ("environment", "clutter"), ("environment", "foliage"),
    ("environment", "scatter"), ("environment", "water_ingress"), ("environment", "corrosion"),
    ("calibration", "boresight"), ("calibration", "thermal_drift"), ("calibration", "extrinsic"),
    ("software", "filtering"), ("software", "data_loss"),
    ("software", "interference_mitigation"), ("software", "temporal_misalignment"),
    ("synergistic", "thermal_multipath"), ("synergistic", "temporal_multipath"),
    ("synergistic", "drift_interference"),
}

# Faults whose random pattern is a persistent physical condition: one seed per burst.
BURST_SEED_FAULTS = {
    ("environment", "radome"), ("environment", "foliage"),
    ("calibration", "boresight"), ("calibration", "thermal_drift"), ("calibration", "extrinsic"),
    ("hardware", "thermal"), ("software", "interference_mitigation"),
    ("synergistic", "thermal_multipath"), ("synergistic", "drift_interference"),
}

# Faults mixing a persistent pattern with transient noise also get a fresh
# per-frame seed (`frame_seed`).
FRAME_SEED_FAULTS = {
    ("environment", "foliage"), ("software", "interference_mitigation"),
    ("synergistic", "drift_interference"),
}

# Affected subsystem of each fault, as in the taxonomy table.
SUBSYSTEM = {
    ("hardware", "lna"): "front_end",
    ("hardware", "antenna"): "antenna",
    ("hardware", "thermal"): "housing",
    ("environment", "rain"): "radome",
    ("environment", "snow"): "radome",
    ("environment", "radome"): "radome",
    ("environment", "condensation"): "radome",
    ("environment", "interference"): "receiver",
    ("environment", "multipath"): "path",
    ("environment", "clutter"): "path",
    ("environment", "foliage"): "path",
    ("environment", "scatter"): "path_radome",
    ("environment", "water_ingress"): "housing",
    ("environment", "corrosion"): "housing",
    ("calibration", "boresight"): "antenna",
    ("calibration", "thermal_drift"): "housing",
    ("calibration", "extrinsic"): "calibration",
    ("software", "filtering"): "dsp_filter",
    ("software", "silent_failure"): "comm_interface",
    ("software", "data_loss"): "firmware",
    ("software", "interference_mitigation"): "dsp",
    ("software", "inference_timing"): "dsp",
    ("software", "temporal_misalignment"): "firmware",
    ("synergistic", "thermal_multipath"): "housing_path",
    ("synergistic", "temporal_multipath"): "sync_path",
    ("synergistic", "drift_interference"): "calib_rx",
}

# Named sensor geometries (see common/radar_loader.py).
GEOMETRIES = {"boreas": BOREAS_GEOMETRY, "boreas-upgraded": BOREAS_GEOMETRY_UPGRADED}


def get_router(family: str):
    if family not in ROUTERS:
        raise ValueError(f"Unknown fault family: {family}")
    return ROUTERS[family]


def severity_to_kwargs(fam: str, kind: str, severity: str) -> Dict:
    """Return the severity kwargs for a fault, or {} if it has no severity scale."""
    if (fam, kind) not in SEVERITY_AWARE:
        return {}
    sev = str(severity).lower().strip()
    if sev not in SEVERITY_LEVELS:
        raise ValueError(f"Unknown severity '{severity}' (expected one of {SEVERITY_LEVELS})")
    return {"severity": sev}


def effective_severity(fam: str, kind: str, severity: str) -> str:
    """Severity label recorded for a fault: 'na' if it has no severity scale."""
    return str(severity).lower().strip() if (fam, kind) in SEVERITY_AWARE else "na"


def resolve_geometry(name: str, range_resolution) -> RadarGeometry:
    geometry = GEOMETRIES[name]
    return replace(geometry, range_resolution_m=range_resolution) if range_resolution else geometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def default_out_dir(in_dir: str) -> str:
    parent = os.path.dirname(in_dir.rstrip(os.sep))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(parent, f"CorruptedRadar_{stamp}")


def load_png_list(in_dir: str, first_n: int) -> List[str]:
    files = sorted((f for f in os.listdir(in_dir) if f.lower().endswith(".png")), key=natural_key)
    if not files:
        raise RuntimeError(f"No .png files found in: {in_dir}")
    return files[:first_n] if first_n > 0 else files


def clamp_intervals(plan: List[Dict], start_ok: int, end_ok_excl: int) -> List[Dict]:
    clamped = []
    for it in plan:
        s, e = max(int(it["start"]), start_ok), min(int(it["end"]), end_ok_excl)
        if s < e:
            item = dict(it)
            item["start"], item["end"] = s, e
            clamped.append(item)
    return clamped


def parse_weights(s: str) -> List[float]:
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4 or any(p < 0 for p in parts) or sum(parts) <= 0:
        raise ValueError("--severity-weights must be 4 non-negative values (mild,moderate,severe,extreme)")
    return parts


def pick_severity(rng: random.Random, mode: str, default: str, burst_idx: int, weights: List[float]) -> str:
    if mode == "fixed":
        return default
    if mode == "cycle":
        return SEVERITY_LEVELS[burst_idx % 4]
    return rng.choices(SEVERITY_LEVELS, weights=weights, k=1)[0]


def normalize_fault_key(fault_key: str) -> Tuple[str, str]:
    if ":" not in fault_key:
        raise ValueError(f"Fault key must be 'family:kind', got: {fault_key}")
    family, kind = (p.strip().lower() for p in fault_key.split(":", 1))
    if family not in FAULT_REGISTRY or kind not in FAULT_REGISTRY[family]:
        raise ValueError(f"Unsupported fault: {fault_key}")
    return family, kind


def parse_progressive_targets(s: str) -> Set[Tuple[str, str]]:
    """Parse the '--progressive-degradation' list into (family, kind) tuples."""
    targets = {normalize_fault_key(item) for item in s.split(",") if item.strip()} if s else set()
    for fam, kind in targets:
        if (fam, kind) not in SEVERITY_AWARE:
            raise ValueError(f"{fam}:{kind} has no severity scale and cannot degrade progressively")
    return targets


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def load_json_plan(path: str, default_severity: str, rng: random.Random) -> List[Dict]:
    """Load a JSON plan and normalize it into the internal schema."""
    with open(path, "r") as f:
        raw_plan = json.load(f)
    if not isinstance(raw_plan, list):
        raise ValueError("The JSON plan must be a list of {fault, start, end, severity?, kwargs?}")

    plan = []
    for item in raw_plan:
        fam, kind = normalize_fault_key(item["fault"])
        raw_severity = item.get("severity", default_severity)
        kwargs = dict(item.get("kwargs") or severity_to_kwargs(fam, kind, raw_severity))

        # Persistent faults keep one seed for the whole burst.
        if (fam, kind) in BURST_SEED_FAULTS and "seed" not in kwargs:
            kwargs["seed"] = rng.randint(0, 2**31 - 1)

        plan.append({
            "fault": f"{fam}:{kind}", "fault_family": fam, "fault_kind": kind,
            "start": int(item["start"]), "end": int(item["end"]),
            "severity": effective_severity(fam, kind, kwargs.get("severity", raw_severity)),
            "kwargs": kwargs,
        })
    return plan


def build_random_plan(allowed_idx: List[int], faults: List[str], ratio: float, n_bursts: int,
                      len_min: int, len_max: int, rng: random.Random, severity_mode: str,
                      default_severity: str, severity_weights: List[float]) -> List[Dict]:
    if not (0.0 < ratio < 1.0):
        raise ValueError("--balance-ratio must be in (0, 1)")
    if len_min <= 0 or len_max <= 0 or len_min > len_max:
        raise ValueError("Invalid --burst-len-min/--burst-len-max")

    allowed_set = set(allowed_idx)
    target_faulty = int(round(len(allowed_idx) * ratio))
    plan: List[Dict] = []
    covered = set()

    for _ in range(n_bursts * 20):
        if len(covered) >= target_faulty or len(plan) >= n_bursts:
            break

        fam, kind = normalize_fault_key(rng.choice(faults))
        raw_severity = pick_severity(rng, severity_mode, default_severity, len(plan), severity_weights)

        length = rng.randint(len_min, len_max)
        start = rng.choice(allowed_idx)
        window = [i for i in range(start, start + length) if i in allowed_set]
        if not window:
            continue
        s, e = window[0], window[-1] + 1

        overlap = sum(1 for i in range(s, e) if i in covered)
        if overlap / max(1, (e - s)) > 0.6:
            continue

        kwargs = severity_to_kwargs(fam, kind, raw_severity)
        if (fam, kind) in BURST_SEED_FAULTS:
            kwargs["seed"] = rng.randint(0, 2**31 - 1)

        plan.append({
            "fault": f"{fam}:{kind}", "fault_family": fam, "fault_kind": kind,
            "start": s, "end": e, "severity": effective_severity(fam, kind, raw_severity),
            "kwargs": kwargs,
        })
        covered.update(range(s, e))

    return plan


def apply_progressive_degradation(plan: List[Dict], targets: Set[Tuple[str, str]]) -> List[Dict]:
    """
    Split each burst whose (family, kind) is in `targets` into 4 contiguous
    sub-segments with increasing severity (mild -> extreme). A burst-level
    seed, if present, is kept so persistent patterns do not change between
    segments. Other bursts are left unchanged.
    """
    if not targets:
        return plan

    expanded: List[Dict] = []
    for item in plan:
        key = (item["fault_family"], item["fault_kind"])
        if key not in targets:
            expanded.append(item)
            continue

        length = item["end"] - item["start"]
        base_len, remainder = divmod(length, len(SEVERITY_LEVELS))

        cursor = item["start"]
        for i, severity in enumerate(SEVERITY_LEVELS):
            seg_len = base_len + (1 if i < remainder else 0)
            if seg_len <= 0:
                continue

            segment = dict(item)
            segment["start"], segment["end"] = cursor, cursor + seg_len
            segment["severity"] = severity
            segment["kwargs"] = severity_to_kwargs(key[0], key[1], severity)
            if "seed" in item["kwargs"]:
                segment["kwargs"]["seed"] = item["kwargs"]["seed"]

            expanded.append(segment)
            cursor += seg_len

    return expanded


def preflight_validate_plan(plan: List[Dict], in_dir: str, files: List[str], geometry: RadarGeometry):
    """Run each planned fault once on the first frame to catch errors before the full run."""
    if not files:
        return
    frame = load_radar_png(os.path.join(in_dir, files[0]), geometry)

    for fault_key in sorted({it["fault"] for it in plan}):
        fam, kind = normalize_fault_key(fault_key)
        if (fam, kind) in TEMPORAL_FAULTS:
            continue  # only copies an existing clean frame

        kwargs = next((dict(it["kwargs"]) for it in plan if it["fault"] == fault_key), {})
        if (fam, kind) in NO_SEED:
            kwargs.pop("seed", None)
        else:
            kwargs.setdefault("seed", 0)

        out = get_router(fam)(frame, kind, **kwargs)
        if not isinstance(out, RadarFrame) or out.power.shape != frame.power.shape:
            raise RuntimeError(f"Preflight: {fault_key} returned an invalid frame")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Radar fault injector: random bursts or JSON plan")

    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--first-n", type=int, default=0)
    p.add_argument("--keep-head", type=int, default=20)
    p.add_argument("--keep-tail", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--geometry", default="boreas", choices=sorted(GEOMETRIES),
                   help="Sensor geometry: 'boreas' (0.0596 m/bin) or 'boreas-upgraded' (0.04381 m/bin)")
    p.add_argument("--range-resolution", type=float, default=None,
                   help="Override the metres per range bin of the chosen geometry")

    p.add_argument("--plan-json", default=None, help="JSON list of {fault,start,end,kwargs?,severity?}")

    p.add_argument("--fault", action="append", choices=ALL_FAULT_KEYS,
                   help="Fault(s) for random mode, 'family:kind', repeatable")
    p.add_argument("--balance-ratio", type=float, default=None, help="Target fraction of faulty frames")
    p.add_argument("--n-bursts", type=int, default=4)
    p.add_argument("--burst-len-min", type=int, default=10)
    p.add_argument("--burst-len-max", type=int, default=25)

    p.add_argument("--severity-mode", default="random", choices=["fixed", "random", "cycle"])
    p.add_argument("--severity", default="severe", choices=SEVERITY_LEVELS)
    p.add_argument("--severity-weights", default="1,1,1,1")

    p.add_argument("--progressive-degradation", default="",
                   help="Comma-separated 'family:kind' list (e.g. 'hardware:lna,calibration:thermal_drift') "
                        "whose bursts are split into 4 sub-segments with severity increasing mild->extreme. "
                        "Random mode only.")

    p.add_argument("--no-preflight", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    rng_sched = random.Random(args.seed)
    geometry = resolve_geometry(args.geometry, args.range_resolution)

    files = load_png_list(args.in_dir, args.first_n)
    total = len(files)

    keep_head = max(0, min(args.keep_head, total))
    keep_tail = max(0, min(args.keep_tail, total - keep_head))
    fault_start, fault_end_excl = keep_head, total - keep_tail
    allowed_idx = list(range(fault_start, fault_end_excl))
    allowed_set = set(allowed_idx)

    if not allowed_idx:
        raise ValueError(
            f"--keep-head ({args.keep_head}) and --keep-tail ({args.keep_tail}) leave no frames "
            f"available for injection out of {total} frames"
        )

    out_dir = args.out_dir or default_out_dir(args.in_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.plan_json:
        if args.progressive_degradation:
            raise ValueError("--progressive-degradation is only supported in random mode")
        plan = load_json_plan(args.plan_json, args.severity, rng_sched)
    else:
        if args.balance_ratio is None or not args.fault:
            raise ValueError("Random mode requires --balance-ratio and at least one --fault")
        plan = build_random_plan(
            allowed_idx, args.fault, args.balance_ratio, args.n_bursts,
            args.burst_len_min, args.burst_len_max, rng_sched,
            args.severity_mode, args.severity, parse_weights(args.severity_weights),
        )
        plan = apply_progressive_degradation(plan, parse_progressive_targets(args.progressive_degradation))

    plan = clamp_intervals(plan, fault_start, fault_end_excl)
    if not plan:
        print("WARNING: the plan contains no faulty frames; all frames will be copied unchanged")

    # Burst identifier: position of the burst in plan_used.json.
    for number, item in enumerate(plan):
        item["burst_id"] = number

    if not args.no_preflight:
        preflight_validate_plan(plan, args.in_dir, files, geometry)

    schedule: Dict[int, Dict] = {}
    for item in plan:
        for i in range(item["start"], item["end"]):
            if i in allowed_set:
                schedule[i] = item

    # A freeze repeats the last healthy frame before the burst, read from the
    # clean input sequence, so even the first frozen frame is stale.
    freeze_reference_by_frame: Dict[int, int] = {}
    for item in plan:
        if (item["fault_family"], item["fault_kind"]) == FREEZE_FAULT:
            ref = max(0, item["start"] - 1)
            for i in range(item["start"], item["end"]):
                if i in allowed_set:
                    freeze_reference_by_frame[i] = ref

    manifest: List[Dict] = []
    failed = 0

    for idx, fname in enumerate(files):
        in_path = os.path.join(args.in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        fault_label, fault_key, fam, kind, severity, subsystem, seed_used = (
            0, "none", "none", "none", "none", "none", None)
        frame_seed_used, burst_id, source_idx = None, None, idx

        try:
            if idx in schedule:
                spec = schedule[idx]
                fam, kind = spec["fault_family"], spec["fault_kind"]
                fault_key, severity = spec["fault"], spec["severity"]
                subsystem = SUBSYSTEM[(fam, kind)]
                burst_id = spec["burst_id"]

                if (fam, kind) in TEMPORAL_FAULTS:
                    if (fam, kind) == FREEZE_FAULT:
                        src_idx = freeze_reference_by_frame.get(idx, idx)
                    else:
                        src_idx = max(0, idx - temporal_delay_frames(severity))
                    source_idx = src_idx
                    shutil.copy2(os.path.join(args.in_dir, files[src_idx]), out_path)
                else:
                    kwargs = dict(spec.get("kwargs") or {})
                    if (fam, kind) in NO_SEED:
                        kwargs.pop("seed", None)
                    else:
                        kwargs.setdefault("seed", rng_sched.randint(0, 2**31 - 1))
                        seed_used = kwargs["seed"]
                    if (fam, kind) in FRAME_SEED_FAULTS:
                        kwargs["frame_seed"] = rng_sched.randint(0, 2**31 - 1)
                        frame_seed_used = kwargs["frame_seed"]

                    # Temporal synergies start from the delayed frame.
                    src_idx = idx
                    if fam == "synergistic" and kind in TEMPORAL_SYNERGIES:
                        src_idx = max(0, idx - temporal_delay_frames(severity))
                    source_idx = src_idx

                    frame = load_radar_png(os.path.join(args.in_dir, files[src_idx]), geometry)
                    save_radar_png(out_path, get_router(fam)(frame, kind, **kwargs))

                fault_label = 1
            else:
                shutil.copy2(in_path, out_path)

        except Exception as e:
            failed += 1
            print(f"[{idx + 1}/{total}] {fname} -> ERROR: {e}")
            try:
                shutil.copy2(in_path, out_path)
            except Exception:
                pass
            fault_label, fault_key, fam, kind, severity, subsystem, seed_used = (
                0, "none", "none", "none", "none", "none", None)
            frame_seed_used, burst_id, source_idx = None, None, idx

        manifest.append({
            "frame_idx": idx, "radar_name": fname, "fault_label": fault_label,
            "fault_key": fault_key, "fault_family": fam, "fault_kind": kind,
            "subsystem": subsystem, "severity": severity, "seed": seed_used,
            "frame_seed": frame_seed_used, "source_frame_idx": source_idx, "burst_id": burst_id,
        })

    manifest_df = pd.DataFrame(manifest)
    for column in ("seed", "frame_seed", "burst_id"):  # integers with empty cells, not floats
        manifest_df[column] = manifest_df[column].astype("Int64")
    manifest_df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    with open(os.path.join(out_dir, "plan_used.json"), "w") as f:
        json.dump(plan, f, indent=2)

    faulty = sum(1 for r in manifest if r["fault_label"] == 1)
    print("\n===== Radar Injector Summary =====")
    print(f"Total frames:      {total}")
    print(f"Faulty frames:     {faulty}")
    print(f"Failed frames:     {failed}")
    print(f"Output directory:  {out_dir}")


if __name__ == "__main__":
    main()