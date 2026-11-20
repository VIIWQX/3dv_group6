#!/usr/bin/env python3
"""Train NeRF with image-estimated SfM cameras, then compare to calibrated poses.

The input training matrices are copied verbatim from transforms_train_sfm.json,
which has already received ONE global Sim(3) gauge alignment upstream.  This
script never reads the original training extrinsics or per-camera GT pose file.
Original test/orbit cameras are used only for evaluation and rendering.  Images
remain the original RGBA PNGs, including alpha, rather than COLMAP RGB composites.

Examples:
    python src/train_sfm_branch.py --prepare-only
    python src/train_sfm_branch.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import traceback
from typing import Any

from run_ablations import (METRICS, baseline_configuration, build_command,
                           cached_configuration_matches, read_json, run_and_log)


GAUGE_COMMENT = (
    "Training cameras are image-estimated COLMAP poses, already aligned upstream "
    "by one global Sim(3) to fix coordinate gauge. Ground truth is used for that "
    "global gauge and held-out evaluation only; no per-camera GT training pose "
    "is read or substituted by this script. This is a gauge-aligned SfM-to-NeRF "
    "experiment, not a completely uncalibrated capture experiment."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_training_path(raw: str) -> Path:
    path = PurePosixPath(raw.replace("\\", "/"))
    parts = path.parts
    if parts and parts[0] == "images":
        parts = parts[1:]
    if path.is_absolute() or ".." in parts or not parts or parts[0] != "train":
        raise ValueError(f"Expected an SfM training PNG path, got {raw!r}")
    local = Path(*parts)
    return local if local.suffix else local.with_suffix(".png")


def ensure_relative_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing original image directory: {source}")
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace a regular path with a symlink: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)


def validate_matrix(matrix: Any, image: str) -> None:
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise ValueError(f"Invalid SfM matrix dimensions for {image}")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"Non-finite SfM matrix for {image}")
    if any(abs(matrix[3][index] - expected) > 1e-6 for index, expected in enumerate((0., 0., 0., 1.))):
        raise ValueError(f"Invalid homogeneous matrix last row for {image}")


def prepare_inputs(data_dir: Path, sfm_dir: Path, destination: Path,
                   expected_training_views: int = 72) -> dict:
    """Prepare training data without opening transforms_train.json or source GLBs."""
    sfm_path = sfm_dir / "transforms_train_sfm.json"
    sfm = read_json(sfm_path)
    if not sfm:
        raise FileNotFoundError(f"No aligned SfM training transforms: {sfm_path}")
    frames = sfm.get("frames", [])
    if len(frames) < 3:
        raise ValueError(f"Only {len(frames)} SfM cameras; at least three registered views are required")
    sfm_metrics = read_json(sfm_dir / "metrics.json")
    expected = int(sfm_metrics.get("input_images") or expected_training_views)
    prepared_frames, seen = [], set()
    for frame in frames:
        relative = normalized_training_path(frame["file_path"])
        if relative.as_posix() in seen:
            raise ValueError(f"Duplicate SfM frame: {relative}")
        seen.add(relative.as_posix())
        source_image = data_dir / relative
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing original RGBA training image: {source_image}")
        # Inspect only the PNG header: color type 6 is RGBA, type 4 is gray+alpha.
        with source_image.open("rb") as handle:
            header = handle.read(26)
        if len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[25] not in (4, 6):
            raise ValueError(f"Expected original alpha-bearing PNG, not SfM RGB composite: {source_image}")
        validate_matrix(frame["transform_matrix"], str(relative))
        record = dict(frame)
        record["file_path"] = relative.as_posix()
        # This assignment preserves the estimated pose values exactly, without
        # fitting or looking up an individual original training camera.
        record["transform_matrix"] = frame["transform_matrix"]
        prepared_frames.append(record)
    destination.mkdir(parents=True, exist_ok=True)
    ensure_relative_symlink(data_dir / "train", destination / "train")
    ensure_relative_symlink(data_dir / "test", destination / "test")
    if (data_dir / "orbit").is_dir():
        ensure_relative_symlink(data_dir / "orbit", destination / "orbit")
    training_document = dict(sfm)
    training_document["frames"] = prepared_frames
    (destination / "transforms_train.json").write_text(json.dumps(training_document, indent=2) + "\n")
    evaluation_fingerprints = {}
    for split in ("test", "orbit"):
        source = data_dir / f"transforms_{split}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        # Copy exact bytes, so all held-out cameras and calibration stay unchanged.
        (destination / source.name).write_bytes(source.read_bytes())
        evaluation_fingerprints[split] = sha256(source)
    input_fingerprint = hashlib.sha256(json.dumps({
        "sfm_transforms": sha256(sfm_path), "evaluation_transforms": evaluation_fingerprints,
        "rgba_paths": sorted(seen)}, sort_keys=True).encode()).hexdigest()
    metadata = {"source_data_directory": str(data_dir), "source_sfm_directory": str(sfm_dir),
                "prepared_directory": str(destination), "registered_count": len(prepared_frames),
                "expected_training_views": expected, "registered_ratio": len(prepared_frames) / max(1, expected),
                "sfm_status": sfm_metrics.get("status"), "sfm_registered_images_reported": sfm_metrics.get("registered_images"),
                "training_pose_source": str(sfm_path), "training_pose_values_copied_verbatim": True,
                "original_training_extrinsics_read": False, "source_geometry_read": False,
                "training_images": "Original alpha-bearing PNGs via relative directory symlink; no COLMAP white RGB composites",
                "test_and_orbit": "Original transforms copied byte-for-byte, for evaluation only",
                "sfm_transforms_sha256": sha256(sfm_path), "evaluation_transforms_sha256": evaluation_fingerprints,
                "input_fingerprint": input_fingerprint, "gauge_comment": GAUGE_COMMENT,
                "comparison_limitation": ("Partial SfM registration changes both camera accuracy and training-view count relative to the full-view baseline"
                                          if len(prepared_frames) < expected else "All original training views registered; comparison primarily measures estimated-camera effects"),
                "camera_center_error": sfm_metrics.get("camera_alignment", {}).get("camera_center_error"),
                "orientation_error_degrees": sfm_metrics.get("camera_alignment", {}).get("orientation_error_degrees")}
    (destination / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def summarize_row(obj: dict, baseline: dict, learned: dict, status: str, preparation: dict,
                  output_dir: Path, error: str | None = None) -> dict:
    row = {"object_id": obj["object_id"], "uid": obj["uid"], "status": status,
           "registered_count": preparation.get("registered_count"),
           "expected_training_views": preparation.get("expected_training_views", 72),
           "registered_ratio": preparation.get("registered_ratio"),
           "steps": learned.get("steps"), "resolution": learned.get("resolution"),
           "train_views": learned.get("train_views"), "output_directory": str(output_dir),
           "baseline_metrics_path": str(preparation.get("baseline_metrics_path", "")),
           "metrics_path": str(output_dir / "metrics.json"), "gauge_comment": GAUGE_COMMENT,
           "comparison_limitation": preparation.get("comparison_limitation"),
           "camera_center_error_rmse": (preparation.get("camera_center_error") or {}).get("rmse"),
           "orientation_error_mean_degrees": (preparation.get("orientation_error_degrees") or {}).get("mean"),
           "error": error}
    for key in METRICS:
        value, reference = learned.get(key), baseline.get(key)
        row[key], row["baseline_" + key] = value, reference
        row["delta_" + key] = float(value - reference) if value is not None and reference is not None else None
    return row


def save_summary(experiments: Path, rows: list[dict]) -> None:
    experiments.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row["status"] == "complete"]
    aggregates = {"completed_objects": len(completed)}
    for name in METRICS:
        values = [row["delta_" + name] for row in completed if row.get("delta_" + name) is not None]
        aggregates["mean_delta_" + name] = sum(values) / len(values) if values else None
    document = {"schema_version": "assignment2-sfm-pose-nerf-v1", "updated_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": "Train the same NeRF with SfM-estimated cameras, compare on the fixed calibrated held-out test split",
                "gauge_comment": GAUGE_COMMENT, "baseline_retrained": False,
                "delta_convention": "SfM-camera NeRF metric minus same-object calibrated-camera baseline metric",
                "missing_results": "Null fields and non-complete status are retained; failed reconstruction is never assigned an invented quality score",
                "aggregate": aggregates, "rows": rows}
    (experiments / "sfm_pose_summary.json").write_text(json.dumps(document, indent=2) + "\n")
    columns = ["object_id", "uid", "status", "registered_count", "expected_training_views", "registered_ratio",
               "steps", "resolution", "train_views", "camera_center_error_rmse", "orientation_error_mean_degrees"]
    columns += [key for metric in METRICS for key in ("baseline_" + metric, metric, "delta_" + metric)]
    columns += ["error", "gauge_comment", "comparison_limitation", "baseline_metrics_path", "metrics_path", "output_directory"]
    with (experiments / "sfm_pose_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--objects", nargs="+", default=["camera", "rock", "telephone"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare RGB/alpha/camera inputs and summaries without training")
    parser.add_argument("--collect-only", action="store_true", help="Prepare inputs and summarize existing matching results without training")
    parser.add_argument("--force", action="store_true", help="Retrain existing completed models")
    for key in ("steps", "resolution", "rays", "samples", "image_size", "seed"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, help="Override inherited baseline setting")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    selection = read_json(root / "manifests" / "selection.json")
    by_name = {obj["object_id"]: obj for obj in selection["objects"]}
    experiments = root / "experiments"
    rows, failed = [], False
    for name in args.objects:
        if name not in by_name:
            raise ValueError(f"Object absent from selection manifest: {name}")
        obj = by_name[name]
        baseline_dir = root / "results" / obj["uid"]
        baseline = read_json(baseline_dir / "metrics.json")
        prepared_dir = experiments / "sfm_inputs" / name
        output_dir = experiments / "sfm_pose" / name
        preparation: dict[str, Any] = {"baseline_metrics_path": str(baseline_dir / "metrics.json")}
        learned = {}
        try:
            preparation.update(prepare_inputs(root / "data" / obj["uid"], baseline_dir / "sfm", prepared_dir))
            config = baseline_configuration(baseline_dir, args)
            # Use every registered SfM frame; a baseline subset must not silently
            # discard successfully estimated cameras in this dedicated branch.
            config["train_views"] = None
            command = build_command(prepared_dir, output_dir, config, args.device)
            previous = read_json(output_dir / "command.json")
            complete = ((output_dir / "metrics.json").is_file() and cached_configuration_matches(output_dir, config)
                        and previous.get("input_fingerprint") == preparation["input_fingerprint"])
            if complete and not args.force:
                learned, status = read_json(output_dir / "metrics.json"), "complete"
                print(f"Reusing {name}: completed SfM-pose NeRF with matching input/configuration", flush=True)
            elif args.prepare_only or args.collect_only:
                status = "prepared" if args.prepare_only else "missing"
                print(f"Prepared {name}: {preparation['registered_count']}/{preparation['expected_training_views']} SfM cameras", flush=True)
            else:
                if not baseline:
                    raise FileNotFoundError(f"Finish calibrated-camera baseline before training: {baseline_dir / 'metrics.json'}")
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "command.json").write_text(json.dumps({"argv": command, "configuration": config,
                    "input_fingerprint": preparation["input_fingerprint"], "baseline_directory": str(baseline_dir),
                    "gauge_comment": GAUGE_COMMENT}, indent=2) + "\n")
                print(f"Training SfM-camera NeRF for {name}: {preparation['registered_count']}/{preparation['expected_training_views']} registered views", flush=True)
                run_and_log(command, output_dir)
                learned = read_json(output_dir / "metrics.json")
                if not learned:
                    raise RuntimeError(f"Training returned without evaluation metrics: {output_dir}")
                status = "complete"
            rows.append(summarize_row(obj, baseline, learned, status, preparation, output_dir))
        except Exception as error:
            failed = True
            rows.append(summarize_row(obj, baseline, {}, "failed", preparation, output_dir, f"{type(error).__name__}: {error}"))
            traceback.print_exc()
        save_summary(experiments, rows)
    print(f"Saved {experiments / 'sfm_pose_summary.json'} and .csv", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
