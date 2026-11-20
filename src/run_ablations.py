#!/usr/bin/env python3
"""Run twelve controlled NeRF ablations and compare held-out metrics to baseline.

Camera, cart and telephone each receive four changes: 12 training views, noisy
training poses, a 32^3 grid, and removal of silhouette density initialization.
All other optimization settings are inherited from each completed baseline's
config.json.  Baselines are read from results/<uid> and never retrained here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


VARIANTS = {
    "views12": {"train_views": 12},
    "pose_noise2deg": {"pose_noise_deg": 2.0},
    "grid32": {"resolution": 32},
    "no_hull_prior": {"no_silhouette_init": True},
}
DESCRIPTIONS = {
    "baseline": "Completed baseline; read without retraining",
    "views12": "Reduce training views to 12, evenly spaced in the original training list",
    "pose_noise2deg": "Perturb training camera rotations: random axis and N(0,2 degrees) angle; clean test poses",
    "grid32": "Keep density and eight-channel feature grids at 32 cubed throughout training",
    "no_hull_prior": "Replace silhouette-derived density initialization with a generic sphere; retain alpha loss and foreground sampling",
}
DEFAULTS = {"steps": 3000, "resolution": 96, "rays": 4096, "samples": 128,
            "image_size": 256, "render_size": 256, "chunk": 4096, "seed": 42,
            "alpha_weight": .1, "tv_weight": 1e-5, "bound": 1.,
            "train_views": None, "pose_noise_deg": 0., "no_alpha_loss": False,
            "no_silhouette_init": False}
METRICS = ("psnr", "foreground_psnr", "ssim", "mask_iou", "training_seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def baseline_configuration(baseline_dir: Path, args: argparse.Namespace) -> dict:
    document = read_json(baseline_dir / "config.json")
    saved = document.get("config", document)
    config = dict(DEFAULTS)
    config.update({key: value for key, value in saved.items() if key in DEFAULTS})
    # Optional CLI overrides are mainly useful for a small reproducibility smoke.
    for key in ("steps", "resolution", "rays", "samples", "image_size", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def build_command(data_dir: Path, output_dir: Path, config: dict, device: str) -> list[str]:
    command = [sys.executable, str(Path(__file__).with_name("nerf.py")),
               "--data", str(data_dir), "--output", str(output_dir), "--device", device, "--skip-orbit"]
    for key, value in config.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend((option, str(value)))
    return command


def metric_row(obj: dict, variant: str, directory: Path, baseline: dict,
               status: str = "complete", command: list[str] | None = None) -> dict:
    metrics = read_json(directory / "metrics.json")
    row: dict[str, Any] = {"object_id": obj["object_id"], "uid": obj["uid"], "variant": variant,
                           "description": DESCRIPTIONS[variant], "status": status,
                           "metrics_path": str(directory / "metrics.json"),
                           "steps": metrics.get("steps"), "train_views": metrics.get("train_views"),
                           "resolution": metrics.get("resolution")}
    for name in METRICS:
        value, base_value = metrics.get(name), baseline.get(name)
        row[name] = value
        row["delta_" + name] = float(value - base_value) if value is not None and base_value is not None else None
    if command is not None:
        row["command"] = shlex.join(command)
    return row


def save_summary(output_root: Path, rows: list[dict], config_by_object: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    aggregates = {}
    for variant in ("baseline", *VARIANTS):
        completed = [row for row in rows if row["variant"] == variant and row["status"] == "complete"]
        aggregate: dict[str, Any] = {"completed_objects": len(completed)}
        for metric in METRICS:
            for key in (metric, "delta_" + metric):
                values = [row[key] for row in completed if row.get(key) is not None]
                aggregate["mean_" + key] = sum(values) / len(values) if values else None
        aggregates[variant] = aggregate
    document = {"schema_version": "assignment2-ablation-v1", "updated_utc": datetime.now(timezone.utc).isoformat(),
                "design": "Four changes relative to each object's completed baseline; same seed and fixed held-out test split",
                "baseline_retrained": False, "delta_convention": "variant metric minus same-object baseline metric",
                "interpretation": "Single-seed controlled comparisons; no significance claims. Pose noise is applied to training cameras only.",
                "base_training_configuration_by_object": config_by_object,
                "aggregate_by_variant": aggregates, "rows": rows}
    (output_root / "ablation_summary.json").write_text(json.dumps(document, indent=2))
    columns = ["object_id", "uid", "variant", "status", "steps", "train_views", "resolution"]
    columns += [key for metric in METRICS for key in (metric, "delta_" + metric)]
    columns += ["description", "metrics_path", "command"]
    with (output_root / "ablation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cached_configuration_matches(directory: Path, expected: dict) -> bool:
    saved_document = read_json(directory / "config.json")
    saved = saved_document.get("config", saved_document)
    return bool(saved) and all(saved.get(key, DEFAULTS.get(key)) == value for key, value in expected.items())


def run_and_log(command: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("w") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objects", nargs="+", default=["camera", "cart", "telephone"])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--force", action="store_true", help="Retrain even when matching completed ablations exist")
    parser.add_argument("--collect-only", action="store_true", help="Read available metrics and write summaries; never launch training")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and write planned summary; never launch training")
    for key in ("steps", "resolution", "rays", "samples", "image_size", "seed"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, help="Override inherited baseline setting")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    manifest = read_json(args.manifest or root / "manifests" / "selection.json")
    by_name = {obj["object_id"]: obj for obj in manifest["objects"]}
    unknown = set(args.objects) - set(by_name)
    if unknown:
        raise ValueError(f"Objects absent from selection manifest: {sorted(unknown)}")
    data_root = (args.data_root or root / "data").resolve()
    results_root = (args.results_root or root / "results").resolve()
    output_root = (args.output_root or root / "experiments").resolve()
    rows: list[dict] = []
    configs = {}
    for name in args.objects:
        obj = by_name[name]
        baseline_dir = results_root / obj["uid"]
        baseline = read_json(baseline_dir / "metrics.json")
        if not baseline and not (args.dry_run or args.collect_only):
            raise FileNotFoundError(f"Finish the baseline before ablation: {baseline_dir / 'metrics.json'}")
        base_config = baseline_configuration(baseline_dir, args)
        configs[name] = base_config
        rows.append(metric_row(obj, "baseline", baseline_dir, baseline, "complete" if baseline else "missing"))
        save_summary(output_root, rows, configs)
        for variant in args.variants:
            config = dict(base_config, **VARIANTS[variant])
            directory = output_root / name / variant
            command = build_command(data_root / obj["uid"], directory, config, args.device)
            complete = (directory / "metrics.json").exists() and cached_configuration_matches(directory, config)
            if args.dry_run:
                print(shlex.join(command), flush=True)
                status = "complete" if complete else "planned"
            elif args.collect_only:
                status = "complete" if complete else "missing"
            elif complete and not args.force:
                print(f"Reusing {name}/{variant}: matching completed configuration", flush=True)
                status = "complete"
            else:
                print(f"Running {name}/{variant}: {DESCRIPTIONS[variant]}", flush=True)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "command.json").write_text(json.dumps({"argv": command, "configuration": config,
                                                                  "baseline_directory": str(baseline_dir),
                                                                  "changed_factor": VARIANTS[variant]}, indent=2))
                try:
                    run_and_log(command, directory)
                    if not (directory / "metrics.json").exists():
                        raise RuntimeError(f"Training finished without metrics: {directory}")
                    status = "complete"
                except Exception:
                    rows.append(metric_row(obj, variant, directory, baseline, "failed", command))
                    save_summary(output_root, rows, configs)
                    raise
            rows.append(metric_row(obj, variant, directory, baseline, status, command))
            save_summary(output_root, rows, configs)
    print(f"Saved {output_root / 'ablation_summary.json'} and .csv", flush=True)


if __name__ == "__main__":
    main()
