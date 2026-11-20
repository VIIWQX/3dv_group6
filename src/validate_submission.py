#!/usr/bin/env python3
"""Independently validate the ten-object assignment artifacts using CPU only.

Run after the pipeline: python src/validate_submission.py
The default requires every mandatory artifact to be present and valid. During
an active run, --allow-pending permits missing artifacts, while still failing
on corrupt files or inconsistent results. This option never marks an unfinished
submission as complete. Identity placeholders are explicitly permitted.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageSequence


class InvalidArtifact(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidArtifact(message)


def read_json(path):
    def reject_constant(value):
        raise InvalidArtifact(f"Non-finite JSON numeric literal {value}: {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_json(value, location="root"):
    if isinstance(value, float):
        require(np.isfinite(value), f"Non-finite metric at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{location}[{index}]")


def close(actual, expected, message, atol=1e-5, rtol=1e-4):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), message)


def check_source_code(project):
    forbidden = {"trimesh", "pyrender", "open3d", "bpy", "blender", "pymeshlab", "render_dataset"}
    checked = []
    for name in ("nerf.py", "export_geometry.py", "match_features.py", "run_sfm.py", "source.py"):
        path = project / "src" / name
        if name == "source.py" and not path.exists():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                require(node.func.attr not in {"read_mesh", "load_mesh", "read_triangle_mesh", "read_point_cloud"},
                        f"Source geometry reader in reconstruction module {name}:{node.lineno}")
        require(not any(module.split(".")[0] in forbidden for module in imports),
                f"Source geometry/renderer import in reconstruction module {name}: {imports}")
        # Check the actual optimization function, excluding held-out evaluation.
        if name == "nerf.py":
            train_function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"), None)
            require(train_function is not None, "NeRF train function is missing")
            loaders = [node for node in ast.walk(train_function)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_dataset"]
            require(bool(loaders), "Cannot locate training data load in NeRF")
            for loader in loaders:
                require(len(loader.args) >= 2 and isinstance(loader.args[1], ast.Constant) and loader.args[1].value == "train",
                        "Optimization function loads a split other than train")
        checked.append({"file": str(path.relative_to(project)), "sha256": sha256(path), "imports": imports})
    return {"checked_modules": checked, "source_geometry_imports_in_reconstruction": False,
            "training_function_loads_train_only": True,
            "limitation": "Static import/call audit supplements artifact checks; it is not a general proof about arbitrary Python execution."}


def load_audit(path, uids):
    selected = {}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("uid") in uids:
                require(record["uid"] not in selected, f"Duplicate selected UID in audit: {path}: {record['uid']}")
                selected[record["uid"]] = record
    return selected


def resolve_audit_context(project, source_root, uids, source_revision=None):
    """Prefer original audits, or verify the archived ten-object evidence subset.

    Bundled records preserve Assignment 1's earlier decisions; reading them is
    not a rerun of cleaning. Source GLBs are independently rehashed afterwards.
    """
    original = {
        "strict_index": source_root / "cleaned/object-paths_000-000.json",
        "old_audit": source_root / "manifests/cleaning_000-000.jsonl",
        "v2_audit": source_root / "version2/results/000-000_to_000-004/manifests/cleaning_000-000.jsonl",
    }
    if all(path.is_file() for path in original.values()):
        paths = original
        evidence_source = "assignment1_original_files"
        details = {"evidence_source": evidence_source, "archived_subset": False,
                   "paths": {name: str(path) for name, path in paths.items()}}
    else:
        folder = project / "manifests/assignment1_evidence"
        manifest = read_json(folder / "evidence_manifest.json")
        require(manifest.get("schema_version") == "assignment1-selected-evidence-v1", "Unsupported archived Assignment 1 evidence schema")
        require(manifest.get("evidence_kind") == "archived_subset_of_existing_assignment1_results"
                and manifest.get("audit_rerun") is False, "Archived evidence must identify itself as a subset, not a new cleaning run")
        require(manifest.get("selected_uid_count") == 10 and len(manifest.get("selected_uids", [])) == 10
                and set(manifest["selected_uids"]) == set(uids), "Archived evidence must contain exactly the ten selected UIDs")
        require(manifest.get("source_shards") == ["000-000"], "Archived evidence has an unexpected shard scope")
        if source_revision is not None:
            require(manifest.get("source_revision") == source_revision, "Archived evidence source revision differs from selection")
        paths = {"strict_index": folder / "strict_index.json", "old_audit": folder / "strict_cleaning.jsonl",
                 "v2_audit": folder / "v2_cleaning.jsonl"}
        for path in paths.values():
            entry = manifest["bundled_files"][path.name]
            require(entry["records"] == 10 and path.stat().st_size == entry["bytes"], f"Archived evidence size/count changed: {path.name}")
            require(sha256(path) == entry["sha256"], f"Archived evidence checksum mismatch: {path.name}")
        for key in ("strict_index", "strict_cleaning", "v2_cleaning"):
            original_entry = manifest["original_source_files"][key]
            require(len(original_entry["sha256"]) == 64 and original_entry["bytes"] > 0,
                    f"Archived evidence lacks original full-file fingerprint: {key}")
        evidence_source = "bundled_assignment1_subset"
        details = {"evidence_source": evidence_source, "archived_subset": True, "audit_rerun": False,
                   "evidence_manifest_sha256": sha256(folder / "evidence_manifest.json"),
                   "original_source_fingerprints": manifest["original_source_files"],
                   "missing_original_files": [str(path) for path in original.values() if not path.is_file()],
                   "paths": {name: str(path) for name, path in paths.items()}}
    strict_index = read_json(paths["strict_index"])
    old_audit = load_audit(paths["old_audit"], uids)
    v2_audit = load_audit(paths["v2_audit"], uids)
    require(set(old_audit) == set(v2_audit) == set(uids), "Cleaning audits do not cover exactly the ten selected UIDs")
    require(set(uids) <= set(strict_index), "Strict cleaned index omits a selected UID")
    if evidence_source == "bundled_assignment1_subset":
        require(set(strict_index) == set(uids), "Archived strict index must contain only the selected ten objects")
        for key in ("old_audit", "v2_audit"):
            with paths[key].open() as stream:
                records = [json.loads(line) for line in stream]
            require(len(records) == 10 and {record["uid"] for record in records} == set(uids),
                    f"Archived {key} contains extra or missing audit records")
        for uid in uids:
            require(old_audit[uid]["sha256"] == v2_audit[uid]["sha256"] == manifest["source_glb_sha256"][uid],
                    f"Archived audit/source fingerprint disagrees for {uid}")
    details.update(selected_original_audit_records=len(old_audit), selected_v2_audit_records=len(v2_audit))
    context = {"strict_index": strict_index, "old_audit": old_audit, "v2_audit": v2_audit,
               "evidence_source": evidence_source}
    return context, details


def check_selection(selection):
    objects = selection["objects"]
    uids = [item["uid"] for item in objects]
    require(len(objects) == 10 and len(set(uids)) == 10, "Selection must contain exactly 10 distinct UIDs")
    require(all(len(uid) == 32 and all(character in "0123456789abcdef" for character in uid) for uid in uids),
            "Selected object UID is malformed")
    require(all(item.get("source_shard") == "000-000" for item in objects), "Every selected object must come from shard 000-000")
    require(len({item["object_id"] for item in objects}) == 10, "Object display IDs are not unique")
    return {"objects": len(objects), "distinct_uids": len(set(uids)), "shards": ["000-000"]}


def check_source(item, source_root, strict_index, old_audit, v2_audit, evidence_source="assignment1_original_files"):
    uid = item["uid"]
    source = source_root / item["source_relative_path"]
    require(source.suffix == ".glb" and source.parent.name == "000-000", "Source path is outside the selected GLB shard")
    require(uid in strict_index, f"{uid}: absent from original strict cleaned index")
    require(uid in old_audit and old_audit[uid]["status"] == "kept", f"{uid}: original strict audit did not keep object")
    require(uid in v2_audit and v2_audit[uid]["status"] == "kept", f"{uid}: V2 audit did not keep object")
    require(v2_audit[uid].get("strict_status") == "kept", f"{uid}: V2 strict status is not kept")
    require(v2_audit[uid].get("shard") == "000-000", f"{uid}: V2 audit has an unexpected shard")
    strict_relative = Path(strict_index[uid])
    require(not strict_relative.is_absolute() and strict_relative.as_posix() == f"glbs/000-000/{uid}.glb",
            f"{uid}: strict index path has an unexpected UID/shard")
    actual_hash = sha256(source)
    require(actual_hash == item["sha256"] == v2_audit[uid]["sha256"] == old_audit[uid]["sha256"],
            f"{uid}: source GLB SHA-256 differs from selection or cleaning audits")
    require(source.stat().st_size == item["file_size_bytes"], f"{uid}: source file size changed")
    strict_copy = source_root / "cleaned" / strict_relative
    strict_copy_exists = strict_copy.is_file()
    if strict_copy_exists:
        require(sha256(strict_copy) == actual_hash, f"{uid}: physical strict cleaned copy differs from the audited source")
    return {"path": str(source), "sha256": actual_hash, "bytes": source.stat().st_size,
            "original_strict_kept": True, "v2_kept": True, "v2_strict_kept": True,
            "evidence_source": evidence_source, "physical_strict_copy_exists": strict_copy_exists,
            "strict_membership_basis": "Original strict kept audit + strict index membership + identical rehashed source SHA-256"}


def png_alpha(path, expected_size):
    with Image.open(path) as image:
        require(image.format == "PNG", f"Input is not a PNG: {path}")
        require(image.size == expected_size, f"Incorrect PNG dimensions: {path}: {image.size}")
        require("A" in image.getbands(), f"Input lost its alpha channel: {path}")
        rgba = np.asarray(image.convert("RGBA"))
    yy, xx = np.where(rgba[:, :, 3] > 127)
    require(len(xx) > 0, f"Empty foreground: {path}")
    width, height = expected_size
    margin = int(min(xx.min(), yy.min(), width - 1 - xx.max(), height - 1 - yy.max()))
    require(margin >= 1, f"Foreground touches image boundary and may be truncated: {path}")
    require(np.any(rgba[:, :, 3] == 0), f"Input alpha has no background: {path}")
    return margin, len(xx)


def check_dataset(data):
    splits, camera_keys, all_paths = {}, {}, set()
    margins, foreground_counts = [], []
    for split, count in (("train", 72), ("test", 12), ("orbit", 60)):
        meta = read_json(data / f"transforms_{split}.json")
        require(len(meta["frames"]) == count, f"{split} has {len(meta['frames'])} views; expected {count}")
        width, height = int(meta["w"]), int(meta["h"])
        require(width > 0 and height > 0, f"Invalid {split} image dimensions")
        require(meta.get("camera_model") == "PINHOLE", f"Unexpected camera model for {split}")
        require(all(np.isfinite(meta[key]) for key in ("fl_x", "fl_y", "cx", "cy")), f"Non-finite {split} calibration")
        require(meta["fl_x"] > 0 and meta["fl_y"] > 0, f"Non-positive {split} focal length")
        keys, paths = [], []
        for index, frame in enumerate(meta["frames"]):
            pose = np.asarray(frame["transform_matrix"], dtype=float)
            require(pose.shape == (4, 4) and np.isfinite(pose).all(), f"Invalid {split} camera {index}")
            close(pose[3], [0, 0, 0, 1], f"Invalid homogeneous {split} camera {index}", atol=1e-7, rtol=0)
            close(pose[:3, :3].T @ pose[:3, :3], np.eye(3), f"Non-orthonormal {split} rotation {index}", atol=1e-6, rtol=0)
            close(np.linalg.det(pose[:3, :3]), 1, f"Improper {split} rotation {index}", atol=1e-6, rtol=0)
            keys.append(tuple(np.round(pose.reshape(-1), 6)))
            relative = Path(frame["file_path"])
            require(not relative.is_absolute() and relative.parts[0] == split, f"Incorrect {split} frame path {relative}")
            actual = (data / relative).resolve()
            require(actual.is_relative_to(data.resolve()), f"Frame leaves dataset directory: {relative}")
            require(actual not in all_paths, f"Image path reused between splits: {relative}")
            all_paths.add(actual)
            paths.append(relative.as_posix())
            if split != "orbit" or meta.get("images_rendered", False):
                margin, foreground = png_alpha(actual, (width, height))
                margins.append(margin)
                foreground_counts.append(foreground)
        require(len(set(keys)) == count, f"Duplicate cameras within {split}")
        for previous, previous_keys in camera_keys.items():
            require(not set(keys) & previous_keys, f"Duplicate camera poses across {previous} and {split}")
        camera_keys[split] = set(keys)
        splits[split] = {"metadata": meta, "paths": paths}
        if split == "orbit":
            expected_azimuths = (np.arange(60) * 6 + 2) % 360
            actual_azimuths = np.asarray([frame["azimuth"] % 360 for frame in meta["frames"]])
            close(actual_azimuths, expected_azimuths, "Orbit cameras must use 60 steps with a 2 degree azimuth offset", atol=1e-6, rtol=0)
    evidence = {"views": {split: len(record["paths"]) for split, record in splits.items()},
                "unique_camera_poses": sum(len(keys) for keys in camera_keys.values()),
                "cross_split_camera_overlap": 0, "orbit_azimuth_offset_degrees": 2,
                "checked_rgba_images": len(margins), "minimum_foreground_border_margin_pixels": min(margins),
                "minimum_foreground_pixels": min(foreground_counts), "input_size": [width, height]}
    return evidence, splits


def npz_arrays(path, required):
    with np.load(path, allow_pickle=False) as archive:
        require(set(required) <= set(archive.files), f"Missing arrays in {path}: {set(required) - set(archive.files)}")
        return {name: archive[name] for name in required}


def check_matching(folder, train_meta):
    summary = read_json(folder / "summary.json")
    finite_json(summary)
    feature_stats = read_json(folder / "feature_stats.json")
    with (folder / "pair_stats.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(summary["views"] == 72 and summary["view_stride"] == 1, "Main matching result does not cover all 72 training views")
    require(len(feature_stats) == 72, "Feature statistics do not cover 72 training images")
    features = {}
    image_size = summary["image_size"]
    scale = image_size / train_meta["w"]
    known_k = np.array([[train_meta["fl_x"] * scale, 0, train_meta["cx"] * scale],
                        [0, train_meta["fl_y"] * scale, train_meta["cy"] * scale], [0, 0, 1]])
    for record in feature_stats:
        frame = int(record["frame"])
        arrays = npz_arrays(folder / "features" / f"{frame:03d}.npz", ["keypoints", "descriptors", "source_image", "K", "camera_to_world"])
        count = len(arrays["keypoints"])
        require(arrays["keypoints"].shape == (count, 7) and arrays["descriptors"].shape == (count, 128), f"Feature shapes disagree for frame {frame}")
        require(np.isfinite(arrays["keypoints"]).all() and np.isfinite(arrays["descriptors"]).all(), f"Non-finite features in frame {frame}")
        require(count == record["keypoints"], f"Feature count disagrees for frame {frame}")
        require(record["file_path"] == train_meta["frames"][frame]["file_path"] == str(arrays["source_image"].item()), f"Feature input path disagrees for frame {frame}")
        close(arrays["K"], known_k, f"Incorrect feature calibration for frame {frame}")
        close(arrays["camera_to_world"], train_meta["frames"][frame]["transform_matrix"], f"Incorrect feature camera for frame {frame}")
        require(frame not in features, f"Duplicate feature statistics for frame {frame}")
        features[frame] = count
    require(set(features) == set(range(72)), "Missing training-frame features")
    require(len(list((folder / "features").glob("*.npz"))) == 72, "Unexpected feature archive count")
    require(len(rows) == summary["pairs"] == 288, "Main matching result must contain 288 scheduled image pairs")
    require(len(list((folder / "pairs").glob("*.npz"))) == len(rows), "Pair archive count differs from pair_stats.csv")
    total_matches = total_inliers = total_triangulated = pairs_with_8 = 0
    rotation_errors = []
    points, frame_pairs, seen = [], [], set()
    for row in rows:
        i, j = int(row["frame_i"]), int(row["frame_j"])
        name = f"{i:03d}_{j:03d}"
        require(0 <= i < j < 72 and (i, j) not in seen, f"Invalid/duplicate image pair {name}")
        seen.add((i, j))
        require(row["pair_file"] == f"pairs/{name}.npz", f"Pair file path disagrees for {name}")
        arrays = npz_arrays(folder / row["pair_file"], ["matches", "distances", "ransac_inliers", "triangulated", "points3d",
                            "reprojection_error", "parallax_degrees", "E_estimated", "E_known", "K", "frame_indices"])
        matches, inliers, tri = arrays["matches"], arrays["ransac_inliers"], arrays["triangulated"]
        count = len(matches)
        require(matches.shape == (count, 2) and np.issubdtype(matches.dtype, np.integer), f"Malformed matches for {name}")
        require(inliers.shape == tri.shape == (count,) and inliers.dtype == tri.dtype == np.dtype(bool), f"Malformed masks for {name}")
        require(arrays["distances"].shape == (count, 2) and arrays["points3d"].shape == (count, 3), f"Malformed correspondence arrays for {name}")
        require(arrays["reprojection_error"].shape == arrays["parallax_degrees"].shape == (count,), f"Malformed geometry arrays for {name}")
        for key in ("distances", "points3d", "reprojection_error", "parallax_degrees", "E_estimated", "E_known", "K"):
            require(np.isfinite(arrays[key]).all(), f"Non-finite {key} for {name}")
        require(arrays["E_estimated"].shape == arrays["E_known"].shape == arrays["K"].shape == (3, 3),
                f"Malformed essential matrix/calibration for {name}")
        require(np.array_equal(arrays["frame_indices"], [i, j]), f"Wrong source frame indices for {name}")
        close(arrays["K"], known_k, f"Incorrect pair calibration for {name}")
        if count:
            require((matches >= 0).all() and (matches[:, 0] < features[i]).all() and (matches[:, 1] < features[j]).all(), f"Match index out of range for {name}")
            require((arrays["distances"][:, 0] < summary["ratio_threshold"] * arrays["distances"][:, 1] + 1e-5).all(), f"Lowe ratio threshold violated for {name}")
        require(not (tri & ~inliers).any(), f"Triangulated points are not a subset of RANSAC inliers for {name}")
        for field, expected in (("features_i", features[i]), ("features_j", features[j]), ("ratio_matches", count),
                                ("ransac_inliers", int(inliers.sum())), ("triangulated_points", int(tri.sum()))):
            require(int(row[field]) == expected, f"CSV {field} disagrees with NPZ for {name}")
        close(float(row["inlier_ratio"]), float(inliers.mean()) if count else 0, f"CSV inlier ratio disagrees for {name}")
        if tri.any():
            require((arrays["reprojection_error"][tri] < 3 + 1e-5).all(), f"Triangulation reprojection threshold violated for {name}")
            require((arrays["parallax_degrees"][tri] > 1 - 1e-5).all(), f"Triangulation parallax threshold violated for {name}")
            require((np.max(abs(arrays["points3d"][tri]), axis=1) < 2 + 1e-5).all(), f"Triangulation world bound violated for {name}")
            for frame_index in (i, j):
                pose = np.asarray(train_meta["frames"][frame_index]["transform_matrix"])
                camera_points = (arrays["points3d"][tri] - pose[:3, 3]) @ pose[:3, :3]
                require((camera_points[:, 2] < 0).all(), f"Triangulated point is behind camera {frame_index} for {name}")
            close(float(row["mean_reprojection_error"]), arrays["reprojection_error"][tri].mean(), f"CSV reprojection mean disagrees for {name}")
            close(float(row["median_parallax_degrees"]), np.median(arrays["parallax_degrees"][tri]), f"CSV parallax median disagrees for {name}")
            points.append(arrays["points3d"][tri])
            frame_pairs.append(np.tile([i, j], (int(tri.sum()), 1)))
        else:
            require(row["mean_reprojection_error"] == row["median_parallax_degrees"] == "", f"CSV reports geometry statistics for empty pair {name}")
        total_matches += count
        total_inliers += int(inliers.sum())
        total_triangulated += int(tri.sum())
        pairs_with_8 += int(inliers.sum() >= 8)
        for key in ("rotation_error_degrees", "translation_error_degrees"):
            if row[key] != "":
                value = float(row[key])
                require(np.isfinite(value) and 0 <= value <= 180.00001, f"Invalid CSV pose error {key} for {name}")
                if key == "rotation_error_degrees":
                    rotation_errors.append(value)
    for field, expected in (("keypoints_total", sum(features.values())), ("ratio_matches_total", total_matches),
                            ("ransac_inliers_total", total_inliers), ("triangulated_observations", total_triangulated),
                            ("pairs_with_8_inliers", pairs_with_8)):
        require(summary[field] == expected, f"Matching summary {field} disagrees with saved evidence")
    close(summary["keypoints_mean"], np.mean(list(features.values())), "Summary mean keypoint count disagrees")
    if rotation_errors:
        close(summary["median_estimated_rotation_error_degrees"], np.median(rotation_errors), "Summary median estimated rotation error disagrees")
    else:
        require(summary["median_estimated_rotation_error_degrees"] is None, "Summary reports rotation error without pose estimates")
    raw = npz_arrays(folder / "sparse_raw.npz", ["points", "colors", "frame_pairs"])
    sparse = npz_arrays(folder / "sparse.npz", ["points", "colors"])
    require(raw["points"].shape == raw["colors"].shape == (total_triangulated, 3), "Raw sparse point/color count disagrees")
    require(raw["frame_pairs"].shape == (total_triangulated, 2), "Raw sparse provenance count disagrees")
    expected_points = np.concatenate(points) if points else np.empty((0, 3))
    expected_pairs = np.concatenate(frame_pairs) if frame_pairs else np.empty((0, 2))
    close(raw["points"], expected_points, "Raw sparse points disagree with saved pair triangulations", atol=1e-7, rtol=0)
    require(np.array_equal(raw["frame_pairs"], expected_pairs), "Raw sparse frame provenance disagrees with pair triangulations")
    require(sparse["points"].shape == sparse["colors"].shape == (summary["sparse_points"], 3), "Deduplicated sparse count disagrees with summary")
    require(np.isfinite(sparse["points"]).all(), "Non-finite deduplicated sparse points")
    raw_set = {tuple(point) for point in raw["points"]}
    require(all(tuple(point) in raw_set for point in sparse["points"]), "Deduplicated point has no raw observation")
    ply = read_ply(folder / "sparse.ply")
    require(ply["vertices"] == summary["sparse_points"], "Sparse PLY count differs from NPZ/summary")
    return {"feature_archives": len(features), "pair_archives": len(rows), "keypoints": sum(features.values()),
            "ratio_matches": total_matches, "ransac_inliers": total_inliers,
            "triangulated_observations": total_triangulated, "sparse_points": summary["sparse_points"],
            "arrays_csv_summary_consistent": True}


def read_ply(path):
    with path.open("rb") as stream:
        header = []
        for _ in range(200):
            line = stream.readline()
            require(bool(line), f"Truncated PLY header: {path}")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        require(header[0] == "ply" and header[-1] == "end_header", f"Invalid PLY header: {path}")
        start = stream.tell()
        payload = stream.read()
    vertices = int(next(line.split()[2] for line in header if line.startswith("element vertex ")))
    faces = int(next((line.split()[2] for line in header if line.startswith("element face ")), 0))
    require(vertices > 0, f"PLY contains no vertices: {path}")
    if "format ascii 1.0" in header:
        lines = payload.decode("ascii").splitlines()
        require(len(lines) >= vertices + faces, f"Truncated ASCII PLY: {path}")
        points = np.asarray([[float(value) for value in line.split()[:3]] for line in lines[:vertices]])
        require(points.shape == (vertices, 3) and np.isfinite(points).all(), f"Invalid PLY coordinates: {path}")
    elif "format binary_little_endian 1.0" in header:
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}
        properties, active = [], False
        for line in header:
            if line.startswith("element "):
                active = line.startswith("element vertex ")
            elif active and line.startswith("property "):
                _, kind, name = line.split()
                require(kind in types, f"Unsupported PLY vertex type {kind}: {path}")
                properties.append((name, types[kind]))
        dtype = np.dtype(properties)
        vertex_bytes = vertices * dtype.itemsize
        require(len(payload) >= vertex_bytes, f"Truncated binary PLY vertices: {path}")
        vertex_data = np.frombuffer(payload, dtype=dtype, count=vertices)
        require(all(np.isfinite(vertex_data[key]).all() for key in ("x", "y", "z")), f"Non-finite PLY coordinates: {path}")
        if faces:
            face_dtype = np.dtype([("size", "u1"), ("indices", "<i4", (3,))])
            require(len(payload) == vertex_bytes + faces * face_dtype.itemsize, f"Binary PLY size does not match vertex/triangle counts: {path}")
            face_data = np.frombuffer(payload, dtype=face_dtype, count=faces, offset=vertex_bytes)
            require((face_data["size"] == 3).all(), f"Non-triangle PLY face: {path}")
            require((face_data["indices"] >= 0).all() and (face_data["indices"] < vertices).all(), f"PLY face index out of range: {path}")
    else:
        raise InvalidArtifact(f"Unsupported PLY format: {path}")
    return {"vertices": vertices, "faces": faces, "bytes": path.stat().st_size, "header_bytes": start}


def check_checkpoint_and_metrics(output, splits, data, project):
    import torch
    torch.set_num_threads(4)
    checkpoint_path = output / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]
    require(bool(state) and "density" in state and "features" in state, "NeRF checkpoint lacks learned density/features")
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor) and tensor.numel() > 0 and bool(torch.isfinite(tensor).all()), f"Invalid checkpoint tensor {name}")
    resolution = checkpoint["resolution"]
    require(state["density"].shape == (1, 1, resolution, resolution, resolution), "Learned density grid shape disagrees with checkpoint metadata")
    require(state["features"].shape == (1, checkpoint["feature_dim"], resolution, resolution, resolution), "Learned feature grid shape disagrees with checkpoint metadata")
    require(checkpoint["train_views"] == 72, "Main checkpoint did not use all 72 training views")
    require(checkpoint["steps"] > 0, "Checkpoint has no optimization steps")
    data_argument = Path(checkpoint["config"]["data"])
    resolutions = [data_argument.resolve()] if data_argument.is_absolute() else [(project / data_argument).resolve(), (project.parent / data_argument).resolve()]
    same_object=(data_argument.name==data.name and data_argument.parent.name=='data')
    require(data.resolve() in resolutions or same_object, "Checkpoint training data points at a different object")
    metrics = read_json(output / "metrics.json")
    finite_json(metrics)
    for key in ("psnr", "foreground_psnr", "mean_view_psnr", "ssim", "mask_iou"):
        require(isinstance(metrics.get(key), (int, float)) and np.isfinite(metrics[key]), f"Missing/non-finite held-out metric {key}")
    require(metrics["test_views"] == len(metrics["per_view"]) == 12, "Metrics do not cover 12 held-out test images")
    require(metrics.get("test_used_for_optimization_or_early_stopping") is False, "Evaluation does not declare held-out isolation")
    expected_paths = set(splits["test"]["paths"])
    actual_paths = [record["file_path"] for record in metrics["per_view"]]
    require(len(set(actual_paths)) == 12 and set(actual_paths) == expected_paths, "Metrics reference non-test or duplicate test inputs")
    require(not set(actual_paths) & set(splits["train"]["paths"]), "Evaluation paths overlap training input paths")
    prediction_files = sorted((output / "test_predictions").glob("*.png"))
    require(len(prediction_files) == 12, "Expected 12 decoded test prediction PNGs")
    expected_size = (metrics["image_width"], metrics["image_height"])
    for path in prediction_files:
        with Image.open(path) as image:
            image.load()
            require(image.size == expected_size, f"Incorrect test prediction size: {path}")
    return {"checkpoint_bytes": checkpoint_path.stat().st_size, "learned_tensors": len(state),
            "all_tensors_finite": True, "train_views": checkpoint["train_views"], "steps": checkpoint["steps"],
            "independent_test_views": len(actual_paths), "test_paths_match_test_split": True,
            "checkpoint_original_data_path": str(data_argument), "relocated_same_object": data.resolve() not in resolutions,
            "psnr": metrics["psnr"], "foreground_psnr": metrics["foreground_psnr"],
            "ssim": metrics["ssim"], "mask_iou": metrics["mask_iou"]}


def check_gif(output):
    path = output / "orbit.gif"
    digests, durations = set(), []
    count = 0
    with Image.open(path) as animation:
        require(animation.format == "GIF", "Orbit animation is not a GIF")
        require(animation.n_frames == 60, f"GIF has {animation.n_frames} frames; expected 60")
        for frame in ImageSequence.Iterator(animation):
            image = frame.convert("RGB")
            image.load()
            require(image.size == (256, 256), f"GIF frame {count} dimensions are {image.size}; expected 256 x 256")
            pixels = np.asarray(image)
            require((pixels < 245).any(), f"GIF frame {count} is empty white background")
            digests.add(hashlib.sha256(image.tobytes()).hexdigest())
            durations.append(frame.info.get("duration", 0))
            count += 1
    require(len(digests) > 1, "GIF repeats a single image")
    require(all(duration > 0 for duration in durations), "GIF has a missing/zero frame duration")
    pngs = sorted((output / "orbit").glob("*.png"))
    require(len(pngs) == 60, "Expected 60 learned orbit PNGs")
    for png in pngs:
        with Image.open(png) as image:
            image.load()
            require(image.size == (256, 256), f"Orbit PNG dimensions differ from GIF: {png}")
    return {"frames": count, "size": [256, 256], "different_frames": len(digests),
            "duration_ms": sum(durations), "decoded_orbit_pngs": len(pngs), "bytes": path.stat().st_size}


def check_geometry(output):
    metadata = read_json(output / "reconstruction.json")
    finite_json(metadata)
    require(metadata.get("source_geometry_used") is False, "Geometry export did not declare learned-density provenance")
    require(Path(metadata["checkpoint"]).name == "checkpoint.pt", "Geometry export references an unexpected checkpoint")
    ply = read_ply(output / "reconstruction.ply")
    require(ply["vertices"] == metadata["vertices"] and ply["faces"] == metadata["faces"] > 0, "Learned PLY counts disagree with reconstruction metadata")
    return dict(ply, source_geometry_used=False, threshold=metadata["density_threshold"],
                adaptive_threshold_used=metadata["adaptive_threshold_used"])


def check_optional_sfm(output):
    folder = output / "sfm"
    path = folder / "metrics.json"
    if not path.exists():
        return {"available": False, "required": False}
    metrics = read_json(path)
    finite_json(metrics)
    require(metrics.get("ground_truth_used_during_mapping") is False, "SfM supplied evaluation poses during mapping")
    require(metrics["status"] in {"complete", "partial", "failed", "running"}, "Unknown SfM status")
    if metrics["status"] in {"complete", "partial"}:
        require((folder / "database.db").is_file(), "Successful SfM has no COLMAP database")
        require(metrics.get("intrinsics_unchanged") is True, "Successful SfM changed fixed intrinsics")
        require(0 < metrics["registered_images"] <= metrics["input_images"], "Invalid SfM registered image count")
        close(metrics["registered_image_ratio"], metrics["registered_images"] / metrics["input_images"], "SfM registration ratio disagrees")
        require((folder / "sparse" / str(metrics["best_model_id"]) / "images.bin").is_file(), "Successful SfM lacks sparse image model")
        read_ply(folder / "sparse_points.ply")
    return {"available": True, "required": False, "status": metrics["status"],
            "registered_image_ratio": metrics.get("registered_image_ratio"),
            "sparse_points": metrics.get("sparse_points"), "reported_reason": metrics.get("reason")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Assignment 2 repository root")
    parser.add_argument("--source-root", type=Path, help="Assignment 1 root; defaults to the repository parent")
    parser.add_argument("--allow-pending", action="store_true", help="Permit missing artifacts while the pipeline is running; inconsistencies still fail")
    args = parser.parse_args()
    started = time.time()
    project = args.root.resolve()
    source_root = (args.source_root or project.parent).resolve()
    checks = []

    def record(scope, stage, function, mandatory=True):
        check = {"scope": scope, "check": stage, "mandatory": mandatory}
        try:
            check.update(status="passed", evidence=function())
        except FileNotFoundError as error:
            check.update(status="pending", error=str(error))
        except Exception as error:
            check.update(status="failed", error=f"{type(error).__name__}: {error}")
        checks.append(check)
        print(f"{scope}: {stage}: {check['status']}" + (f" ({check['error']})" if "error" in check else ""), flush=True)
        return check

    selection = read_json(project / "manifests/selection.json")
    record("submission", "selection", lambda: check_selection(selection))
    record("submission", "reconstruction_source_code", lambda: check_source_code(project))
    uids = {item["uid"] for item in selection["objects"]}
    source_context = {}

    def audit_context():
        context, details = resolve_audit_context(project, source_root, uids, selection.get("source_revision"))
        source_context.update(context)
        return details

    record("submission", "cleaning_audits_available", audit_context)
    for item in selection["objects"]:
        uid, scope = item["uid"], item["object_id"]
        data, output = project / "data" / uid, project / "results" / uid
        if source_context:
            record(scope, "source_hash_and_cleaning_intersection", lambda item=item: check_source(item, source_root, **source_context))
        dataset = {}

        def dataset_check():
            evidence, splits = check_dataset(data)
            dataset.update(splits)
            return evidence

        record(scope, "dataset_cameras_and_rgba", dataset_check)
        # Let missing/corrupt transforms be visible in dependent checks as well.
        record(scope, "matching_arrays_csv_summary", lambda: check_matching(output / "matching", read_json(data / "transforms_train.json")))

        def learning_check():
            splits = dataset or {split: {"paths": [frame["file_path"] for frame in read_json(data / f"transforms_{split}.json")["frames"]]}
                                 for split in ("train", "test")}
            return check_checkpoint_and_metrics(output, splits, data, project)

        record(scope, "checkpoint_and_independent_test_metrics", learning_check)
        record(scope, "gif_and_orbit_decode", lambda: check_gif(output))
        record(scope, "learned_reconstruction_ply", lambda: check_geometry(output))
        record(scope, "optional_colmap_metrics", lambda: check_optional_sfm(output), mandatory=False)
    failed = [check for check in checks if check["status"] == "failed"]
    pending = [check for check in checks if check["mandatory"] and check["status"] == "pending"]
    complete_objects = []
    for item in selection["objects"]:
        object_checks = [check for check in checks if check["scope"] == item["object_id"] and check["mandatory"]]
        if len(object_checks) == 6 and all(check["status"] == "passed" for check in object_checks):
            complete_objects.append(item["uid"])
    complete = not failed and not pending and len(complete_objects) == 10
    result = {"schema_version": "assignment2-validation-v1", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "complete": complete, "status": "passed" if complete else ("failed" if failed else "pending"),
              "allow_pending_requested": args.allow_pending, "source_root": str(source_root),
              "evidence_source": source_context.get("evidence_source", "unavailable"),
              "required_objects": 10, "complete_objects": len(complete_objects), "complete_object_uids": complete_objects,
              "failed_checks": len(failed), "pending_required_checks": len(pending),
              "identity_placeholders_allowed": True,
              "scope_note": "CPU artifact and data-independence validation; optional SfM outcomes are reported, and the later delivery ZIP is outside this check.",
              "checks": checks, "seconds": time.time() - started}
    destination = project / "manifests/validation.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "complete", "complete_objects", "failed_checks", "pending_required_checks", "seconds")}))
    print(f"Saved {destination}")
    return 0 if complete or (args.allow_pending and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())
