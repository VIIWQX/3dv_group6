#!/usr/bin/env python3
"""Estimate unknown camera poses from synthetic training RGB with CPU COLMAP.

The PINHOLE intrinsics are known and fixed. Ground-truth extrinsics from
transforms_train.json are used only AFTER mapping, to measure Sim(3)-aligned
camera errors; they are never supplied to the mapper. Source GLBs, depths and
geometry are not read. Requires pycolmap==3.13.0.

Example:
  python src/run_sfm.py --data data/<uid> --output results/<uid>/sfm --threads 4

The script preserves the COLMAP database, all sparse models and a PLY of the
largest model. A failed or partial reconstruction is reported in metrics.json.
An existing database is reused only when its recorded input fingerprint and
configuration agree. Use a new --output directory to change the protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image


CV_FROM_GL = np.diag([1.0, -1.0, -1.0])


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(data: Path, image_output: Path):
    """Composite RGBA over white, without supplying extrinsics to COLMAP."""
    transform_path = data / "transforms_train.json"
    metadata = json.loads(transform_path.read_text())
    frames = metadata["frames"]
    if len(frames) < 3:
        raise ValueError("SfM requires at least three training images")
    if metadata.get("camera_model", "PINHOLE") != "PINHOLE":
        raise ValueError("This branch supports fixed PINHOLE intrinsics only")
    records, ground_truth = [], {}
    image_names, sizes = [], set()
    for frame in frames:
        relative = Path(frame["file_path"])
        if not relative.suffix:
            relative = relative.with_suffix(".png")
        source = (data / relative).resolve()
        if not source.is_relative_to(data):
            raise ValueError(f"Frame path leaves dataset directory: {relative}")
        if relative.parts[0] != "train":
            raise ValueError(f"Only train/ images are allowed: {relative}")
        name = relative.as_posix()
        if name in ground_truth:
            raise ValueError(f"Duplicate image name: {name}")
        with Image.open(source) as image:
            image = image.convert("RGBA")
            sizes.add(image.size)
            white = Image.new("RGBA", image.size, (255, 255, 255, 255))
            white.alpha_composite(image)
            target = image_output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            white.convert("RGB").save(target)
        image_names.append(name)
        # Kept in a separate evaluation-only dictionary; never a COLMAP input.
        pose = np.asarray(frame["transform_matrix"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid evaluation pose for {name}")
        ground_truth[name] = pose
        records.append({"image": name, "source_sha256": fingerprint(source)})
    if len(sizes) != 1:
        raise ValueError("A single camera requires identical image dimensions")
    width, height = sizes.pop()
    if metadata.get("w", width) != width or metadata.get("h", height) != height:
        raise ValueError("Metadata dimensions disagree with training PNG files")
    if "fl_x" in metadata:
        fx = float(metadata["fl_x"])
    else:
        fx = 0.5 * width / math.tan(float(metadata["camera_angle_x"]) / 2)
    fy = float(metadata.get("fl_y", fx))
    cx, cy = float(metadata.get("cx", width / 2)), float(metadata.get("cy", height / 2))
    calibration = {"model": "PINHOLE", "width": width, "height": height,
                   "params": [fx, fy, cx, cy]}
    if not np.isfinite(calibration["params"]).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid fixed PINHOLE calibration")
    return metadata, ground_truth, image_names, calibration, records


def similarity_align(estimated, reference):
    """Least-squares Umeyama transform: reference = s * R * estimated + t."""
    x, y = np.asarray(estimated), np.asarray(reference)
    xmean, ymean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xmean, y - ymean
    variance = np.mean(np.sum(xc * xc, axis=1))
    if len(x) < 3 or variance < 1e-12 or np.linalg.matrix_rank(xc) < 2:
        raise ValueError("At least three non-collinear registered camera centers are needed")
    u, singular_values, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.dot(singular_values, sign) / variance)
    translation = ymean - scale * rotation @ xmean
    return scale, rotation, translation


def summarize_errors(errors):
    values = np.asarray(errors, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values * values))),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def evaluate_cameras(reconstruction, ground_truth, calibration, output):
    records = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        if image.name not in ground_truth:
            continue
        world_to_cam = np.asarray(image.cam_from_world().matrix())
        rotation_cv = world_to_cam[:, :3].T
        center = -rotation_cv @ world_to_cam[:, 3]
        pose_gl = np.eye(4)
        pose_gl[:3, :3] = rotation_cv @ CV_FROM_GL
        pose_gl[:3, 3] = center
        records.append({"image_id": int(image_id), "name": image.name,
                        "cam_from_world_cv": world_to_cam.tolist(),
                        "raw_c2w_opengl": pose_gl.tolist(),
                        "ground_truth_c2w_opengl": ground_truth[image.name].tolist()})
    records.sort(key=lambda record: record["name"])
    if len(records) < 3:
        write_json(output / "camera_poses.json", records)
        return {"status": "insufficient_registered_cameras", "compared_cameras": len(records)}
    estimated = np.asarray([np.asarray(r["raw_c2w_opengl"])[:3, 3] for r in records])
    reference = np.asarray([ground_truth[r["name"]][:3, 3] for r in records])
    try:
        scale, rotation, translation = similarity_align(estimated, reference)
    except ValueError as error:
        write_json(output / "camera_poses.json", records)
        return {"status": "alignment_failed", "reason": str(error), "compared_cameras": len(records)}
    frames, position_errors, angle_errors = [], [], []
    for record in records:
        pose = np.asarray(record["raw_c2w_opengl"])
        pose[:3, :3] = rotation @ pose[:3, :3]
        pose[:3, 3] = scale * rotation @ pose[:3, 3] + translation
        gt = ground_truth[record["name"]]
        pos_error = float(np.linalg.norm(pose[:3, 3] - gt[:3, 3]))
        relative_rotation = gt[:3, :3].T @ pose[:3, :3]
        angle_error = float(np.degrees(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))))
        position_errors.append(pos_error)
        angle_errors.append(angle_error)
        record.update({"aligned_c2w_opengl": pose.tolist(), "center_error": pos_error,
                       "orientation_error_degrees": angle_error})
        frames.append({"file_path": "images/" + record["name"], "transform_matrix": pose.tolist()})
    write_json(output / "camera_poses.json", records)
    fx, fy, cx, cy = calibration["params"]
    write_json(output / "transforms_train_sfm.json", {
        "w": calibration["width"], "h": calibration["height"], "fl_x": fx,
        "fl_y": fy, "cx": cx, "cy": cy, "camera_model": "PINHOLE",
        "camera_convention": "OpenGL camera-to-world, globally Sim(3)-aligned for evaluation",
        "pose_source": "COLMAP image-only pose estimation; ground truth used only for a global gauge alignment",
        "frames": frames})
    gt_radius = float(np.sqrt(np.mean(np.sum((reference - reference.mean(axis=0)) ** 2, axis=1))))
    return {"status": "aligned", "method": "least-squares Umeyama Sim(3), no per-camera fitting",
            "compared_cameras": len(records), "scale": scale, "rotation": rotation.tolist(),
            "translation": translation.tolist(), "camera_center_error": summarize_errors(position_errors),
            "orientation_error_degrees": summarize_errors(angle_errors),
            "center_rmse_over_gt_trajectory_radius": float(np.sqrt(np.mean(np.square(position_errors))) / max(gt_radius, 1e-12)),
            "ground_truth_used_during_mapping": False}


def database_statistics(path, pycolmap):
    database = pycolmap.Database.open(str(path))
    try:
        cameras = database.read_all_cameras()
        _, geometry_inliers = database.read_two_view_geometry_num_inliers()
        return {"images": int(database.num_images()), "cameras": int(database.num_cameras()),
                "keypoints": int(database.num_keypoints()), "raw_matches": int(database.num_matches()),
                "matched_image_pairs": int(database.num_matched_image_pairs()),
                "verification_records": int(database.num_verified_image_pairs()),
                "verified_image_pairs_with_inliers": sum(int(count > 0) for count in geometry_inliers),
                "inlier_matches": int(database.num_inlier_matches()),
                "guided_matching_note": "Guided matching may add correspondences, so final inliers can exceed raw descriptor matches.",
                "camera_parameters": [camera.params.tolist() for camera in cameras]}
    finally:
        database.close()


def run(args):
    import pycolmap

    started = time.time()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == data or data.is_relative_to(output):
        raise ValueError("Output must be separate from input dataset")
    metrics = {"status": "running", "data": str(data), "output": str(output),
               "pycolmap_version": pycolmap.__version__, "device": "CPU",
               "ground_truth_used_during_mapping": False, "input_images": 0,
               "registered_images": 0, "registered_image_ratio": 0.0,
               "sparse_points": 0, "mean_reprojection_error_pixels": None,
               "stage": "prepare_inputs"}
    metrics_path = output / "metrics.json"
    try:
        if pycolmap.__version__ != "3.13.0":
            raise ValueError(f"This recorded protocol requires pycolmap 3.13.0, found {pycolmap.__version__}")
        image_output = output / "images"
        metadata, ground_truth, image_names, calibration, inputs = read_inputs(data, image_output)
        metrics["input_images"] = len(image_names)
        metrics["fixed_calibration"] = calibration
        protocol = {"pycolmap": pycolmap.__version__, "threads": args.threads,
                    "device": "CPU", "camera_mode": "SINGLE", "calibration": calibration,
                    "intrinsics_refined": False, "background": "white alpha composite",
                    "transform_sha256": fingerprint(data / "transforms_train.json"), "inputs": inputs}
        manifest_path = output / "input_manifest.json"
        database_path = output / "database.db"
        if database_path.exists():
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != protocol:
                raise ValueError("Existing database has incompatible or unrecorded inputs; choose a new output directory")
        else:
            write_json(manifest_path, protocol)
        write_json(metrics_path, metrics)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "PINHOLE"
        reader.camera_params = ",".join(str(value) for value in calibration["params"])
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = args.threads
        extraction.use_gpu = False
        extraction.max_image_size = max(calibration["width"], calibration["height"])
        extraction.sift.max_num_features = 8192
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = args.threads
        matching.use_gpu = False
        matching.guided_matching = True
        mapping = pycolmap.IncrementalPipelineOptions()
        mapping.num_threads = args.threads
        mapping.random_seed = 42
        mapping.min_model_size = 3
        mapping.multiple_models = True
        mapping.max_num_models = 10
        mapping.ba_refine_focal_length = False
        mapping.ba_refine_principal_point = False
        mapping.ba_refine_extra_params = False
        mapping.ba_use_gpu = False
        mapping.mapper.num_threads = args.threads
        mapping.mapper.random_seed = 42
        mapping.mapper.abs_pose_refine_focal_length = False
        mapping.mapper.abs_pose_refine_extra_params = False
        mapping.mapper.init_min_num_inliers = 50
        mapping.mapper.init_min_tri_angle = 8.0
        mapping.mapper.abs_pose_min_num_inliers = 20
        mapping.triangulation.random_seed = 42
        write_json(output / "colmap_options.json", {"reader": reader.todict(),
                   "extraction_summary": extraction.summary(), "matching": matching.todict(),
                   "mapping_summary": mapping.summary()})
        metrics["stage"] = "extract_features"
        write_json(metrics_path, metrics)
        pycolmap.extract_features(str(database_path), str(image_output), image_names=image_names,
                                  camera_mode=pycolmap.CameraMode.SINGLE, camera_model="PINHOLE",
                                  reader_options=reader, extraction_options=extraction,
                                  device=pycolmap.Device.cpu)
        metrics["stage"] = "exhaustive_matching"
        write_json(metrics_path, metrics)
        pycolmap.match_exhaustive(str(database_path), matching_options=matching,
                                 device=pycolmap.Device.cpu)
        metrics["database"] = database_statistics(database_path, pycolmap)
        metrics["stage"] = "incremental_mapping"
        write_json(metrics_path, metrics)
        sparse_path = output / "sparse"
        sparse_path.mkdir(exist_ok=True)
        models = pycolmap.incremental_mapping(str(database_path), str(image_output), str(sparse_path), options=mapping)
        model_summaries, all_registered = [], set()
        for model_id, reconstruction in models.items():
            model_path = sparse_path / str(model_id)
            model_path.mkdir(exist_ok=True)
            reconstruction.write(str(model_path))
            reconstruction.write_text(str(model_path))
            names = [reconstruction.image(image_id).name for image_id in reconstruction.reg_image_ids()]
            all_registered.update(names)
            model_summaries.append({"model_id": int(model_id), "registered_images": len(names),
                                    "registered_image_ratio": len(names) / len(image_names),
                                    "sparse_points": int(reconstruction.num_points3D()),
                                    "mean_reprojection_error_pixels": float(reconstruction.compute_mean_reprojection_error()),
                                    "registered_image_names": sorted(names)})
        metrics["models"] = model_summaries
        metrics["all_models_registered_images"] = len(all_registered)
        metrics["all_models_registered_image_ratio"] = len(all_registered) / len(image_names)
        if not models:
            metrics.update({"status": "failed", "reason": "COLMAP returned no reconstructable sparse model",
                            "camera_alignment": {"status": "no_model"}})
        else:
            best_id, best = max(models.items(), key=lambda pair: (pair[1].num_reg_images(), pair[1].num_points3D()))
            best_summary = next(model for model in model_summaries if model["model_id"] == best_id)
            metrics.update({key: best_summary[key] for key in ["registered_images", "registered_image_ratio",
                            "sparse_points", "mean_reprojection_error_pixels"]})
            metrics["best_model_id"] = int(best_id)
            metrics["status"] = "complete" if metrics["registered_images"] == len(image_names) else "partial"
            best.export_PLY(str(output / "sparse_points.ply"))
            metrics["camera_alignment"] = evaluate_cameras(best, ground_truth, calibration, output)
            # Verify that single-camera intrinsics really remained fixed.
            estimated_params = [camera.params.tolist() for camera in best.cameras.values()]
            metrics["estimated_camera_parameters"] = estimated_params
            metrics["intrinsics_unchanged"] = all(np.allclose(params, calibration["params"], atol=1e-8, rtol=0)
                                                    for params in estimated_params)
            if not metrics["intrinsics_unchanged"]:
                raise RuntimeError("COLMAP unexpectedly changed fixed camera intrinsics")
        metrics["stage"] = "finished"
    except Exception as error:
        metrics.update({"status": "failed", "reason": str(error), "exception": type(error).__name__})
        (output / "error_traceback.txt").write_text(traceback.format_exc())
    metrics["seconds"] = time.time() - started
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, allow_nan=False), flush=True)
    return 1 if metrics["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Object directory with transforms_train.json and train/*.png")
    parser.add_argument("--output", type=Path, required=True, help="Separate SfM result directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
