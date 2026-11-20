#!/usr/bin/env python3
"""Extract an approximate surface from a *learned* NeRF density checkpoint.

This diagnostic does not consume images or source geometry.  Its isosurface is
threshold dependent and is not a claim of ground-truth geometric accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from nerf import load_model


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray,
              normals: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY, readable by MeshLab, Blender and Open3D."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment Surface extracted from learned NeRF density; no source mesh used\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    vertex_data = np.empty(len(vertices), dtype=[("xyz", "<f4", 3), ("normal", "<f4", 3), ("rgb", "u1", 3)])
    vertex_data["xyz"], vertex_data["normal"], vertex_data["rgb"] = vertices, normals, colors
    face_data = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    face_data["count"], face_data["indices"] = 3, faces
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex_data.tofile(handle)
        face_data.tofile(handle)


@torch.no_grad()
def export_geometry(checkpoint: str | Path, output: str | Path,
                    alpha_threshold: float = 0.2, density_threshold: float | None = None,
                    min_component_voxels: int = 8) -> dict:
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise SystemExit("Geometry export requires scikit-image: pip install scikit-image") from error
    model, _ = load_model(checkpoint, "cpu")
    sigma = F.softplus(model.density).squeeze().numpy()
    voxel_size = 2 * model.bound / (model.resolution - 1)
    requested = float(density_threshold) if density_threshold is not None else -math.log(1 - alpha_threshold) / voxel_size
    threshold = requested
    fallback = False
    # Sparse/short-run fields may never reach the requested opacity.  A documented
    # fraction of their learned density range still yields a useful diagnostic.
    lower, upper = float(sigma.min()), float(sigma.max())
    if not lower < threshold < upper:
        threshold = lower + 0.35 * (upper - lower)
        fallback = True
    if upper - lower < 1e-7:
        raise ValueError("The learned field has constant density; no isosurface can be extracted")
    removed_voxels = 0
    if min_component_voxels > 0:
        try:
            from scipy.ndimage import label
            labels, component_count = label(sigma >= threshold)
            sizes = np.bincount(labels.reshape(-1))
            keep = sizes >= min_component_voxels
            keep[0] = False
            small_components = (labels != 0) & ~keep[labels]
            removed_voxels = int(small_components.sum())
            sigma = sigma.copy()
            sigma[small_components] = lower
        except ImportError:
            component_count = None
    else:
        component_count = None
    # Pad with empty space so a surface also closes at the reconstruction bounds.
    padded = np.pad(sigma, 1, mode="constant", constant_values=min(lower, threshold - 1e-6))
    vertices_zyx, faces, normals_zyx, _ = marching_cubes(padded, level=threshold, spacing=(voxel_size,) * 3)
    vertices = (vertices_zyx[:, ::-1].copy() - model.bound - voxel_size).astype(np.float32)
    normals = normals_zyx[:, ::-1].copy()
    faces = faces[:, [0, 2, 1]].copy()
    # Appearance features contain spatial RGB logits plus five learned channels.
    # These vertex colors use the spatial term only and omit view-dependent RGB.
    grid = torch.from_numpy(vertices / model.bound).reshape(1, 1, 1, -1, 3)
    features = F.grid_sample(model.features, grid, padding_mode="border", align_corners=True).reshape(model.feature_dim, -1).T
    colors = np.uint8(torch.sigmoid(features[:, :3]).numpy().clip(0, 1) * 255 + .5)
    output = Path(output)
    write_ply(output, vertices, faces, normals, colors)
    metadata = {"checkpoint": str(checkpoint), "output": str(output), "vertices": len(vertices), "faces": len(faces),
                "resolution": model.resolution, "bound": model.bound, "voxel_size": voxel_size,
                "requested_density_threshold": requested, "density_threshold": threshold,
                "equivalent_alpha_per_voxel": 1 - math.exp(-threshold * voxel_size),
                "adaptive_threshold_used": fallback, "density_min": lower, "density_max": upper,
                "removed_small_component_voxels": removed_voxels, "components_before_filter": component_count,
                "colors": "sigmoid of learned spatial RGB feature logits; view-dependent decoder omitted",
                "source_geometry_used": False,
                "interpretation": "Threshold-dependent isosurface diagnostic of the learned volume, not a ground-truth geometry evaluation"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output reconstruction.ply")
    parser.add_argument("--alpha-threshold", type=float, default=.2, help="Target opacity across one voxel")
    parser.add_argument("--density-threshold", type=float, help="Override opacity-based density threshold")
    parser.add_argument("--min-component-voxels", type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.alpha_threshold < 1:
        parser.error("alpha-threshold must lie strictly between zero and one")
    result = export_geometry(args.checkpoint, args.output, args.alpha_threshold, args.density_threshold, args.min_component_voxels)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
