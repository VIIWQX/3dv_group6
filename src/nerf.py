#!/usr/bin/env python3
"""Train a compact, explicit voxel-feature NeRF from calibrated RGBA photographs.

Only PNG images and Blender-style camera transforms are inputs.  The learned
representation is a density grid, an eight-channel feature grid, and a small
view-conditioned RGB network.  Rendering uses differentiable volume integration.
No mesh, depth map, normal map, or ground-truth geometry is read by this module.

Example::
    python assignment2/src/nerf.py --data assignment2/data/OBJECT \
        --output assignment2/results/OBJECT --device cuda

For downstream rendering use ``load_model`` and ``render_camera`` below.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def camera_rays(c2w: torch.Tensor, width: int, height: int,
                fx: float, fy: float, cx: float, cy: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-center rays for Blender camera coordinates: -Z forward, +Y up."""
    yy, xx = torch.meshgrid(torch.arange(height, device=c2w.device),
                            torch.arange(width, device=c2w.device), indexing="ij")
    local = torch.stack(((xx + 0.5 - cx) / fx, -(yy + 0.5 - cy) / fy,
                         -torch.ones_like(xx)), dim=-1).float()
    directions = F.normalize(local @ c2w[:3, :3].T, dim=-1)
    origins = c2w[:3, 3].expand_as(directions)
    return origins.reshape(-1, 3), directions.reshape(-1, 3)


def _intrinsics(meta: dict, frame: dict, width: int, height: int) -> tuple[float, float, float, float]:
    orig_w, orig_h = int(meta.get("w", width)), int(meta.get("h", height))
    fx = frame.get("fl_x", meta.get("fl_x"))
    if fx is None:
        fx = 0.5 * orig_w / math.tan(0.5 * meta["camera_angle_x"])
    fy = frame.get("fl_y", meta.get("fl_y", fx))
    cx, cy = frame.get("cx", meta.get("cx", orig_w / 2)), frame.get("cy", meta.get("cy", orig_h / 2))
    return float(fx) * width / orig_w, float(fy) * height / orig_h, float(cx) * width / orig_w, float(cy) * height / orig_h


def _rotation_noise(degrees: float, rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    a = math.radians(degrees) * rng.normal()
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + math.sin(a) * cross + (1 - math.cos(a)) * (cross @ cross)


def load_dataset(data_dir: str | Path, split: str, image_size: int = 192,
                 device: str | torch.device = "cpu", max_views: int | None = None,
                 pose_noise_deg: float = 0.0, seed: int = 0,
                 load_images: bool = True) -> dict[str, Any]:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / f"transforms_{split}.json").read_text())
    frames = meta["frames"]
    if max_views and max_views < len(frames):
        indices = np.linspace(0, len(frames) - 1, max_views).round().astype(int)
        frames = [frames[i] for i in indices]
    orig_w, orig_h = int(meta.get("w", image_size)), int(meta.get("h", image_size))
    scale = image_size / max(orig_w, orig_h)
    width, height = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    result: dict[str, Any] = {"width": width, "height": height, "frames": [], "meta": meta}
    rng = np.random.default_rng(seed)
    for frame in frames:
        matrix = np.array(frame["transform_matrix"], dtype=np.float32)
        if pose_noise_deg:
            matrix[:3, :3] = matrix[:3, :3] @ _rotation_noise(pose_noise_deg, rng)
        c2w = torch.tensor(matrix, dtype=torch.float32, device=device)
        fx, fy, cx, cy = _intrinsics(meta, frame, width, height)
        record: dict[str, Any] = {"c2w": c2w, "intrinsics": (fx, fy, cx, cy), "file_path": frame.get("file_path", "")}
        if load_images:
            path = data_dir / frame["file_path"]
            if not path.exists() and not path.suffix:
                path = path.with_suffix(".png")
            rgba = np.array(Image.open(path).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
            alpha = torch.tensor(rgba[..., 3:4], device=device)
            rgb = torch.tensor(rgba[..., :3], device=device) * alpha + 1 - alpha
            ro, rd = camera_rays(c2w, width, height, fx, fy, cx, cy)
            record.update(rgb=rgb, alpha=alpha, rays_o=ro, rays_d=rd)
        result["frames"].append(record)
    return result


class VoxelNeRF(nn.Module):
    def __init__(self, resolution: int = 32, feature_dim: int = 8, bound: float = 1.0):
        super().__init__()
        self.resolution, self.feature_dim, self.bound = resolution, feature_dim, bound
        self.density = nn.Parameter(torch.full((1, 1, resolution, resolution, resolution), -6.0))
        self.features = nn.Parameter(torch.zeros(1, feature_dim, resolution, resolution, resolution))
        # Position is represented by the spatial feature grid; only view direction
        # needs an explicit encoding in the tiny appearance decoder.
        self.decoder = nn.Sequential(nn.Linear(feature_dim + 15, 32), nn.ReLU(inplace=True),
                                     nn.Linear(32, 32), nn.ReLU(inplace=True), nn.Linear(32, 3))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def upsample(self, resolution: int) -> None:
        if resolution == self.resolution:
            return
        self.density = nn.Parameter(F.interpolate(self.density.detach(), size=(resolution,) * 3, mode="trilinear", align_corners=True))
        self.features = nn.Parameter(F.interpolate(self.features.detach(), size=(resolution,) * 3, mode="trilinear", align_corners=True))
        self.resolution = resolution

    @staticmethod
    def encode_direction(direction: torch.Tensor) -> torch.Tensor:
        return torch.cat((direction, torch.sin(math.pi * direction), torch.cos(math.pi * direction),
                          torch.sin(2 * math.pi * direction), torch.cos(2 * math.pi * direction)), dim=-1)

    def render_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor, samples: int = 96,
                    perturb: bool = False) -> dict[str, torch.Tensor]:
        inverse = 1.0 / torch.where(rays_d.abs() < 1e-8, torch.full_like(rays_d, 1e-8), rays_d)
        t0, t1 = (-self.bound - rays_o) * inverse, (self.bound - rays_o) * inverse
        near = torch.minimum(t0, t1).amax(-1).clamp_min(0)
        far = torch.maximum(t0, t1).amin(-1)
        valid = far > near
        far = torch.where(valid, far, near)
        offsets = torch.arange(samples, device=rays_o.device, dtype=rays_o.dtype)[None]
        offsets = offsets + (torch.rand(len(rays_o), samples, device=rays_o.device) if perturb else 0.5)
        step = (far - near) / samples
        t = near[:, None] + offsets * step[:, None]
        positions = rays_o[:, None] + t[..., None] * rays_d[:, None]
        grid = (positions / self.bound).reshape(1, 1, 1, -1, 3)
        sigma_raw = F.grid_sample(self.density, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(len(rays_o), samples)
        optical = F.softplus(sigma_raw) * step[:, None]
        alpha = -torch.expm1(-optical)
        transmittance = torch.exp(-torch.cat((torch.zeros_like(optical[:, :1]), optical[:, :-1].cumsum(-1)), dim=-1))
        weights = alpha * transmittance
        feature = F.grid_sample(self.features, grid, mode="bilinear", padding_mode="border", align_corners=True)
        feature = feature.reshape(self.feature_dim, -1).T.reshape(len(rays_o), samples, self.feature_dim)
        view = self.encode_direction(rays_d)[:, None].expand(-1, samples, -1)
        color = torch.sigmoid(feature[..., :3] + self.decoder(torch.cat((feature, view), dim=-1)))
        opacity = weights.sum(-1)
        rgb = (weights[..., None] * color).sum(-2) + (1 - opacity[..., None])
        depth = (weights * t).sum(-1)
        return {"rgb": rgb, "alpha": opacity, "depth": depth}

    def tv_loss(self) -> torch.Tensor:
        loss = self.density.new_tensor(0.)
        for grid, coefficient in ((self.density, 0.1), (self.features, 1.0)):
            for axis in (2, 3, 4):
                left = grid.narrow(axis, 1, grid.shape[axis] - 1)
                right = grid.narrow(axis, 0, grid.shape[axis] - 1)
                loss = loss + coefficient * (left - right).abs().mean()
        return loss


@torch.no_grad()
def initialize_from_silhouettes(model: VoxelNeRF, dataset: dict, use_alpha: bool = True) -> dict:
    """RGB/alpha-only visual-hull initialization; it is a soft, trainable prior.

    Alpha is dilated by one pixel before testing consistency, avoiding clipping
    thin silhouettes due to sampling.  Density remains trainable outside the hull.
    """
    device = model.density.device
    r = model.resolution
    zz, yy, xx = torch.meshgrid(*(torch.linspace(-model.bound, model.bound, r, device=device) for _ in range(3)), indexing="ij")
    xyz = torch.stack((xx, yy, zz), -1).reshape(-1, 3)
    hits = torch.zeros(len(xyz), device=device)
    colors = torch.zeros(len(xyz), 3, device=device)
    valid_colors = torch.zeros(len(xyz), 1, device=device)
    frames = dataset["frames"]
    # At most 24 training views are sufficient for this initialization.
    frame_ids = np.linspace(0, len(frames) - 1, min(24, len(frames))).round().astype(int)
    for index in frame_ids:
        frame = frames[index]
        c2w = frame["c2w"]
        camera = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
        z = (-camera[:, 2]).clamp_min(1e-5)
        fx, fy, cx, cy = frame["intrinsics"]
        px, py = fx * camera[:, 0] / z + cx, -fy * camera[:, 1] / z + cy
        # align_corners=False maps normalized coordinates to pixel centers.
        uv = torch.stack((2 * px / dataset["width"] - 1, 2 * py / dataset["height"] - 1), -1).reshape(1, 1, -1, 2)
        image = frame["rgb"].permute(2, 0, 1)[None]
        mask = F.max_pool2d(frame["alpha"].permute(2, 0, 1)[None], 3, 1, 1)
        a = F.grid_sample(mask, uv, align_corners=False).reshape(-1)
        color = F.grid_sample(image, uv, align_corners=False).reshape(3, -1).T
        hit = (a > 0.05) & (camera[:, 2] < 0)
        hits += hit.float()
        colors += color * hit[:, None]
        valid_colors += hit[:, None]
    inside = hits >= len(frame_ids) * 0.95 if use_alpha else xyz.norm(dim=-1) < 0.75
    model.density.data.reshape(-1).copy_(torch.where(inside, 1.0, -7.0))
    rgb = (colors / valid_colors.clamp_min(1)).clamp(0.02, 0.98)
    logits = torch.logit(rgb).T.reshape(1, 3, r, r, r)
    model.features.data[:, :3].copy_(logits)
    model.features.data[:, 3:].normal_(std=0.01)
    return {"initialization": "soft_visual_hull_from_training_alpha" if use_alpha else "generic_sphere_rgb_initialization",
            "initial_voxels": int(inside.sum().item()), "initialization_views": len(frame_ids)}


@torch.no_grad()
def render_camera(model: VoxelNeRF, c2w: torch.Tensor | np.ndarray, width: int, height: int,
                  fx: float, fy: float, cx: float | None = None, cy: float | None = None,
                  samples: int = 96, chunk: int = 4096) -> dict[str, np.ndarray]:
    """Render one camera; returns float32 HWC RGB, HW alpha, and HW depth arrays."""
    model.eval()
    device = next(model.parameters()).device
    c2w = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    ro, rd = camera_rays(c2w, width, height, fx, fy, width / 2 if cx is None else cx, height / 2 if cy is None else cy)
    outputs: dict[str, list] = {"rgb": [], "alpha": [], "depth": []}
    for start in range(0, len(ro), chunk):
        rendered = model.render_rays(ro[start:start + chunk], rd[start:start + chunk], samples=samples)
        for key in outputs:
            outputs[key].append(rendered[key].detach().cpu())
    return {key: torch.cat(value).reshape((height, width, 3) if key == "rgb" else (height, width)).numpy()
            for key, value in outputs.items()}


def load_model(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[VoxelNeRF, dict]:
    """Load a checkpoint without the original training images or any geometry."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VoxelNeRF(checkpoint["resolution"], checkpoint.get("feature_dim", 8), checkpoint.get("bound", 1.0))
    model.load_state_dict({key: value.float() for key, value in checkpoint["state_dict"].items()})
    return model.to(device).eval(), checkpoint


def _psnr(mse: float) -> float:
    return float(-10 * math.log10(max(mse, 1e-12)))


def _ssim(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(reference, prediction, channel_axis=-1, data_range=1.0))
    except ImportError:
        # Standard local SSIM formula, averaged over channels and valid pixels.
        x = torch.from_numpy(prediction).permute(2, 0, 1)[None]
        y = torch.from_numpy(reference).permute(2, 0, 1)[None]
        mu_x, mu_y = F.avg_pool2d(x, 7, 1), F.avg_pool2d(y, 7, 1)
        var_x, var_y = F.avg_pool2d(x * x, 7, 1) - mu_x.square(), F.avg_pool2d(y * y, 7, 1) - mu_y.square()
        cov = F.avg_pool2d(x * y, 7, 1) - mu_x * mu_y
        return float((((2 * mu_x * mu_y + .01 ** 2) * (2 * cov + .03 ** 2)) /
                      ((mu_x.square() + mu_y.square() + .01 ** 2) * (var_x + var_y + .03 ** 2))).mean())


@torch.no_grad()
def evaluate(model: VoxelNeRF, data: dict, output: Path, samples: int, chunk: int = 4096) -> dict:
    rows, panels = [], []
    prediction_dir = output / "test_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    total_error, total_values, foreground_error, foreground_values = 0., 0, 0., 0
    for index, frame in enumerate(data["frames"]):
        rendered = render_camera(model, frame["c2w"], data["width"], data["height"], *frame["intrinsics"], samples=samples, chunk=chunk)
        prediction = rendered["rgb"].clip(0, 1)
        reference = frame["rgb"].cpu().numpy()
        mask = frame["alpha"].cpu().numpy()[..., 0] > 0.5
        errors = (prediction - reference) ** 2
        mse = float(errors.mean())
        fg_mse = float(errors[mask].mean()) if mask.any() else mse
        total_error += float(errors.sum())
        total_values += errors.size
        foreground_error += float(errors[mask].sum())
        foreground_values += int(mask.sum()) * 3
        iou = float(((rendered["alpha"] > 0.5) & mask).sum() / max(1, ((rendered["alpha"] > 0.5) | mask).sum()))
        rows.append({"view": index, "file_path": frame["file_path"], "psnr": _psnr(mse),
                     "foreground_psnr": _psnr(fg_mse), "ssim": _ssim(prediction, reference), "mask_iou": iou})
        Image.fromarray(np.uint8(prediction * 255 + .5)).save(prediction_dir / f"{index:03d}.png")
        if len(panels) < 4:
            error_image = np.clip(np.abs(prediction - reference) * 3, 0, 1)
            panels.append(np.concatenate((reference, prediction, error_image), axis=1))
    metrics = {"test_views": len(rows), "image_width": data["width"], "image_height": data["height"],
               "psnr": _psnr(total_error / max(1, total_values)),
               "foreground_psnr": _psnr(foreground_error / max(1, foreground_values)),
               "mean_view_psnr": float(np.mean([x["psnr"] for x in rows])),
               "ssim": float(np.mean([x["ssim"] for x in rows])),
               "mask_iou": float(np.mean([x["mask_iou"] for x in rows])),
               "background": "white", "foreground_threshold": 0.5,
               "test_used_for_optimization_or_early_stopping": False, "per_view": rows}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if panels:
        canvas = Image.fromarray(np.uint8(np.concatenate(panels, axis=0) * 255 + .5))
        labeled = Image.new("RGB", (canvas.width, canvas.height + 24), "white")
        labeled.paste(canvas, (0, 24))
        draw = ImageDraw.Draw(labeled)
        for x, title in zip((8, data["width"] + 8, 2 * data["width"] + 8), ("Held-out RGB", "NeRF prediction", "Absolute error x3")):
            draw.text((x, 6), title, fill="black")
        labeled.save(output / "comparison.png")
    return metrics


@torch.no_grad()
def render_orbit(model: VoxelNeRF, dataset: dict, output: Path, samples: int = 96,
                 max_frames: int = 60, chunk: int = 4096) -> None:
    frames = dataset["frames"]
    indices = np.linspace(0, len(frames), min(max_frames, len(frames)), endpoint=False).astype(int)
    images = []
    orbit_dir = output / "orbit"
    orbit_dir.mkdir(parents=True, exist_ok=True)
    for count, index in enumerate(indices):
        frame = frames[index]
        rendered = render_camera(model, frame["c2w"], dataset["width"], dataset["height"], *frame["intrinsics"], samples=samples, chunk=chunk)
        image = Image.fromarray(np.uint8(rendered["rgb"].clip(0, 1) * 255 + .5))
        image.save(orbit_dir / f"{count:03d}.png")
        images.append(image)
    if images:
        images[0].save(output / "orbit.gif", save_all=True, append_images=images[1:],
                       duration=80, loop=0, optimize=False, disposal=2)


def _make_optimizer(model: VoxelNeRF, scale: float = 1.0) -> torch.optim.Optimizer:
    return torch.optim.Adam([{"params": [model.density, model.features], "lr": .08 * scale, "initial_lr": .08},
                             {"params": model.decoder.parameters(), "lr": .002 * scale, "initial_lr": .002}],
                            betas=(.9, .99), eps=1e-8)


def train(args: argparse.Namespace) -> tuple[VoxelNeRF, dict]:
    device = torch.device(args.device)
    dataset = load_dataset(args.data, "train", args.image_size, device, args.train_views,
                           args.pose_noise_deg, args.seed)
    if not dataset["frames"]:
        raise ValueError("Training split contains no images")
    rays_o = torch.cat([frame["rays_o"] for frame in dataset["frames"]])
    rays_d = torch.cat([frame["rays_d"] for frame in dataset["frames"]])
    rgb = torch.cat([frame["rgb"].reshape(-1, 3) for frame in dataset["frames"]])
    alpha = torch.cat([frame["alpha"].reshape(-1) for frame in dataset["frames"]])
    foreground_indices = torch.where(alpha > .05)[0]
    schedule_resolutions = sorted(set([min(32, args.resolution), min(64, args.resolution), args.resolution]))
    milestones = [0, round(args.steps * .35), round(args.steps * .65)][:len(schedule_resolutions)]
    # For target resolutions <=64, the refinement happens at 35% of training.
    schedule = dict(zip(milestones, schedule_resolutions))
    model = VoxelNeRF(schedule_resolutions[0], bound=args.bound).to(device)
    initialization = initialize_from_silhouettes(model, dataset, use_alpha=not (args.no_alpha_loss or args.no_silhouette_init))
    optimizer = _make_optimizer(model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with (output / "training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "resolution", "loss", "rgb_mse", "batch_psnr", "alpha_mse", "tv", "seconds"])
        writer.writeheader()
        for step in range(args.steps):
            if step in schedule and model.resolution != schedule[step]:
                model.upsample(schedule[step])
                optimizer = _make_optimizer(model, scale=.1 ** (step / max(1, args.steps)))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * .1 ** (step / max(1, args.steps))
            n_foreground = int(args.rays * .65) if len(foreground_indices) else 0
            ids_fg = foreground_indices[torch.randint(len(foreground_indices), (n_foreground,), device=device)] if n_foreground else torch.empty(0, dtype=torch.long, device=device)
            ids = torch.cat((ids_fg, torch.randint(len(rgb), (args.rays - n_foreground,), device=device)))
            prediction = model.render_rays(rays_o[ids], rays_d[ids], args.samples, perturb=True)
            rgb_loss = F.mse_loss(prediction["rgb"], rgb[ids])
            alpha_loss = F.mse_loss(prediction["alpha"], alpha[ids])
            tv = model.tv_loss() if step % 8 == 0 and args.tv_weight > 0 else rgb_loss.new_tensor(0.)
            loss = rgb_loss + (0. if args.no_alpha_loss else args.alpha_weight) * alpha_loss + args.tv_weight * 8 * tv
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == args.steps - 1:
                values = {"step": step + 1, "resolution": model.resolution, "loss": float(loss.detach()),
                          "rgb_mse": float(rgb_loss.detach()), "batch_psnr": _psnr(float(rgb_loss.detach())),
                          "alpha_mse": float(alpha_loss.detach()), "tv": float(tv.detach()),
                          "seconds": time.perf_counter() - start_time}
                writer.writerow(values)
                handle.flush()
                print(f"step={step + 1:5d}/{args.steps} res={model.resolution} loss={values['loss']:.5f} "
                      f"psnr={values['batch_psnr']:.2f} alpha={values['alpha_mse']:.5f} seconds={values['seconds']:.1f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metadata = {"resolution": model.resolution, "feature_dim": model.feature_dim, "bound": model.bound,
                "training_seconds": time.perf_counter() - start_time, "train_views": len(dataset["frames"]),
                "training_width": dataset["width"], "training_height": dataset["height"],
                "steps": args.steps, "initialization": initialization, "config": vars(args),
                "method": "voxel-feature NeRF with differentiable alpha compositing",
                "training_inputs": ["RGB PNG", "alpha PNG", "calibrated camera poses and intrinsics"],
                "resolution_schedule": [{"step": k, "resolution": v} for k, v in schedule.items()]}
    # Half precision is a storage format only; training and rendering use float32.
    checkpoint = dict(metadata, state_dict={key: value.detach().cpu().half() for key, value in model.state_dict().items()})
    torch.save(checkpoint, output / "checkpoint.pt")
    (output / "config.json").write_text(json.dumps(metadata, indent=2))
    return model.eval(), metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Directory with transforms_train/test/orbit.json and RGBA PNGs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--rays", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=192, help="Longest side for training and held-out evaluation")
    parser.add_argument("--render-size", type=int, default=192, help="Longest side for orbit renders")
    parser.add_argument("--orbit-frames", type=int, default=60)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--train-views", type=int)
    parser.add_argument("--pose-noise-deg", type=float, default=0.)
    parser.add_argument("--no-alpha-loss", action="store_true", help="Disable alpha loss and silhouette-hull initialization")
    parser.add_argument("--no-silhouette-init", action="store_true", help="Use generic sphere density initialization; retain alpha loss and foreground sampling")
    parser.add_argument("--alpha-weight", type=float, default=.1)
    parser.add_argument("--tv-weight", type=float, default=1e-5)
    parser.add_argument("--bound", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-only", action="store_true", help="Load OUTPUT/checkpoint.pt, evaluate test and render orbit")
    parser.add_argument("--checkpoint", help="Optional checkpoint path for --render-only")
    parser.add_argument("--skip-orbit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.steps, args.resolution, args.rays, args.samples, args.image_size, args.render_size, args.chunk) < 1:
        raise ValueError("Steps, sizes, rays, samples and chunk must all be positive")
    set_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        model, metadata = load_model(args.checkpoint or output / "checkpoint.pt", args.device)
    else:
        model, metadata = train(args)
    test = load_dataset(args.data, "test", args.image_size, args.device)
    metrics = evaluate(model, test, output, args.samples, args.chunk)
    metrics.update(training_seconds=metadata.get("training_seconds"), train_views=metadata.get("train_views"),
                   steps=metadata.get("steps"), resolution=model.resolution)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Held-out PSNR={metrics['psnr']:.3f}, foreground PSNR={metrics['foreground_psnr']:.3f}, SSIM={metrics['ssim']:.4f}", flush=True)
    if not args.skip_orbit:
        orbit = load_dataset(args.data, "orbit", args.render_size, args.device, load_images=False)
        render_orbit(model, orbit, output, args.samples, args.orbit_frames, args.chunk)
        print(f"Saved {output / 'orbit.gif'}", flush=True)


if __name__ == "__main__":
    main()
