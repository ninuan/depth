"""Validate a converted ScanNet chunk and write an RGB-depth alignment preview."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "test"), default="train")
    parser.add_argument("--output", type=Path, default=Path("outputs/scannet_data_check.png"))
    return parser.parse_args()


def decode_image(data: torch.Tensor) -> np.ndarray:
    with Image.open(io.BytesIO(data.numpy().tobytes())) as image:
        return np.array(image.convert("RGB"), copy=True)


def decode_depth(data: torch.Tensor, scale: float) -> np.ndarray:
    with Image.open(io.BytesIO(data.numpy().tobytes())) as image:
        return np.array(image, dtype=np.float32, copy=True) / scale


def main() -> None:
    args = parse_args()
    stage_dir = args.dataset_dir / args.stage
    index = json.loads((stage_dir / "index.json").read_text())
    if not index:
        raise ValueError(f"No scenes found in {stage_dir / 'index.json'}")
    scene, chunk_name = next(iter(index.items()))
    chunk = torch.load(stage_dir / chunk_name, map_location="cpu", weights_only=False)
    example = next(item for item in chunk if item["key"] == scene)

    image = decode_image(example["images"][0])
    scale = float(example.get("depth_scale", 1000.0))
    depth = decode_depth(example["depths"][0], scale)
    valid = np.isfinite(depth) & (depth > 0)
    cameras = example["cameras"]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3] = cameras[0, 6:].numpy().reshape(3, 4)

    if image.shape[:2] != depth.shape:
        raise ValueError(f"RGB/depth mismatch: {image.shape[:2]} vs {depth.shape}")
    if not np.isfinite(cameras.numpy()).all():
        raise ValueError("Camera tensor contains NaN or Inf")
    if not valid.any():
        raise ValueError("The sampled depth map has no valid pixels")

    lo, hi = np.percentile(depth[valid], (2, 98))
    normalized = np.zeros_like(depth)
    normalized[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_rgb = np.stack(
        (255 * normalized, 255 * (1 - normalized), 128 * np.ones_like(normalized)),
        axis=-1,
    ).astype(np.uint8)
    depth_rgb[~valid] = 0

    # Red depth discontinuities over RGB make calibration errors easy to spot.
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1]))
    edge = valid & ((gx + gy) > 0.08)
    overlay = image.copy()
    overlay[edge] = (255, 32, 32)
    preview = np.concatenate((image, depth_rgb, overlay), axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(args.output)

    print(f"scene: {scene}")
    print(f"frames: {len(example['images'])}")
    print(f"RGB/depth shape: {image.shape[:2]}")
    print(f"valid depth: {valid.mean():.2%}")
    print(f"valid depth range: {depth[valid].min():.3f} to {depth[valid].max():.3f} m")
    print(f"normalized intrinsics: {cameras[0, :4].tolist()}")
    print(f"pose rotation determinant: {np.linalg.det(w2c[:3, :3]):.6f}")
    print(f"preview: {args.output.resolve()}")


if __name__ == "__main__":
    main()
