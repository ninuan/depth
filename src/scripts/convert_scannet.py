"""Pack the registered ScanNet 25k export into reproducible RGB-D scene chunks.

Input: sceneXXXX_YY/{color,depth,pose}, intrinsics_color.txt, intrinsics_depth.txt.
This exporter requires color/depth cameras already registered to a common optical
frame (the official 25k export). Raw unregistered sensors need registration first.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


FORMAT_VERSION = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--train_split", type=Path)
    p.add_argument("--test_split", type=Path,
                   help="Held-out scan list, e.g. official scannetv2_val.txt")
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Hold out this fraction of training physical locations for model selection")
    p.add_argument("--test_fraction", type=float, default=0.1,
                   help="Used only without explicit split files")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_scenes", type=int)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--min_frames", type=int, default=3)
    p.add_argument("--scenes_per_chunk", "--frames_per_chunk", dest="scenes_per_chunk", type=int, default=1)
    p.add_argument("--depth_scale", type=float, default=1000.)
    p.add_argument("--min_depth", type=float, default=0.1)
    p.add_argument("--max_depth", type=float, default=10.)
    p.add_argument("--max_gap", type=int, default=4, help="Maximum context gap in retained frames")
    p.add_argument("--min_baseline", type=float, default=0.02, help="Context camera distance in metres")
    p.add_argument("--min_overlap", type=float, default=0.3,
                   help="Minimum depth-consistent overlap per context/target pair")
    return p.parse_args()


def space_id(scene):
    return scene.split("_")[0]


def split_scenes(names, seed, val_fraction, test_fraction, train_names=None, test_names=None):
    """Never split repeat scans of one physical location across partitions."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    names = set(names)
    if (train_names is None) != (test_names is None):
        raise ValueError("Provide train_split and test_split together")
    if train_names is not None:
        if {space_id(s) for s in train_names} & {space_id(s) for s in test_names}:
            raise ValueError("Train/test split files share a physical location")
        train_pool, test = names & set(train_names), names & set(test_names)
    else:
        spaces = np.array(sorted({space_id(n) for n in names}))
        rng.shuffle(spaces)
        if len(spaces) < 3:
            raise ValueError("Need at least three physical locations for train/val/test")
        n_test = min(max(1, round(len(spaces) * test_fraction)), len(spaces) - 2)
        test_spaces = set(spaces[:n_test])
        test = {n for n in names if space_id(n) in test_spaces}
        train_pool = names - test
    spaces = np.array(sorted({space_id(n) for n in train_pool}))
    rng.shuffle(spaces)
    if len(spaces) < 2 or not test:
        raise ValueError("Not enough locations for nonempty splits; increase max_scenes")
    n_val = min(max(1, round(len(spaces) * val_fraction)), len(spaces) - 1)
    val_spaces = set(spaces[:n_val])
    val = {n for n in train_pool if space_id(n) in val_spaces}
    return {"train": train_pool - val, "val": val, "test": test}


def read_intrinsics(path):
    k = np.loadtxt(path, dtype=np.float64)
    if k.shape not in ((3, 3), (4, 4)) or not np.isfinite(k).all():
        raise ValueError(f"Invalid intrinsics: {path}")
    k = k[:3, :3]
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError(f"Invalid camera model: {path}")
    return k


def read_pose(path):
    c2w = np.loadtxt(path, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError(f"Invalid pose: {path}")
    r = c2w[:3, :3]
    if (not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-2)
            or abs(np.linalg.det(r) - 1) > 1e-2):
        raise ValueError(f"Non-rigid pose: {path}")
    return np.linalg.inv(c2w)


def encoded(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return torch.from_numpy(np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy())


def align_registered_rgb(rgb, color_k, depth_k, depth_shape):
    """Ray-based resampling, not resizing then substituting the other camera's K.

    Calibration uses OpenCV integer pixel centers. Output normalized K is converted
    to DepthSplat's half-pixel convention when packed, not during this remapping.
    """
    h, w = depth_shape
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(depth_k).T
    uv = rays @ color_k.T
    uv = uv[..., :2] / uv[..., 2:]
    valid = ((uv[..., 0] >= 0) & (uv[..., 0] <= rgb.shape[1] - 1)
             & (uv[..., 1] >= 0) & (uv[..., 1] <= rgb.shape[0] - 1))
    # Fail instead of supervising black borders with plausible depth.
    if valid.mean() < 0.98:
        raise ValueError("Color/depth cameras are not a registered ScanNet pair (coverage <98%)")
    aligned = cv2.remap(rgb, uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned, valid


def depth_overlap(source, target, k, source_w2c, target_w2c, stride=12, target_k=None):
    """Fraction of valid sampled source pixels also visible at the target surface."""
    h, w = source.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = source[::stride, ::stride].reshape(-1)
    uv1 = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=0)
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return 0.
    xyz = (np.linalg.inv(k) @ uv1[:, valid]) * z[valid]
    transform = target_w2c @ np.linalg.inv(source_w2c)
    xyz = transform[:3, :3] @ xyz + transform[:3, 3:4]
    positive = xyz[2] > 1e-5
    uv = (k if target_k is None else target_k) @ xyz
    xy = np.rint(uv[:2] / np.maximum(uv[2:], 1e-5)).astype(np.int64)
    inside = positive & (xy[0] >= 0) & (xy[0] < w) & (xy[1] >= 0) & (xy[1] < h)
    target_z = target[xy[1, inside], xy[0, inside]]
    consistent = (target_z > 0) & (np.abs(target_z - xyz[2, inside]) < np.maximum(0.05, 0.05 * target_z))
    return float(consistent.sum() / valid.sum())


def build_triplets(depths, k, poses, max_gap=4, min_baseline=.02, min_overlap=.3):
    # Registered RGB-D datasets may have frame-dependent intrinsics.
    ks = [k] * len(poses) if np.asarray(k).ndim == 2 else k
    centers = [np.linalg.inv(p)[:3, 3] for p in poses]
    cache = {}
    def overlap(a, b):
        if (a, b) not in cache:
            cache[a, b] = depth_overlap(depths[a], depths[b], ks[a], poses[a], poses[b], target_k=ks[b])
        return cache[a, b]
    scored = []
    for left in range(len(poses)):
        for right in range(left + 2, min(len(poses), left + max_gap + 1)):
            if np.linalg.norm(centers[left] - centers[right]) < min_baseline:
                continue
            for target in range(left + 1, right):
                score = min(overlap(target, left), overlap(left, target),
                            overlap(target, right), overlap(right, target))
                if score >= min_overlap:
                    scored.append((score, [left, target, right]))
    # One deterministic evaluation triplet per scene; training samples all retained triplets.
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[1] for x in scored], [x[0] for x in scored]


def scene_example(scene_dir, frame_stride=1, min_frames=3, depth_scale=1000.,
                  max_gap=4, min_baseline=.02, min_overlap=.3, min_depth=.1, max_depth=10.):
    color_dir, depth_dir, pose_dir = (scene_dir / n for n in ("color", "depth", "pose"))
    frame_ids = sorted(set(p.stem for p in color_dir.glob("*.jpg"))
                       & set(p.stem for p in depth_dir.glob("*.png"))
                       & set(p.stem for p in pose_dir.glob("*.txt")), key=int)[::frame_stride]
    kc = read_intrinsics(scene_dir / "intrinsics_color.txt")
    kd = read_intrinsics(scene_dir / "intrinsics_depth.txt")
    # Do not silently treat raw unregistered camera exports as registered input.
    ec, ed = scene_dir / "extrinsics_color.txt", scene_dir / "extrinsics_depth.txt"
    if ec.exists() != ed.exists() or (ec.exists() and not np.allclose(np.loadtxt(ec), np.loadtxt(ed), atol=1e-5)):
        raise ValueError("Non-coincident RGB/depth extrinsics; register this raw export first")
    poses, images, depths, metric_depths, retained, bad_frames = [], [], [], [], [], []
    shape = None
    for frame in frame_ids:
        try:
            pose = read_pose(pose_dir / f"{frame}.txt")
            with Image.open(depth_dir / f"{frame}.png") as image:
                raw = np.array(image, copy=True)
            if raw.ndim != 2 or raw.min() < 0 or raw.max() > 65535:
                raise ValueError("Expected unsigned 16-bit camera-Z depth")
            if shape is not None and raw.shape != shape:
                raise ValueError("Inconsistent depth image dimensions")
            shape = raw.shape
            with Image.open(color_dir / f"{frame}.jpg") as image:
                rgb = np.array(image.convert("RGB"))
            rgb, coverage = align_registered_rgb(rgb, kc, kd, raw.shape)
            raw = raw.astype(np.uint16)
            raw[~coverage] = 0
            metric = raw.astype(np.float32) / depth_scale
            metric[(metric < min_depth) | (metric > max_depth)] = 0
            if (metric > 0).mean() < .05:
                raise ValueError("Too few valid depth pixels")
        except (OSError, ValueError, np.linalg.LinAlgError) as error:
            bad_frames.append({"frame": frame, "reason": str(error)})
            continue
        retained.append(int(frame))
        poses.append(pose)
        images.append(encoded(Image.fromarray(rgb), "JPEG"))
        depths.append(encoded(Image.fromarray(raw), "PNG"))
        metric_depths.append(metric)
    if len(retained) < min_frames:
        raise ValueError(f"Only {len(retained)} valid frames; rejected {len(bad_frames)}")
    triplets, overlaps = build_triplets(metric_depths, kd, poses, max_gap, min_baseline, min_overlap)
    if not triplets:
        raise ValueError("No triplets satisfy baseline and bidirectional depth overlap thresholds")
    h, w = shape
    # OpenCV centers: integer i. DepthSplat centers: (i + .5) / size.
    params = [kd[0, 0] / w, kd[1, 1] / h, (kd[0, 2] + .5) / w, (kd[1, 2] + .5) / h, 0., 0.]
    cameras = np.stack([np.concatenate([params, p[:3].ravel()]) for p in poses])
    return dict(format_version=FORMAT_VERSION, key=scene_dir.name,
                timestamps=torch.tensor(retained), cameras=torch.tensor(cameras, dtype=torch.float32),
                images=images, depths=depths, depth_scale=float(depth_scale), image_shape=[h, w],
                valid_triplets=triplets, triplet_overlaps=overlaps, rejected_frames=bad_frames,
                color_intrinsics=torch.tensor(kc), depth_intrinsics=torch.tensor(kd))


def main():
    args = parse_args()
    if (args.frame_stride < 1 or args.scenes_per_chunk < 1 or args.min_frames < 3
            or args.max_gap < 2 or args.min_baseline < 0 or not 0 < args.min_overlap <= 1
            or not np.isfinite(args.depth_scale) or args.depth_scale <= 0
            or not 0 < args.min_depth < args.max_depth):
        raise ValueError("Invalid conversion/sampling parameters")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory; old chunks must not mix with a new protocol")
    scenes = sorted(p for p in args.input_dir.glob("scene*_*" ) if p.is_dir())
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    def read_split(path):
        return None if path is None else {x.strip() for x in path.read_text().splitlines() if x.strip()}
    splits = split_scenes([p.name for p in scenes], args.seed, args.val_fraction,
                          args.test_fraction, read_split(args.train_split), read_split(args.test_split))
    assigned = {s: stage for stage, names in splits.items() for s in names}
    buffers = {s: [] for s in splits}
    indices = {s: {} for s in splits}
    evaluation = {s: {} for s in splits}
    chunks = {s: 0 for s in splits}
    rejected, rejected_frames = {}, {}
    for stage in splits:
        (args.output_dir / stage).mkdir(parents=True, exist_ok=True)
    def save(stage):
        if not buffers[stage]:
            return
        name = f"{chunks[stage]:06d}.torch"
        torch.save(buffers[stage], args.output_dir / stage / name)
        for item in buffers[stage]:
            indices[stage][item["key"]] = name
        buffers[stage].clear()
        chunks[stage] += 1
    for scene in tqdm(scenes, desc="Registering and checking ScanNet"):
        if scene.name not in assigned:
            continue
        stage = assigned[scene.name]
        try:
            item = scene_example(scene, args.frame_stride, args.min_frames, args.depth_scale,
                                 args.max_gap, args.min_baseline, args.min_overlap,
                                 args.min_depth, args.max_depth)
        except (OSError, ValueError, np.linalg.LinAlgError) as error:
            rejected[scene.name] = str(error)
            continue
        a, t, b = item["valid_triplets"][0]
        evaluation[stage][scene.name] = dict(context=[a, b], target=[t],
                                            frame_ids=item["timestamps"][[a, t, b]].tolist(),
                                            overlap=item["triplet_overlaps"][0])
        if item["rejected_frames"]:
            rejected_frames[scene.name] = item["rejected_frames"]
        buffers[stage].append(item)
        if len(buffers[stage]) >= args.scenes_per_chunk:
            save(stage)
    for stage in splits:
        save(stage)
        (args.output_dir / stage / "index.json").write_text(json.dumps(indices[stage], indent=2))
        (args.output_dir / stage / "evaluation_index.json").write_text(json.dumps(evaluation[stage], indent=2))
    manifest = dict(format_version=FORMAT_VERSION,
                    source="ScanNet registered RGB-D frames; camera-Z metres",
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    split_file_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (args.train_split, args.test_split) if p is not None},
                    requested_splits={s: sorted(v) for s, v in splits.items()},
                    retained_splits={s: sorted(v) for s, v in indices.items()},
                    scene_counts={s: len(v) for s, v in indices.items()},
                    rejected_scenes=rejected, rejected_frames=rejected_frames)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["scene_counts"], indent=2))
    print(f"Rejected scenes: {len(rejected)}; details in {args.output_dir / 'manifest.json'}")
    if any(not values for values in indices.values()):
        raise ValueError("An output split is empty after filtering; inspect manifest before training")


if __name__ == "__main__":
    main()
