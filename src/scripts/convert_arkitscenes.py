"""Convert official ARKitScenes RAW lowres RGB-D assets to the RGB-D v2 loader.

Trajectory angle-axis/translation represent world-to-camera (as in Apple's
TenFpsDataLoader). Images, depths, and confidence use the same lowres grid.
No highres FARO labels or pseudo-depth are mixed into this protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .arkitscenes_data import ASSETS, validate_plan
from .convert_scannet import build_triplets, encoded


def timestamp(path):
    value = float(path.stem.rsplit("_", 1)[-1])
    if not np.isfinite(value):
        raise ValueError(f"Invalid timestamp: {path}")
    return value


def timed_files(folder, suffix):
    result = {}
    for p in sorted(folder.glob(f"*.{suffix}")):
        t = timestamp(p)
        if t in result:
            raise ValueError(f"Duplicate timestamp in {folder}: {t}")
        result[t] = p
    if not result:
        raise ValueError(f"No {suffix} files in {folder}")
    return result


def nearest(times, value, tolerance):
    i = int(np.searchsorted(times, value))
    candidates = [j for j in (i-1, i) if 0 <= j < len(times)]
    j = min(candidates, key=lambda j: (abs(times[j] - value), times[j]))
    if abs(times[j] - value) > tolerance + 1e-9:
        raise ValueError(f"No timestamp within {tolerance}s of {value}")
    return j


def read_trajectory(path):
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError(f"Invalid trajectory: {path}")
    values = values[np.argsort(values[:, 0])]
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f"Duplicate trajectory timestamp: {path}")
    poses = []
    for row in values:
        w2c = np.eye(4)
        w2c[:3, :3] = cv2.Rodrigues(row[1:4])[0]
        w2c[:3, 3] = row[4:7]
        poses.append(w2c)
    return values[:, 0], poses


def read_intrinsics(path, shape):
    values = np.loadtxt(path).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid pincam file: {path}")
    w, h, fx, fy, cx, cy = values
    if (h, w) != shape or fx <= 0 or fy <= 0:
        raise ValueError(f"Intrinsic/image size mismatch or invalid focal length: {path}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])


def scene_example(folder, *, frame_interval=1., max_frames=200, pose_tolerance=.005,
                  intrinsics_tolerance=.001, min_confidence=2, min_depth=.1, max_depth=10.,
                  max_gap=4, max_time_gap=4.5, min_baseline=.02, min_overlap=.3):
    files = {n: timed_files(folder / n, "pincam" if n.endswith("intrinsics") else "png")
             for n in ASSETS if not n.endswith(".traj")}
    pose_times, all_poses = read_trajectory(folder / "lowres_wide.traj")
    k_times = np.array(sorted(files["lowres_wide_intrinsics"]))
    # Only exactly synchronized image/depth/confidence timestamps are paired.
    candidates = sorted(set(files["lowres_depth"]) & set(files["lowres_wide"]) & set(files["confidence"]))
    retained_times, rejected, selected, last = [], [], [], -np.inf
    for t in candidates:
        if t - last < frame_interval - 1e-9:
            continue
        try:
            pose_id = nearest(pose_times, t, pose_tolerance)
            ki = nearest(k_times, t, intrinsics_tolerance)
        except ValueError as e:
            rejected.append(dict(timestamp=t, reason=str(e)))
            continue
        selected.append((t, pose_id, k_times[ki]))
        last = t
    # Use the first bounded temporal window rather than sparse whole-video frames:
    # this maintains usable baselines in long videos (cap selection is recorded).
    selected = selected[:max_frames]
    images, depths, metrics, ks, poses, residuals = [], [], [], [], [], []
    shape = None
    for t, pose_id, kt in selected:
        try:
            with Image.open(files["lowres_wide"][t]) as im:
                rgb = np.array(im.convert("RGB"))
            with Image.open(files["lowres_depth"][t]) as im:
                raw = np.array(im)
            with Image.open(files["confidence"][t]) as im:
                confidence = np.array(im)
            if raw.ndim != 2 or raw.min() < 0 or raw.max() > 65535:
                raise ValueError("Expected uint16 millimeter depth")
            if rgb.shape[:2] != raw.shape or confidence.shape != raw.shape:
                raise ValueError("Lowres RGB/depth/confidence grids differ")
            if (shape is not None and raw.shape != shape) or confidence.max() > 2:
                raise ValueError("Inconsistent resolution or invalid confidence values")
            k = read_intrinsics(files["lowres_wide_intrinsics"][kt], raw.shape)
            raw = raw.astype(np.uint16)
            mask = (confidence >= min_confidence) & (raw >= min_depth*1000) & (raw <= max_depth*1000)
            raw[~mask] = 0
            if mask.mean() < .05:
                raise ValueError("Less than 5% valid depth after confidence/range filtering")
            shape = raw.shape
        except (OSError, ValueError) as e:
            rejected.append(dict(timestamp=t, reason=str(e)))
            continue
        retained_times.append(t)
        poses.append(all_poses[pose_id]); ks.append(k)
        images.append(encoded(Image.fromarray(rgb), "PNG"))
        depths.append(encoded(Image.fromarray(raw), "PNG"))
        metrics.append(raw.astype(np.float32) / 1000)
        residuals.append(abs(float(pose_times[pose_id])-t))
    if len(poses) < 3:
        raise ValueError(f"Only {len(poses)} valid frames; {len(rejected)} rejected")
    triplets, overlaps = build_triplets(metrics, ks, poses, max_gap, min_baseline, min_overlap)
    pairs = [(tri, score) for tri, score in zip(triplets, overlaps)
             if retained_times[tri[2]] - retained_times[tri[0]] <= max_time_gap + 1e-9]
    if not pairs:
        raise ValueError("No triplets satisfy overlap/baseline/time-gap constraints")
    h, w = shape
    cameras = []
    for k, pose in zip(ks, poses):
        params = [k[0, 0]/w, k[1, 1]/h, (k[0, 2]+.5)/w, (k[1, 2]+.5)/h, 0., 0.]
        cameras.append(np.concatenate([params, pose[:3].ravel()]))
    return dict(format_version=2, key=folder.name,
                timestamps=torch.tensor(np.rint(np.array(retained_times)*1e6), dtype=torch.int64),
                timestamps_seconds=retained_times, cameras=torch.tensor(np.array(cameras), dtype=torch.float32),
                images=images, depths=depths, depth_scale=1000., image_shape=[h, w],
                valid_triplets=[x[0] for x in pairs], triplet_overlaps=[x[1] for x in pairs],
                rejected_frames=rejected, max_pose_time_residual=max(residuals),
                unmatched_image_timestamps=len(files["lowres_depth"])-len(candidates))


def convert(args):
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    records = [r for r in plan["records"] if r["split"] in args.splits]
    if not records or any(not any(r["split"] == s for r in records) for s in args.splits):
        raise ValueError("Requested split has no planned records")
    settings = {k: v for k, v in vars(args).items() if k not in ("plan", "input_dir", "output_dir", "resume", "splits")}
    protocol = dict(dataset="arkitscenes", settings=settings, splits=sorted(args.splits),
                    plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    state_path = args.output_dir / "conversion_protocol.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume or not state_path.exists() or json.loads(state_path.read_text()) != protocol:
            raise ValueError("Use an empty directory or --resume with the identical plan/settings/splits")
    # Missing downloads are operational errors, never silently filtered scenes.
    for r in records:
        for asset in ASSETS:
            path = args.input_dir / r["fold"] / r["video_id"] / asset
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; finish the download first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(protocol, indent=2))
    indices = {s: {} for s in ("train", "val", "test")}
    evaluation = {s: {} for s in indices}
    rejected, frame_rejections, visits = {}, {}, {}
    for r in tqdm(records, desc="Converting ARKitScenes"):
        s, video = r["split"], r["video_id"]
        dest = args.output_dir / s / f"{video}.torch"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and dest.exists():
            item, = torch.load(dest, map_location="cpu", weights_only=False)
            if item["key"] != video or item.get("visit_id") != r["visit_id"]:
                raise ValueError(f"Mismatched cached chunk: {dest}")
        else:
            try:
                item = scene_example(args.input_dir / r["fold"] / video, **settings)
            except (OSError, ValueError, np.linalg.LinAlgError) as e:
                rejected[video] = str(e)
                continue
            item["visit_id"] = r["visit_id"]
            tmp = dest.with_suffix(".torch.part")
            torch.save([item], tmp)
            tmp.replace(dest)
        indices[s][video] = dest.name
        visits[video] = r["visit_id"]
        a, t, b = item["valid_triplets"][0]
        evaluation[s][video] = dict(context=[a, b], target=[t],
                                   frame_ids=item["timestamps"][[a, t, b]].tolist(),
                                   overlap=item["triplet_overlaps"][0])
        if item["rejected_frames"]:
            frame_rejections[video] = item["rejected_frames"]
    for s in indices:
        folder = args.output_dir / s
        folder.mkdir(exist_ok=True)
        (folder / "index.json").write_text(json.dumps(indices[s], indent=2))
        (folder / "evaluation_index.json").write_text(json.dumps(evaluation[s], indent=2))
    manifest = dict(format_version=2, dataset="arkitscenes", protocol=protocol,
                    source="Official RAW lowres Apple LiDAR; camera-Z metres; confidence filtered",
                    depth_reference="sensor_depth_not_FARO", physical_locations=visits,
                    plan=plan, scene_counts={s: len(v) for s, v in indices.items()},
                    retained_splits={s: sorted(v) for s, v in indices.items()},
                    requested_splits={s: [r["video_id"] for r in records if r["split"] == s] for s in indices},
                    rejected_scenes=rejected, rejected_frames=frame_rejections)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["scene_counts"], indent=2))
    print(f"Rejected {len(rejected)} videos; inspect manifest.json before evaluating")
    if any(not indices[s] for s in args.splits):
        raise ValueError("Requested split empty after filtering")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--input_dir", type=Path, required=True, help="Root containing Training/ and Validation/")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["test"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--frame_interval", type=float, default=1.)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--pose_tolerance", type=float, default=.005)
    p.add_argument("--intrinsics_tolerance", type=float, default=.001)
    p.add_argument("--min_confidence", type=int, choices=[0, 1, 2], default=2)
    p.add_argument("--min_depth", type=float, default=.1)
    p.add_argument("--max_depth", type=float, default=10.)
    p.add_argument("--max_gap", type=int, default=4)
    p.add_argument("--max_time_gap", type=float, default=4.5)
    p.add_argument("--min_baseline", type=float, default=.02)
    p.add_argument("--min_overlap", type=float, default=.3)
    args = p.parse_args()
    if (args.frame_interval <= 0 or args.max_frames < 3 or args.max_gap < 2 or args.max_time_gap <= 0
            or args.pose_tolerance < 0 or args.intrinsics_tolerance < 0 or args.min_baseline < 0
            or not 0 < args.min_depth < args.max_depth or not 0 < args.min_overlap <= 1):
        p.error("Invalid sampling/range/tolerance settings")
    convert(args)


if __name__ == "__main__":
    main()
