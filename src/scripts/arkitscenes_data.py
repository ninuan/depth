"""Freeze a visit-disjoint ARKitScenes subset, then download only its RGB-D assets.

Uses Python's standard library and curl; planning does not download image archives.
Source format: https://github.com/apple/ARKitScenes/blob/main/DATA.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
ASSETS = ("lowres_wide", "lowres_depth", "confidence", "lowres_wide_intrinsics", "lowres_wide.traj")
# Official download_data.py lists these videos as lacking lowres_wide.traj.
MISSING_TRAJECTORIES = set("""47334522 47334523 42897421 45261582 47333152 47333155
48458535 48018733 47429677 48458541 42897848 47895482 47333960 47430089 42899148
42897612 42899153 42446164 48018149 47332198 47334515 45663223 45663226 45663227""".split())


def rank(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def make_plan(metadata, train_visits=200, val_visits=30, test_visits=50, seed=1234):
    counts = (train_visits, val_visits, test_visits)
    if any(n < 0 for n in counts) or test_visits < 1 or (train_visits == 0) != (val_visits == 0):
        raise ValueError("Need test_visits > 0 and either both or neither of train/val")
    visits, seen_videos, unknown = {}, set(), 0
    for row in csv.DictReader(io.StringIO(metadata)):
        video, visit, fold = (row[k].strip() for k in ("video_id", "visit_id", "fold"))
        if not video.isdigit() or video in seen_videos:
            raise ValueError(f"Invalid/duplicate video ID: {video}")
        seen_videos.add(video)
        if fold not in ("Training", "Validation"):
            raise ValueError(f"Unknown official fold: {fold}")
        if not visit.isdigit():
            unknown += 1
            continue
        visits.setdefault(visit, []).append((fold, video))
    ambiguous = sorted(v for v, rows in visits.items() if len({f for f, _ in rows}) > 1)
    visits = {v: (rows[0][0], [video for _, video in rows if video not in MISSING_TRAJECTORIES])
              for v, rows in visits.items() if v not in ambiguous}
    visits = {v: row for v, row in visits.items() if row[1]}
    pools = {f: sorted((v for v, (fold, _) in visits.items() if fold == f),
                       key=lambda v: rank(v, seed)) for f in ("Training", "Validation")}
    if len(pools["Training"]) < train_visits + val_visits or len(pools["Validation"]) < test_visits:
        raise ValueError("Requested more visits than metadata provides")
    selected = {"train": pools["Training"][:train_visits],
                "val": pools["Training"][train_visits:train_visits + val_visits],
                "test": pools["Validation"][:test_visits]}
    records = []
    for split, ids in selected.items():
        for visit in ids:
            fold, videos = visits[visit]
            video = min(videos, key=lambda v: rank(v, seed))
            records.append(dict(video_id=video, visit_id=visit, fold=fold, split=split))
    return dict(dataset="arkitscenes", protocol="arkitscenes_rgbd_subset_v1", seed=seed,
                metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(),
                selection="Hash-ranked visits; one hash-ranked video per visit; unknown/cross-fold visits excluded",
                cross_fold_visits_excluded=ambiguous,
                known_missing_trajectory_videos=sorted(MISSING_TRAJECTORIES),
                unknown_visit_scans_excluded=unknown, assets=list(ASSETS), records=records,
                requested_counts=dict(zip(("train", "val", "test"), counts)))


def validate_plan(plan):
    if plan.get("dataset") != "arkitscenes" or not plan.get("records"):
        raise ValueError("Expected a nonempty ARKitScenes plan")
    videos, visits = set(), set()
    for r in plan["records"]:
        if not all(str(r[k]).isdigit() for k in ("video_id", "visit_id")):
            raise ValueError("Non-numeric video/visit ID")
        if r["video_id"] in MISSING_TRAJECTORIES:
            raise ValueError("Plan includes video with known missing trajectory")
        if r["video_id"] in videos or r["visit_id"] in visits:
            raise ValueError("Repeated video or physical visit in plan")
        videos.add(r["video_id"]); visits.add(r["visit_id"])
        expected = {"train": "Training", "val": "Training", "test": "Validation"}
        if r["split"] not in expected or r["fold"] != expected[r["split"]]:
            raise ValueError("Invalid split / official fold mapping")


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "3",
                    "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
                    "--continue-at", "-", "--output", str(part), url], check=True)
    part.replace(path)


def extract_asset(archive, destination, asset):
    """Validate CRC and containment before publishing the expected asset directory."""
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            p = Path(member.filename)
            if p.is_absolute() or ".." in p.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive entry: {member.filename}")
        if z.testzip() is not None:
            raise ValueError(f"Corrupt archive: {archive}; remove it and download again")
        with tempfile.TemporaryDirectory(dir=destination, prefix=".extract-") as temp:
            z.extractall(temp)
            source = Path(temp) / asset
            if not source.is_dir() or not any(source.iterdir()):
                raise ValueError(f"Archive lacks nonempty {asset}/ directory: {archive}")
            if (destination / asset).exists():
                raise FileExistsError(f"Unverified directory exists: {destination / asset}")
            (source / ".arkitscenes_complete").write_text("CRC verified\n")
            shutil.move(str(source), destination / asset)


def download(plan, output, splits, keep_zip=False):
    validate_plan(plan)
    records = [r for r in plan["records"] if r["split"] in splits]
    if not records:
        raise ValueError("No planned videos in requested splits")
    for i, r in enumerate(records):
        folder = output / r["fold"] / r["video_id"]
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(records)}] {r['split']} video={r['video_id']}", flush=True)
        for asset in ASSETS:
            marker = folder / f".{asset}.complete"
            if (folder / asset / ".arkitscenes_complete").is_file():
                continue
            if marker.exists() and (folder / asset).exists():
                continue
            filename = asset if asset.endswith(".traj") else asset + ".zip"
            path = folder / filename
            if not path.exists():
                fetch(f"{BASE_URL}/{r['fold']}/{r['video_id']}/{filename}", path)
            if filename.endswith(".zip"):
                extract_asset(path, folder, asset)
            elif not path.stat().st_size:
                raise ValueError(f"Empty trajectory: {path}")
            marker.write_text("Downloaded and validated by arkitscenes_data.py\n")
            if filename.endswith(".zip") and not keep_zip:
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path, help="Use an already downloaded official raw metadata.csv")
    p.add_argument("--train_visits", type=int, default=200)
    p.add_argument("--val_visits", type=int, default=30)
    p.add_argument("--test_visits", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p = sub.add_parser("download")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["test"])
    p.add_argument("--keep_zip", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        if args.output.exists():
            raise FileExistsError("Plan already exists; reuse it instead of silently changing the protocol")
        metadata_path = args.metadata or args.output.with_suffix(".metadata.csv")
        if not metadata_path.exists():
            if args.metadata:
                raise FileNotFoundError(metadata_path)
            fetch(f"{BASE_URL}/metadata.csv", metadata_path)
        plan = make_plan(metadata_path.read_text(), args.train_visits, args.val_visits,
                         args.test_visits, args.seed)
        validate_plan(plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({k: v for k, v in plan.items() if k != "records"}, indent=2))
    else:
        download(json.loads(args.plan.read_text()), args.output_dir, args.splits, args.keep_zip)


if __name__ == "__main__":
    main()
