"""ARKit geometry, confidence masking, visit splits and real loader integration."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

from dacite import Config, from_dict
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

from src.dataset import get_dataset
from src.dataset.dataset_scannet import DatasetScanNetCfg
from src.scripts.arkitscenes_data import make_plan, validate_plan, extract_asset
from src.scripts.convert_arkitscenes import scene_example, read_trajectory, nearest, convert
from src.scripts.convert_scannet import depth_overlap


def fixture_video(root):
    folder = root / "Validation" / "123"
    for asset in ("lowres_wide", "lowres_depth", "confidence", "lowres_wide_intrinsics"):
        (folder / asset).mkdir(parents=True)
    trajectory = []
    for i in range(5):
        name = f"123_{i:.3f}"
        rgb = np.full((48, 64, 3), 80+i, np.uint8)
        depth = np.full((48, 64), 2000, np.uint16)
        conf = np.full((48, 64), 2, np.uint8)
        conf[:4] = 0
        Image.fromarray(rgb).save(folder / "lowres_wide" / f"{name}.png")
        Image.fromarray(depth).save(folder / "lowres_depth" / f"{name}.png")
        Image.fromarray(conf).save(folder / "confidence" / f"{name}.png")
        # Per-frame focal length variation must survive conversion.
        (folder / "lowres_wide_intrinsics" / f"{name}.pincam").write_text(f"64 48 {55+i} 55 31 23")
        # w2c camera translation is negative for a camera moving right.
        trajectory.append(f"{i+.0008} 0 0 0 {-i*.04} 0 0")
    (folder / "lowres_wide.traj").write_text("\n".join(trajectory))
    return folder


def test_plan_stable_and_visit_disjoint():
    header = "video_id,visit_id,fold\n"
    rows = [f"{i},{i//2},Training" for i in range(10, 22)]
    rows += [f"{i},{i},Validation" for i in range(30, 34)]
    rows += ["90,NA,Training", "91,99,Training", "92,99,Validation"]
    a = make_plan(header + "\n".join(rows), 2, 1, 2)
    b = make_plan(header + "\n".join(reversed(rows)), 2, 1, 2)
    assert a["records"] == b["records"]
    assert a["unknown_visit_scans_excluded"] == 1
    assert a["cross_fold_visits_excluded"] == ["99"]
    validate_plan(a)
    a["records"].append(a["records"][0])
    with pytest.raises(ValueError, match="Repeated"):
        validate_plan(a)


def test_pose_direction_and_time_bound(tmp_path):
    p = tmp_path / "pose.traj"
    p.write_text(f"1.0 0 0 {np.pi/2} 1 2 3\n")
    _, poses = read_trajectory(p)
    np.testing.assert_allclose(poses[0][:3] @ [1, 0, 0, 1], [1, 3, 3], atol=1e-7)
    np.testing.assert_allclose(np.linalg.inv(poses[0])[:3, 3], [-2, 1, -3], atol=1e-7)
    assert nearest(np.array([1., 2.]), 1.004, .005) == 0
    with pytest.raises(ValueError):
        nearest(np.array([1., 2.]), 1.006, .005)


def test_frame_dependent_intrinsics_in_overlap():
    depth = np.ones((48, 64), np.float32)
    a = np.array([[55., 0, 31], [0, 55, 23], [0, 0, 1]])
    b = a.copy(); b[0, 2] += 1000
    assert depth_overlap(depth, depth, a, np.eye(4), np.eye(4)) == 1
    assert depth_overlap(depth, depth, a, np.eye(4), np.eye(4), target_k=b) == 0


def test_converter_loader_and_resume(tmp_path):
    folder = fixture_video(tmp_path / "raw")
    item = scene_example(folder)
    assert item["max_pose_time_residual"] < .001
    assert item["cameras"][0, 0] != item["cameras"][1, 0]
    np.testing.assert_allclose(item["cameras"][0, 2], 31.5/64)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(dict(dataset="arkitscenes", records=[
        dict(video_id="123", visit_id="42", fold="Validation", split="test")])))
    args = argparse.Namespace(plan=plan_path, input_dir=tmp_path / "raw", output_dir=tmp_path / "out",
                              splits=["test"], resume=False)
    convert(args)
    with initialize_config_dir(config_dir=str(Path("config").resolve()), version_base=None):
        raw = compose(config_name="main", overrides=["+experiment=arkitscenes"])
    assert raw.dataset.name == "arkitscenes"
    assert list(raw.dataset.image_shape) == [192, 256]
    assert raw.train.depth_loss_weight == .1
    raw.dataset.roots = [str(args.output_dir)]
    raw.dataset.image_shape = [48, 64]
    cfg = from_dict(DatasetScanNetCfg, OmegaConf.to_container(raw.dataset), config=Config(type_hooks={Path: Path}))
    ds = get_dataset(cfg, "test", None)
    sample = ds[0]
    assert not sample["context"]["depth_mask"][:, :4].any()
    assert torch.all(sample["context"]["depth"][:, 4:] == 2)
    ids = sample["context"]["index"]
    torch.testing.assert_close(sample["context"]["extrinsics"][:, 0, 3], ids.float()*.04)
    assert torch.equal(ds[0]["context"]["index"], ids)
    chunk = args.output_dir / "test" / "123.torch"
    original = chunk.read_bytes()
    args.resume = True
    convert(args)
    assert chunk.read_bytes() == original
    args.min_confidence = 1
    with pytest.raises(ValueError, match="identical"):
        convert(args)


def test_missing_download_not_silently_filtered(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(dict(dataset="arkitscenes", records=[
        dict(video_id="123", visit_id="42", fold="Validation", split="test")])))
    args = argparse.Namespace(plan=p, input_dir=tmp_path, output_dir=tmp_path / "out", splits=["test"], resume=False)
    with pytest.raises(FileNotFoundError, match="finish the download"):
        convert(args)
    assert not args.output_dir.exists()


def test_archive_containment_and_expected_layout(tmp_path):
    archive = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_asset(archive, tmp_path, "confidence")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("confidence/frame.png", "fixture")
    extract_asset(archive, tmp_path, "confidence")
    assert (tmp_path / "confidence" / "frame.png").is_file()


def test_pinned_plan():
    plan = json.loads(Path("assets/arkitscenes_subset_v1.json").read_text())
    validate_plan(plan)
    assert {s: sum(r["split"] == s for r in plan["records"]) for s in ("train", "val", "test")} == {
        "train": 200, "val": 30, "test": 50}
