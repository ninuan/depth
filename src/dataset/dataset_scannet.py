"""ScanNet map dataset: scene-level random sampling with standard DDP sharding."""
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetScanNetCfg(DatasetCfgCommon):
    name: Literal["scannet", "arkitscenes"]
    roots: list[Path]
    near: float = 0.1
    far: float = 10.0
    augment: bool = True
    train_times_per_scene: int = 1
    test_len: int = -1
    test_split: Literal["val", "test"] = "test"


class DatasetScanNet(Dataset):
    def __init__(self, cfg: DatasetScanNetCfg, stage: Stage, view_sampler: ViewSampler):
        self.cfg, self.stage, self.view_sampler = cfg, stage, view_sampler
        if not 0 < cfg.near < cfg.far or cfg.train_times_per_scene < 1:
            raise ValueError("Require 0 < near < far and positive train_times_per_scene")
        self.entries = []
        self._cache_path, self._cache = None, None
        split = cfg.test_split if stage == "test" else stage
        seen = set()
        for root in cfg.roots:
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest.get("format_version") != 2:
                raise ValueError("ScanNet format v2 required; regenerate using convert_scannet.py")
            if cfg.name == "arkitscenes" and manifest.get("dataset") != "arkitscenes":
                raise ValueError("Expected an ARKitScenes manifest")
            index = json.loads((root / split / "index.json").read_text())
            for scene, chunk in sorted(index.items()):
                if cfg.overfit_to_scene is not None and scene != cfg.overfit_to_scene:
                    continue
                if scene in seen:
                    raise ValueError(f"Duplicate scene across roots: {scene}")
                seen.add(scene)
                self.entries.append((scene, root / split / chunk))
        if stage == "test" and cfg.test_len > 0:
            self.entries = self.entries[:cfg.test_len]
        if not self.entries:
            raise ValueError(f"No {cfg.name} scenes in {split}; check roots, split and conversion report")

    def __len__(self):
        return len(self.entries) * (self.cfg.train_times_per_scene if self.stage == "train" else 1)

    def __getitem__(self, index):
        repeats = self.cfg.train_times_per_scene if self.stage == "train" else 1
        scene, path = self.entries[index // repeats]
        # Bounded per-worker cache; conversion defaults to one scene per chunk.
        if self._cache_path != path:
            chunk = torch.load(path, map_location="cpu", weights_only=False)
            self._cache = {item["key"]: item for item in chunk}
            self._cache_path = path
        item = self._cache[scene]
        if item.get("format_version") != 2:
            raise ValueError(f"Legacy chunk {path}; regenerate the processed data")
        cameras = item["cameras"].float()
        w2c = torch.eye(4).repeat(len(cameras), 1, 1)
        w2c[:, :3] = cameras[:, 6:].reshape(-1, 3, 4)
        extrinsics = torch.linalg.inv(w2c)
        intrinsics = torch.eye(3).repeat(len(cameras), 1, 1)
        intrinsics[:, 0, 0] = cameras[:, 0]
        intrinsics[:, 1, 1] = cameras[:, 1]
        intrinsics[:, 0, 2] = cameras[:, 2]
        intrinsics[:, 1, 2] = cameras[:, 3]
        context, target = self.view_sampler.sample(
            scene, extrinsics, intrinsics, valid_triplets=item["valid_triplets"])
        if set(context.tolist()) & set(target.tolist()):
            raise ValueError(f"Context/target leakage in {scene}")

        def make_views(indices):
            images, depths = [], []
            for i in indices.tolist():
                with Image.open(BytesIO(item["images"][i].numpy().tobytes())) as image:
                    images.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True))
                                  .permute(2, 0, 1).float() / 255)
                with Image.open(BytesIO(item["depths"][i].numpy().tobytes())) as image:
                    depths.append(torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
                                  / item["depth_scale"])
            images, depths = torch.stack(images), torch.stack(depths)
            if images.shape[-2:] != depths.shape[-2:]:
                raise ValueError(f"RGB/depth shape mismatch in {scene}")
            valid = torch.isfinite(depths) & (depths >= self.cfg.near) & (depths <= self.cfg.far)
            depths = torch.where(valid, depths, torch.zeros_like(depths))
            return dict(extrinsics=extrinsics[indices], intrinsics=intrinsics[indices],
                        image=images, depth=depths, depth_mask=valid,
                        near=torch.full((len(indices),), self.cfg.near),
                        far=torch.full((len(indices),), self.cfg.far), index=indices,
                        frame_id=item["timestamps"][indices])

        example = dict(context=make_views(context), target=make_views(target), scene=scene)
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        return apply_crop_shim(example, tuple(self.cfg.image_shape))
