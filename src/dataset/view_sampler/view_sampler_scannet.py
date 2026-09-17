from dataclasses import dataclass
from typing import Literal

import torch

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerScanNetCfg:
    name: Literal["scannet"]
    num_context_views: int = 2
    num_target_views: int = 1


class ViewSamplerScanNet(ViewSampler[ViewSamplerScanNetCfg]):
    """Use geometry-checked triplets saved by convert_scannet; evaluation is fixed."""

    def sample(self, scene, extrinsics, intrinsics, device=torch.device("cpu"),
               valid_triplets=None, **kwargs):
        if self.cfg.num_context_views != 2 or self.cfg.num_target_views != 1:
            raise ValueError("The ScanNet protocol uses two context views and one held-out target")
        if not valid_triplets:
            raise ValueError(f"No valid triplets for {scene}; regenerate the dataset")
        i = (torch.randint(len(valid_triplets), ()).item()
             if self.stage == "train" and not self.is_overfitting else 0)
        left, target, right = valid_triplets[i]
        if not 0 <= left < target < right < len(extrinsics):
            raise ValueError(f"Invalid triplet in {scene}: {valid_triplets[i]}")
        return (torch.tensor([left, right], device=device),
                torch.tensor([target], device=device))

    @property
    def num_context_views(self):
        return self.cfg.num_context_views

    @property
    def num_target_views(self):
        return self.cfg.num_target_views
