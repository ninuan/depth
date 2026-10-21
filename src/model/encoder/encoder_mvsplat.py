"""Official MVSplat default RE10K model adapted to our metric RGB-D protocol.

Backbone, predictor, and Gaussian parameterization are vendored separately from
DepthSplat. See mvsplat/UPSTREAM.json and mvsplat/README.md for provenance.
"""
from dataclasses import dataclass, field
from typing import Literal

from einops import rearrange
import torch

from .encoder import Encoder
from .mvsplat.backbone import BackboneMultiview
from .mvsplat.costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .mvsplat.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from ..types import Gaussians
from ...geometry.projection import sample_image_grid
from ...dataset.shims.patch_shim import apply_patch_shim


@dataclass
class EncoderMVSplatCfg:
    name: Literal["mvsplat"]
    gaussian_adapter: GaussianAdapterCfg
    d_feature: int = 128
    num_depth_candidates: int = 128
    downscale_factor: int = 4
    shim_patch_size: int = 4
    multiview_trans_attn_split: int = 2
    costvolume_unet_feat_dim: int = 128
    costvolume_unet_channel_mult: list[int] = field(default_factory=lambda: [1, 1, 1])
    costvolume_unet_attn_res: list[int] = field(default_factory=lambda: [4])
    depth_unet_feat_dim: int = 32
    depth_unet_attn_res: list[int] = field(default_factory=lambda: [16])
    depth_unet_channel_mult: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    # Common wrapper contract: only final RGB/depth outputs, no depth training.
    return_depth: bool = True
    train_depth_only: bool = False
    supervise_intermediate_depth: bool = False


class EncoderMVSplat(Encoder[EncoderMVSplatCfg]):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.return_depth or cfg.train_depth_only or cfg.supervise_intermediate_depth:
            raise ValueError("MVSplat comparison requires final RGB/depth outputs only")
        self.backbone = BackboneMultiview(feature_channels=cfg.d_feature,
                                         downscale_factor=cfg.downscale_factor)
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature, upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=self.gaussian_adapter.d_in + 2,
            gaussians_per_pixel=1, num_views=2,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=tuple(cfg.depth_unet_attn_res),
            depth_unet_channel_mult=tuple(cfg.depth_unet_channel_mult),
        )

    def forward(self, context, global_step=0, deterministic=False,
                visualization_dump=None, scene_names=None):
        b, v, _, h, w = context["image"].shape
        if v != 2:
            raise ValueError("This fixed MVSplat comparison uses exactly two context views")
        # Both existing RGB-D resolutions satisfy the 16-pixel U-Net grid.
        if h % 16 or w % 16:
            raise ValueError("MVSplat input dimensions must be multiples of 16")
        features, cnn = self.backbone(context["image"],
                                     attn_splits=self.cfg.multiview_trans_attn_split,
                                     return_cnn_features=True)
        # Our stored K uses normalized half-pixel centers; CNN warping uses
        # integer input-pixel centers, as for the DepthSplat adapter.
        matching_k = context["intrinsics"].clone()
        matching_k[..., 0, 2] -= .5 / w
        matching_k[..., 1, 2] -= .5 / h
        depths, densities, raw = self.depth_predictor(
            features, matching_k, context["extrinsics"], context["near"], context["far"],
            deterministic=deterministic, cnn_features=cnn,
            extra_info={"images": rearrange(context["image"], "b v c h w -> (v b) c h w"),
                        "scene_names": scene_names},
        )
        xy, _ = sample_image_grid((h, w), context["image"].device)
        xy = rearrange(xy, "h w xy -> (h w) () xy")
        raw = rearrange(raw, "b v r c -> b v r () c")
        pixel_size = 1 / torch.tensor((w, h), device=xy.device, dtype=xy.dtype)
        xy = xy + (raw[..., :2].sigmoid() - .5) * pixel_size
        # Upstream default opacity exponent is exactly 1: mapping is identity.
        g = self.gaussian_adapter(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy, "b v r s xy -> b v r s () xy"), depths, densities,
            rearrange(raw[..., 2:], "b v r s c -> b v r s () c"), (h, w),
        )
        gaussians = Gaussians(
            rearrange(g.means, "b v r s p c -> b (v r s p) c"),
            rearrange(g.covariances, "b v r s p i j -> b (v r s p) i j"),
            rearrange(g.harmonics, "b v r s p c d -> b (v r s p) c d"),
            rearrange(g.opacities, "b v r s p -> b (v r s p)"),
        )
        # Official Gaussian placement uses UNIT rays: its scalar is ray distance.
        # Compare camera-Z to sensor camera-Z, preserving official Gaussian means.
        origin = context["extrinsics"][:, :, None, None, None, :3, 3]
        forward = context["extrinsics"][:, :, None, None, None, :3, 2]
        camera_z = ((g.means - origin) * forward).sum(-1)
        return {"gaussians": gaussians,
                "depths": rearrange(camera_z, "b v (h w) 1 1 -> b v h w", h=h, w=w)}

    def get_data_shim(self):
        return lambda batch: apply_patch_shim(batch, patch_size=16)
