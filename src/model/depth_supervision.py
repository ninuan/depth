"""Metric camera-Z supervision; ground truth is used only by losses/metrics."""

import torch
import torch.nn.functional as F
from torch import Tensor


def valid_depth_mask(views: dict) -> Tensor:
    depth = views["depth"]
    valid = views.get("depth_mask", torch.ones_like(depth, dtype=torch.bool)).bool()
    return (valid & torch.isfinite(depth) & (depth > 0)
            & (depth >= views["near"][..., None, None])
            & (depth <= views["far"][..., None, None]))


def resize_prediction(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.ndim != 4 or prediction.shape[:2] != target.shape[:2]:
        raise ValueError(f"Depth shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.shape[-2:] != target.shape[-2:]:
        prediction = F.interpolate(
            prediction.flatten(0, 1).unsqueeze(1), target.shape[-2:],
            mode="bilinear", align_corners=False,
        ).reshape_as(target)
    return prediction.float()


def masked_depth_loss(prediction: Tensor, views: dict, mode: str = "log_l1") -> Tensor:
    if mode not in ("log_l1", "l1", "inverse_l1"):
        raise ValueError(f"Unknown depth loss mode: {mode}")
    gt = views["depth"].float()
    prediction = resize_prediction(prediction, gt)
    if not torch.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Predicted metric depths must be finite and positive")
    valid = valid_depth_mask(views)
    # Select BEFORE log/division: NaN labels in invalid pixels must never enter autograd.
    p, g = prediction[valid], gt[valid]
    if p.numel() == 0:
        return prediction.sum() * 0.0
    if mode == "log_l1":
        error = (p.log() - g.log()).abs()
    elif mode == "inverse_l1":
        error = (p.reciprocal() - g.reciprocal()).abs()
    else:
        error = (p - g).abs()
    return error.mean()


def supervised_depth_loss(
    predictions: Tensor, views: dict, mode: str = "log_l1",
    intermediate_weight: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Encoder packs coarse-to-fine predictions along batch; final B entries are final."""
    batch_size = views["depth"].shape[0]
    if predictions.ndim != 4 or predictions.shape[0] % batch_size:
        raise ValueError("Expected depth predictions packed as [scales * B, V, H, W]")
    if not 0 <= intermediate_weight <= 1:
        raise ValueError("intermediate_weight must be in [0, 1]")
    stages = predictions.split(batch_size)
    final = masked_depth_loss(stages[-1], views, mode)
    total, weight_sum = final, 1.0
    for i, stage in enumerate(stages[:-1]):
        weight = intermediate_weight ** (len(stages) - 1 - i)
        if weight:
            total = total + weight * masked_depth_loss(stage, views, mode)
            weight_sum += weight
    # Keep lambda comparable across architectures with different refinement counts.
    return total / weight_sum, final


@torch.no_grad()
def depth_metrics(prediction: Tensor, views: dict) -> dict[str, Tensor]:
    gt = views["depth"].float()
    prediction = resize_prediction(prediction, gt)
    valid = valid_depth_mask(views)
    if not valid.any():
        return {}
    p, g = prediction[valid], gt[valid]
    if not torch.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Invalid predicted depth during evaluation")
    return {
        "abs_rel": ((p - g).abs() / g).mean(),
        "rmse": (p - g).square().mean().sqrt(),
        "a1": (torch.maximum(p / g, g / p) < 1.25).float().mean(),
    }
