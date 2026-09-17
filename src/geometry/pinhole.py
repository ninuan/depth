"""Projection from normalized, half-pixel-center pinhole intrinsics."""
import torch


def projection_from_intrinsics(intrinsics, near, far):
    matrix = torch.zeros((*near.shape, 4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    matrix[..., 0, 0] = 2 * intrinsics[..., 0, 0]
    matrix[..., 1, 1] = 2 * intrinsics[..., 1, 1]
    matrix[..., 0, 2] = 2 * intrinsics[..., 0, 2] - 1
    matrix[..., 1, 2] = 2 * intrinsics[..., 1, 2] - 1
    matrix[..., 2, 2] = far / (far - near)
    matrix[..., 2, 3] = -(far * near) / (far - near)
    matrix[..., 3, 2] = 1
    return matrix
