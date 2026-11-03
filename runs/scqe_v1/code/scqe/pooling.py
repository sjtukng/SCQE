"""Region sampling phi per method spec (Eq. grid/pooling, validity rules)."""
from typing import List, Tuple

import torch

from .config import GRID, EPS_NORM


def nu(v: torch.Tensor, eps: float = EPS_NORM) -> torch.Tensor:
    """v / max(||v||_2, eps) along last dim."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def grid_centers(boxes_xyxy: torch.Tensor, grid: int = GRID) -> torch.Tensor:
    """Cell centers of an n_g x n_g grid in each box, [N, grid*grid, 2]."""
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    steps = torch.linspace(0.5 / grid, 1 - 0.5 / grid, grid, device=boxes_xyxy.device)
    gx, gy = torch.meshgrid(steps, steps, indexing="xy")
    px = x1[:, None, None] + gx[None] * (x2 - x1)[:, None, None]
    py = y1[:, None, None] + gy[None] * (y2 - y1)[:, None, None]
    return torch.stack([px, py], dim=-1).view(boxes_xyxy.shape[0], -1, 2)


def _sample_level(feat: torch.Tensor, pts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample pts ([N,P,2] feature coords) on feat [1,C,Hf,Wf].

    Boundary-clamped sampling; invalid points zeroed; mean over valid points.
    Returns [N, C].
    """
    C = feat.shape[1]
    Hf, Wf = feat.shape[-2:]
    grid = pts.clone()
    grid[..., 0] = 2.0 * grid[..., 0].clamp(0, Wf - 1) / (Wf - 1) - 1
    grid[..., 1] = 2.0 * grid[..., 1].clamp(0, Hf - 1) / (Hf - 1) - 1
    N, P = pts.shape[:2]
    sampled = torch.nn.functional.grid_sample(
        feat, grid.view(1, N * P, 1, 2), mode="bilinear",
        padding_mode="zeros", align_corners=True).view(C, N, P).permute(1, 2, 0)
    sampled = sampled * valid[..., None]
    count = valid.sum(dim=-1).clamp(min=1)
    return sampled.sum(dim=1) / count[:, None]


def phi(features: Tuple[torch.Tensor, ...],
        boxes_xyxy: torch.Tensor,
        img_W: float,
        img_H: float,
        valid_feature_hw: torch.Tensor,
        exclude_masks: List[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """phi(F, G) per Eq. (pooling); returns (vec [N,C] float32, valid [N] bool).

    boxes in non-padded resized image coords (already clipped copies are fine,
    clipping happens inside). Points outside the valid image are discarded.
    valid_feature_hw: [L,2] (Hf_valid, Wf_valid) per level.
    """
    N = boxes_xyxy.shape[0]
    C = features[0].shape[1]
    if N == 0:
        return boxes_xyxy.new_zeros((0, C)), torch.zeros(0, dtype=torch.bool, device=boxes_xyxy.device)
    if N > 256:
        chunks = []
        for start in range(0, N, 256):
            masks = None if exclude_masks is None else [
                mask[start:start + 256] for mask in exclude_masks]
            chunks.append(phi(features, boxes_xyxy[start:start + 256], img_W, img_H,
                              valid_feature_hw, masks))
        return torch.cat([x[0] for x in chunks]), torch.cat([x[1] for x in chunks])

    boxes = boxes_xyxy.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0, max=img_W)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0, max=img_H)
    centers = grid_centers(boxes)
    x1, y1, x2, y2 = boxes.unbind(-1)
    valid = (centers[..., 0] >= x1[:, None]) & (centers[..., 0] <= x2[:, None]) & \
            (centers[..., 1] >= y1[:, None]) & (centers[..., 1] <= y2[:, None]) & \
            (centers[..., 0] >= 0) & (centers[..., 0] <= img_W) & \
            (centers[..., 1] >= 0) & (centers[..., 1] <= img_H)
    if exclude_masks is not None:
        for em in exclude_masks:
            valid = valid & (~em)

    level_vecs = []
    for lvl, feat in enumerate(features):
        Hf, Wf = valid_feature_hw[lvl].tolist()
        pts = centers.clone()
        pts[..., 0] = centers[..., 0] * (Wf / img_W) - 0.5
        pts[..., 1] = centers[..., 1] * (Hf / img_H) - 0.5
        mu = _sample_level(feat, pts, valid)          # [N, C]
        level_vecs.append(nu(mu))
    stacked = torch.stack(level_vecs, dim=1)          # [N, L, C]
    mean = stacked.mean(dim=1)                        # [N, C]
    pre_norm = mean.norm(dim=-1)
    valid_rep = valid.any(dim=-1) & torch.isfinite(mean).all(dim=-1) & (pre_norm > EPS_NORM)
    out = nu(mean)
    out = torch.where(torch.isfinite(out).all(dim=-1, keepdim=True), out, torch.zeros_like(out))
    return out, valid_rep
