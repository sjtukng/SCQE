"""Box geometry helpers (normalized cxcywh <-> image xyxy, crop, expand, IoU)."""
import torch


def norm01_to_image(boxes_norm: torch.Tensor, W: float, H: float) -> torch.Tensor:
    """normalized cxcywh in [0,1] -> image xyxy."""
    cx, cy, w, h = boxes_norm.unbind(-1)
    x1 = (cx - 0.5 * w) * W
    y1 = (cy - 0.5 * h) * H
    x2 = (cx + 0.5 * w) * W
    y2 = (cy + 0.5 * h) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


def image_to_norm01(boxes_xyxy: torch.Tensor, W: float, H: float) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    return torch.stack([cx, cy, w, h], dim=-1)


def crop_to_valid(boxes: torch.Tensor, W: float, H: float) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0::2] = out[..., 0::2].clamp(min=0, max=W)
    out[..., 1::2] = out[..., 1::2].clamp(min=0, max=H)
    return out


def expand_box(boxes: torch.Tensor, beta: float, W: float, H: float) -> torch.Tensor:
    """Expand about center by beta, then clip to valid image."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * beta
    h = (y2 - y1) * beta
    return crop_to_valid(torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1), W, H)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N,4] x [M,4] xyxy -> [N,M] IoU."""
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    inter_x1 = ax1[:, None].clamp(min=bx1[None, :])
    inter_y1 = ay1[:, None].clamp(min=by1[None, :])
    inter_x2 = ax2[:, None].clamp(max=bx2[None, :])
    inter_y2 = ay2[:, None].clamp(max=by2[None, :])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = ((ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0))[:, None]
    area_b = ((bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0))[None, :]
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-12), torch.zeros_like(union))
