"""Exact class-balanced coverage with bounded-memory top-two cache.

The atom set is the full V_s x C+ Cartesian product. Only temporary tensors
are blocked; no demand atoms are sampled or discarded.
"""
from typing import List, Tuple

import torch

from .geometry import iou_matrix


class Coverage:
    def __init__(self, vs_idx: torch.Tensor, vs_boxes: torch.Tensor,
                 alpha: torch.Tensor, cls_plus: List[int], plan: torch.Tensor,
                 atom_block: int = 256):
        if vs_idx.numel() == 0 or alpha.numel() == 0:
            raise ValueError("coverage requires nonempty evidence")
        if not bool(torch.isfinite(alpha).all()) or not bool(torch.isfinite(vs_boxes).all()):
            raise ValueError("nonfinite coverage input")
        self.vs_idx, self.vs_boxes, self.alpha = vs_idx, vs_boxes, alpha
        self.Nv, self.C = alpha.shape
        self.plan = plan.clone()
        self.atom_block = atom_block
        n_index = max(int(vs_idx.max()), int(plan.max())) + 1
        self.pos_in_vs = torch.full((n_index,), -1, dtype=torch.long,
                                    device=vs_idx.device)
        self.pos_in_vs[vs_idx] = torch.arange(self.Nv, device=vs_idx.device)
        Z = alpha.to(torch.float64).sum(0)
        if self.C != len(cls_plus) or not bool((Z > 0).all()):
            raise ValueError("all active classes must have positive evidence")
        self.w = alpha.to(torch.float64) / (self.C * Z)[None]
        self.cache = {
            "m1": torch.zeros_like(alpha),
            "m2": torch.zeros_like(alpha),
            "o1": torch.zeros_like(alpha, dtype=torch.long),
        }
        self._rebuild_classes(torch.arange(self.C, device=alpha.device))

    def _rebuild_classes(self, cidx: torch.Tensor):
        """Recompute affected columns over every slot, including the third owner."""
        if cidx.numel() == 0:
            return
        pos = self.pos_in_vs[self.plan]
        valid = pos >= 0
        q = self.plan.numel()
        for lo in range(0, self.Nv, self.atom_block):
            hi = min(lo + self.atom_block, self.Nv)
            k = self.alpha.new_zeros((q, hi - lo, cidx.numel()))
            if bool(valid.any()):
                iou = iou_matrix(self.vs_boxes[pos[valid]], self.vs_boxes[lo:hi])
                k[valid] = iou[:, :, None] * self.alpha[pos[valid]][:, None, cidx]
            best, owner = k.max(dim=0)  # torch.max resolves ties by first slot
            rows = torch.arange(q, device=k.device)[:, None, None]
            second, _ = k.masked_fill(rows == owner[None], -1).max(dim=0)
            self.cache["m1"][lo:hi, cidx] = best
            self.cache["m2"][lo:hi, cidx] = second
            self.cache["o1"][lo:hi, cidx] = owner
        if not bool(torch.isfinite(self.cache["m1"]).all()):
            raise FloatingPointError("nonfinite coverage cache")

    def F(self) -> float:
        return float((self.w * self.cache["m1"].to(torch.float64)).sum())

    def removal_losses(self) -> torch.Tensor:
        m1, m2, o1 = (self.cache[k] for k in ("m1", "m2", "o1"))
        contrib = self.w * (m1 - m2).to(torch.float64)
        out = torch.zeros(self.plan.numel(), dtype=torch.float64, device=m1.device)
        out.index_add_(0, o1.reshape(-1), contrib.reshape(-1))
        return out

    def add_only_gain(self, in_pos: torch.Tensor) -> torch.Tensor:
        out = torch.empty(in_pos.numel(), dtype=torch.float64, device=self.alpha.device)
        m1 = self.cache["m1"]
        for lo in range(0, in_pos.numel(), 16):
            rows = in_pos[lo:lo + 16]
            iou = iou_matrix(self.vs_boxes[rows], self.vs_boxes)
            k = iou[:, :, None] * self.alpha[rows][:, None, :]
            out[lo:lo + len(rows)] = (
                self.w[None] * (torch.maximum(m1[None], k) - m1[None]).to(torch.float64)
            ).sum((1, 2))
        return out

    def pair_net_gain(self, r: int, in_pos: int) -> float:
        m1, m2, o1 = (self.cache[k] for k in ("m1", "m2", "o1"))
        base = torch.where(o1 == r, m2, m1)
        iou = iou_matrix(self.vs_boxes[in_pos:in_pos + 1], self.vs_boxes)[0]
        k = iou[:, None] * self.alpha[in_pos][None]
        return float((self.w * (torch.maximum(base, k) - m1).to(torch.float64)).sum())

    def commit(self, r: int, enc_idx: int, affected_classes: torch.Tensor):
        old = int(self.plan[r])
        self.plan[r] = enc_idx
        self._rebuild_classes(affected_classes)
        return old


def compute_evidence(z: torch.Tensor, z_valid: torch.Tensor, p: torch.Tensor,
                     t: torch.Tensor, c_valid: torch.Tensor
                     ) -> Tuple[torch.Tensor, List[int]]:
    if not bool(torch.isfinite(z[z_valid]).all()) or not bool(torch.isfinite(p[c_valid]).all()) \
            or not bool(torch.isfinite(t[c_valid]).all()):
        raise FloatingPointError("nonfinite support evidence")
    alpha = ((z @ p.T - z @ t.T) / 2).clamp(0, 1)
    alpha = torch.where(z_valid[:, None] & c_valid[None], alpha,
                        torch.zeros_like(alpha))
    Z = alpha.to(torch.float64).sum(0)
    cls_plus = [c for c in range(p.shape[0]) if bool(c_valid[c]) and float(Z[c]) > 0]
    return alpha, cls_plus
