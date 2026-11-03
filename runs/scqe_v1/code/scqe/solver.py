"""Deterministic bounded ordered reference exchange and internal selectors."""
from typing import Dict, List, Optional

import torch

from .config import L_IN, L_OUT, M_MAX
from .coverage import Coverage
from .geometry import iou_matrix
from .risk import pair_risk, step_zeta, proxy_from_zeta


def _lexsort(keys):
    """Stable lexicographic sort; the last key is primary."""
    idx = torch.arange(keys[0].numel(), device=keys[0].device)
    for key in keys:
        idx = idx[torch.argsort(key[idx], stable=True)]
    return idx


class QueryCtx:
    def __init__(self, state):
        vs = state.V_enc & state.geom_valid & state.z_valid & \
            torch.isfinite(state.xi_all).all(dim=-1)
        self.vs_idx = torch.nonzero(vs).flatten()
        xy = torch.stack([
            (state.xi_all[:, 0] - state.xi_all[:, 2] / 2) * state.img_W,
            (state.xi_all[:, 1] - state.xi_all[:, 3] / 2) * state.img_H,
            (state.xi_all[:, 0] + state.xi_all[:, 2] / 2) * state.img_W,
            (state.xi_all[:, 1] + state.xi_all[:, 3] / 2) * state.img_H], -1)
        xy[..., 0::2] = xy[..., 0::2].clamp(0, state.img_W)
        xy[..., 1::2] = xy[..., 1::2].clamp(0, state.img_H)
        self.vs_boxes = xy[self.vs_idx]
        self.xi_vs = state.xi_all[self.vs_idx]
        self.z_vs = state.z_all[self.vs_idx]
        self.R0, self.A0, self.H = state.R0, state.A0, state.H_slots
        self.g = None


def _round_robin_incoming(alpha: torch.Tensor, classes: List[int],
                          used: torch.Tensor, limit: int) -> torch.Tensor:
    lists = []
    for c in classes:
        a = alpha[:, c]
        idx = torch.nonzero((a > 0) & ~used).flatten()
        order = _lexsort([idx.to(torch.float64), -a[idx].to(torch.float64)])
        lists.append(idx[order])
    cursor = [0] * len(lists)
    out, seen = [], set()
    while len(out) < limit:
        consumed = False
        for c, items in enumerate(lists):
            if cursor[c] >= len(items):
                continue
            j = int(items[cursor[c]])
            cursor[c] += 1
            consumed = True
            if j not in seen:
                seen.add(j)
                out.append(j)
            if len(out) == limit:
                break
        if not consumed:
            break
    return torch.tensor(out, device=alpha.device, dtype=torch.long)


def _make_coverage(ctx, alpha, classes):
    return Coverage(ctx.vs_idx, ctx.vs_boxes, alpha, classes, ctx.A0)


def netgreedy(ctx: QueryCtx, alpha: torch.Tensor, cls_plus_cols: List[int],
              g: Optional[torch.Tensor], m_max: int = M_MAX,
              eps_proxy: Optional[float] = None, delta_F: float = 0.0,
              l_in: int = L_IN, l_out: int = L_OUT,
              add_only: bool = False, isolated_risk: bool = False,
              fixed_slot: bool = False) -> Dict:
    """Exact net/add-only gain; cumulative or isolated proxy; one-use slots.

    Numeric faults are raised to the caller, which must restore the full B0.
    """
    if not bool(torch.isfinite(alpha).all()) or g is not None and not bool(torch.isfinite(g).all()):
        raise FloatingPointError("nonfinite evidence or gradients")
    cov = _make_coverage(ctx, alpha, cls_plus_cols)
    F0 = cov.F()
    plan = ctx.A0.clone()
    used_slots = set()
    protected = set(ctx.H.tolist())
    omega = torch.zeros(g.shape[0], dtype=torch.float64, device=plan.device) if g is not None else None
    isolated = 0.0
    steps = []
    status = "budget_reached"
    while len(steps) < m_max:
        eligible = [r for r in range(len(plan)) if r not in protected and r not in used_slots]
        if not eligible:
            status = "no_slots"
            break
        losses = cov.removal_losses()
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError("nonfinite removal loss")
        E = torch.tensor(eligible, dtype=torch.long, device=plan.device)
        out = E[_lexsort([E.to(torch.float64), losses[E]])[:l_out]]
        if fixed_slot:
            out = out[:1]
        used = torch.zeros(len(ctx.vs_idx), dtype=torch.bool, device=plan.device)
        positions = cov.pos_in_vs[plan]
        used[positions[positions >= 0]] = True
        original_positions = cov.pos_in_vs[ctx.A0]
        used[original_positions[original_positions >= 0]] = True
        incoming = _round_robin_incoming(alpha, cls_plus_cols, used, l_in)
        if not len(incoming):
            status = "no_candidates"
            break
        m1, m2, owner = (cov.cache[k] for k in ("m1", "m2", "o1"))
        gain = torch.empty((len(out), len(incoming)), dtype=torch.float64, device=plan.device)
        add_gain = torch.empty(len(incoming), dtype=torch.float64, device=plan.device)
        for start in range(0, len(incoming), 8):
            part = incoming[start:start + 8]
            iou = iou_matrix(ctx.vs_boxes[part], ctx.vs_boxes)
            k = iou[:, :, None] * alpha[part][:, None, :]
            add = (cov.w[None] * (torch.maximum(m1[None], k) - m1[None]).to(torch.float64)).sum((1, 2))
            add_gain[start:start + len(part)] = add
            for oi, slot in enumerate(out.tolist()):
                if add_only:
                    gain[oi, start:start + len(part)] = add
                else:
                    base = torch.where(owner == slot, m2, m1)
                    gain[oi, start:start + len(part)] = (
                        cov.w[None] * (torch.maximum(base[None], k) - m1[None]).to(torch.float64)
                    ).sum((1, 2))
        if not bool(torch.isfinite(gain).all()):
            raise FloatingPointError("nonfinite candidate gain")
        risk = None
        d2 = None
        if g is not None:
            risk = pair_risk(g, omega, out, ctx.xi_vs[incoming], ctx.R0)
            if isolated_risk:
                zeros = torch.zeros_like(omega)
                d2 = pair_risk(g, zeros, out, ctx.xi_vs[incoming], ctx.R0)
                risk = d2 + isolated
            if not bool(torch.isfinite(risk).all()):
                raise FloatingPointError("nonfinite proxy risk")
        feasible = gain > delta_F
        if eps_proxy is not None:
            if risk is None:
                raise ValueError("proxy threshold requires VJP gradients")
            feasible &= risk <= eps_proxy
        if not bool(feasible.any()):
            status = "no_feasible_pair"
            break
        pairs = torch.nonzero(feasible).tolist()
        pairs.sort(key=lambda ij: (-float(gain[ij[0], ij[1]]),
                                   float(risk[ij[0], ij[1]]) if risk is not None else 0.0,
                                   int(out[ij[0]]), int(ctx.vs_idx[incoming[ij[1]]])))
        oi, ji = pairs[0]
        slot, pos = int(out[oi]), int(incoming[ji])
        j, old = int(ctx.vs_idx[pos]), int(plan[slot])
        actual_gain = cov.pair_net_gain(slot, pos)
        score_gain = float(gain[oi, ji])
        old_pos = int(cov.pos_in_vs[old])
        old_alpha = cov.alpha[old_pos] if old_pos >= 0 else torch.zeros_like(alpha[pos])
        affected = torch.nonzero((old_alpha > 0) | (alpha[pos] > 0)).flatten()
        cov.commit(slot, j, affected)
        plan[slot] = j
        used_slots.add(slot)
        if g is not None:
            next_omega = step_zeta(g, omega, slot, ctx.xi_vs[pos], ctx.R0)
            isolated += float(((next_omega - omega) ** 2).mean())
            omega = next_omega
        steps.append(dict(slot=slot, out=old, incoming=j, score_gain=score_gain,
                          gain=actual_gain, add_gain=float(add_gain[ji]),
                          removal_loss=float(losses[slot]), proxy=float(risk[oi, ji]) if risk is not None else None,
                          proxy_cum=proxy_from_zeta(omega) if omega is not None else None,
                          proxy_isolated=isolated))
        if not torch.isfinite(torch.tensor(cov.F(), device=plan.device)):
            raise FloatingPointError("nonfinite coverage cache")
    return dict(plan=plan, m=len(steps), F0=F0, Fplan=cov.F(),
                steps=steps, status=status, zeta=omega,
                proxy_cum=proxy_from_zeta(omega) if omega is not None else None,
                proxy_isolated=isolated)


def b1_independent_rank(ctx: QueryCtx, alpha: torch.Tensor,
                        cls_plus_cols: List[int], budget: int,
                        l_in: int = L_IN, l_out: int = L_OUT) -> Dict:
    cov = _make_coverage(ctx, alpha, cls_plus_cols)
    F0 = cov.F()
    losses = cov.removal_losses()
    protected = set(ctx.H.tolist())
    E = torch.tensor([r for r in range(len(ctx.A0)) if r not in protected],
                     dtype=torch.long, device=ctx.A0.device)
    out = E[_lexsort([E.to(torch.float64), losses[E]])[:l_out]]
    used = torch.zeros(len(ctx.vs_idx), dtype=torch.bool, device=ctx.A0.device)
    positions = cov.pos_in_vs[ctx.A0]
    used[positions[positions >= 0]] = True
    initial = _round_robin_incoming(alpha, cls_plus_cols, used, l_in)
    scores = alpha[initial].max(-1).values
    incoming = initial[_lexsort([ctx.vs_idx[initial].to(torch.float64), -scores.to(torch.float64)])]
    n = min(budget, len(out), len(incoming))
    plan = ctx.A0.clone()
    steps = []
    for i in range(n):
        slot, pos = int(out[i]), int(incoming[i])
        old, j = int(plan[slot]), int(ctx.vs_idx[pos])
        actual = cov.pair_net_gain(slot, pos)
        old_pos = int(cov.pos_in_vs[old])
        old_alpha = alpha[old_pos] if old_pos >= 0 else torch.zeros_like(alpha[pos])
        affected = torch.nonzero((old_alpha > 0) | (alpha[pos] > 0)).flatten()
        cov.commit(slot, j, affected)
        plan[slot] = j
        steps.append(dict(slot=slot, out=old, incoming=j, gain=actual))
    return dict(plan=plan, m=n, F0=F0, Fplan=cov.F(), steps=steps,
                status="ok" if n else "no_candidates", zeta=None)
