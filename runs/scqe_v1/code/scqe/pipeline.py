"""Per-query orchestration with complete native baseline restoration."""
import math
from typing import Optional

import torch

from .config import M_MAX, P_PROBES, SEED_PROBE_BASE
from .coverage import compute_evidence
from .pf_runtime import PFRuntime, QueryState, probe_seed_int, gen_probes
from .risk import true_risk
from .solver import QueryCtx, b1_independent_rank, netgreedy


class QueryEngine:
    _no_risk_cache_checks = 0

    def __init__(self, runtime: PFRuntime, dataset: str = "DIOR",
                 parent_id: str = "", chip_id: Optional[str] = None,
                 probes: int = P_PROBES, search_cache: bool = True):
        self.rt, self.dataset = runtime, dataset
        self.parent_id, self.chip_id, self.probes = str(parent_id), chip_id, probes
        self.state: Optional[QueryState] = None
        self.ctx: Optional[QueryCtx] = None
        self.g = None
        self._no_risk_greedy = None
        self.search_cache = search_cache
        self.bypass_reason = None
        self.probe_seed = probe_seed_int(dataset, self.parent_id, chip_id, SEED_PROBE_BASE)

    def prepare(self, img_path: str):
        st = self.rt.native_pass(img_path)
        self.state = st
        if st.A0.numel() != len(st.R0) or not bool(((st.A0 >= 0) & (st.A0 < len(st.V_enc))).all()):
            self.bypass_reason = "invalid_baseline_indices"
            return self
        if len(torch.unique(st.A0)) != len(st.A0) or not bool(st.V_enc[st.A0].all()):
            self.bypass_reason = "invalid_baseline_membership"
            return self
        if not bool(torch.isfinite(st.R0).all()) or not bool(((st.R0 >= 0) & (st.R0 <= 1)).all()):
            self.bypass_reason = "invalid_baseline_reference"
            return self
        if st.T < 1:
            self.bypass_reason = "no_valid_tokens"
            return self
        self.ctx = QueryCtx(st)
        if not len(self.ctx.vs_idx):
            self.bypass_reason = "empty_candidate_pool"
        return self

    def prepare_baseline(self, img_path: str):
        """Capture the native B0 output without computing selection evidence."""
        self.state = self.rt.native_pass(img_path)
        return self

    def get_vjps(self):
        if self.g is None:
            if self.state is None or self.bypass_reason:
                raise RuntimeError("VJP requested on bypassed query")
            rows = gen_probes(self.probe_seed, self.state.y0.numel(), max(32, self.probes))
            self.g = self.rt.compute_vjps(self.state, self.probes, rows[:self.probes])
            if not bool(torch.isfinite(self.g).all()):
                raise FloatingPointError("nonfinite VJP")
        return self.g

    def build_alpha(self, p, t, c_valid, no_context=False):
        if self.bypass_reason:
            return None, [], self.bypass_reason
        ctx = self.ctx
        if no_context:
            if not bool(torch.isfinite(ctx.z_vs).all()) or not bool(torch.isfinite(p[c_valid]).all()):
                raise FloatingPointError("nonfinite no-context evidence")
            all_alpha = ((ctx.z_vs @ p.T) / 2).clamp(0, 1)
            all_alpha = torch.where(c_valid[None], all_alpha, torch.zeros_like(all_alpha))
            cols = [c for c in range(len(c_valid)) if bool(c_valid[c]) and float(all_alpha[:, c].sum()) > 0]
        else:
            all_alpha, cols = compute_evidence(
                ctx.z_vs, torch.ones(len(ctx.z_vs), dtype=torch.bool, device=ctx.z_vs.device),
                p, t, c_valid)
        if not cols:
            return None, [], "no_evidence"
        return all_alpha[:, cols], list(range(len(cols))), "ok"

    def R_of_plan(self, plan):
        return self.state.xi_all[plan].unsqueeze(0)

    def _baseline(self, method, status, **extra):
        st = self.state
        return dict(method=method, status=status, planned_m=0, accepted_m=0,
                    F0=None, Fplan=None, true_risk=None,
                    boxes=st.b0_boxes, scores=st.b0_scores,
                    A0=st.A0.clone(), Astar=st.A0.clone(), Aout=st.A0.clone(),
                    plan=st.A0.clone(), steps=[], probe_seed=self.probe_seed,
                    fallback_verified=True, **extra)

    def _decode_plan(self, plan):
        st = self.state
        y, pf, outputs = self.rt.native_decode_R(st, self.R_of_plan(plan))
        if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(pf).all()):
            raise FloatingPointError("nonfinite changed decoder output")
        risk = true_risk(y, st.y0, st.wsqrt)
        if not math.isfinite(risk):
            raise FloatingPointError("nonfinite true risk")
        T = st.T
        weighted = ((y - st.y0) * st.wsqrt).reshape(len(st.H_slots), T + 5)
        grouped = weighted.square().sum(0)
        groups = dict(box=float(grouped[:4].sum()),
                      token=float(grouped[4:4 + T].sum()),
                      confidence=float(grouped[-1]))
        boxes, scores = self.rt.proposals_from_outputs(st, outputs[0], pf)
        if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("nonfinite changed proposals")
        return risk, groups, boxes, scores

    def run_method(self, method: str, alpha=None, cols=None, budget: int = M_MAX,
                   eps_proxy: Optional[float] = None, eps_true: Optional[float] = None,
                   delta_F: float = 0.0):
        method = "SCQE" if method == "M" else method
        if method == "B0":
            return self._baseline(method, "ok")
        if self.bypass_reason or alpha is None:
            return self._baseline(method, "bypass_" + (self.bypass_reason or "no_evidence"))
        if method in ("SCQE", "A1", "A2", "A3", "A3-Cal", "A4", "A5"):
            if eps_proxy is None or (eps_true is None and method != "A4"):
                raise ValueError("locked risk thresholds required")
        if method == "B3" and eps_true is None:
            raise ValueError("B3 requires its own locked true-risk threshold")
        try:
            if method == "B1":
                out = b1_independent_rank(self.ctx, alpha, cols, budget)
            else:
                use_risk = method not in ("B2", "B3")
                g = self.get_vjps() if use_risk else None
                m_max = M_MAX if method == "B3" else budget
                cached = self._no_risk_greedy
                same_no_risk_search = (
                    self.search_cache and method == "B3" and cached is not None and
                    cached["alpha"] is alpha and cached["cols"] == tuple(cols) and
                    cached["m_max"] == m_max and cached["delta_F"] == delta_F)
                if same_no_risk_search:
                    out = cached["out"]
                    if QueryEngine._no_risk_cache_checks < 3:
                        fresh = netgreedy(self.ctx, alpha, cols, None,
                                          m_max=m_max, eps_proxy=None,
                                          delta_F=delta_F)
                        if not torch.equal(out["plan"], fresh["plan"]) or \
                                any(out[key] != fresh[key] for key in
                                    ("m", "F0", "Fplan", "steps", "status")):
                            raise RuntimeError("B2/B3 GPU search cache mismatch")
                        QueryEngine._no_risk_cache_checks += 1
                else:
                    out = netgreedy(self.ctx, alpha, cols, g,
                                    m_max=m_max,
                                    eps_proxy=eps_proxy if use_risk else None,
                                    delta_F=delta_F,
                                    add_only=method == "A2",
                                    isolated_risk=method in ("A3", "A3-Cal"),
                                    fixed_slot=method == "A5")
                    if method == "B2" and self.search_cache:
                        self._no_risk_greedy = dict(alpha=alpha, cols=tuple(cols),
                                                    m_max=m_max, delta_F=delta_F,
                                                    out=out)
            res = self._baseline(method, out["status"])
            res.update(planned_m=out["m"], F0=out["F0"], Fplan=out["Fplan"],
                       Astar=out["plan"].clone(), steps=out["steps"],
                       proxy_cum=out.get("proxy_cum"),
                       proxy_isolated=out.get("proxy_isolated"))
            if out["m"] == 0:
                res["status"] = "no_exchange"
                return res
            if method == "B3":
                return self._finish_b3(res, out, eps_true)
            risk, groups, boxes, scores = self._decode_plan(out["plan"])
            res.update(true_risk=risk, risk_groups=groups,
                       planned_boxes=boxes, planned_scores=scores)
            if method not in ("B1", "B2", "A4") and risk > eps_true:
                res["status"] = "true_reject"
                return res
            res.update(status="accepted" if method not in ("B1", "B2", "A4") else "deployed_no_true_gate",
                       accepted_m=out["m"], Aout=out["plan"].clone(),
                       plan=out["plan"].clone(), boxes=boxes, scores=scores)
            res["fallback_verified"] = None
            return res
        except (FloatingPointError, OverflowError) as exc:
            return self._baseline(method, "numeric_bypass", reason=str(exc))

    def _finish_b3(self, res, out, eps_true):
        st = self.state
        k = out["m"]
        checkpoints = sorted({min(x, k) for x in (1, 4, 8, 16, 32)})
        best = None
        prefixes = []
        for m in checkpoints:
            plan = st.A0.clone()
            for step in out["steps"][:m]:
                plan[step["slot"]] = step["incoming"]
            try:
                risk, groups, boxes, scores = self._decode_plan(plan)
                feasible = risk <= eps_true
            except FloatingPointError:
                risk, groups, boxes, scores, feasible = None, None, None, None, False
            F = out["F0"] + sum(step["gain"] for step in out["steps"][:m])
            prefixes.append(dict(m=m, F=F, true_risk=risk,
                                 risk_groups=groups, feasible=feasible))
            if feasible and (best is None or F > best[0] or (F == best[0] and m < best[1])):
                best = (F, m, plan, risk, groups, boxes, scores)
        res["prefix_info"] = prefixes
        if best is None:
            res["status"] = "true_reject"
            return res
        _, m, plan, risk, groups, boxes, scores = best
        res.update(status="accepted_prefix", accepted_m=m, Aout=plan.clone(),
                   plan=plan.clone(), true_risk=risk, risk_groups=groups,
                   boxes=boxes, scores=scores)
        res["fallback_verified"] = None
        return res
