"""PF-RPN runtime: native state capture, functional decoder, VJP, native output.

All structural facts verified against third_party/PF-RPN @ 52d3959:
- pre_decoder (pf_rpn.py:423) builds A0 (450 conf + 450 cls, _select_score_guided_queries)
  and feeds detached sigmoid references to the decoder.
- DINO.forward_decoder (dino.py:216) calls the decoder with the captured kwargs.
- DinoTransformerDecoder.forward (dino_layers.py:26) detaches iterative references
  at line 105; the functional copy below removes only that detach.
- Native eval scoring (grounding_dino_head.py:486-520): cls sigmoid -> max over T
  tokens; conf sigmoided twice (head forward + predict); output_score='both' sum;
  topk(max_per_img=900); boxes cxcywh->xyxy * img_shape, clamp, / scale_factor.
"""
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import ETA, Q
from .geometry import norm01_to_image
from .pooling import phi

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.models.layers.transformer.utils import (coordinate_to_encoding,
                                                   inverse_sigmoid)


# ---------------------------------------------------------------------------
# functional decoder: exact copy of DinoTransformerDecoder.forward with the
# single change `reference_points = new_reference_points` (no .detach()).
# ---------------------------------------------------------------------------
def functional_decoder_forward(decoder, query, value, key_padding_mask,
                               self_attn_mask, reference_points, spatial_shapes,
                               level_start_index, valid_ratios, reg_branches,
                               **kwargs):
    intermediate = []
    intermediate_reference_points = [reference_points]
    for lid, layer in enumerate(decoder.layers):
        if reference_points.shape[-1] == 4:
            reference_points_input = \
                reference_points[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = \
                reference_points[:, :, None] * valid_ratios[:, None]

        query_sine_embed = coordinate_to_encoding(
            reference_points_input[:, :, 0, :])
        query_pos = decoder.ref_point_head(query_sine_embed)

        query = layer(
            query,
            query_pos=query_pos,
            value=value,
            key_padding_mask=key_padding_mask,
            self_attn_mask=self_attn_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points_input,
            **kwargs)

        if reg_branches is not None:
            tmp = reg_branches[lid](query)
            assert reference_points.shape[-1] == 4
            new_reference_points = tmp + inverse_sigmoid(
                reference_points, eps=1e-3)
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points  # functional: no detach

        if decoder.return_intermediate:
            intermediate.append(decoder.norm(query))
            intermediate_reference_points.append(new_reference_points)

    if decoder.return_intermediate:
        return torch.stack(intermediate), torch.stack(
            intermediate_reference_points)
    return query, reference_points


def split_memory_to_features(memory: torch.Tensor,
                             spatial_shapes: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """[B, sum(H*W), C] -> per-level [1, C, H, W]; spatial_shapes rows are (H, W)."""
    B, _, C = memory.shape
    feats = []
    start = 0
    for Hf, Wf in spatial_shapes.tolist():
        n = int(Hf * Wf)
        feat = memory[:, start:start + n, :].view(B, Hf, Wf, C).permute(0, 3, 1, 2)
        feats.append(feat.contiguous())
        start += n
    return tuple(feats)


class QueryState:
    """Support-independent per-query cached state (normal tensors)."""

    def __init__(self):
        self.A0: Optional[torch.Tensor] = None          # [Q] long
        self.R0: Optional[torch.Tensor] = None          # [Q,4]
        self.xi_all: Optional[torch.Tensor] = None      # [N,4] sigmoid(coord_unact)
        self.coord_unact: Optional[torch.Tensor] = None  # [N,4] raw logits
        self.V_enc: Optional[torch.Tensor] = None       # [N] bool
        self.cls_score_enc: Optional[torch.Tensor] = None
        self.conf_enc: Optional[torch.Tensor] = None
        # decoder call context (cloned normal tensors) + outputs
        self.decoder_call: Optional[Tuple[list, Dict]] = None
        self.text_token_mask: Optional[torch.Tensor] = None
        self.hidden0: Optional[torch.Tensor] = None     # [L,1,Q,256]
        self.refs0: Optional[torch.Tensor] = None       # [L+1,1,Q,4]
        # image geometry
        self.img_W: float = 0.0   # non-padded resized
        self.img_H: float = 0.0
        self.scale_factor: Tuple[float, float] = (1.0, 1.0)
        self.ori_W: int = 0
        self.ori_H: int = 0
        # features for phi
        self.features: Optional[Tuple[torch.Tensor, ...]] = None
        self.valid_feature_hw: Optional[torch.Tensor] = None  # [L,2] (Hf,Wf) valid
        # candidate support-independent validity + representations
        self.z_all: Optional[torch.Tensor] = None       # [N,C]
        self.z_valid: Optional[torch.Tensor] = None     # [N] bool
        self.geom_valid: Optional[torch.Tensor] = None  # [N] bool (finite, pos area)
        # native outputs
        self.b0_boxes: Optional[torch.Tensor] = None    # [N0,4] xyxy original coords
        self.b0_scores: Optional[torch.Tensor] = None   # [N0]
        # protected set + y0 (built lazily)
        self.H_slots: Optional[torch.Tensor] = None     # [h] long
        self.y0: Optional[torch.Tensor] = None          # [d]
        self.wsqrt: Optional[torch.Tensor] = None       # [d]
        self.T: int = 0
        self.s0: Optional[torch.Tensor] = None          # [Q] native PF scores

    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
            elif isinstance(v, dict):
                setattr(self, k, {kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                                  for kk, vv in v.items()})
            elif isinstance(v, tuple) and v and isinstance(v[0], torch.Tensor):
                setattr(self, k, tuple(t.to(device) for t in v))
        return self


class PFRuntime:
    """Drives a PFRPN model, capturing state and running functional/native decodes."""

    def __init__(self, model, device="cuda:0"):
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("PF-RPN requires CUDA; CPU execution is disabled")
        self.model = model
        self.device = device
        self._capture: Dict = {}
        self._inject_R: Optional[torch.Tensor] = None
        self._hooked = False
        self._ref64 = None

    def ref64(self):
        """Cache the CUDA FP64 decoder used for SCQE VJPs."""
        if self._ref64 is None:
            from .ref64 import Ref64
            self._ref64 = Ref64(self)
        return self._ref64

    # -- hooks ---------------------------------------------------------------
    def _enable_hooks(self):
        if self._hooked:
            return
        m = self.model
        self._orig_forward_encoder = m.forward_encoder
        self._orig_pre_decoder = m.pre_decoder
        self._orig_decoder_forward = m.decoder.forward

        def enc_hook(*args, **kwargs):
            out = self._orig_forward_encoder(*args, **kwargs)
            self._capture["encoder_out"] = {
                k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in out.items()}
            return out

        def pre_hook(memory, memory_mask, spatial_shapes, memory_text,
                     text_token_mask, batch_data_samples=None):
            # capture encoder-stage proposal quantities (identical recompute)
            with torch.no_grad():
                output_memory, output_proposals = m.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
                coord_unact = m.bbox_head.reg_branches[m.decoder.num_layers](
                    output_memory) + output_proposals
                conf = m.bbox_head.conf_branches[m.decoder.num_layers](
                    output_memory).squeeze(-1).sigmoid()
                cls_logits = m.bbox_head.cls_branches[m.decoder.num_layers](
                    output_memory, memory_text, text_token_mask)
                cls_score = cls_logits.max(-1)[0].sigmoid()
                # native proposal-validity flags BEFORE box-regression correction:
                # proposals-valid = all four proposal coords in (0.01, 0.99)
                bs = memory.shape[0]
                if memory_mask is None:
                    memory_mask = torch.zeros(bs, memory.shape[1], dtype=torch.bool,
                                              device=memory.device)
                valids = []
                _cur = 0
                for Hf, Wf in spatial_shapes.tolist():
                    n = int(Hf * Wf)
                    mask_lvl = memory_mask[:, _cur:_cur + n]
                    valid_H = torch.sum(~mask_lvl.view(bs, Hf, Wf)[:, :, 0], 1)
                    valid_W = torch.sum(~mask_lvl.view(bs, Hf, Wf)[:, 0, :], 1)
                    scale = torch.stack([valid_W, valid_H], 1).view(bs, 1, 1, 2).float()
                    gy, gx = torch.meshgrid(
                        torch.linspace(0, Hf - 1, Hf, device=memory.device),
                        torch.linspace(0, Wf - 1, Wf, device=memory.device))
                    grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)
                    grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
                    wh = torch.ones_like(grid) * 0.05 * (2.0 ** len(valids))
                    proposal = torch.cat((grid, wh), -1).view(bs, n, 4)
                    valid = ((proposal > 0.01) & (proposal < 0.99)).sum(-1) == 4
                    valids.append(valid)
                    _cur += n
                e_prop = torch.cat(valids, dim=1)  # [bs, N] proposal-validity
                a0 = m._select_score_guided_queries(conf, cls_score)
                self._capture["enc_stage"] = dict(
                    coord_unact=coord_unact.detach().clone(),
                    conf=conf.detach().clone(),
                    cls_score=cls_score.detach().clone(),
                    e_prop=e_prop.detach().clone(),
                    m_pad=memory_mask.detach().clone(),
                    A0=a0.detach().clone())
            return self._orig_pre_decoder(memory, memory_mask, spatial_shapes,
                                          memory_text, text_token_mask,
                                          batch_data_samples)

        def dec_hook(*args, **kwargs):
            self._capture["decoder_call"] = (
                [v.detach().clone() if isinstance(v, torch.Tensor) else v
                 for v in args],
                {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                 for k, v in kwargs.items()})
            if self._inject_R is not None:
                kwargs = dict(kwargs)
                kwargs["reference_points"] = self._inject_R
            out = self._orig_decoder_forward(*args, **kwargs)
            self._capture["decoder_out"] = tuple(
                o.detach().clone() if isinstance(o, torch.Tensor) else o
                for o in out)
            return out

        m.forward_encoder = enc_hook
        m.pre_decoder = pre_hook
        m.decoder.forward = dec_hook
        self._hooked = True

    def _disable_hooks(self):
        if not self._hooked:
            return
        m = self.model
        m.forward_encoder = self._orig_forward_encoder
        m.pre_decoder = self._orig_pre_decoder
        m.decoder.forward = self._orig_decoder_forward
        self._hooked = False

    # -- decoder outputs -> y / native scores ---------------------------------
    def decode_outputs(self, state: QueryState, hidden, refs):
        """Last-layer head outputs: b_norm [1,Q,4], cls_logits [1,Q,T], conf1 [1,Q].

        b_norm follows the head formula sigmoid(reg(norm(q)) + inv_sig(ref)),
        i.e. the box consumed by the native scoring interface (NOT the decoder's
        internal raw-query reference update).
        """
        head = self.model.bbox_head
        lid = self.model.decoder.num_layers - 1
        hq = hidden[lid]  # already layer-normed in decoder
        _, dkw = state.decoder_call
        cls_logits = head.cls_branches[lid](hq, dkw["memory_text"],
                                            state.text_token_mask)
        conf1 = head.conf_branches[lid](hq).squeeze(-1).sigmoid()
        b_norm = (head.reg_branches[lid](hq) +
                  inverse_sigmoid(refs[lid])).sigmoid()
        return b_norm, cls_logits, conf1

    def build_y(self, state: QueryState, b_norm, cls_logits, conf1):
        """Protected output y_H and native PF scores s_r^PF. All slots [1,Q,*]."""
        valid_tokens = state.text_token_mask[0].bool()
        pi = cls_logits[:, :, valid_tokens].sigmoid()  # [1,Q,T valid]
        gamma = conf1.sigmoid()                 # [1,Q]
        if pi.shape[-1] < 1:
            # Keep the native proposal path available for the mandated B0
            # bypass, even when no protected token output can be constructed.
            state.T = 0
            state.H_slots = torch.empty(0, dtype=torch.long, device=gamma.device)
            state.wsqrt = torch.empty(0, dtype=torch.float64, device=gamma.device)
            return state.wsqrt, cls_logits.sigmoid().max(-1).values + gamma
        s_pf = pi.max(-1).values + gamma        # [1,Q]
        if state.H_slots is None:
            h = int(np.ceil(ETA * Q))
            state.T = pi.shape[-1]
            state.H_slots = torch.sort(
                torch.argsort(-s_pf[0], stable=True)[:h]).values  # ascending
            # tie-break: torch.argsort stable keeps ascending slot order on ties
            T = state.T
            h = state.H_slots.numel()
            blk = ([1.0 / np.sqrt(12.0 * h)] * 4 +
                   [1.0 / np.sqrt(3.0 * T * h)] * T +
                   [1.0 / np.sqrt(3.0 * h)])
            state.wsqrt = torch.tensor(blk * h, dtype=torch.float64,
                                       device=s_pf.device)
        idx = state.H_slots
        y = torch.cat([b_norm[0, idx], pi[0, idx], gamma[0, idx, None]],
                      dim=-1).reshape(-1).to(torch.float64)
        return y, s_pf

    def _replay_decoder(self, state: QueryState, R: torch.Tensor,
                        functional: bool):
        """Replay the captured native decoder call with reference_points=R."""
        args, kwargs = state.decoder_call
        args = list(args)
        if "reference_points" in kwargs:
            kwargs = dict(kwargs)
            kwargs["reference_points"] = R
        else:
            # reference_points is positional arg 4 per DinoTransformerDecoder.forward
            args[4] = R
        fn = functional_decoder_forward if functional else self._orig_decoder_forward
        if functional:
            return fn(self.model.decoder, *args, **kwargs)
        return fn(*args, **kwargs)

    def y_of_R(self, state: QueryState, R: torch.Tensor):
        """Functional decoder at reference R [1,Q,4] -> (y, s_pf, extras)."""
        hidden, refs = self._replay_decoder(state, R, functional=True)
        b_norm, cls_logits, conf1 = self.decode_outputs(state, hidden, refs)
        y, s_pf = self.build_y(state, b_norm, cls_logits, conf1)
        return y, s_pf, (b_norm, cls_logits, conf1, refs)

    @torch.no_grad()
    def native_decode_R(self, state: QueryState, R: torch.Tensor):
        """Original frozen decoder at R (values identical to native forward)."""
        hidden, refs = self._replay_decoder(state, R, functional=False)
        b_norm, cls_logits, conf1 = self.decode_outputs(state, hidden, refs)
        y, s_pf = self.build_y(state, b_norm, cls_logits, conf1)
        return y, s_pf, (b_norm, cls_logits, conf1, refs)

    # -- VJP ------------------------------------------------------------------
    def compute_vjps(self, state: QueryState, P: int, probe_rows: np.ndarray):
        """g_b = grad_R [v_b^T W^{1/2} y(R)] at R0. probe_rows: [P, d] +-1 int8."""
        reference = self.ref64()
        gs = []
        for b in range(P):
            v = torch.from_numpy(probe_rows[b].astype(np.float64)).to(state.R0.device)
            gs.append(reference.vjp(state, state.R0, v, state.wsqrt))
        return torch.stack(gs)  # [P,Q,4] float64

    # -- B_out postprocessing --------------------------------------------------
    @torch.no_grad()
    def proposals_from_outputs(self, state: QueryState, b_norm, s_pf):
        """Native proposal postprocessing: score order, box conversion, filter.

        Returns boxes [N,4] xyxy original coords, scores [N] (descending).
        """
        order = torch.argsort(-s_pf[0], stable=True)
        boxes_norm = b_norm[0, order]              # [N,4] cxcywh normalized
        scores = s_pf[0, order]
        boxes = bbox_cxcywh_to_xyxy(boxes_norm)
        boxes[:, 0::2] *= state.img_W
        boxes[:, 1::2] *= state.img_H
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, state.img_W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, state.img_H)
        sx, sy = state.scale_factor
        boxes[:, 0::2] /= sx
        boxes[:, 1::2] /= sy
        finite = torch.isfinite(boxes).all(dim=-1)
        pos = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        keep = finite & pos
        return boxes[keep], scores[keep]

    # -- native pass ------------------------------------------------------------
    @torch.no_grad()
    def native_pass(self, img_path: str) -> QueryState:
        """Run the full native PF-RPN predict path, capturing all state."""
        from mmengine.dataset import pseudo_collate
        from mmcv.transforms import Compose
        cfg = self.model.cfg
        pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        data = pipeline(dict(img_path=str(img_path), img_id=0))
        data = pseudo_collate([data])
        data = self.model.data_preprocessor(data, training=False)
        batch_inputs = data["inputs"]
        data_samples = data["data_samples"]

        self._enable_hooks()
        self._capture = {}
        self._inject_R = None
        try:
            results = self.model.predict(batch_inputs, data_samples, rescale=True)
        finally:
            self._disable_hooks()

        sample = results[0]
        pred = sample.pred_instances
        st = QueryState()
        dev = self.device
        # image geometry from metainfo
        ori = sample.metainfo["ori_shape"]
        img_shape = sample.metainfo["img_shape"]
        st.ori_H, st.ori_W = int(ori[0]), int(ori[1])
        st.img_H, st.img_W = float(img_shape[0]), float(img_shape[1])
        sf = sample.metainfo.get("scale_factor", (1.0, 1.0))
        if isinstance(sf, (int, float)):
            sf = (sf, sf)
        st.scale_factor = (float(sf[0]), float(sf[1]))

        enc = self._capture["enc_stage"]
        enc_out = self._capture["encoder_out"]
        st.A0 = enc["A0"][0].to(dev)
        if st.A0.numel() != Q:
            raise RuntimeError(f"unexpected PF-RPN query count: {st.A0.numel()}")
        st.coord_unact = enc["coord_unact"][0].to(dev)
        st.xi_all = st.coord_unact.sigmoid()
        st.conf_enc = enc["conf"][0].to(dev)
        st.cls_score_enc = enc["cls_score"][0].to(dev)
        st.V_enc = (enc["e_prop"][0] & (~enc["m_pad"][0]) &
                    torch.isfinite(st.coord_unact).all(dim=-1)).to(dev)

        # decoder call context (cloned normal tensors, detached) + native outputs
        args, kwargs = self._capture["decoder_call"]
        st.decoder_call = (
            [v.to(dev) if isinstance(v, torch.Tensor) else v for v in args],
            {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
             for k, v in kwargs.items()})
        st.text_token_mask = enc_out["text_token_mask"].to(dev)
        hidden, refs = self._capture["decoder_out"]
        st.hidden0 = hidden.to(dev)
        st.refs0 = refs.to(dev)

        st.R0 = refs[0][0].to(dev)  # [Q,4]
        # candidate-side support-independent quantities
        memory = enc_out["memory"].to(dev)
        spatial_shapes = enc_out["spatial_shapes"].to(dev)
        st.features = split_memory_to_features(memory, spatial_shapes)
        # valid feature extents per level from padding mask
        vm = enc_out["memory_mask"]
        if vm is None:
            vm = torch.zeros(memory.shape[1], dtype=torch.bool, device=memory.device)
        else:
            vm = vm[0]
        vf = []
        _cur = 0
        for Hf, Wf in spatial_shapes.tolist():
            n = int(Hf * Wf)
            lvl = vm[_cur:_cur + n].view(Hf, Wf)
            vf.append((int((~lvl[:, 0]).sum()), int((~lvl[0, :]).sum())))
            _cur += n
        st.valid_feature_hw = torch.tensor(vf, device=dev)

        N = st.xi_all.shape[0]
        xi_img = norm01_to_image(st.xi_all, st.img_W, st.img_H)
        boxes_c = xi_img.clone()
        boxes_c[..., 0::2] = boxes_c[..., 0::2].clamp(0, st.img_W)
        boxes_c[..., 1::2] = boxes_c[..., 1::2].clamp(0, st.img_H)
        st.geom_valid = torch.isfinite(boxes_c).all(dim=-1) & \
            (boxes_c[:, 2] > boxes_c[:, 0]) & (boxes_c[:, 3] > boxes_c[:, 1])
        z, z_valid = phi(st.features, xi_img, st.img_W, st.img_H,
                         st.valid_feature_hw)
        st.z_all, st.z_valid = z, z_valid

        # native y0 + scores + B0 proposals
        b_norm, cls_logits, conf1 = self.decode_outputs(st, st.hidden0, st.refs0)
        y0, s0 = self.build_y(st, b_norm, cls_logits, conf1)
        st.y0 = y0
        st.s0 = s0[0]
        boxes, scores = self.proposals_from_outputs(st, b_norm, s0)
        st.b0_boxes, st.b0_scores = boxes, scores

        # consistency: my B0 postprocess vs model's own pred_instances
        mb = pred.bboxes
        ms = pred.scores
        n = min(len(mb), len(boxes))
        st.native_match_max_box_diff = float(
            (mb[:n] - boxes[:n]).abs().max()) if n > 0 else 0.0
        st.native_match_max_score_diff = float(
            (ms[:n] - scores[:n]).abs().max()) if n > 0 else 0.0
        st.native_n_out = len(mb)
        return st


def probe_seed_int(dataset: str, parent_id: str, chip_id: Optional[str],
                   base: str = "20261012") -> int:
    """SHA256 of 'dataset|parent_id|chip_id|base', first 8 bytes big-endian mod 2^32."""
    s = f"{dataset}|{parent_id}|{chip_id if chip_id else ''}|{base}"
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def gen_probes(seed_int: int, d: int, n_rows: int = 32) -> np.ndarray:
    """NumPy PCG64; n_rows x d independent +-1 Rademacher rows."""
    rng = np.random.Generator(np.random.PCG64(seed_int))
    return rng.integers(0, 2, size=(n_rows, d), dtype=np.int8) * 2 - 1
