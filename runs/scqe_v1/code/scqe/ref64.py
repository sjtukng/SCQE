"""Float64 decoder reference for derivative diagnostics.

The CUDA MSDA extension supports only float32. Its pure-PyTorch reference
supports float64 forward and autograd, using the same frozen weights and
captured inputs. Float64 reduces subtraction error in finite differences,
but a secant across nonlinear or nonsmooth regions need not equal the local
derivative even in float64. Check convergence for each fixed direction.
"""
import copy
from contextlib import contextmanager

import torch

from mmcv.ops import multi_scale_deform_attn as _msa
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from .msda_manual import msda_manual

_active_msda_op = multi_scale_deformable_attn_pytorch
_active_msda_shapes = None


def _msda_forward_pytorch(self, query, key=None, value=None, identity=None,
                          query_pos=None, key_padding_mask=None,
                          reference_points=None, spatial_shapes=None,
                          level_start_index=None, **kwargs):
    if value is None:
        value = query
    if identity is None:
        identity = query
    if query_pos is not None:
        query = query + query_pos
    if not self.batch_first:
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
    bs, num_query, _ = query.shape
    bs, num_value, _ = value.shape
    value = self.value_proj(value)
    if key_padding_mask is not None:
        value = value.masked_fill(key_padding_mask[..., None], 0.0)
    value = value.view(bs, num_value, self.num_heads, -1)
    sampling_offsets = self.sampling_offsets(query).view(
        bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
    attention_weights = self.attention_weights(query).view(
        bs, num_query, self.num_heads, self.num_levels * self.num_points)
    attention_weights = attention_weights.softmax(-1).view(
        bs, num_query, self.num_heads, self.num_levels, self.num_points)
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]
    elif reference_points.shape[-1] == 4:
        sampling_locations = reference_points[:, :, None, :, None, :2] \
            + sampling_offsets / self.num_points \
            * reference_points[:, :, None, :, None, 2:] * 0.5
    else:
        raise ValueError("reference_points last dim must be 2 or 4")
    sizes = _active_msda_shapes
    if sizes is None:
        sizes = [tuple(int(x) for x in s) for s in spatial_shapes.tolist()]
    output = _active_msda_op(
        value, sizes,
        sampling_locations, attention_weights)
    output = self.output_proj(output)
    if not self.batch_first:
        output = output.permute(1, 0, 2)
    return self.dropout(output) + identity


@contextmanager
def use_pytorch_msda():
    global _active_msda_op, _active_msda_shapes
    previous_op = _active_msda_op
    previous_shapes = _active_msda_shapes
    previous_forward = _msa.MultiScaleDeformableAttention.forward
    _active_msda_op = multi_scale_deformable_attn_pytorch
    _active_msda_shapes = None
    _msa.MultiScaleDeformableAttention.forward = _msda_forward_pytorch
    try:
        yield
    finally:
        _active_msda_op = previous_op
        _active_msda_shapes = previous_shapes
        _msa.MultiScaleDeformableAttention.forward = previous_forward


@contextmanager
def use_manual_msda(sizes):
    """Reference attention with a separate bilinear JVP implementation."""
    global _active_msda_op, _active_msda_shapes
    previous_op = _active_msda_op
    previous_shapes = _active_msda_shapes
    previous_forward = _msa.MultiScaleDeformableAttention.forward
    _active_msda_op = msda_manual
    _active_msda_shapes = sizes
    _msa.MultiScaleDeformableAttention.forward = _msda_forward_pytorch
    try:
        yield
    finally:
        _active_msda_op = previous_op
        _active_msda_shapes = previous_shapes
        _msa.MultiScaleDeformableAttention.forward = previous_forward


class Ref64:
    """Float64 replica of the functional decoder and final head."""

    def __init__(self, runtime):
        model = runtime.model
        self.dec = copy.deepcopy(model.decoder).double().eval()
        head = model.bbox_head
        lid = model.decoder.num_layers - 1
        self.cls64 = copy.deepcopy(head.cls_branches[lid]).double().eval()
        self.conf64 = copy.deepcopy(head.conf_branches[lid]).double().eval()
        self.reg64 = copy.deepcopy(head.reg_branches[lid]).double().eval()
        self.reg_branches64 = copy.deepcopy(head.reg_branches).double().eval()

    def _y64(self, state, R, manual: bool = False, manual_sizes=None):
        from mmdet.models.layers.transformer.utils import inverse_sigmoid
        from .pf_runtime import functional_decoder_forward
        args, kwargs = state.decoder_call
        args = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args]
        kwargs = {k: (t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t)
                  for k, t in kwargs.items()}
        kwargs["reference_points"] = R.double().reshape(1, -1, 4)
        kwargs["reg_branches"] = self.reg_branches64
        if manual and manual_sizes is None:
            original_shapes = kwargs.get("spatial_shapes", args[5] if len(args) > 5 else None)
            manual_sizes = [tuple(int(x) for x in s) for s in original_shapes.tolist()]
        sizes = manual_sizes
        with (use_manual_msda(sizes) if manual else use_pytorch_msda()):
            hidden, refs = functional_decoder_forward(self.dec, *args, **kwargs)
        hq = hidden[-1]
        cls_logits = self.cls64(hq, kwargs["memory_text"],
                                state.text_token_mask)
        conf1 = self.conf64(hq).squeeze(-1).sigmoid()
        b_norm = (self.reg64(hq) +
                  inverse_sigmoid(refs[-2])).sigmoid()
        pi = cls_logits[:, :, state.text_token_mask[0].bool()].sigmoid()
        gamma = conf1.sigmoid()
        idx = state.H_slots
        return torch.cat([b_norm[0, idx], pi[0, idx], gamma[0, idx, None]],
                         dim=-1).reshape(-1)

    def phi(self, state, R: torch.Tensor, v: torch.Tensor,
            wsqrt: torch.Tensor) -> float:
        """v^T W^{1/2} y(R) in float64 (forward only). R: [Q,4] or [1,Q,4]."""
        with torch.no_grad():
            y = self._y64(state, R)
            return float((v.double() * wsqrt * y).sum())

    def ad(self, state, R: torch.Tensor, U: torch.Tensor,
           v: torch.Tensor, wsqrt: torch.Tensor) -> float:
        """Reverse-mode directional derivative <grad_R phi, U> in float64."""
        Rd = R.double().reshape(1, -1, 4).clone().requires_grad_(True)
        y = self._y64(state, Rd)
        obj = (v.double() * wsqrt * y).sum()
        g = torch.autograd.grad(obj, Rd)[0][0]
        return float((g * U.double()).sum())

    def vjp(self, state, R: torch.Tensor, v: torch.Tensor,
            wsqrt: torch.Tensor) -> torch.Tensor:
        """Production FP64 VJP of the real-valued decoder, shape [Q,4]."""
        Rd = R.double().reshape(1, -1, 4).clone().requires_grad_(True)
        y = self._y64(state, Rd)
        objective = (v.double() * wsqrt * y).sum()
        return torch.autograd.grad(objective, Rd)[0][0].detach()

    def jvp(self, state, R: torch.Tensor, U: torch.Tensor,
            v: torch.Tensor, wsqrt: torch.Tensor) -> float:
        """Independent forward-mode JVP via manual bilinear attention on CUDA."""
        Rd = R.double().reshape(1, -1, 4)
        Ud = U.double().reshape(1, -1, 4)
        call_args, call_kwargs = state.decoder_call
        shapes = call_kwargs.get("spatial_shapes", call_args[5] if len(call_args) > 5 else None)
        sizes = [tuple(int(x) for x in s) for s in shapes.tolist()]
        def objective(x):
            y = self._y64(state, x, manual=True, manual_sizes=sizes)
            return (v.double() * wsqrt * y).sum()
        _, tangent = torch.func.jvp(objective, (Rd,), (Ud,))
        return float(tangent)
