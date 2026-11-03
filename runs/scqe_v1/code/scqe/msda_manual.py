"""CUDA tensor bilinear sampler with forward-mode AD support.

PyTorch 2.1 grid_sample has no forward-AD rule. This implementation is used
only as an independent derivative oracle; production inference keeps MMCV's
native CUDA op. Inputs and arithmetic remain on the caller's CUDA device.
"""
import torch


def bilinear_sample_zeros(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Match grid_sample(..., bilinear, zeros, align_corners=False) for 4-D tensors."""
    if input.ndim != 4 or grid.ndim != 4 or input.shape[0] != grid.shape[0]:
        raise ValueError("expected input [N,C,H,W] and grid [N,Hout,Wout,2]")
    n, c, height, width = input.shape
    out_h, out_w = grid.shape[1:3]
    x = ((grid[..., 0] + 1) * width - 1) * 0.5
    y = ((grid[..., 1] + 1) * height - 1) * 0.5
    x0, y0 = x.floor(), y.floor()
    x1, y1 = x0 + 1, y0 + 1
    flat = input.reshape(n, c, height * width)
    result = torch.zeros(n, c, out_h * out_w, dtype=input.dtype, device=input.device)

    def corner(xi, yi, weight):
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        index = (yi.clamp(0, height - 1).to(torch.long) * width +
                 xi.clamp(0, width - 1).to(torch.long)).reshape(n, -1)
        values = flat.gather(2, index[:, None, :].expand(-1, c, -1))
        return values * (weight * valid).reshape(n, 1, -1)

    result = result + corner(x0, y0, (x1 - x) * (y1 - y))
    result = result + corner(x1, y0, (x - x0) * (y1 - y))
    result = result + corner(x0, y1, (x1 - x) * (y - y0))
    result = result + corner(x1, y1, (x - x0) * (y - y0))
    return result.reshape(n, c, out_h, out_w)


def msda_manual(value: torch.Tensor, value_spatial_shapes,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
    """Match mmcv.multi_scale_deformable_attn_pytorch using manual sampling."""
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    sizes = [tuple(int(x) for x in shape) for shape in value_spatial_shapes]
    values = value.split([h * w for h, w in sizes], dim=1)
    grids = 2 * sampling_locations - 1
    sampled = []
    for level, (height, width) in enumerate(sizes):
        vl = values[level].flatten(2).transpose(1, 2).reshape(
            batch * heads, channels, height, width)
        gl = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(bilinear_sample_zeros(vl, gl))
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return output.reshape(batch, heads * channels, queries).transpose(1, 2).contiguous()
