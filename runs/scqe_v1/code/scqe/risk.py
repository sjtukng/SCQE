"""Risk: cumulative VJP proxy risk and true output perturbation (float64 scalars)."""
import torch


def pair_risk(g: torch.Tensor, zeta: torch.Tensor, r_list: torch.Tensor,
              xi: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
    """Cumulative proxy risk for all pairs (r in r_list, j in xi rows): [R, J].

    R_hat = (1/P) sum_b (zeta_b + d_b(r,j))^2, d_b = <g_b[r], xi_j - R0[r]>.
    """
    g64 = g.to(torch.float64)
    G = g64[:, r_list, :]                        # [P,R,4]
    D = (xi[None, :, :] - R0[r_list][:, None, :]).to(torch.float64)  # [R,J,4]
    d = torch.einsum("prc,rjc->prj", G, D)       # [P,R,J]
    tot = zeta.to(torch.float64)[:, None, None] + d
    return (tot * tot).mean(dim=0)               # [R,J]


def step_zeta(g: torch.Tensor, zeta: torch.Tensor, r: int,
              xi_j: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
    """zeta after committing exchange (r, j): [P] float64."""
    g64 = g.to(torch.float64)
    d = (g64[:, r, :] * (xi_j.to(torch.float64) - R0[r].to(torch.float64))[None, :]).sum(-1)
    return zeta.to(torch.float64) + d


def proxy_from_zeta(zeta: torch.Tensor) -> float:
    z = zeta.to(torch.float64)
    return float((z * z).mean())


def true_risk(y: torch.Tensor, y0: torch.Tensor, wsqrt: torch.Tensor) -> float:
    """|| W^{1/2} (y - y0) ||_2^2 in float64."""
    diff = (y.to(torch.float64) - y0.to(torch.float64)) * wsqrt
    return float((diff * diff).sum())
