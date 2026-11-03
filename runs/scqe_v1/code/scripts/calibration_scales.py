"""E3 measured D-protocol proxy/true scales on frozen first 12 calibration images."""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, DIOR_DIR
from scqe.acceptance_gate import require_scqe_acceptance
from scqe.coverage import Coverage
from scqe.data import support_instances
from scqe.pf_runtime import PFRuntime, gen_probes
from scqe.pipeline import QueryEngine
from scqe.risk import true_risk
from scqe.solver import netgreedy
from scqe.support import SupportEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    require_scqe_acceptance(args.acceptance)
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA GPUs required; CPU fallback disabled")
    args.out.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device="cuda:0").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = PFRuntime(model, "cuda:0")
    encoder = SupportEncoder(runtime)
    supports = {}
    for seed in (2023, 42):
        supports[seed] = encoder.encode_episode(support_instances(seed, 5), 20)
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    entries = entries[:12]
    if len(entries) != 12:
        raise ValueError("D calibration manifest needs at least 12 images")
    records, proxy_positive, isolated_positive, true_positive = [], [], [], []
    true_noise, proxy_noise, isolated_noise, coverage_repeat = [], [], [], []
    for entry in entries:
        qid = str(entry["image_id"])
        engine = QueryEngine(runtime, "DIOR", qid).prepare(
            str(DIOR_DIR / "images" / entry["image_path"]))
        for seed in (2023, 42):
            p, t, cv = supports[seed]
            alpha, cols, status = engine.build_alpha(p, t, cv)
            rec = dict(image_id=qid, seed=seed, status=status, planned_m=0)
            if alpha is None:
                records.append(rec)
                continue
            ctx = engine.ctx
            cov1 = Coverage(ctx.vs_idx, ctx.vs_boxes, alpha, cols, ctx.A0)
            cov2 = Coverage(ctx.vs_idx, ctx.vs_boxes, alpha, cols, ctx.A0)
            coverage_repeat.append(abs(cov1.F() - cov2.F()))
            out = netgreedy(ctx, alpha, cols, None, m_max=32, delta_F=0.0)
            rec.update(planned_m=out["m"], F0=out["F0"], Fplan=out["Fplan"])
            if not out["m"]:
                records.append(rec)
                continue
            g = engine.get_vjps()
            delta = (engine.state.xi_all[out["plan"]] - ctx.R0).to(torch.float64)
            projections = torch.einsum("prc,rc->p", g.to(torch.float64), delta)
            proxy = float(projections.square().mean())
            probe_rows = gen_probes(engine.probe_seed, engine.state.y0.numel(),
                                    max(32, engine.probes))[:engine.probes]
            g_repeat = runtime.compute_vjps(engine.state, engine.probes, probe_rows)
            repeated = torch.einsum("prc,rc->p", g_repeat.to(torch.float64), delta)
            proxy_repeat = float(repeated.square().mean())
            proxy_noise.append(abs(proxy - proxy_repeat))
            isolated = 0.0
            isolated_repeat = 0.0
            for step in out["steps"]:
                slot = step["slot"]
                direction = (engine.state.xi_all[step["incoming"]] -
                             ctx.R0[slot]).to(torch.float64)
                d = (g[:, slot, :].to(torch.float64) * direction).sum(-1)
                dr = (g_repeat[:, slot, :].to(torch.float64) * direction).sum(-1)
                isolated += float(d.square().mean())
                isolated_repeat += float(dr.square().mean())
            isolated_noise.append(abs(isolated - isolated_repeat))
            y_values = [runtime.native_decode_R(engine.state, engine.R_of_plan(out["plan"]))[0]
                        for _ in range(3)]
            risks = [true_risk(y, engine.state.y0, engine.state.wsqrt) for y in y_values]
            noise = max(risks) - min(risks)
            true_noise.append(noise)
            rec.update(proxy=proxy, isolated_proxy=isolated,
                       proxy_repeat_range=abs(proxy - proxy_repeat),
                       isolated_repeat_range=abs(isolated - isolated_repeat),
                       true_risk=risks[0], true_repeat_range=noise)
            records.append(rec)
            if proxy > 0:
                proxy_positive.append(proxy)
            if isolated > 0:
                isolated_positive.append(isolated)
            if risks[0] > 0:
                true_positive.append(risks[0])
    noise_p = 10 * max(proxy_noise, default=0.0)
    noise_i = 10 * max(isolated_noise, default=0.0)
    noise_t = 10 * max(true_noise, default=0.0)
    pos_p = [x for x in proxy_positive if x > noise_p]
    pos_i = [x for x in isolated_positive if x > noise_i]
    pos_t = [x for x in true_positive if x > noise_t]
    delta_F = max(10 * max(coverage_repeat, default=0.0),
                  10 * torch.finfo(torch.float32).eps)
    if not pos_p or not pos_i or not pos_t:
        raise RuntimeError("positive calibration scales unavailable; inspect records")
    scales = dict(protocol="D", dataset="DIOR", shot=5, images=12,
                  support_seeds=[2023, 42], records=records,
                  s_p=statistics.median(pos_p),
                  s_p_isolated=statistics.median(pos_i),
                  s_t=statistics.median(pos_t),
                  proxy_positive_n=len(pos_p),
                  isolated_positive_n=len(pos_i),
                  true_positive_n=len(pos_t),
                  proxy_noise_floor=noise_p, isolated_noise_floor=noise_i,
                  true_noise_floor=noise_t, delta_F=delta_F,
                  delta_F_rule="max(10x repeated coverage difference, 10x FP32 machine epsilon)",
                  empty_plans=sum(r["planned_m"] == 0 for r in records))
    (args.out / "scales.json").write_text(json.dumps(scales, indent=2))


if __name__ == "__main__":
    main()
