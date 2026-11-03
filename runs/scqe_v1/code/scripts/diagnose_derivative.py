"""Read-only, fixed-input E2 derivative diagnosis on a CUDA device.

This script does not change acceptance criteria or produce an acceptance pass.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, DIOR_DIR
from scqe.pf_runtime import PFRuntime, gen_probes, probe_seed_int
from scqe.ref64 import Ref64, use_pytorch_msda
from scqe.pf_runtime import functional_decoder_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-id", default="8091")
    ap.add_argument("--direction", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--jvp", action="store_true")
    args = ap.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rt = PFRuntime(model, args.device)
    ref = Ref64(rt)
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    entry = next(e for e in entries if str(e["image_id"]) == args.image_id)
    st = rt.native_pass(str(DIOR_DIR / "images" / entry["image_path"]))
    seed = probe_seed_int("DIOR", args.image_id, None)
    probes = gen_probes(seed, st.y0.numel(), 2)
    generator = torch.Generator(device=args.device).manual_seed(seed)
    if not 0 <= args.direction < 5:
        raise ValueError("direction must be 0..4")
    for _ in range(args.direction + 1):
        U = torch.randn(st.R0.shape, generator=generator, device=args.device)
    U = U / U.norm().clamp(min=1e-12)
    boundary = torch.minimum(st.R0, 1 - st.R0).clamp(min=0)
    U *= boundary
    v = [torch.as_tensor(row, device=args.device, dtype=torch.float64) for row in probes]
    w = [row * st.wsqrt for row in v]
    steps = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4,
             3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8]
    with torch.no_grad():
        baseline = [ref._y64(st, st.R0.double()).detach() for _ in range(3)]
    repeat_error = [float(((baseline[i] - baseline[0]) * w[0]).sum()) for i in range(1, 3)]
    ad = [ref.ad(st, st.R0, U, row, st.wsqrt) for row in v]
    jvp = [ref.jvp(st, st.R0, U, row, st.wsqrt) for row in v] if args.jvp else None
    manual_forward_error = float((ref._y64(st, st.R0, manual=True)-baseline[0]).abs().max()) \
        if args.jvp else None
    implementation_ad = {"native_cuda": [], "pytorch_fp32": []}
    implementation_y = {}
    for backend in ("native_cuda", "pytorch_fp32"):
        R = st.R0.unsqueeze(0).clone().requires_grad_(True)
        if backend == "pytorch_fp32":
            with use_pytorch_msda():
                y, _, _ = rt.y_of_R(st, R)
        else:
            y, _, _ = rt.y_of_R(st, R)
        implementation_y[backend] = y.detach()
        for row in w:
            grad = torch.autograd.grad((row*y).sum(), R, retain_graph=True)[0][0]
            implementation_ad[backend].append(float((grad*U).sum()))
    implementation_ad["pytorch_fp32_forward_max_abs_error"] = float(
        (implementation_y["pytorch_fp32"]-implementation_y["native_cuda"]).abs().max())
    production_vjp = rt.compute_vjps(st, 2, probes)
    implementation_ad["production_fp64"] = [
        float((production_vjp[i] * U.double()).sum()) for i in range(2)]
    rows = []
    for h in steps:
        with torch.no_grad():
            yp = ref._y64(st, st.R0.double() + h * U.double())
            ym = ref._y64(st, st.R0.double() - h * U.double())
        rows.append({"h": h, "fd": [float((row * (yp - ym)).sum() / (2*h)) for row in w],
                     "max_y_change": float(torch.maximum((yp-baseline[0]).abs(),
                                                          (ym-baseline[0]).abs()).max()),
                     "max_y_index_plus": int((yp-baseline[0]).abs().argmax())})
    def decoder_stages(R):
        call_args, call_kwargs = st.decoder_call
        call_args = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                     for a in call_args]
        call_kwargs = {k: (t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t)
                       for k, t in call_kwargs.items()}
        call_kwargs["reference_points"] = R.reshape(1, -1, 4)
        call_kwargs["reg_branches"] = ref.reg_branches64
        with torch.no_grad(), use_pytorch_msda():
            return functional_decoder_forward(ref.dec, *call_args, **call_kwargs)
    hidden0, refs0 = decoder_stages(st.R0.double())
    stages = []
    for h in [3e-3, 1e-5]:
        hp, rp = decoder_stages(st.R0.double() + h * U.double())
        hm, rm = decoder_stages(st.R0.double() - h * U.double())
        stages.append({"h": h,
                       "hidden_max_abs_change_by_layer": [
                           float(torch.maximum((hp[i]-hidden0[i]).abs(),
                                               (hm[i]-hidden0[i]).abs()).max())
                           for i in range(len(hidden0))],
                       "reference_max_abs_change_by_layer": [
                           float(torch.maximum((rp[i]-refs0[i]).abs(),
                                               (rm[i]-refs0[i]).abs()).max())
                           for i in range(len(refs0))]})
    out = {"image_id": args.image_id, "direction": args.direction,
           "allow_tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn": torch.backends.cudnn.allow_tf32},
           "seed": seed, "U_norm": float(U.norm()),
           "U_max": float(U.abs().max()), "R0_min": float(st.R0.min()),
           "R0_max": float(st.R0.max()), "ad64": ad,
           "jvp_manual_fp64": jvp, "manual_forward_error": manual_forward_error,
           "implementation_ad": implementation_ad,
           "repeat_phi_error": repeat_error, "steps": rows,
           "decoder_stages": stages, "protected_vector_width": st.T+5}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
