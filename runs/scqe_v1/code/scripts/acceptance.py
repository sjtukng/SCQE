"""E2 CUDA implementation acceptance; never evaluates AP or picks thresholds."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scqe.config import CHECKPOINT_PATH, CONFIG_PATH, DIOR_DIR
from scqe.coverage import Coverage
from scqe.pf_runtime import PFRuntime, gen_probes, probe_seed_int
from scqe.ref64 import Ref64
from scqe.msda_manual import bilinear_sample_zeros
from scqe.pipeline import QueryEngine
from scqe.data import support_instances
from scqe.solver import netgreedy
from scqe.support import SupportEncoder

LEGACY_STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
LOCAL_STEPS = (3e-5, 1e-5, 3e-6, 1e-6)
STEPS = LEGACY_STEPS + LOCAL_STEPS
PROTOCOL_VERSION = "E2-2026-10-16-fp64-vjp-manual-jvp"


def synthetic_sampler_check(device):
    generator = torch.Generator(device=device).manual_seed(20261016)
    value = torch.randn(2, 3, 7, 9, generator=generator,
                        device=device, dtype=torch.float64)
    grid = (torch.rand(2, 4, 5, 2, generator=generator,
                       device=device, dtype=torch.float64) * 2.8 - 1.4).requires_grad_(True)
    direction = torch.randn(grid.shape, generator=generator,
                            device=device, dtype=torch.float64)
    standard = F.grid_sample(value, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)
    manual = bilinear_sample_zeros(value, grid)
    forward_error = float((standard - manual).abs().max())
    standard_grad = torch.autograd.grad(standard.sum(), grid, retain_graph=True)[0]
    manual_grad = torch.autograd.grad(manual.sum(), grid)[0]
    backward_error = float((standard_grad - manual_grad).abs().max())
    _, tangent = torch.func.jvp(lambda x: bilinear_sample_zeros(value, x).sum(),
                                (grid.detach(),), (direction,))
    duality_error = abs(float(tangent) - float((manual_grad * direction).sum()))
    return dict(forward_max_abs=forward_error,
                backward_max_abs=backward_error,
                jvp_vjp_abs=duality_error,
                passed=max(forward_error, backward_error, duality_error) <= 1e-10)


def synthetic_cache_check(device):
    # The old top two (slots 0,1) become (1,2) after replacing slot 0.
    idx = torch.arange(4, device=device)
    boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 8, 8],
                          [0, 0, 6, 6], [20, 20, 30, 30]],
                         device=device, dtype=torch.float32)
    alpha = torch.ones(4, 1, device=device)
    before = Coverage(idx, boxes, alpha, [0], idx[:3])
    before.commit(0, 3, torch.tensor([0], device=device))
    direct = Coverage(idx, boxes, alpha, [0],
                      torch.tensor([3, 1, 2], device=device))
    assert torch.equal(before.cache["o1"], direct.cache["o1"])
    assert torch.equal(before.cache["m2"], direct.cache["m2"])
    assert abs(before.F() - direct.F()) < 1e-12
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-index", type=int, default=0,
                    help="three-image calibration window; 0 is the frozen E2 set")
    ap.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA GPUs required; CPU fallback disabled")
    device = args.device
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    if args.start_index < 0:
        raise ValueError("start-index must be nonnegative")
    selected = entries[args.start_index:args.start_index + 3]
    if len(selected) != 3:
        raise ValueError("calibration manifest needs at least 3 images")
    from mmdet.apis import init_detector
    model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rt = PFRuntime(model, device)
    ref64 = rt.ref64()
    p, t, class_valid = SupportEncoder(rt).encode_episode(
        support_instances(2023, 5), 20)
    table = []
    direction_specs = []
    forward_rows = []
    solver_rows = []
    sampler_check = synthetic_sampler_check(device)
    all_ok = synthetic_cache_check(device) and sampler_check["passed"]
    nonempty_solver_plans = 0
    for entry in selected:
        qid = str(entry["image_id"])
        st = rt.native_pass(str(DIOR_DIR / "images" / entry["image_path"]))
        R0 = st.R0.unsqueeze(0)
        native_y, native_pf, native_out = rt.native_decode_R(st, R0)
        functional_y, functional_pf, functional_out = rt.y_of_R(st, R0)
        valid = st.text_token_mask[0].bool()
        native_vec = torch.cat([
            native_out[0].flatten(), native_out[1][:, :, valid].sigmoid().flatten(),
            native_out[2].sigmoid().flatten()])
        func_vec = torch.cat([
            functional_out[0].flatten(),
            functional_out[1][:, :, valid].sigmoid().flatten(),
            functional_out[2].sigmoid().flatten()])
        delta = (native_vec - func_vec).abs()
        allowed = 1e-6 + 1e-5 * native_vec.abs()
        with torch.no_grad():
            reference_y0 = ref64._y64(st, R0)
            manual_y0 = ref64._y64(st, R0, manual=True)
        reference_error = (reference_y0 - st.y0).abs()
        manual_error = (manual_y0 - reference_y0).abs()
        manual_ok = bool((manual_error <= 1e-9).all())
        reference_allowed = 1e-4 + 1e-3 * st.y0.abs()
        reference_ok = bool((reference_error <= reference_allowed).all())
        f_ok = bool((delta <= allowed).all()) and \
            st.native_n_out == len(st.b0_boxes) and \
            st.native_match_max_box_diff <= 1e-5 and \
            st.native_match_max_score_diff <= 1e-5 and reference_ok and manual_ok
        forward_rows.append(dict(image_id=qid, max_abs=float(delta.max()),
                                 max_relative=float((delta / allowed).max()),
                                 reference_max_abs=float(reference_error.max()),
                                 reference_forward_ok=reference_ok,
                                 manual_reference_max_abs=float(manual_error.max()),
                                 manual_reference_ok=manual_ok,
                                 pass_forward=f_ok,
                                 native_n_out=st.native_n_out,
                                 implementation_n_out=len(st.b0_boxes),
                                 native_prediction_match_max_box=st.native_match_max_box_diff,
                                 native_prediction_match_max_score=st.native_match_max_score_diff))
        all_ok &= f_ok
        native_repeat = [rt.native_decode_R(st, R0)[0] for _ in range(3)]
        with torch.no_grad():
            reference_repeat = [ref64._y64(st, R0) for _ in range(3)]
        seed = probe_seed_int("DIOR", qid, None)
        rows = gen_probes(seed, st.y0.numel(), 2)
        production_grads = rt.compute_vjps(st, 2, rows)
        native_grads = []
        for probe in range(2):
            v = torch.as_tensor(rows[probe], device=device, dtype=torch.float64)
            R = R0.clone().requires_grad_(True)
            y, _, _ = rt.y_of_R(st, R)
            native_grads.append(torch.autograd.grad((v * st.wsqrt * y).sum(), R)[0][0])
        generator = torch.Generator(device=device).manual_seed(seed)
        derivative_passes = []
        for direction in range(5):
            U = torch.randn(st.R0.shape, generator=generator, device=device)
            U = U / U.norm().clamp(min=1e-12)
            boundary = torch.minimum(st.R0, 1 - st.R0).clamp(min=0)
            U = U * boundary
            direction_specs.append(dict(
                image_id=qid, direction=direction, seed=seed,
                formula="randn_cuda_seeded/unit_l2 * min(R0,1-R0).clamp_min(0)",
                Q=int(st.R0.shape[0]),
                norm=float(U.norm()), max_abs=float(U.abs().max()),
                sha256_float32=hashlib.sha256(U.contiguous().cpu().numpy().tobytes()).hexdigest()))
            # CUDA MSDA computes the forward in FP32. Subtracting two near
            # equal FP32 outputs can swamp the smallest central differences.
            # Keep the native result for diagnosis, and compute the official
            # FD with the same frozen decoder/weights in FP64 using mmcv's
            # pure-PyTorch MSDA reference on cuda:0. The production AD remains
            # the CUDA functional decoder; its tolerance is unchanged.
            differences = []
            for h in STEPS:
                plus = (st.R0 + h * U).unsqueeze(0)
                minus = (st.R0 - h * U).unsqueeze(0)
                yp_native = rt.native_decode_R(st, plus)[0]
                ym_native = rt.native_decode_R(st, minus)[0]
                with torch.no_grad():
                    yp64 = ref64._y64(st, st.R0.double() + h * U.double())
                    ym64 = ref64._y64(st, st.R0.double() - h * U.double())
                differences.append((h, yp_native - ym_native, yp64 - ym64))
            for probe in range(2):
                v = torch.as_tensor(rows[probe], device=device, dtype=torch.float64)
                native_noise = max(abs(float((v * st.wsqrt * (value - native_repeat[0])).sum()))
                                   for value in native_repeat[1:])
                noise = max(abs(float((v * st.wsqrt * (value - reference_repeat[0])).sum()))
                            for value in reference_repeat[1:])
                ad = float((production_grads[probe] * U.double()).sum())
                ad_native = float((native_grads[probe] * U).sum())
                jvp = ref64.jvp(st, st.R0, U, v, st.wsqrt)
                grad_allowed = 1e-5 + 0.05 * max(abs(ad), abs(jvp))
                gradient_ok = math.isfinite(jvp) and abs(ad - jvp) <= grad_allowed
                good = []
                for h, native_delta, reference_delta in differences:
                    fd_native = float((v * st.wsqrt * native_delta).sum() / (2 * h))
                    fd = float((v * st.wsqrt * reference_delta).sum() / (2 * h))
                    threshold = 1e-5 + 0.05 * max(abs(fd), abs(ad))
                    usable = math.isfinite(fd) and abs(fd) * h > 2 * noise
                    passed = usable and gradient_ok and abs(fd - ad) <= threshold
                    table.append(dict(image_id=qid, direction=direction, probe=probe,
                                      h=h, fd=fd, fd_native_fp32=fd_native,
                                      ad=ad, ad_reference_fp64=jvp,
                                      ad_native_fp32=ad_native,
                                      jvp_manual_fp64=jvp,
                                      jvp_vjp_abs_error=abs(ad-jvp),
                                      gradient_reference_ok=gradient_ok,
                                      abs_error=abs(fd - ad),
                                      reference_nonlinearity=abs(fd - jvp),
                                      fp32_fd_error=abs(fd_native - fd),
                                      allowed=threshold, noise=noise,
                                      native_repeat_noise=native_noise,
                                      reference_repeat_noise=noise,
                                      local_step=h in LOCAL_STEPS,
                                      usable=usable, passed=passed))
                    good.append(passed)
                derivative_passes.append(sum(good[len(LEGACY_STEPS):]) >= 1)
        all_ok &= all(derivative_passes)
        # Numeric bypass must return the exact cached B0 tensors and assignment.
        engine = QueryEngine(rt, "DIOR", qid).prepare(
            str(DIOR_DIR / "images" / entry["image_path"]))
        if engine.ctx is not None:
            bad = torch.full((len(engine.ctx.vs_idx), 1), float("nan"), device=device)
            fallback = engine.run_method("SCQE", bad, [0], eps_proxy=1.0, eps_true=1.0)
            all_ok &= fallback["status"] == "numeric_bypass" and \
                torch.equal(fallback["Aout"], st.A0) and \
                torch.equal(fallback["boxes"], st.b0_boxes) and \
                torch.equal(fallback["scores"], st.b0_scores)
            alpha, cols, _ = engine.build_alpha(p, t, class_valid)
            if alpha is not None:
                gradients = engine.get_vjps()
                out = netgreedy(engine.ctx, alpha, cols, gradients,
                                eps_proxy=1e9, delta_F=0.0)
                if out["m"]:
                    nonempty_solver_plans += 1
                    direct_cov = Coverage(engine.ctx.vs_idx, engine.ctx.vs_boxes,
                                          alpha, cols, out["plan"])
                    delta_R = (engine.state.xi_all[out["plan"]] - engine.state.R0).to(torch.float64)
                    projections = torch.einsum("prc,rc->p", gradients.to(torch.float64), delta_R)
                    direct_proxy = float(projections.square().mean())
                    protected_ok = torch.equal(out["plan"][engine.ctx.H],
                                               engine.state.A0[engine.ctx.H])
                    unique_ok = len(torch.unique(out["plan"])) == len(out["plan"])
                    incoming_ok = all(step["incoming"] not in set(engine.state.A0.tolist())
                                      for step in out["steps"])
                    coverage_ok = abs(direct_cov.F() - out["Fplan"]) <= 1e-8
                    proxy_ok = abs(direct_proxy - out["proxy_cum"]) <= \
                        1e-8 * max(1, direct_proxy)
                    rejected = engine.run_method("SCQE", alpha, cols,
                                                 eps_proxy=1e9, eps_true=-1.0,
                                                 delta_F=0.0)
                    fallback_ok = rejected["status"] == "true_reject" and \
                        torch.equal(rejected["Aout"], engine.state.A0) and \
                        torch.equal(rejected["boxes"], engine.state.b0_boxes) and \
                        torch.equal(rejected["scores"], engine.state.b0_scores)
                    check_ok = protected_ok and unique_ok and incoming_ok and \
                        coverage_ok and proxy_ok and fallback_ok
                    all_ok &= check_ok
                    solver_rows.append(dict(image_id=qid, exchanges=out["m"],
                                            direct_coverage=direct_cov.F(),
                                            cached_coverage=out["Fplan"],
                                            direct_proxy=direct_proxy,
                                            cached_proxy=out["proxy_cum"],
                                            protected_ok=protected_ok,
                                            unique_ok=unique_ok,
                                            incoming_ok=incoming_ok,
                                            fallback_ok=fallback_ok,
                                            passed=check_ok))
    all_ok &= nonempty_solver_plans > 0
    (args.out / "forward.json").write_text(json.dumps(forward_rows, indent=2))
    (args.out / "finite_differences.json").write_text(json.dumps(table, indent=2))
    (args.out / "direction_specs.json").write_text(json.dumps(direction_specs, indent=2))
    failures = []
    legacy_failures = []
    for image_id in sorted({r["image_id"] for r in table}):
        for direction in range(5):
            for probe in range(2):
                group = [r for r in table if r["image_id"] == image_id and
                         r["direction"] == direction and r["probe"] == probe]
                record = dict(image_id=image_id, direction=direction, probe=probe,
                              usable_steps=sum(r["usable"] for r in group),
                              passing_local_steps=sum(r["passed"] for r in group
                                                      if r["local_step"]),
                              gradient_reference_ok=group[0]["gradient_reference_ok"],
                              min_reference_nonlinearity=min(r["reference_nonlinearity"]
                                                             for r in group),
                              max_fp32_fd_error=max(r["fp32_fd_error"] for r in group))
                if record["passing_local_steps"] < 1:
                    failures.append(record)
                if not any(group[i]["passed"] and group[i + 1]["passed"]
                           for i in range(len(LEGACY_STEPS)-1)):
                    legacy_failures.append(record)
    (args.out / "derivative_failures.json").write_text(json.dumps(failures, indent=2))
    (args.out / "legacy_derivative_failures.json").write_text(
        json.dumps(legacy_failures, indent=2))
    (args.out / "solver_checks.json").write_text(json.dumps(solver_rows, indent=2))
    (args.out / "sampler_check.json").write_text(json.dumps(sampler_check, indent=2))
    result = dict(status="COMPLETE" if all_ok else "FAILED", images=[e["image_id"] for e in selected],
                  calibration_offset=args.start_index,
                  synthetic_cache_test=True, forward_checks=forward_rows,
                  synthetic_sampler_check=sampler_check,
                  production_gradient_backend="CUDA_FP64_PyTorch_MSDA",
                  protocol_version=PROTOCOL_VERSION,
                  legacy_steps=LEGACY_STEPS, local_steps=LOCAL_STEPS,
                  derivative_rows=len(table), derivative_failed_groups=len(failures),
                  legacy_derivative_failed_groups=len(legacy_failures),
                  direction_scheme="boundary_weighted_seeded_unit_vector",
                  nonempty_solver_plans=nonempty_solver_plans,
                  solver_checks=solver_rows, pass_all=bool(all_ok))
    (args.out / "implementation_acceptance.json").write_text(json.dumps(result, indent=2))
    if not all_ok:
        raise RuntimeError(
            f"E2 acceptance failed: {len(failures)}/30 derivative groups failed; "
            f"inspect {args.out / 'derivative_failures.json'} and forward/solver checks. "
            "Do not start E3 or formal SCQE evaluation.")


if __name__ == "__main__":
    main()
