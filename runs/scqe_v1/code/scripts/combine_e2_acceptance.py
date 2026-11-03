"""Join the two immutable three-image CUDA E2 runs into a six-image gate."""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "E2-2026-10-16-fp64-vjp-manual-jvp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--extra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    entries = json.loads(args.split.read_text())
    entries = entries.get("queries", entries) if isinstance(entries, dict) else entries
    expected = [e["image_id"] for e in entries[:6]]
    if len(expected) != 6 or len(set(expected)) != 6:
        raise ValueError("calibration manifest must start with six distinct images")
    sources = []
    for path, offset in ((args.primary, 0), (args.extra, 3)):
        raw = path.read_bytes()
        result = json.loads(raw)
        if result.get("protocol_version") != PROTOCOL or \
                result.get("production_gradient_backend") != "CUDA_FP64_PyTorch_MSDA" or \
                result.get("calibration_offset") != offset or \
                result.get("images") != expected[offset:offset+3]:
            raise ValueError(f"incompatible E2 source: {path}")
        source_dir = path.parent
        for filename in ("finite_differences.json", "direction_specs.json",
                         "forward.json", "solver_checks.json", "sampler_check.json"):
            if not (source_dir / filename).is_file():
                raise FileNotFoundError(source_dir / filename)
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                            result=result))
    primary, extra = (s["result"] for s in sources)
    forward = primary["forward_checks"] + extra["forward_checks"]
    solver = primary["solver_checks"] + extra["solver_checks"]
    all_ok = all(s["result"].get("pass_all") for s in sources) and \
        all(r["pass_forward"] for r in forward) and \
        all(r["passed"] for r in solver)
    result = dict(protocol_version=PROTOCOL,
                  production_gradient_backend="CUDA_FP64_PyTorch_MSDA",
                  status="COMPLETE" if all_ok else "FAILED",
                  pass_all=bool(all_ok), images=expected,
                  forward_checks=forward, solver_checks=solver,
                  synthetic_cache_test=bool(primary["synthetic_cache_test"] and
                                            extra["synthetic_cache_test"]),
                  synthetic_sampler_check=dict(
                      passed=bool(primary["synthetic_sampler_check"]["passed"] and
                                  extra["synthetic_sampler_check"]["passed"]),
                      gpu0=primary["synthetic_sampler_check"],
                      gpu1=extra["synthetic_sampler_check"]),
                  derivative_rows=primary["derivative_rows"]+extra["derivative_rows"],
                  derivative_failed_groups=primary["derivative_failed_groups"]+
                                           extra["derivative_failed_groups"],
                  legacy_derivative_failed_groups=
                      primary["legacy_derivative_failed_groups"]+
                      extra["legacy_derivative_failed_groups"],
                  source_runs=[dict(path=s["path"], sha256=s["sha256"])
                               for s in sources])
    args.out.mkdir(parents=True)
    (args.out / "implementation_acceptance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result["status"], images=expected,
                          derivative_failed_groups=result["derivative_failed_groups"],
                          legacy_derivative_failed_groups=
                              result["legacy_derivative_failed_groups"])))
    if not all_ok:
        raise RuntimeError("six-image E2 gate failed; inspect source runs")


if __name__ == "__main__":
    main()
