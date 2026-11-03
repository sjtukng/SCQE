"""Validate the six-image E2 gate before SCQE calibration or evaluation."""
import hashlib
import json
from pathlib import Path

from .config import RUN_DIR

PROTOCOL = "E2-2026-10-16-fp64-vjp-manual-jvp"


def require_scqe_acceptance(path: Path) -> dict:
    result = json.loads(path.read_text())
    manifest = json.loads((RUN_DIR / "splits/dior_dcal40.json").read_text())
    expected = [e["image_id"] for e in manifest["queries"][:6]]
    if result.get("protocol_version") != PROTOCOL or \
            result.get("production_gradient_backend") != "CUDA_FP64_PyTorch_MSDA" or \
            result.get("images") != expected or \
            result.get("derivative_rows") != 6 * 5 * 2 * 9 or \
            result.get("derivative_failed_groups") != 0 or \
            not result.get("pass_all") or \
            not result.get("synthetic_sampler_check", {}).get("passed") or \
            len(result.get("forward_checks", [])) != 6 or \
            not all(r.get("pass_forward") for r in result["forward_checks"]):
        raise RuntimeError("six-image FP64 E2 acceptance has not passed")
    sources = result.get("source_runs", [])
    if len(sources) != 2:
        raise RuntimeError("E2 gate must identify both immutable GPU source runs")
    for source in sources:
        content = Path(source["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise RuntimeError(f"E2 source hash changed: {source['path']}")
    return result
