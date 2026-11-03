"""Record available DIOR D support adaptation artifacts and their hashes."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/scqe_server_20261014"
REPO = ROOT / "third_party/CDFSOD-benchmark"


@lru_cache(maxsize=None)
def sha(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for shot in (1, 5, 10):
        for seed in (2023, 42, 1234, 7, 99):
            backend = ROOT / f"runs/scqe_v1/backend/seed{seed}_shot{shot}"
            support = ROOT / f"runs/sqr_v1/splits/dior_support_{shot}shot_seed{seed}.json"
            coco = ROOT / f"c0_datasets/DIOR/annotations/{shot}_shot_seed{seed}.json"
            proto = backend / f"DIOR_{shot}shot_seed{seed}.vitl14.bbox.p{shot}.sk.pkl"
            weights = backend / "adapt/model_final.pth"
            config = backend / "adapt/config.yaml"
            metrics = backend / "adapt/metrics.json"
            iterations = []
            if metrics.is_file():
                for line in metrics.read_text().splitlines():
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if "iteration" in record and "total_loss" in record:
                        iterations.append(int(record["iteration"]))
            rows.append(dict(protocol="D", dataset="DIOR", seed=seed, shot=shot,
                             status="COMPLETE" if weights.is_file() and proto.is_file()
                             else "INCOMPLETE",
                             origin="new_gpu_run_2026-10-15" if shot == 5 and seed in (7, 99)
                             else "pre_existing" if weights.is_file() and proto.is_file()
                             else "not_run",
                             support_path=str(support), support_sha256=sha(support),
                             backend_coco_path=str(coco), backend_coco_sha256=sha(coco),
                             prototype_path=str(proto) if proto.is_file() else None,
                             prototype_sha256=sha(proto),
                             checkpoint_path=str(weights) if weights.is_file() else None,
                             checkpoint_sha256=sha(weights),
                             checkpoint_bytes=weights.stat().st_size if weights.is_file() else None,
                             config_path=str(config) if config.is_file() else None,
                             config_sha256=sha(config),
                             last_logged_training_iteration=max(iterations) if iterations else None))
    out = RUN / "adaptation_summary.json"
    out.write_text(json.dumps(dict(
        status="COMPLETE" if all(x["status"] == "COMPLETE" for x in rows) else "PARTIAL",
        initial_checkpoint_path=str(REPO / "weights/trained/few-shot/vitl_0089999.pth"),
        initial_checkpoint_sha256=sha(REPO / "weights/trained/few-shot/vitl_0089999.pth"),
        support_episodes=rows,
        note="These are support adaptation artifacts, not formal Table 1 AP measurements."),
        indent=2))
    print(f"{sum(x['status'] == 'COMPLETE' for x in rows)}/{len(rows)} adaptations complete: {out}")


if __name__ == "__main__":
    main()
