"""Copy a complete DIOR producer prefix into a fresh, auditable run directory."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_prefix(source, entries, methods, record_state):
    count = 0
    for index, entry in enumerate(entries):
        qid = str(entry["image_id"])
        state = source / "states" / f"{qid}.pt"
        if record_state and (not state.is_file() or state.stat().st_size == 0):
            break
        try:
            for method in methods:
                job = source / "jobs" / f"{qid}__{method}.ready.json"
                plan = source / "plans" / f"{qid}__{method}.json"
                if not job.is_file() or not plan.is_file():
                    raise ValueError("missing job or plan")
                job_data = json.loads(job.read_text())
                plan_data = json.loads(plan.read_text())
                if (job_data.get("image_id") != int(qid) or
                        job_data.get("method") != method or
                        plan_data.get("image_id") != int(qid) or
                        plan_data.get("method") != method or
                        plan_data.get("query_index") != index):
                    raise ValueError("job or plan identity mismatch")
                proposal = source / "planned_proposals" / f"{qid}__{method}.json"
                if proposal.exists():
                    json.loads(proposal.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            break
        count += 1
    return count


def stage_prefix(source, target, split, seed, shot, methods, record_state, lock,
                 support, checkpoint, backend_checkpoint, backend_prototype):
    source = Path(source)
    launch = json.loads((source / "launch.json").read_text())
    meta = json.loads((source / "metadata.json").read_text())
    if launch.get("status") not in ("RUNNING", "FAILED", "PAUSED"):
        raise RuntimeError("resume source must be an interrupted run")
    entries_data = json.loads(Path(split).read_text())
    entries = (entries_data.get("queries", entries_data)
               if isinstance(entries_data, dict) else entries_data)
    checks = {
        "split_sha256": split,
        "support_sha256": support,
        "lock_sha256": lock,
        "checkpoint_sha256": checkpoint,
        "backend_checkpoint_sha256": backend_checkpoint,
        "backend_prototype_sha256": backend_prototype,
    }
    if meta.get("seed") != seed or meta.get("shot") != shot or \
            meta.get("methods") != methods or meta.get("expected_queries") != len(entries) or \
            meta.get("execution_mode") != "sequential_single_gpu":
        raise RuntimeError("resume source episode or execution mode differs")
    for key, path in checks.items():
        if meta.get(key) != digest(Path(path)):
            raise RuntimeError(f"resume source {key} differs from current input")
    prefix = inspect_prefix(source, entries, methods, record_state)
    if prefix < 1 or prefix >= len(entries):
        raise RuntimeError(f"no partial complete prefix to resume: {prefix}")
    for dirname in ("jobs", "plans", "states", "planned_proposals"):
        (target / dirname).mkdir()
    files = []
    for entry in entries[:prefix]:
        qid = str(entry["image_id"])
        paths = [source / "jobs" / f"{qid}__{method}.ready.json"
                 for method in methods]
        paths += [source / "plans" / f"{qid}__{method}.json"
                  for method in methods]
        if record_state:
            paths.append(source / "states" / f"{qid}.pt")
        paths += [source / "planned_proposals" / f"{qid}__{method}.json"
                  for method in methods
                  if (source / "planned_proposals" / f"{qid}__{method}.json").is_file()]
        for src in paths:
            relative = src.relative_to(source)
            dst = target / relative
            h = hashlib.sha256()
            with src.open("rb") as inp, dst.open("xb") as out:
                for chunk in iter(lambda: inp.read(1024 * 1024), b""):
                    out.write(chunk)
                    h.update(chunk)
            files.append({"path": str(relative), "sha256": h.hexdigest(),
                          "bytes": dst.stat().st_size})
    manifest = {
        "source": str(source),
        "source_launch_sha256": digest(source / "launch.json"),
        "source_metadata_sha256": digest(source / "metadata.json"),
        "split_sha256": meta["split_sha256"],
        "prefix_queries": prefix,
        "next_image_id": entries[prefix]["image_id"],
        "files": files,
    }
    (target / "resume_manifest.json").write_text(json.dumps(manifest, indent=2))
    return prefix
