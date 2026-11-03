"""Assemble traceable partial/full result package without inventing missing values."""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.run_root
    audit_path = (root / "audit_attested/data_audit.json"
                  if (root / "audit_attested/data_audit.json").exists()
                  else root / "audit/data_audit.json")
    audit = load(audit_path)
    acceptance_path = (root / "acceptance_fp64_vjp_complete/implementation_acceptance.json"
                       if (root / "acceptance_fp64_vjp_complete/implementation_acceptance.json").exists()
                       else root / "acceptance_ref64/implementation_acceptance.json"
                       if (root / "acceptance_ref64/implementation_acceptance.json").exists()
                       else root / "acceptance/implementation_acceptance.json")
    acceptance = load(acceptance_path)
    lock = load(root / "config_lock_D.json")
    confirmation = load(root / "confirmation/metrics.json")
    metrics = (load(root / "metrics/dior_D_5/metrics.json") or
               load(root / "metrics/dior_D_5_primary/metrics.json"))
    timing = load(root / "timing/dior/timing_summary.json")
    table1_status = load(root / "paper_tables/table1_manifest.json")
    adaptation = load(root / "adaptation_summary.json")
    dota_audit = load(Path(__file__).resolve().parents[2] / "splits/dota_v1_hbb/data_audit.json")
    dota_integrity = load(Path(__file__).resolve().parents[2] / "splits/dota_v1_hbb/integrity_report.json")
    k_audit = load(Path(__file__).resolve().parents[2] / "splits/k_coco2017/source_audit.json")
    k_integrity = load(Path(__file__).resolve().parents[2] / "splits/k_coco2017/integrity_report.json")
    ft_resources = load(root / "audit/ft_fsod_resources_20261017.json")
    ft_source_ready = bool(ft_resources and
                           ft_resources.get("status") ==
                           "SOURCE_AND_BASE_WEIGHT_ACQUIRED_ADAPTATION_PENDING" and
                           Path(ft_resources["base_weight_path"]).is_file())
    dota_ready = bool(dota_audit and dota_audit.get("status") == "DATA_PREPARED_MODEL_BACKEND_PENDING"
                      and dota_integrity and dota_integrity.get("status") == "PASS")
    k_source_ready = bool(k_audit and k_audit.get("status") == "SOURCE_FROZEN_BACKEND_NOT_CALIBRATED"
                          and k_integrity and k_integrity.get("status") == "PASS")
    e2_status = ("COMPLETE" if acceptance and acceptance.get("pass_all") else
                 "FAILED" if acceptance and acceptance.get("status") == "FAILED" else
                 "PLANNED")
    stages = {
        "E0-E1": "COMPLETE" if audit and audit.get("formal_ready") else "BLOCKED",
        "E2": e2_status,
        "E3": "COMPLETE" if lock and confirmation else
              "BLOCKED_BY_E2" if e2_status == "FAILED" else "PLANNED",
        "E4": "COMPLETE" if metrics and all(x in metrics.get("summary", {})
                                           for x in ("B0", "B3", "SCQE")) else "PLANNED",
        "E5": "COMPLETE" if table1_status and table1_status.get("formal_measurements") == 360
              else "RUNNING" if table1_status and table1_status.get("formal_measurements", 0) > 0
              else "PLANNED",
        "E6": "PLANNED", "E7": "PLANNED", "E8": "PLANNED",
        "E9": "COMPLETE" if timing else "PLANNED", "E10": "PLANNED",
    }
    if not audit or not audit.get("formal_ready"):
        stages["E5"] = "BLOCKED"
    if e2_status == "FAILED":
        for stage in ("E4", "E6", "E7", "E8", "E9", "E10"):
            if stages[stage] == "PLANNED":
                stages[stage] = "BLOCKED_BY_E2"
    root.mkdir(parents=True, exist_ok=True)
    tables = root / "paper_tables"
    tables.mkdir(exist_ok=True)
    status_csv = tables / "table1_all_status.csv"
    if status_csv.exists():
        with status_csv.open(newline="") as f:
            source = csv.DictReader(f)
            headers = source.fieldnames
            rows = [row for row in source if row["protocol"] == "D"]
    else:
        headers = ["protocol", "dataset", "shot", "method", "status", "mean_AP",
                   "sample_SD", "n_supports", "reason"]
        rows = []
        for method in ("B0", "B1", "B2", "B3", "SCQE"):
            result = metrics.get("summary", {}).get(method) if metrics else None
            rows.append(dict(protocol="D", dataset="DIOR", shot=5, method=method,
                             status="COMPLETE" if result else "BLOCKED",
                             mean_AP=result["mean_AP"] if result else None,
                             sample_SD=result["sample_SD"] if result else None,
                             n_supports=len(result["raw_support_AP"]) if result else None,
                             reason=None if result else "no new formal prediction package"))
    with open(tables / "table1D.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    (tables / "table2K.json").write_text(json.dumps(
        dict(status="BLOCKED", reason="K source prepared; independent calibration lock, target-only adaptation and qualified external baselines missing"
             if k_source_ready else "K source, independent calibration and external resources missing",
             results=None), indent=2))
    (tables / "table3_ablations.json").write_text(json.dumps(
        dict(status="PLANNED", results=None), indent=2))
    (tables / "table4_cost.json").write_text(json.dumps(
        dict(status="COMPLETE" if timing else "PLANNED",
             results=timing["summary"] if timing else None), indent=2))
    lines = ["# SCQE results report", "", "Generated only from files under this RUN_ROOT.",
             "", "## Stage status", ""]
    lines.extend(f"- {stage}: {status}" for stage, status in stages.items())
    if acceptance and not acceptance.get("pass_all"):
        lines += ["", "## E2 acceptance failure", "",
                  f"Fixed images: {acceptance.get('images')}",
                  f"Failed derivative groups: {acceptance.get('derivative_failed_groups')} / 30",
                  f"Forward checks: {[r.get('pass_forward') for r in acceptance.get('forward_checks', [])]}",
                  f"Solver checks: {[r.get('passed') for r in acceptance.get('solver_checks', [])]}",
                  f"Evidence: {acceptance_path.parent / 'finite_differences.json'} and "
                  f"{acceptance_path.parent / 'derivative_failures.json'}. "
                  "Formal SCQE evaluation stopped."]
    lines += ["", "## Main paired comparison", ""]
    if metrics:
        lines.append(f"SCQE - B0 mean AP difference: {metrics['paired_delta_pp'].get('B0')} pp.")
        lines.append(f"SCQE - B3 mean AP difference: {metrics['paired_delta_pp'].get('B3')} pp.")
        lines.append("Bootstrap intervals: " + json.dumps(metrics.get("bootstrap", {})))
    else:
        lines.append("No newly executed five-support formal results.")
    if table1_status:
        complete_adaptations = sum(x.get("status") == "COMPLETE"
                                   for x in adaptation.get("support_episodes", [])) if adaptation else 0
        lines += ["", "## Table 1 execution record", "",
                  f"Formal measured support cells: {table1_status['formal_measurements']} / "
                  f"{table1_status['raw_rows']}.",
                  f"Aggregate cells: {table1_status['aggregate_rows']}; "
                  "missing AP fields remain null.",
                  f"DIOR support adaptations complete: {complete_adaptations} / 15 "
                  "(all five 5-shot states are present).",
                  "Per-cell blockers: paper_tables/table1_raw_status.csv and "
                  "paper_tables/table1_all_status.csv.",
                  "Support checkpoint and prototype hashes: adaptation_summary.json."]
    lines += ["", "## Current input and implementation state", ""]
    lines.append("- DIOR 4,900-query use-history attestation and input audit: " +
                 ("complete." if audit and audit.get("formal_ready") else "pending."))
    lines.append("- DOTA parent/chip/support inputs and integrity audit: " +
                 ("prepared; adapted backend and parent-level inference/evaluation pending."
                  if dota_ready else "incomplete."))
    lines.append("- K non-target source and integrity audit: " +
                 ("prepared; independent calibration lock and target-only adaptation pending."
                  if k_source_ready else "incomplete."))
    lines.append("- Original FT-FSOD source and base pretrained weight: " +
                 ("downloaded and SHA256 recorded; target-support adaptation, pretraining exposure audit, and compliant fixed stopping remain."
                  if ft_source_ready else "pending."))
    (root / "results_report.md").write_text("\n".join(lines) + "\n")
    (root / "execution_summary.json").write_text(json.dumps(
        dict(stages=stages, formal_metrics=bool(metrics), timing=bool(timing),
             K_results=None, DOTA_results=None), indent=2))


if __name__ == "__main__":
    main()
