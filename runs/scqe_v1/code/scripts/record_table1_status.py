"""Record every Table 1 cell without substituting unpublished or simulated AP."""
import csv
import hashlib
import json
import statistics
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/scqe_server_20261014"
SEEDS = (2023, 42, 1234, 7, 99)
DOTA = ROOT / "runs/scqe_v1/splits/dota_v1_hbb"
K_SOURCE = ROOT / "runs/scqe_v1/splits/k_coco2017"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=None)
def sha(path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def blockers(protocol, dataset, shot, method, seed, audit, acceptance):
    missing = []
    if dataset == "DOTA-HBB":
        dota_audit = read(DOTA / "data_audit.json") or {}
        dota_integrity = read(DOTA / "integrity_report.json") or {}
        if dota_audit.get("status") != "DATA_PREPARED_MODEL_BACKEND_PENDING" or \
                dota_integrity.get("status") != "PASS":
            missing.append("DOTA parent/chip inputs or integrity audit are incomplete")
        if protocol == "D":
            backend = ROOT / f"runs/scqe_v1/backend_dota/seed{seed}_shot{shot}"
            proto = backend / f"DOTA_HBB_{shot}shot_seed{seed}.vitl14.bbox.p{shot}.sk.pkl"
            weights = backend / "adapt/model_final.pth"
            if not proto.is_file() or not weights.is_file():
                missing.append("DOTA support-specific adaptation/prototypes incomplete")
            run = RUN / f"runs/dota_D_{shot}_seed{seed}/launch.json"
            if not run.is_file() or (read(run) or {}).get("status") != "COMPLETE":
                missing.append("DOTA parent-level inference/evaluation incomplete")
    else:
        if not audit or not audit.get("formal_ready"):
            missing.append("DIOR 4,900-query use-history attestation is absent")
        if method != "FT-FSOD":
            backend = ROOT / f"runs/scqe_v1/backend/seed{seed}_shot{shot}"
            proto = backend / f"DIOR_{shot}shot_seed{seed}.vitl14.bbox.p{shot}.sk.pkl"
            weights = backend / "adapt/model_final.pth"
            if not proto.is_file() or not weights.is_file():
                missing.append("support-specific CD-ViTO adaptation/prototypes incomplete")
    if protocol == "D":
        if not (RUN / "config_lock_D.json").is_file():
            missing.append("D threshold/budget calibration lock is absent")
    else:
        k_audit = read(K_SOURCE / "source_audit.json") or {}
        k_integrity = read(K_SOURCE / "integrity_report.json") or {}
        if k_audit.get("status") != "SOURCE_FROZEN_BACKEND_NOT_CALIBRATED" or \
                k_integrity.get("status") != "PASS":
            missing.append("K non-target calibration source or integrity audit is incomplete")
        if not (RUN / "config_lock_K.json").is_file():
            missing.append("K independent calibration lock is absent")
        if method in ("B0", "B1", "B2", "B3", "SCQE", "CD-ViTO"):
            missing.append("independent K target-only adaptation is absent")
        if method in ("CD-ViTO", "FT-FSOD"):
            missing.append(f"qualified {method} external baseline is absent")
    if method == "SCQE" and not (acceptance and acceptance.get("pass_all")):
        missing.append("SCQE E2 acceptance failed")
    return missing


def main():
    audit_path = (RUN / "audit_attested/data_audit.json"
                  if (RUN / "audit_attested/data_audit.json").exists()
                  else RUN / "audit/data_audit.json")
    audit = read(audit_path)
    acceptance_path = (RUN / "acceptance_fp64_vjp_complete/implementation_acceptance.json"
                       if (RUN / "acceptance_fp64_vjp_complete/implementation_acceptance.json").exists()
                       else RUN / "acceptance_ref64/implementation_acceptance.json")
    acceptance = read(acceptance_path)
    measured_rows = {}
    dota_measured_rows = {}
    historical_b0_rows = {}
    for shot in (1, 5, 10):
        for seed in SEEDS:
            name = (f"B0_seed{seed}.json" if shot == 5 else
                    f"B0_shot{shot}_seed{seed}.json")
            single = read(RUN / "metrics/per_support" / name)
            if single and single.get("status") == "COMPLETE":
                historical_b0_rows[(shot, seed, "B0")] = single
        measured = read(RUN / f"metrics/dior_D_{shot}_B0/metrics.json")
        if measured:
            historical_b0_rows.update({(shot, x["support_seed"], x["method"]): x
                                       for x in measured.get("per_support", [])})
    for shot in (1, 5, 10):
        full = (read(RUN / f"metrics/dior_D_{shot}/metrics.json") or
                read(RUN / f"metrics/dior_D_{shot}_primary/metrics.json"))
        if full:
            measured_rows.update({(shot, x["support_seed"], x["method"]): x
                                  for x in full.get("per_support", [])})
        if not full:
            for seed in SEEDS:
                single = read(RUN / f"metrics/per_support_full/shot{shot}_seed{seed}.json")
                if single is None and shot == 5:
                    single = read(RUN / f"metrics/per_support_full/seed{seed}.json")
                if single and single.get("status") == "COMPLETE":
                    measured_rows.update({(shot, x["support_seed"], x["method"]): x
                                          for x in single.get("per_support", [])})
        dota_full = (read(RUN / f"metrics/dota_D_{shot}/metrics.json") or
                     read(RUN / f"metrics/dota_D_{shot}_primary/metrics.json"))
        if dota_full:
            dota_measured_rows.update({(shot, x["support_seed"], x["method"]): x
                                       for x in dota_full.get("per_support", [])})
        else:
            for seed in SEEDS:
                single = read(RUN / f"metrics/per_support_dota/shot{shot}_seed{seed}.json")
                if single and single.get("status") == "COMPLETE":
                    dota_measured_rows.update({(shot, x["support_seed"], x["method"]): x
                                               for x in single.get("per_support", [])})
    rows = []
    for protocol, methods in (("D", ("B0", "B1", "B2", "B3", "SCQE")),
                              ("K", ("B0", "B1", "B2", "B3", "SCQE",
                                     "CD-ViTO", "FT-FSOD"))):
        for dataset in ("DIOR", "DOTA-HBB"):
            for shot in (1, 5, 10):
                for method in methods:
                    for seed in SEEDS:
                        reasons = blockers(protocol, dataset, shot, method,
                                           seed, audit, acceptance)
                        support = (ROOT / "runs/sqr_v1/splits" /
                                   f"dior_support_{shot}shot_seed{seed}.json"
                                   if dataset == "DIOR" else
                                   DOTA / f"dota_support_{shot}shot_seed{seed}.json")
                        checkpoint = ROOT / (f"runs/scqe_v1/backend/seed{seed}_shot{shot}/adapt/model_final.pth"
                                             if dataset == "DIOR" else
                                             f"runs/scqe_v1/backend_dota/seed{seed}_shot{shot}/adapt/model_final.pth")
                        actual = (measured_rows.get((shot, seed, method)) if
                                  protocol == "D" and dataset == "DIOR" else
                                  dota_measured_rows.get((shot, seed, method)) if
                                  protocol == "D" and dataset == "DOTA-HBB" else None)
                        full_run = RUN / f"runs/{'dior' if dataset == 'DIOR' else 'dota'}_D_{shot}_seed{seed}"
                        if dataset == "DIOR" and shot == 5 and seed == 1234 and \
                                (RUN / "runs/dior_D_5_seed1234_resume1").exists():
                            full_run = RUN / "runs/dior_D_5_seed1234_resume1"
                        if dataset == "DIOR" and shot == 5 and seed == 1234 and \
                                (RUN / "runs/dior_D_5_seed1234_resume2").exists():
                            full_run = RUN / "runs/dior_D_5_seed1234_resume2"
                        run = full_run if protocol == "D" and full_run.exists() else None
                        launch = read(run / "launch.json") if run else None
                        status = ("BLOCKED" if reasons else
                                  "COMPLETE" if actual else
                                  "RUNNING" if launch and launch.get("status") == "COMPLETE" else
                                  launch.get("status", "READY") if launch else "READY")
                        rows.append(dict(run_id=f"{protocol}_{dataset}_{shot}_{method}_{seed}",
                                         protocol=protocol, dataset=dataset,
                                         split_hash=sha(ROOT / "runs/scqe_v1/splits/dior_final_queries.json")
                                         if dataset == "DIOR" else
                                         sha(DOTA / "val_chips_1024_gap200.json"),
                                         support_hash=sha(support),
                                         support_seed=seed, shot=shot, method=method,
                                         config_hash=sha(RUN / "config_lock_D.json")
                                         if protocol == "D" else sha(RUN / "config_lock_K.json"),
                                         checkpoint_hash=sha(checkpoint)
                                         if protocol == "D" else None,
                                         status=status,
                                         reason="; ".join(reasons) if reasons else
                                         "prediction complete; evaluation pending" if
                                         launch and launch.get("status") == "COMPLETE" and not actual else None,
                                         AP=actual["AP"] if actual else None,
                                         AP50=actual["AP50"] if actual else None,
                                         AP75=actual["AP75"] if actual else None,
                                         AP_small=actual["AP_small"] if actual else None,
                                         AP_medium=actual["AP_medium"] if actual else None,
                                         AP_large=actual["AP_large"] if actual else None,
                                         prediction_path=actual["prediction_path"] if actual else None))
    table = RUN / "paper_tables"
    table.mkdir(parents=True, exist_ok=True)
    with (table / "table1_raw_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for i in range(0, len(rows), len(SEEDS)):
        group = rows[i:i + len(SEEDS)]
        head = group[0]
        n_measured = sum(x["AP"] is not None for x in group)
        status = ("COMPLETE" if n_measured == 5 else
                  "FAILED" if any(x["status"] == "FAILED" for x in group) else
                  "RUNNING" if any(x["status"] == "RUNNING" for x in group) else
                  "BLOCKED" if any(x["status"] == "BLOCKED" for x in group) else
                  "READY")
        summary.append(dict(protocol=head["protocol"], dataset=head["dataset"],
                            shot=head["shot"], method=head["method"],
                            status=status, n_measured=n_measured,
                            mean_AP=statistics.mean(x["AP"] for x in group) if n_measured == 5 else None,
                            sample_SD=statistics.stdev(x["AP"] for x in group) if n_measured == 5 else None,
                            mean_AP50=statistics.mean(x["AP50"] for x in group) if n_measured == 5 else None,
                            mean_AP75=statistics.mean(x["AP75"] for x in group) if n_measured == 5 else None,
                            reason="; ".join(sorted({part.strip()
                                                      for x in group for part in
                                                      (x["reason"] or "").split(";")
                                                      if part.strip()}))))
    with (table / "table1_all_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    measured_count = sum(x["AP"] is not None for x in rows)
    dior_d_measured = sum(x["AP"] is not None for x in rows
                          if x["protocol"] == "D" and x["dataset"] == "DIOR")
    dota_d_measured = sum(x["AP"] is not None for x in rows
                          if x["protocol"] == "D" and x["dataset"] == "DOTA-HBB")
    manifest = dict(status="RUNNING" if any(x["status"] == "RUNNING" for x in rows)
                    else "BLOCKED", formal_measurements=measured_count,
                    raw_rows=len(rows), aggregate_rows=len(summary),
                    audit_path=str(audit_path),
                    acceptance_path=str(acceptance_path),
                    table1_raw_status=str(table / "table1_raw_status.csv"),
                    table1_all_status=str(table / "table1_all_status.csv"))
    (table / "table1_manifest.json").write_text(json.dumps(manifest, indent=2))
    completed_adaptation = {
        shot: sum((ROOT / f"runs/scqe_v1/backend/seed{seed}_shot{shot}/adapt/model_final.pth").is_file()
                  for seed in SEEDS)
        for shot in (1, 5, 10)
    }
    report = [
        "# Table 1 execution status", "",
        f"Formal measured AP: {measured_count} / {len(rows)} support cells; "
        f"complete aggregates: {sum(x['status'] == 'COMPLETE' for x in summary)} / {len(summary)}.",
        "Paper target numbers and previous development results are excluded.", "",
        "## Completed preparation", "",
        "- Frozen DIOR support COCO inputs: 15 / 15 checked against source instances.",
        "- DIOR D adapted checkpoints currently present: " +
        ", ".join(f"{shot}-shot {completed_adaptation[shot]}/5" for shot in (1, 5, 10)) + ".",
        "- E0/E1 history and data audit passed; " +
        ("six-image E2 FP64 VJP/JVP acceptance passed."
         if acceptance and acceptance.get("pass_all") else
         "E2 acceptance has not passed."),
        "", "## Formal progress and blockers", "",
        "- DIOR history was attested by the user and the 4,900-query input audit passed.",
        ("- SCQE: E2 passed and the D calibration lock is present; see the raw CSV for each held-out support AP."
         if acceptance and acceptance.get("pass_all") and (RUN / "config_lock_D.json").is_file() else
         "- SCQE: E2 passed; D calibration lock and formal SCQE evaluation remain."
         if acceptance and acceptance.get("pass_all") else
         "- SCQE: E2 acceptance failed " +
         f"{acceptance.get('derivative_failed_groups') if acceptance else 'unknown'} derivative groups."),
        f"- DIOR D: {dior_d_measured}/75 support-method cells measured with the deterministic backend; prior B0 values are historical and excluded.",
        f"- DOTA D: {dota_d_measured}/75 support-method cells measured at parent level.",
        "- DOTA-HBB: parent images, annotations, chips, supports and integrity audit are prepared; adapted backend and parent-level inference/evaluation are pending.",
        "- K: non-target source and integrity audit are prepared. FT-FSOD source and base weight were acquired; independent calibration lock, target-only adaptation, pretraining exposure audit and qualified external baseline results are pending.",
        "", "## Files", "",
        "- table1_raw_status.csv: one row per support; AP appears only after formal evaluation.",
        "- table1_all_status.csv: one row per Table 1 protocol/dataset/shot/method cell.",
        "- table1_manifest.json: counts and provenance paths.",
    ]
    for shot in (1, 5, 10):
        b0_rows = [historical_b0_rows[(shot, seed, "B0")] for seed in SEEDS
                   if (shot, seed, "B0") in historical_b0_rows]
        if b0_rows:
            report.extend(["", f"## Historical DIOR D {shot}-shot B0 measurements (old backend)", "",
                           "| Support seed | AP | AP50 | AP75 |",
                           "| ---: | ---: | ---: | ---: |"])
            report.extend(f"| {row['support_seed']} | {row['AP']:.4f} | "
                          f"{row['AP50']:.4f} | {row['AP75']:.4f} |"
                          for row in b0_rows)
            if len(b0_rows) == len(SEEDS):
                report.append(f"| Mean ± sample SD | "
                              f"{statistics.mean(row['AP'] for row in b0_rows):.4f} ± "
                              f"{statistics.stdev(row['AP'] for row in b0_rows):.4f} | "
                              f"{statistics.mean(row['AP50'] for row in b0_rows):.4f} | "
                              f"{statistics.mean(row['AP75'] for row in b0_rows):.4f} |")
            report.extend(["", "Each row uses 4,900 held-out images, but the "
                           "recognition backend has since changed to the deterministic "
                           "standard GPU attention path. These values are preserved for "
                           "audit and excluded from the current unified Table 1."])
    (table / "table1_execution_report.md").write_text("\n".join(report) + "\n")
    print(f"recorded {len(rows)} support cells and {len(summary)} aggregate cells")


if __name__ == "__main__":
    main()
