"""Render only recorded SCQE results; missing diagnostics remain visibly missing."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


def finish(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def missing(ax, label):
    ax.text(0.5, 0.5, f"Missing recorded data: {label}", ha="center", va="center",
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _safe_name(value):
    """Return a stable filename component without exposing free-form text."""
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "case"


def _draw_detection_image(image_path, boxes, out_path, color):
    """Save one clean detection image: boxes only, with no labels or scores."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    line_width = max(2, min(width, height) // 500)
    draw = ImageDraw.Draw(image)
    for box in boxes or []:
        if len(box) != 4:
            continue
        values = [float(value) for value in box]
        if not all(math.isfinite(value) for value in values):
            continue
        x1, y1, x2, y2 = values
        x1 = max(0, min(width - 1, round(x1)))
        y1 = max(0, min(height - 1, round(y1)))
        x2 = max(0, min(width - 1, round(x2)))
        y2 = max(0, min(height - 1, round(y2)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    image.save(out_path, format="PNG", optimize=True)


def save_standalone_detection_images(cases, out):
    """Export B0 and deployed detections beside the qualitative composite."""
    manifest = []
    for index, case in enumerate(cases, start=1):
        stem = f"case_{index:02d}_{_safe_name(case.get('type', 'case'))}_query_{case['query_id']}"
        for field, suffix, color in (
                ("b0_detections", "b0_detections", (0, 220, 255)),
                ("deployed_detections", "deployed_detections", (255, 70, 210))):
            if field not in case:
                continue
            output = out / f"{stem}_{suffix}.png"
            _draw_detection_image(case["image_path"], case[field], output, color)
            manifest.append({
                "source_image": case["image_path"],
                "case_type": case.get("type"),
                "query_id": case["query_id"],
                "field": field,
                "box_count": len(case[field] or []),
                "output": str(output),
                "boxes_have_labels_or_scores": False,
            })
    (out / "standalone_detection_images.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path)
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--risk-run", type=Path)
    ap.add_argument("--lock", type=Path)
    ap.add_argument("--coverage", type=Path)
    ap.add_argument("--trajectories", type=Path)
    ap.add_argument("--assignments", type=Path)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--sensitivity", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    sources = {name: str(getattr(args, name)) if getattr(args, name) else None
               for name in ("metrics", "timing", "risk_run", "lock", "coverage",
                            "trajectories", "assignments", "cases", "sensitivity")}
    (args.out / "figure_sources.json").write_text(json.dumps(sources, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    if args.metrics and args.timing:
        metrics = json.loads(args.metrics.read_text())["summary"]
        costs = json.loads(args.timing.read_text())["summary"]
        common = sorted(set(metrics) & set(costs))
        for method in common:
            x, y = costs[method]["mean_ms"], metrics[method]["mean_AP"]
            ax.scatter(x, y)
            ax.annotate(method, (x, y))
        ax.set_xlabel("DIOR 50-query timing: mean wall time (ms)")
        ax.set_ylabel("DIOR full-query mean AP (0–100)")
        ax.set_title("Accuracy and cost use distinct declared samples")
    else:
        missing(ax, "global AP and fresh timing")
    finish(fig, args.out, "scqe_accuracy_cost")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    labels = ["Per-GT coverage", "Paired trajectories", "Proxy vs true risk",
              "Fixed-member assignment"]
    for axis, label in zip(axes.flat, labels):
        axis.set_title(label)
    for axis, path, key in ((axes[0, 0], args.coverage, "coverage"),
                            (axes[0, 1], args.trajectories, "trajectories"),
                            (axes[1, 1], args.assignments, "assignments")):
        if path is None:
            missing(axis, key)
        else:
            rows = json.loads(path.read_text())
            if not rows:
                missing(axis, key + " (empty)")
            elif key == "coverage":
                values = [r["deployed_minus_B0"] for r in rows]
                axis.hist(values, bins=30)
                axis.set_xlabel("Deployed − B0 maximum IoU per GT")
            elif key == "trajectories":
                stages = (("planned proposal", "planned_proposal_iou"),
                          ("deployed proposal", "deployed_proposal_iou"),
                          ("final detection", "deployed_final_iou"))
                baseline_fields = ("B0_proposal_iou", "B0_proposal_iou", "B0_final_iou")
                gained = [sum(r[base] < .5 <= r[field] for r in rows)
                          for (_, field), base in zip(stages, baseline_fields)]
                lost = [sum(r[field] < .5 <= r[base] for r in rows)
                        for (_, field), base in zip(stages, baseline_fields)]
                xs = list(range(len(stages)))
                axis.bar([x - .2 for x in xs], gained, width=.4, label="gained @.50")
                axis.bar([x + .2 for x in xs], lost, width=.4, label="lost @.50")
                axis.set_xticks(xs, [name for name, _ in stages], rotation=15)
                axis.set_ylabel("GT instances")
                axis.legend()
            else:
                values = [r["true_risk_delta"] for r in rows if r["assignment"] != "original"]
                if values:
                    axis.hist(values, bins=30)
                    axis.set_xlabel("True risk − original under fixed members")
                else:
                    missing(axis, "non-original assignments")
    risk_ax = axes[1, 0]
    if args.risk_run and args.lock:
        lock = json.loads(args.lock.read_text())
        ep = lock["selections"]["SCQE"]["eps_proxy"]
        et = lock["selections"]["SCQE"]["eps_true"]
        points = []
        for path in sorted((args.risk_run / "plans").glob("*__SCQE.json")):
            row = json.loads(path.read_text())
            if row.get("planned_m", 0) and row.get("proxy_cum") is not None and \
                    row.get("true_risk") is not None:
                points.append(row)
        if points:
            colors = ["tab:green" if p["accepted_m"] else "tab:red" for p in points]
            risk_ax.scatter([p["proxy_cum"] / ep for p in points],
                            [p["true_risk"] / et for p in points], c=colors, s=10)
            risk_ax.axvline(1, color="black", linestyle="--")
            risk_ax.axhline(1, color="black", linestyle="--")
            risk_ax.set_xlabel("Proxy risk / locked threshold")
            risk_ax.set_ylabel("Pre-gate true risk / locked threshold")
        else:
            missing(risk_ax, "nonempty SCQE plans")
    else:
        missing(risk_ax, "SCQE plan records and lock")
    fig.tight_layout()
    finish(fig, args.out, "scqe_mechanisms")

    if args.cases and json.loads(args.cases.read_text()):
        cases = json.loads(args.cases.read_text())
        save_standalone_detection_images(cases, args.out)
        fig, axes_case = plt.subplots(max(len(cases), 1), 4,
                                      figsize=(16, 4 * max(len(cases), 1)),
                                      squeeze=False)
        columns = [("initial_refs", "B0 references"),
                   ("planned_refs", "Planned references"),
                   ("b0_detections", "B0 detections"),
                   ("deployed_detections", "Deployed detections")]
        for ri, case in enumerate(cases):
            image = plt.imread(case["image_path"])
            for ci, (field, title) in enumerate(columns):
                axis = axes_case[ri, ci]
                axis.imshow(image)
                for bi, box in enumerate(case[field]):
                    x1, y1, x2, y2 = box
                    axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             fill=False, edgecolor="yellow" if ci < 2 else "cyan",
                                             linewidth=1))
                    if ci < 2:
                        label = str(case["slots"][bi])
                        if ci == 1:
                            label += f": {case['outgoing'][bi]}→{case['incoming'][bi]}"
                        axis.text(x1, y1, label, color="yellow", fontsize=6)
                axis.set_title(f"{case['type']} / {title}\n"
                               f"seed {case['support_seed']} query {case['query_id']} "
                               f"{case['status']} {case['planned_m']}→{case['accepted_m']}",
                               fontsize=9)
                if ci == 0:
                    axis.text(0, -0.04,
                              f"proxy {case['proxy_risk']} / {case['epsilon_proxy']}; "
                              f"true {case['true_risk']} / {case['epsilon_true']}; "
                              f"B0 restored {case['fallback_verified']}",
                              transform=axis.transAxes, fontsize=7, va="top")
                axis.set_axis_off()
        fig.suptitle("Yellow: exchanged slot IDs; planned refs retain rejected changes. "
                     "Full risk and thresholds are in qualitative_cases.json", fontsize=10)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        missing(ax, "preselected qualitative case records")
    finish(fig, args.out, "scqe_qualitative_cases")

    fig, ax = plt.subplots(figsize=(6, 4))
    if args.sensitivity:
        rows = json.loads(args.sensitivity.read_text())
        xs = list(range(len(rows)))
        ax.plot(xs, [r["AP"] for r in rows], marker="o")
        ax.set_xticks(xs, [f"P={r['P']}, M={r['M']}" for r in rows], rotation=45)
        ax.set_ylabel("Global AP")
    else:
        missing(ax, "five measured P/M configurations")
    finish(fig, args.out, "scqe_sensitivity")


if __name__ == "__main__":
    main()
