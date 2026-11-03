"""E3 stage-2: classify pilot proposals, evaluate AP per (method, seed),
LOO single-image deletion check for M vs B0, and dump Go/No-Go inputs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/root-user/projects/PF_RPN_SR")
E3 = ROOT / "runs/scqe_v1/e3_pilot"
BACKEND = ROOT / "runs/scqe_v1/code/backend"
GT = ROOT / "runs/scqe_v1/splits/dcheck60_gt.json"
SEEDS = (2023, 42, 1234)
METHODS = ("B0", "B1", "B2", "B3", "M")


def classify(seed, run_key):
    pf = E3 / "proposals" / f"{seed}__{run_key}.json"
    det = E3 / "detections" / f"{seed}__{run_key}.json"
    if not det.exists():
        r = subprocess.run(
            [sys.executable, str(BACKEND / "classify_proposals.py"),
             "--seed", str(seed), "--shot", "5",
             "--proposals", str(pf), "--out", str(det), "--device", "cuda:1"],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"classify failed {seed}/{run_key}: {r.stderr[-400:]}")
    return det


def evaluate_det(det, gt=GT):
    ev = subprocess.run(
        [sys.executable, "-c",
         "import sys,json; sys.path.insert(0,r'%s'); from offline_eval import evaluate; "
         "print(json.dumps(evaluate(json.load(open(sys.argv[1])), sys.argv[2])))"
         % BACKEND, str(det), str(gt)],
        capture_output=True, text=True)
    return json.loads(ev.stdout.strip().splitlines()[-1])


def main():
    ap_all = {}
    dets = {}
    for seed in SEEDS:
        for meth in METHODS:
            det = classify(seed, meth)
            m = evaluate_det(det)
            ap_all[(seed, meth)] = m
            dets[(seed, meth)] = json.load(open(det))
            print(f"seed{seed} {meth:3s} AP={m['AP']:.2f} AP50={m['AP50']:.2f}",
                  flush=True)

    # LOO for M vs B0: drop the image with the largest influence, recompute
    def ap_without(seed, meth, drop_img, keep_imgs):
        sub = [d for d in dets[(seed, meth)] if d["image_id"] in keep_imgs
               and d["image_id"] != drop_img]
        gt = ROOT / "runs/scqe_v1/splits/dcheck60_gt.json"
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(sub, f)
            tmp = f.name
        return evaluate_det(tmp, gt)["AP"]

    keep = {}
    for seed in SEEDS:
        imgs = sorted({d["image_id"] for d in dets[(seed, "B0")]})
        keep[seed] = set(imgs)
    loo = {}
    for seed in SEEDS:
        imgs = sorted(keep[seed])
        diffs = []
        base_diff = ap_all[(seed, "M")]["AP"] - ap_all[(seed, "B0")]["AP"]
        best_img, worst = None, None
        for img in imgs:
            d = (ap_without(seed, "M", img, keep[seed])
                 - ap_without(seed, "B0", img, keep[seed]))
            diffs.append((d, img))
        diffs.sort()
        loo[seed] = dict(base_diff=base_diff, min_diff_after_drop=diffs[0][0],
                         drop_img=diffs[0][1],
                         max_diff_after_drop=diffs[-1][0])
        print(f"LOO seed{seed}: base={base_diff:.3f} "
              f"min-after-drop={diffs[0][0]:.3f} (img {diffs[0][1]})", flush=True)

    json.dump(dict(ap={f"{s}/{m}": v for (s, m), v in ap_all.items()},
                   loo=loo),
              open(E3 / "metrics.json", "w"), indent=1)
    print("wrote", E3 / "metrics.json")


if __name__ == "__main__":
    main()
