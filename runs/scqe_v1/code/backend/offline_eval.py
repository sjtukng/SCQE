"""Offline COCO AP evaluation + synthetic evaluator self-tests (spec 5.4).

GT is used ONLY here, offline, after predictions exist. Detectors output
COCO-format detections json. AP reported in 0-100.
"""
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(detections, gt_coco, iou_type="bbox"):
    """detections: COCO-format list; gt_coco: path or COCO. Returns metrics dict."""
    coco = COCO(str(gt_coco)) if isinstance(gt_coco, (str, Path)) else gt_coco
    if len(detections) == 0:
        res = COCO()
        res.dataset = dict(images=coco.dataset["images"],
                           categories=coco.dataset["categories"], annotations=[])
        res.createIndex()
    else:
        res = coco.loadRes(detections)
    ev = COCOeval(coco, res, iou_type)
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = getattr(ev, "stats", None)
    if s is None:
        s = ev.eval["stats"]
    scalar = lambda x: None if x < 0 else float(100 * x)
    precision = ev.eval["precision"]  # [IoU, recall, class, area, max_det]
    per_class = {}
    for ci, category_id in enumerate(ev.params.catIds):
        values = precision[:, :, ci, 0, -1]
        valid = values[values >= 0]
        per_class[str(category_id)] = float(valid.mean() * 100) if valid.size else None
    return dict(AP=scalar(s[0]), AP50=scalar(s[1]), AP75=scalar(s[2]),
                AP_small=scalar(s[3]), AP_medium=scalar(s[4]), AP_large=scalar(s[5]),
                APs=scalar(s[3]), APm=scalar(s[4]), APl=scalar(s[5]),
                per_class_AP=per_class, n_img=len(coco.getImgIds()))


def make_gt(tmp_path, images, anns, cats):
    gt = dict(images=images, annotations=anns, categories=cats)
    p = Path(tmp_path) / "gt.json"
    json.dump(gt, open(p, "w"))
    return p


def synthetic_selftest(tmp_dir):
    """Behavioral self-tests of the evaluator on synthetic data (no real GT).

    - perfect detections -> AP 100
    - high-scoring spatially-distinct false positives -> AP drops
    - wrong class -> 0 AP on that GT
    - predictions on empty-GT images count as FP -> AP drops
    - maxDets=100 caps recall with 150 GTs -> AP reflects the cap
    """
    cats = [dict(id=1, name="a"), dict(id=2, name="b")]
    imgs = [dict(id=1, width=200, height=200), dict(id=2, width=100, height=100)]
    anns = [dict(id=1, image_id=1, category_id=1, bbox=[10, 10, 20, 20], area=400,
                 iscrowd=0)]
    gt = make_gt(tmp_dir, imgs, anns, cats)

    perfect = [dict(image_id=1, category_id=1, bbox=[10, 10, 20, 20],
                    score=0.9)]
    r = evaluate(perfect, gt)
    assert abs(r["AP"] - 100.0) < 1e-6, ("perfect AP", r)

    many_fp = [dict(image_id=1, category_id=1,
                    bbox=[60 + (i % 5) * 20, 60 + (i // 5) * 20, 15, 15],
                    score=0.95 + i * 1e-4) for i in range(20)] + perfect
    r2 = evaluate(many_fp, gt)
    assert r2["AP"] < r["AP"] - 20, ("fp should lower AP", r2)

    wrong_cls = [dict(image_id=1, category_id=2, bbox=[10, 10, 20, 20], score=0.9)]
    r3 = evaluate(wrong_cls, gt)
    assert r3["AP"] == 0.0, ("wrong class AP", r3)

    empty_gt_fp = perfect + [dict(image_id=2, category_id=1, bbox=[0, 0, 10, 10],
                                  score=0.99)]
    r4 = evaluate(empty_gt_fp, gt)
    assert r4["AP"] < r["AP"] - 20, ("empty-GT FP should lower AP", r4)

    # cap: 150 GTs, 150 perfect detections, maxDets=100 -> recall capped at 100/150
    imgs150 = [dict(id=k, width=500, height=500) for k in range(1, 2)]
    anns150 = [dict(id=k, image_id=1, category_id=1,
                    bbox=[(k % 12) * 40, (k // 12) * 40, 30, 30], area=900,
                    iscrowd=0) for k in range(1, 151)]
    gt150 = make_gt(tmp_dir, imgs150, anns150, [cats[0]])
    dets150 = [dict(image_id=1, category_id=1,
                    bbox=[(k % 12) * 40, (k // 12) * 40, 30, 30],
                    score=1.0 - k * 1e-3) for k in range(150)]
    r5 = evaluate(dets150, gt150)
    assert 50.0 < r5["AP"] < 80.0, ("cap100 behavior", r5)
    return dict(perfect=r["AP"], with_fp=r2["AP"], wrong_cls=r3["AP"],
                empty_gt_fp=r4["AP"], capped150=r5["AP"])


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        print(synthetic_selftest(d))
