import argparse
import numpy as np
import pickle
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def fast_computeOks(self, imgId, catId):
    """Vectorized OKS computation to replace the slow pycocotools double for-loop."""
    p = self.params
    gts = self._gts[imgId, catId]
    dts = self._dts[imgId, catId]
    inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
    dts = [dts[i] for i in inds]
    if len(dts) > p.maxDets[-1]:
        dts = dts[0 : p.maxDets[-1]]
    if len(gts) == 0 or len(dts) == 0:
        return []

    sigmas = p.kpt_oks_sigmas
    vars = (sigmas * 2) ** 2

    # Extract all GT data -> Shape: (M, k)
    xg = np.array([gt["keypoints"][0::3] for gt in gts])
    yg = np.array([gt["keypoints"][1::3] for gt in gts])
    vg = np.array([gt["keypoints"][2::3] for gt in gts])
    areas = np.array([gt["area"] for gt in gts])

    # Extract all DT data -> Shape: (N, k)
    xd = np.array([dt["keypoints"][0::3] for dt in dts])
    yd = np.array([dt["keypoints"][1::3] for dt in dts])

    # Broadcasting to compute all pairwise distances -> Shape: (N, M, k)
    dx = xd[:, np.newaxis, :] - xg[np.newaxis, :, :]
    dy = yd[:, np.newaxis, :] - yg[np.newaxis, :, :]

    # Compute OKS components
    e = (
        (dx**2 + dy**2)
        / vars
        / (areas[np.newaxis, :, np.newaxis] + np.spacing(1))
        / 2
    )

    # Mask out invisible keypoints
    vg_mask = vg[np.newaxis, :, :] > 0
    oks_all = np.exp(-e) * vg_mask

    # Sum over keypoints and divide by number of visible keypoints
    num_visible = np.sum(vg > 0, axis=1)
    valid_gt = num_visible > 0

    ious = np.zeros((len(dts), len(gts)))
    ious[:, valid_gt] = (
        np.sum(oks_all[:, valid_gt, :], axis=2) / num_visible[valid_gt]
    )

    return ious


def fast_evaluateImg(self, imgId, catId, aRng, maxDet):
    """Vectorized evaluateImg to replace the slow pycocotools triple for-loop."""
    p = self.params
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return None

    for g in gt:
        if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
            g["_ignore"] = 1
        else:
            g["_ignore"] = 0

    gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = np.array([int(o["iscrowd"]) for o in gt], dtype=bool)

    ious = (
        self.ious[imgId, catId][:, gtind]
        if len(self.ious[imgId, catId]) > 0
        else self.ious[imgId, catId]
    )

    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g["_ignore"] for g in gt])
    dtIg = np.zeros((T, D))
    gtIds = np.array([g["id"] for g in gt])
    dtIds = np.array([d["id"] for d in dt])

    if len(ious) > 0:
        for tind, t in enumerate(p.iouThrs):
            matched_gt = np.zeros(G, dtype=bool)
            for dind in range(D):
                d_ious = ious[dind, :]
                # Valid GTs: IoU >= threshold AND (not matched OR is crowd)
                valid_mask = (d_ious >= t) & (~(matched_gt & ~iscrowd))

                if not np.any(valid_mask):
                    continue

                valid_ious = np.where(valid_mask, d_ious, -1.0)

                # Pycocotools prefers non-ignore GTs over ignore GTs
                non_ignore_mask = valid_mask & (gtIg == 0)
                if np.any(non_ignore_mask):
                    m = np.argmax(np.where(non_ignore_mask, valid_ious, -1.0))
                else:
                    ignore_mask = valid_mask & (gtIg == 1)
                    m = np.argmax(np.where(ignore_mask, valid_ious, -1.0))

                matched_gt[m] = True
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gtIds[m]
                gtm[tind, m] = dtIds[dind]

    a = np.array(
        [d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]
    ).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

    return {
        "image_id": imgId,
        "category_id": catId,
        "aRng": aRng,
        "maxDet": maxDet,
        "dtIds": [d["id"] for d in dt],
        "gtIds": [g["id"] for g in gt],
        "dtMatches": dtm,
        "gtMatches": gtm,
        "dtScores": [d["score"] for d in dt],
        "gtIgnore": gtIg,
        "dtIgnore": dtIg,
    }


def evaluate_modality(coco_gt, pred_path, iou_type, cat_ids, out_dir, prefix):
    print(f"\n--- Evaluating {prefix} ({iou_type}) ---")
    if not pred_path.exists():
        print(f"Prediction file {pred_path} not found. Skipping.")
        return {}

    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    # Monkey-patch the slow functions
    if iou_type == "keypoints":
        coco_eval.computeOks = fast_computeOks.__get__(coco_eval, COCOeval)

    # Patch evaluateImg for BOTH symbols and lines (it speeds up both!)
    coco_eval.evaluateImg = fast_evaluateImg.__get__(coco_eval, COCOeval)

    coco_eval.params.catIds = cat_ids
    # Lowered maxDets to 2000 to speed up evaluation, included 20 for summary
    coco_eval.params.maxDets = [20, 100, 2000]

    if iou_type == "keypoints":
        # Custom sigmas for start/end points
        coco_eval.params.kpt_oks_sigmas = np.array([0.1, 0.1])

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval["precision"]
    cats = coco_gt.loadCats(cat_ids)

    per_cat_stats = {}
    for i, cat in enumerate(cats):
        p_50 = precisions[0, :, i, 0, 2]
        p_50 = p_50[p_50 > -1]

        p_all = precisions[:, :, i, 0, 2]
        p_all = p_all[p_all > -1]

        map_50 = np.mean(p_50) if len(p_50) > 0 else None
        map_all = np.mean(p_all) if len(p_all) > 0 else None

        per_cat_stats[cat["name"]] = {
            "category_id": cat["id"],
            "mAP_0.5": float(map_50) if map_50 is not None else None,
            "mAP_0.5_0.95": float(map_all) if map_all is not None else None,
        }

        if map_50 is not None:
            print(f"{cat['name']:<30}: {map_50:.4f}")
        else:
            print(f"{cat['name']:<30}: N/A (No ground truth)")

    out_pkl = out_dir / f"coco_eval_raw_{prefix}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"eval": coco_eval.eval, "stats": coco_eval.stats}, f)

    return {
        "global_stats": coco_eval.stats.tolist()
        if coco_eval.stats is not None
        else [],
        "per_category": per_cat_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO metrics for Symbols and Lines"
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        required=True,
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory containing preds_symbols.json and preds_lines.json",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save detailed results. Defaults to pred_dir.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")
    coco_gt = COCO(str(args.anno_path))

    # Separate categories
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    line_cat_ids = []
    sym_cat_ids = []
    for cat in cats:
        if (
            "keypoints" in cat
            and "start" in cat["keypoints"]
            and "end" in cat["keypoints"]
        ):
            line_cat_ids.append(cat["id"])
        else:
            sym_cat_ids.append(cat["id"])

    summary = {}

    # Evaluate Symbols
    sym_pred_path = args.pred_dir / "preds_symbols.json"
    summary["symbols"] = evaluate_modality(
        coco_gt, sym_pred_path, "bbox", sym_cat_ids, out_dir, "symbols"
    )

    # Evaluate Lines
    line_pred_path = args.pred_dir / "preds_lines.json"
    summary["lines"] = evaluate_modality(
        coco_gt, line_pred_path, "keypoints", line_cat_ids, out_dir, "lines"
    )

    out_json = out_dir / "coco_eval_summary.json"
    print(f"\nSaving human-readable summary to {out_json}...")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
