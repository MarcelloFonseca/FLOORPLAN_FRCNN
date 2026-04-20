import csv
import time
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


CHECKPOINTS = [
    ("v4_hn2", "floorplan_door_only4_hn2.pth"),
    ("v5_hn3", "floorplan_door_only5_hn3.pth"),
]
TEST_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\test_set")
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"

GT_DOOR_CLASS_IDS = {3}

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
INFERENCE_THR = 0.30
IOU_THRESHOLD = 0.50
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_CSV = "sweep_results.csv"


def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(path):
    model = get_model(NUM_CLASSES)
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def load_yolo_gt(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        if int(cls) not in GT_DOOR_CLASS_IDS:
            continue
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gts, iou_thr):
    preds_sorted = sorted(preds, key=lambda p: -p[1])
    gt_used = [False] * len(gts)
    tp = fp = 0
    for pbox, _score in preds_sorted:
        best_iou = 0.0
        best_idx = -1
        for j, gbox in enumerate(gts):
            if gt_used[j]: continue
            i = iou(pbox, gbox)
            if i > best_iou:
                best_iou = i
                best_idx = j
        if best_iou >= iou_thr and best_idx >= 0:
            gt_used[best_idx] = True
            tp += 1
        else:
            fp += 1
    fn = gt_used.count(False)
    return tp, fp, fn


def run_inference(model, imgs, lbl_dir):
    from torchvision.transforms.functional import to_tensor
    cache = []
    for img_path in imgs:
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        gts = load_yolo_gt(lbl_dir / f"{img_path.stem}.txt", W, H)
        t = to_tensor(pil).to(DEVICE)
        with torch.no_grad():
            out = model([t])[0]
        boxes = out["boxes"].cpu().numpy().tolist()
        scores = out["scores"].cpu().numpy().tolist()
        preds = [(b, s) for b, s in zip(boxes, scores) if s >= INFERENCE_THR]
        cache.append({"name": img_path.name, "gts": gts, "preds": preds})
    return cache


def evaluate_at_threshold(cache, thr):
    tp_tot = fp_tot = fn_tot = 0
    for item in cache:
        filtered = [(b, s) for b, s in item["preds"] if s >= thr]
        tp, fp, fn = match_predictions(filtered, item["gts"], IOU_THRESHOLD)
        tp_tot += tp; fp_tot += fp; fn_tot += fn
    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0.0
    rec  = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp_tot, fp_tot, fn_tot, prec, rec, f1


def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] GT door class_ids: {sorted(GT_DOOR_CLASS_IDS)}")

    img_dir = TEST_DIR / IMAGES_SUBDIR
    lbl_dir = TEST_DIR / LABELS_SUBDIR
    imgs = sorted([p for p in img_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    print(f"[INFO] {len(imgs)} images\n")

    all_rows = []

    for tag, ckpt_path in CHECKPOINTS:
        print(f"=" * 70)
        print(f"CHECKPOINT: {tag}  ({ckpt_path})")
        print(f"=" * 70)

        model = load_checkpoint(ckpt_path)

        t0 = time.time()
        cache = run_inference(model, imgs, lbl_dir)
        dt_inf = time.time() - t0
        total_preds = sum(len(it["preds"]) for it in cache)
        total_gt = sum(len(it["gts"]) for it in cache)
        print(f"[INFO] Inference: {dt_inf:.1f}s  ({total_preds} preds >= {INFERENCE_THR}, {total_gt} GT)\n")

        print(f"{'thr':>6} | {'TP':>4} {'FP':>4} {'FN':>4} | {'P':>6} {'R':>6} {'F1':>6}")
        print("-" * 50)
        best = {"f1": -1.0}
        for thr in THRESHOLDS:
            tp, fp, fn, p, r, f1 = evaluate_at_threshold(cache, thr)
            print(f"{thr:>6.2f} | {tp:>4d} {fp:>4d} {fn:>4d} | {p:>6.4f} {r:>6.4f} {f1:>6.4f}")
            all_rows.append({
                "model": tag, "threshold": thr,
                "TP": tp, "FP": fp, "FN": fn,
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            })
            if f1 > best["f1"]:
                best = {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}
        print(f"\n[BEST {tag}]  thr={best['thr']:.2f}  P={best['p']:.4f}  R={best['r']:.4f}  F1={best['f1']:.4f}\n")

        del model
        torch.cuda.empty_cache()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"[OK] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()