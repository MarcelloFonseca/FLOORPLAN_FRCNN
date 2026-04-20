import csv
import time
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CHECKPOINT_PATH = "floorplan_door_only4_hn2.pth"
TEST_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\test_set")
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"

# GT_DOOR_CLASS_IDS = {3}
GT_DOOR_CLASS_IDS = {1}

# SCORE_THRESHOLD = 0.70
SCORE_THRESHOLD = 0.75
IOU_THRESHOLD = 0.50 
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_SAHI = True
SAHI_SLICE = 640
SAHI_OVERLAP = 0.2

OUTPUT_CSV = "eval_results.csv"

def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
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
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gts, iou_thr):
    preds_sorted = sorted(preds, key=lambda p: -p[1])
    gt_used = [False] * len(gts)
    tp = 0
    fp = 0
    for pbox, _score in preds_sorted:
        best_iou = 0.0
        best_idx = -1
        for j, gbox in enumerate(gts):
            if gt_used[j]:
                continue
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


def predict_plain(model, img_tensor):
    with torch.no_grad():
        out = model([img_tensor.to(DEVICE)])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep = scores >= SCORE_THRESHOLD
    return list(zip(boxes[keep].tolist(), scores[keep].tolist()))


def predict_sahi(model, pil_img):
    from torchvision.transforms.functional import to_tensor
    from torchvision.ops import nms

    W, H = pil_img.size
    stride = int(SAHI_SLICE * (1 - SAHI_OVERLAP))
    all_boxes, all_scores = [], []

    xs = list(range(0, max(1, W - SAHI_SLICE + 1), stride))
    ys = list(range(0, max(1, H - SAHI_SLICE + 1), stride))
    if xs[-1] + SAHI_SLICE < W: xs.append(W - SAHI_SLICE)
    if ys[-1] + SAHI_SLICE < H: ys.append(H - SAHI_SLICE)

    for y in ys:
        for x in xs:
            crop = pil_img.crop((x, y, x + SAHI_SLICE, y + SAHI_SLICE))
            t = to_tensor(crop).to(DEVICE)
            with torch.no_grad():
                out = model([t])[0]
            b = out["boxes"].cpu().numpy()
            s = out["scores"].cpu().numpy()
            for (x1, y1, x2, y2), sc in zip(b, s):
                if sc < SCORE_THRESHOLD:
                    continue
                all_boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                all_scores.append(float(sc))

    if not all_boxes:
        return []
    tb = torch.tensor(all_boxes, dtype=torch.float32)
    ts = torch.tensor(all_scores, dtype=torch.float32)
    keep = nms(tb, ts, iou_threshold=0.5)
    return [(tb[i].tolist(), ts[i].item()) for i in keep.tolist()]


def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Checkpoint: {CHECKPOINT_PATH}")
    print(f"[INFO] SAHI: {USE_SAHI}  |  Score thr: {SCORE_THRESHOLD}  |  IoU thr: {IOU_THRESHOLD}")
    print(f"[INFO] GT door class_ids: {sorted(GT_DOOR_CLASS_IDS)}")

    model = get_model(NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        class_names = ckpt.get("class_names", None)
        if class_names:
            print(f"[INFO] Classes du checkpoint: {class_names}")
    else:
        state = ckpt

    model.load_state_dict(state)
    model.to(DEVICE).eval()

    img_dir = TEST_DIR / IMAGES_SUBDIR
    lbl_dir = TEST_DIR / LABELS_SUBDIR
    imgs = sorted([p for p in img_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not imgs:
        print(f"[ERROR] Aucune image trouvée dans {img_dir}")
        return
    print(f"[INFO] {len(imgs)} images de test trouvées.\n")

    from torchvision.transforms.functional import to_tensor
    rows = []
    total_tp = total_fp = total_fn = 0
    t0 = time.time()

    for img_path in imgs:
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        gts = load_yolo_gt(lbl_dir / f"{img_path.stem}.txt", W, H)

        if USE_SAHI:
            preds = predict_sahi(model, pil)
        else:
            preds = predict_plain(model, to_tensor(pil))

        tp, fp, fn = match_predictions(preds, gts, IOU_THRESHOLD)
        total_tp += tp; total_fp += fp; total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        rows.append({
            "image": img_path.name,
            "gt":    len(gts),
            "pred":  len(preds),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        })
        print(f"  {img_path.name:35s}  GT={len(gts):3d}  Pred={len(preds):3d}  "
            f"TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print(f"AGGREGATE — {len(imgs)} images  ({dt:.1f}s, {dt/len(imgs):.2f}s/img)")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision = {prec:.4f}")
    print(f"  Recall    = {rec:.4f}")
    print(f"  F1        = {f1:.4f}")
    print("=" * 70)

    rows.append({
        "image": "__AGGREGATE__",
        "gt": total_tp + total_fn,
        "pred": total_tp + total_fp,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    })
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[OK] Résultats écrits dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()