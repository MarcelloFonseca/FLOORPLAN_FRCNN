import cv2
import torch
import torchvision
from PIL import Image
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CHECKPOINT_PATH = "floorplan_door_only4_hn2.pth"
TEST_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\test_set")
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"
OUTPUT_DIR = Path("error_analysis")

GT_DOOR_CLASS_IDS = {3}
FN_SCORE_THR = 0.30
FP_SCORE_THR = 0.90
IOU_THRESHOLD = 0.50
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(n):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(f, n)
    return m


def load_yolo_gt(lp, W, H):
    boxes = []
    if not lp.exists(): return boxes
    for line in lp.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5: continue
        cls, cx, cy, w, h = parts
        if int(cls) not in GT_DOOR_CLASS_IDS: continue
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        boxes.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
    return boxes


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    ua = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ub = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0


def crop_with_box(img, box, margin=80, color=(0, 0, 255), label=""):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx1 = max(0, x1 - margin); cy1 = max(0, y1 - margin)
    cx2 = min(W, x2 + margin); cy2 = min(H, y2 + margin)
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), color, 2)
    if label:
        cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return crop


def main():
    (OUTPUT_DIR / "FN_hard").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "FP_hard").mkdir(parents=True, exist_ok=True)

    model = get_model(NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state); model.to(DEVICE).eval()

    from torchvision.transforms.functional import to_tensor
    img_dir = TEST_DIR / IMAGES_SUBDIR
    lbl_dir = TEST_DIR / LABELS_SUBDIR
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

    n_fn = n_fp = 0

    for img_path in imgs:
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        img_np = cv2.cvtColor(__import__("numpy").array(pil), cv2.COLOR_RGB2BGR)

        gts = load_yolo_gt(lbl_dir / f"{img_path.stem}.txt", W, H)

        with torch.no_grad():
            out = model([to_tensor(pil).to(DEVICE)])[0]
        boxes = out["boxes"].cpu().numpy().tolist()
        scores = out["scores"].cpu().numpy().tolist()

        preds_low = [(b, s) for b, s in zip(boxes, scores) if s >= FN_SCORE_THR]
        matched = [False] * len(gts)
        for pbox, _ in sorted(preds_low, key=lambda p: -p[1]):
            best, bidx = 0, -1
            for j, g in enumerate(gts):
                if matched[j]: continue
                i = iou(pbox, g)
                if i > best: best, bidx = i, j
            if best >= IOU_THRESHOLD and bidx >= 0:
                matched[bidx] = True
        for j, g in enumerate(gts):
            if not matched[j]:
                n_fn += 1
                crop = crop_with_box(img_np, g, color=(0, 255, 0), label="FN (GT non trouvee)")
                cv2.imwrite(str(OUTPUT_DIR / "FN_hard" / f"{img_path.stem}_fn{j}.png"), crop)

        preds_high = [(b, s) for b, s in zip(boxes, scores) if s >= FP_SCORE_THR]
        for pbox, sc in preds_high:
            best = 0
            for g in gts:
                i = iou(pbox, g)
                if i > best: best = i
            if best < IOU_THRESHOLD:
                n_fp += 1
                crop = crop_with_box(img_np, pbox, color=(0, 0, 255), label=f"FP score={sc:.2f}")
                cv2.imwrite(str(OUTPUT_DIR / "FP_hard" / f"{img_path.stem}_fp_{int(sc*100)}.png"), crop)

    print(f"\n[OK] {n_fn} FN durs -> {OUTPUT_DIR/'FN_hard'}")
    print(f"[OK] {n_fp} FP durs -> {OUTPUT_DIR/'FP_hard'}")


if __name__ == "__main__":
    main()