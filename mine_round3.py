import torch
from PIL import Image
from pathlib import Path
from train_quick import get_model
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CHECKPOINT = "floorplan_door_only4_hn2.pth"

FP_PLANS = [
    "TEST-PLAN4.png",
    "TEST-PLAN5.png",
    "TEST-PLAN6.png",
]

OUT_DIR = Path("hard_negatives_round3")
OUT_DIR.mkdir(exist_ok=True)
MARGIN = 40

CONF_THRESHOLD = 0.30

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
class_names = ckpt["class_names"]
model = get_model(len(class_names) + 1)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

det_model = AutoDetectionModel.from_pretrained(
    model_type="torchvision", model=model,
    confidence_threshold=CONF_THRESHOLD, device=device,
    category_mapping={str(i + 1): n for i, n in enumerate(class_names)},
)

total_crops = 0
for plan_path in FP_PLANS:
    if not Path(plan_path).exists():
        print(f"[skip] {plan_path} introuvable")
        continue

    print(f"Traitement : {plan_path}")
    img = Image.open(plan_path).convert("RGB")
    W, H = img.size

    result = get_sliced_prediction(
        plan_path, det_model,
        auto_slice_resolution=True,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
    )

    stem = Path(plan_path).stem
    for i, obj in enumerate(result.object_prediction_list):
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        cx1 = max(0, int(x1) - MARGIN)
        cy1 = max(0, int(y1) - MARGIN)
        cx2 = min(W, int(x2) + MARGIN)
        cy2 = min(H, int(y2) + MARGIN)
        crop = img.crop((cx1, cy1, cx2, cy2))
        score_str = f"{obj.score.value:.2f}"
        crop.save(OUT_DIR / f"{stem}_det{i:03d}_score{score_str}.png")
        total_crops += 1

print(f"\nRound 3 : {total_crops} crops générés dans {OUT_DIR}")
print("TRI MANUEL : garde uniquement les FP flèches/boussoles/fermes/hachures.")
print("Supprime impérativement toutes les vraies portes.")