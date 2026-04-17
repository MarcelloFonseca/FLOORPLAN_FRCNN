# import torch
# from train_quick import get_model
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction

# CHECKPOINT_PATH = "floorplan_door_only2.pth"
# IMAGE_PATH = "TEST-PLAN4.png"

# checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
# class_names = checkpoint["class_names"]
# num_classes = len(class_names) + 1

# model = get_model(num_classes)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model.to(device)

# category_mapping = {
#     "1": "door"
# }

# detection_model = AutoDetectionModel.from_pretrained(
#     model_type="torchvision",
#     model=model,
#     confidence_threshold=0.50,
#     device=device,
#     category_mapping=category_mapping,
# )

# result = get_sliced_prediction(
#     IMAGE_PATH,
#     detection_model=detection_model,
#     auto_slice_resolution=True,
#     # slice_height=640,
#     # slice_width=640,
#     overlap_height_ratio=0.25,
#     overlap_width_ratio=0.25,
#     postprocess_type="NMS",
#     perform_standard_pred=True,  
#     postprocess_match_threshold=0.5,
#     verbose=2, 
#     # overlap_height_ratio=0.2,
#     # overlap_width_ratio=0.2,
# )

# print(result)

# result.export_visuals(
#     export_dir="sahi_output",
#     file_name="prediction_sahi",
# )

# for obj in result.object_prediction_list[:20]:
#     print(
#         obj.category.name,
#         round(obj.score.value, 3),
#         obj.bbox.to_xyxy()
#     )

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from train_quick import get_model
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CHECKPOINT_PATH = "floorplan_door_only3_hn.pth"
IMAGE_PATH = "TEST-PLAN4.png"
SR_SCALE = 2

DOOR_MIN_SIZE = 15   
DOOR_MAX_SIZE = 250  
DOOR_MAX_RATIO = 2.0
DOOR_MIN_AREA = 400


def upscale_image(image_path: str, scale: int) -> tuple[str, float]:
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        print(f"[SR] Real-ESRGAN x{scale} activé")

        model_map = {
            2: ("RealESRGAN_x2plus.pth", 2),
            4: ("RealESRGAN_x4plus.pth", 4),
        }
        weights_name, actual_scale = model_map.get(scale, ("RealESRGAN_x2plus.pth", 2))
        weights_path = Path(weights_name)

        if not weights_path.exists():
            print(f"[SR] Poids '{weights_name}' introuvables → fallback PIL")
            raise FileNotFoundError

        arch = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=actual_scale
        )
        upsampler = RealESRGANer(
            scale=actual_scale,
            model_path=str(weights_path),
            model=arch,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        output_np, _ = upsampler.enhance(img_np, outscale=actual_scale)
        output_img = Image.fromarray(output_np)

        out_path = f"sr_upscaled_x{actual_scale}.png"
        output_img.save(out_path)
        print(f"[SR] Image upscalée : {img.size} → {output_img.size}")
        return out_path, float(actual_scale)

    except (ImportError, FileNotFoundError):
        print(f"[SR] Fallback PIL Lanczos x{scale}")
        img = Image.open(image_path).convert("RGB")
        new_w = img.width * scale
        new_h = img.height * scale
        upscaled = img.resize((new_w, new_h), Image.LANCZOS)
        out_path = f"sr_upscaled_pil_x{scale}.png"
        upscaled.save(out_path)
        print(f"[SR] Image upscalée : {img.size} → {upscaled.size}")
        return out_path, float(scale)


def is_valid_door(obj, actual_scale: float) -> tuple[bool, str]:
    """Filtre géométrique : rejette les détections aux dimensions aberrantes.
    Retourne (valide, raison_rejet)."""
    x1, y1, x2, y2 = obj.bbox.to_xyxy()
    w = (x2 - x1) / actual_scale
    h = (y2 - y1) / actual_scale
    area = w * h

    if w < DOOR_MIN_SIZE or h < DOOR_MIN_SIZE:
        return False, f"trop_petit(w={w:.0f},h={h:.0f})"
    if w > DOOR_MAX_SIZE or h > DOOR_MAX_SIZE:
        return False, f"trop_grand(w={w:.0f},h={h:.0f})"
    if area < DOOR_MIN_AREA:
        return False, f"aire_faible({area:.0f})"

    ratio = max(w, h) / max(1.0, min(w, h))
    if ratio > DOOR_MAX_RATIO:
        return False, f"ratio_allonge({ratio:.2f})"

    return True, ""


checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]
num_classes = len(class_names) + 1

model = get_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

category_mapping = {str(i + 1): name for i, name in enumerate(class_names)}

detection_model = AutoDetectionModel.from_pretrained(
    model_type="torchvision",
    model=model,
    confidence_threshold=0.60,
    device=device,
    category_mapping=category_mapping,
)

sr_image_path, actual_scale = upscale_image(IMAGE_PATH, SR_SCALE)

result = get_sliced_prediction(
    sr_image_path,
    detection_model=detection_model,
    auto_slice_resolution=True,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    postprocess_type="NMS",
    perform_standard_pred=True,
    postprocess_match_threshold=0.5,
    verbose=2,
)

raw_detections = result.object_prediction_list
kept = []
rejected = []
for obj in raw_detections:
    valid, reason = is_valid_door(obj, actual_scale)
    if valid:
        kept.append(obj)
    else:
        rejected.append((obj, reason))

print(f"\n[FILTRE] Brut : {len(raw_detections)} | Gardés : {len(kept)} | Rejetés : {len(rejected)}")
for obj, reason in rejected[:20]:
    xyxy = [round(v / actual_scale) for v in obj.bbox.to_xyxy()]
    print(f"  ✗ {obj.category.name} {obj.score.value:.2f} {xyxy} → {reason}")

result.export_visuals(
    export_dir="sahi_output",
    file_name="prediction_sahi_sr_raw",
)

original_img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(original_img)

for obj in kept:
    xyxy = [v / actual_scale for v in obj.bbox.to_xyxy()]
    draw.rectangle(xyxy, outline="red", width=3)
    draw.text(
        (xyxy[0] + 4, xyxy[1] + 4),
        f"{obj.category.name} {obj.score.value:.2f}",
        fill="red"
    )

original_img.save("sahi_output/prediction_original_space.png")
print("\nExport filtré sur image originale sauvegardé.")

print(f"\nTotal détections finales : {len(kept)}")
print(f"[SR] Coords rescalées ÷ {actual_scale} (espace image originale)\n")

for obj in kept[:20]:
    xyxy = obj.bbox.to_xyxy()
    scaled = [round(v / actual_scale) for v in xyxy]
    print(obj.category.name, round(obj.score.value, 3), scaled)