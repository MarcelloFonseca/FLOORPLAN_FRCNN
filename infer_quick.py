import torch
import random
import torchvision
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CHECKPOINT_PATH = "floorplan_quick.pth" #chemin vers le checkpoint du modèle entrainé. IMPORTANT.
IMAGE_INDEX = 10
IMAGE_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_dataset\Floor_plan_multiple.yolov8\train\images")
images = sorted(IMAGE_DIR.glob("*.*"))
IMAGE_PATH = images[IMAGE_INDEX]
OUTPUT_PATH = "prediction_output.png"
SCORE_THRESHOLD = 0.10

def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names) + 1

    model = get_model(num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        pred = model([image_tensor])[0]

    boxes = pred["boxes"].cpu()
    labels = pred["labels"].cpu()
    scores = pred["scores"].cpu()

    print("num boxes:", len(boxes))
    print("Using image:", IMAGE_PATH)
    print("top 20 scores:", scores[:20].tolist())
    print("top 20 labels:", labels[:20].tolist())

    draw = ImageDraw.Draw(image)

    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        if score.item() < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.tolist()
        class_index = int(label.item()) - 1
        class_name = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{label.item()}"
        text = f"{class_name} {score.item():.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 4, y1 + 4), text, fill="red")
        kept += 1

    image.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH} with {kept} detections")