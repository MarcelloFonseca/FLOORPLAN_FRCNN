import torch
import random
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

CHECKPOINT_PATH = "floorplan_door_only4_hn2.pth" #chemin vers le checkpoint du modèle entrainé. IMPORTANT.
# CHECKPOINT_PATH ="floorplan_door_only5_hn3.pth"
IMAGE_INDEX = 4
# IMAGE_DIR = Path(r"C:\Users\MarcelloFonseca\Desktop\floorplan_dataset\train\images")
# IMAGE_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\TEST_PLAN4.png")
# images = sorted(IMAGE_DIR.glob("*.*"))
#IMAGE_PATH = images[IMAGE_INDEX]
# IMAGE_PATH = r"C:\Users\MarcelloFonseca\Desktop\TestML\TestPlanMachineLearning6.png"
IMAGE_PATH = r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\TEST-PLAN5.png"
OUTPUT_PATH = "prediction_output.png" 
#SCORE_THRESHOLD = 0.55
#SCORE_THRESHOLD = 0.50
SCORE_THRESHOLD = 0.75
# SCORE_THRESHOLD = 0.85

# def get_model(num_classes: int):
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
#         weights=None,
#         weights_backbone=None
#     )
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.transform = GeneralizedRCNNTransform(
        min_size=1600, max_size=2666,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
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
    print(f"pixel[0,0]: R={image_tensor[0,0,0]:.4f} G={image_tensor[1,0,0]:.4f} B={image_tensor[2,0,0]:.4f}")

    with torch.no_grad():
        pred = model([image_tensor])[0] #Les predictions se font ici...

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