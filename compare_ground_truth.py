import yaml
from pathlib import Path
from PIL import Image, ImageDraw

YAML_PATH = Path(r"C:\Users\MarcelloFonseca\Desktop\floorplan_dataset\data.yaml")
IMAGE_DIR = Path(r"C:\Users\MarcelloFonseca\Desktop\floorplan_dataset\train\images")
LABELS_DIR = Path(r"C:\Users\MarcelloFonseca\Desktop\floorplan_dataset\train\labels")

IMAGE_INDEX = 4
OUTPUT_PATH = "ground_truth_output.png"

with open(YAML_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

CLASS_NAMES = cfg["names"]

images = sorted(IMAGE_DIR.glob("*.*"))
IMAGE_PATH = images[IMAGE_INDEX]

image = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(image)
w, h = image.size

label_path = LABELS_DIR / f"{IMAGE_PATH.stem}.txt"

print("Using image:", IMAGE_PATH)
print("Using label:", label_path)

if label_path.exists():
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print("Ground truth boxes:", len(lines))

    for line in lines:
        class_id, x_center, y_center, bw, bh = map(float, line.split())
        class_id = int(class_id)

        x1 = (x_center - bw / 2) * w
        y1 = (y_center - bh / 2) * h
        x2 = (x_center + bw / 2) * w
        y2 = (y_center + bh / 2) * h

        class_name = CLASS_NAMES[class_id]
        text = f"GT {class_name}"

        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1 + 4, y1 + 4), text, fill="green")
else:
    print("No label file found.")

image.save(OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH}")