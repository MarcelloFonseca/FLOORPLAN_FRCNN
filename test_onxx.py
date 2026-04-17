import onnxruntime as ort
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

IMAGE_INDEX = 5
IMAGE_DIR = Path(r"C:\Users\MarcelloFonseca\Desktop\floorplan_dataset\train\images")
MODEL_PATH = "floorplan_quick.onnx"
OUTPUT_PATH = "onnx_prediction_output.png"
SCORE_THRESHOLD = 0.10

CLASS_NAMES = ["2door", "door", "window"]

images = sorted(IMAGE_DIR.glob("*.*"))
IMAGE_PATH = images[IMAGE_INDEX]

print("Using image:", IMAGE_PATH)

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = F.to_tensor(image)
input_array = image_tensor.numpy().astype(np.float32)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_array})

boxes = outputs[0]
labels = outputs[1]
scores = outputs[2]

print("num boxes:", len(boxes))
print("top 20 scores:", scores[:20].tolist())
print("top 20 labels:", labels[:20].tolist())

draw = ImageDraw.Draw(image)

kept = 0
for box, label, score in zip(boxes, labels, scores):
    if float(score) < SCORE_THRESHOLD:
        continue

    x1, y1, x2, y2 = box.tolist()
    class_index = int(label) - 1
    class_name = CLASS_NAMES[class_index] if 0 <= class_index < len(CLASS_NAMES) else f"class_{label}"

    text = f"{class_name} {float(score):.2f}"

    draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
    draw.text((x1 + 4, y1 + 4), text, fill="blue")
    kept += 1

image.save(OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH} with {kept} detections")