import torch
from train_quick import get_model
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

CHECKPOINT_PATH = "floorplan_door_only2.pth"
IMAGE_PATH = "TEST-PLAN4.png"

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
class_names = checkpoint["class_names"]
num_classes = len(class_names) + 1

model = get_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

category_mapping = {
    "1": "door"
}

detection_model = AutoDetectionModel.from_pretrained(
    model_type="torchvision",
    model=model,
    confidence_threshold=0.50,
    device=device,
    category_mapping=category_mapping,
)

result = get_sliced_prediction(
    IMAGE_PATH,
    detection_model=detection_model,
    auto_slice_resolution=True,
    # slice_height=640,
    # slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    postprocess_type="NMS",
    perform_standard_pred=True,  
    postprocess_match_threshold=0.5,
    verbose=2, 
    # overlap_height_ratio=0.2,
    # overlap_width_ratio=0.2,
)

print(result)

result.export_visuals(
    export_dir="sahi_output",
    file_name="prediction_sahi",
)

for obj in result.object_prediction_list[:20]:
    print(
        obj.category.name,
        round(obj.score.value, 3),
        obj.bbox.to_xyxy()
    )