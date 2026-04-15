import torch
from train_quick import get_model

checkpoint = torch.load("floorplan_door_only2.pth", map_location="cpu")
class_names = checkpoint["class_names"]
num_classes = len(class_names) + 1

model = get_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(3, 640, 640)

torch.onnx.export(
    model,
    ([dummy_input],),
    "floorplan_door_only2.onnx",
    opset_version=17,
    input_names=["images"],
    dynamic_axes={
        "images": {1: "height", 2: "width"}
    }
)
print(class_names)
print("ONNX model saved.")
