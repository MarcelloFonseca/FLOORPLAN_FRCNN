import torch
from train_quick import get_model

checkpoint = torch.load("floorplan_quick.pth", map_location="cpu")

class_names = checkpoint["class_names"]
num_classes = len(class_names) + 1

model = get_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(3, 640, 640)

torch.onnx.export(
    model,
    ([dummy_input],),
    "floorplan_quick.onnx",
    opset_version=11,
    input_names=["images"],
)

print("ONNX model saved: floorplan_quick.onnx")