import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# checkpoint = torch.load("floorplan_door_only_MERGED.pth", map_location="cpu")
checkpoint = torch.load("floorplan_door_only4_hn2.pth", map_location="cpu")
print(checkpoint["class_names"])  # doit afficher ['door']
class_names = checkpoint["class_names"]
num_classes = len(class_names) + 1

model = get_model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(3, 640, 640)

torch.onnx.export(
    model,
    ([dummy_input],),
    "floorplan_door_only4_hn2.onnx",
    opset_version=17,
    input_names=["images"],
    dynamic_axes={
        "images": {1: "height", 2: "width"}
    }
)
print(f"Classes: {class_names}")
print("ONNX model saved.")

# import torch
# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def get_model(num_classes):
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

# checkpoint = torch.load("floorplan_door_resnet50.pth", map_location="cpu")
# class_names = checkpoint["class_names"]
# num_classes = len(class_names) + 1

# model = get_model(num_classes)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# dummy_input = torch.randn(3, 640, 640)

# torch.onnx.export(
#     model,
#     ([dummy_input],),
#     "floorplan_door_resnet502.onnx",
#     opset_version=11,             
#     do_constant_folding=True,     
#     input_names=["images"],
#     output_names=["boxes", "labels", "scores"],
#     dynamic_axes={
#         "images": {1: "height", 2: "width"},
#         "boxes":  {0: "num_det"},
#         "labels": {0: "num_det"},
#         "scores": {0: "num_det"},
#     }
# )

# print(f"Classes: {class_names}")
# print("ONNX model saved.")

# import torch
# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def get_model(num_classes):
#     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
#         weights=None,
#         min_size=768,  
#         max_size=768   
#     )
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

# checkpoint = torch.load("floorplan_door_only_MERGED.pth", map_location="cpu")
# class_names = checkpoint["class_names"]
# num_classes = len(class_names) + 1

# model = get_model(num_classes)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# dummy_input = torch.randn(3, 768, 768)  

# torch.onnx.export(
#     model,
#     ([dummy_input],),
#     "floorplan_door_only_MERGED.onnx",
#     opset_version=11,
#     do_constant_folding=False,
#     input_names=["images"],
#     output_names=["boxes", "labels", "scores"],
#     dynamic_axes={
#         "images": {1: "height", 2: "width"},
#         "boxes":  {0: "num_det"},
#         "labels": {0: "num_det"},
#         "scores": {0: "num_det"},
#     }
# )

# print(f"Classes: {class_names}")
# print("ONNX model saved.")