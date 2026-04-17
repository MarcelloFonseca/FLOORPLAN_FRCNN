import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import YoloDetectionDataset, collate_fn, load_class_names

# j'utilise le SummaryWriter pour enregistrer les métriques et les visualiser dans TensorBoard du RCNN

YAML_PATH = r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\data.yaml"
ROOT_DIR  = r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged"

PREV_CHECKPOINT = "floorplan_door_only4_hn2.pth"
OUT_CHECKPOINT  = "floorplan_door_only5_hn3.pth"

TRAIN_MAX_ITEMS = None
EPOCHS = 3
BATCH_SIZE = 2
LR = 0.00005 
writer = SummaryWriter()


def get_model(num_classes: int):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = load_class_names(YAML_PATH)
    num_classes = len(class_names) + 1

    train_ds = YoloDetectionDataset(
        yaml_path=YAML_PATH,
        root_dir=ROOT_DIR,
        split="train",
        max_items=TRAIN_MAX_ITEMS,
    )

    print(f"Train images: {len(train_ds)}")
    print(f"Classes: {class_names}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = get_model(num_classes)

    prev = torch.load(PREV_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(prev["model_state_dict"])
    print(f"Fine-tuning depuis {PREV_CHECKPOINT}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_loader) + batch_idx)

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(checkpoint, OUT_CHECKPOINT)
    print(f"Saved {OUT_CHECKPOINT}")
    writer.close()