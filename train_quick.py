import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import YoloDetectionDataset, collate_fn, load_class_names

#j'utilise le SummaryWrite pour enregistrer les métriques et les visualiser dans TensorBoard du RCNN

# YAML_PATH = r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_dataset\Floor_plan_multiple.yolov8\data.yaml"
# ROOT_DIR = r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_dataset\Floor_plan_multiple.yolov8"

YAML_PATH = r"C:\Users\MarcelloFonseca\Desktop\Floor_plan_Dataset_DoorOnly\data.yaml"
ROOT_DIR = r"C:\Users\MarcelloFonseca\Desktop\Floor_plan_Dataset_DoorOnly"

TRAIN_MAX_ITEMS = None
EPOCHS = 20
BATCH_SIZE = 2
LR = 0.005 #Pour le modele de resnet50 (ne peut pas s'appliquer a un autre modele plus leger comme mobilenet, etc. !)

writer = SummaryWriter()

def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT") #j'utilise les weights par defaut pour faire un transfer learning et ne pas repartir de 0
    in_features = model.roi_heads.box_predictor.cls_score.in_features #model pre-entraine sur COCO (Common Objects in Context)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #j'utilise le GPU si disponible pour accélérer l'entraînement, sinon je tombe sur le CPU
    print(f"Using device: {device}")

    class_names = load_class_names(YAML_PATH)
    num_classes = len(class_names) + 1

    train_ds = YoloDetectionDataset(  yaml_path=YAML_PATH, root_dir=ROOT_DIR, split="train", max_items=TRAIN_MAX_ITEMS)

    print(f"Train images: {len(train_ds)}")
    print(f"Classes: {class_names}")

    #Ici le train_loader est créé à partir dee notre class YoloDetectionDataset qui represente notre dataset de plans
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)

    #A OPTIMISER PLUS TARD DANS LE PROCESSUS 
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
    # torch.save(checkpoint, "floorplan_quick.pth")
    torch.save(checkpoint, "floorplan_door_only.pth")
    print("Saved floorplan_quick.pth")
    writer.close()