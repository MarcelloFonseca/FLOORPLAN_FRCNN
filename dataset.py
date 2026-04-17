import yaml
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

def load_class_names(yaml_path: str):
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)

#Creation du dataset pour la detection d'objets
class YoloDetectionDataset(Dataset):
    def __init__(self, yaml_path: str, root_dir: str, split: str = "train", max_items: int | None = None, transform=None):
        self.yaml_path = Path(yaml_path).resolve()
        self.root_dir = Path(root_dir).resolve()
        self.split = split
        self.transform = transform

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.configuration = yaml.safe_load(f)

        split_key = "train" if split == "train" else "val"
        image_relative_dir = self.configuration[split_key]

        self.images_dir = self.resolve_path(image_relative_dir)
        self.labels_dir = self.images_dir.parent / "labels"

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in extensions]
        )

        if max_items is not None:
            self.image_paths = self.image_paths[:max_items]

    def resolve_path(self, relative_path: str) -> Path:
        raw_path = Path(relative_path)

        candidates = [
            self.root_dir / raw_path,
            self.yaml_path.parent / raw_path,
            self.yaml_path.parent / relative_path.replace("../", "").replace("..\\", ""),
        ]

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not resolve path: {relative_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, bw, bh = map(float, parts)
                    class_id = int(class_id)

                    x1 = (x_center - bw / 2.0) * w
                    y1 = (y_center - bh / 2.0) * h
                    x2 = (x_center + bw / 2.0) * w
                    y2 = (y_center + bh / 2.0) * h

                    x1 = max(0.0, min(x1, w))
                    y1 = max(0.0, min(y1, h))
                    x2 = max(0.0, min(x2, w))
                    y2 = max(0.0, min(y2, h))

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id + 1)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        image = F.to_tensor(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))