from dataset import YoloDetectionDataset

YAML_PATH = r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_dataset\Floor_plan_multiple.yolov8\data.yaml"
ROOT_DIR = r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_dataset\Floor_plan_multiple.yolov8"

ds = YoloDetectionDataset(
    yaml_path=YAML_PATH,
    root_dir=ROOT_DIR,
    split="train",
    max_items=10
)

print("dataset size:", len(ds))
print("images dir:", ds.images_dir)
print("labels dir:", ds.labels_dir)

print("\nFirst few image files:")
for p in ds.image_paths[:5]:
    print(" -", p.name)

print("\nInspecting first samples:")
for i in range(min(5, len(ds))):
    image, target = ds[i]
    print(f"\nSample {i}")
    print("image shape:", tuple(image.shape))
    print("num boxes:", len(target["boxes"]))
    if len(target["boxes"]) > 0:
        print("first boxes:", target["boxes"][:3].tolist())
        print("first labels:", target["labels"][:3].tolist())