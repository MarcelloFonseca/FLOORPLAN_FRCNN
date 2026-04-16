import shutil
from pathlib import Path

def copy_split(src_dir, out_dir):
    for sub in ["images", "labels"]:
        src = src_dir / sub
        dst = out_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue
        for f in src.glob("*.*"):
            dest = dst / f.name
            if dest.exists():
                dest = dst / f"ds2_{f.name}"
            shutil.copy(f, dest)

def merge_yolo(ds1_path, ds2_path, output_path):
    ds1 = Path(ds1_path)
    ds2 = Path(ds2_path)
    out = Path(output_path)

    copy_split(ds1, out / "train")

    for split in ["train", "valid", "test"]:
        copy_split(ds2 / split, out / split)

merge_yolo(
    ds1_path=    r"C:\Users\MarcelloFonseca\Desktop\train",
    ds2_path=    r"C:\Users\MarcelloFonseca\Desktop\Floor_plan_multiple.yolov8DoorOnly2",
    output_path= r"C:\Users\MarcelloFonseca\Desktop\MERGED_DATASET_DOORS_V1"
)