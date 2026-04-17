from pathlib import Path
import shutil

HN_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\hard_negatives_raw")
DATASET_IMG = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\train\images")
DATASET_LBL = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\train\labels")

count = 0
for img_path in HN_DIR.glob("*.png"):
    new_name = f"hn_{img_path.name}"
    shutil.copy2(img_path, DATASET_IMG / new_name)
    (DATASET_LBL / f"hn_{img_path.stem}.txt").write_text("")
    count += 1
print(f"Copié {count} hard negatives.")