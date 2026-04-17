# copy_hard_negatives_round3.py
from pathlib import Path
import shutil

HN3_DIR = Path("hard_negatives_round3")
DATASET_IMG = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\train\images")
DATASET_LBL = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\train\labels")

count = 0
for img_path in HN3_DIR.glob("*.png"):
    new_name = f"hn3_{img_path.name}"   
    shutil.copy2(img_path, DATASET_IMG / new_name)
    (DATASET_LBL / f"hn3_{img_path.stem}.txt").write_text("")
    count += 1
print(f"Copié {count} HN round 3.")