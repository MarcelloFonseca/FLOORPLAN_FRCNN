import shutil
from pathlib import Path

HN2_DIR = Path("hard_negatives_round2") 
DATASET_IMG = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\train\images")
DATASET_LBL = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\train\labels")

count = 0
for img_path in HN2_DIR.glob("*.png"):
    new_name = f"hn2_{img_path.name}"
    shutil.copy2(img_path, DATASET_IMG / new_name)
    (DATASET_LBL / f"hn2_{img_path.stem}.txt").write_text("")
    count += 1
print(f"Copié {count} HN round 2.")