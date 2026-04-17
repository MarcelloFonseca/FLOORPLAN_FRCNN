from pathlib import Path
from collections import Counter

LBL = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\TrainMerged\train\labels")

total_files = 0
empty_files = 0
class_counter = Counter()

for txt in LBL.glob("*.txt"):
    total_files += 1
    content = txt.read_text().strip()
    if not content:
        empty_files += 1
        continue
    for line in content.splitlines():
        class_counter[line.split()[0]] += 1

print(f"Total fichiers label : {total_files}")
print(f"Fichiers vides (hard negatives implicites) : {empty_files}")
print(f"Classes rencontrées : {dict(class_counter)}")