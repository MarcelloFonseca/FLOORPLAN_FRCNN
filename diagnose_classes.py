from pathlib import Path
from collections import Counter

TEST_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\test_set")
lbl_dir = TEST_DIR / "labels"

classes = Counter()
for txt in lbl_dir.glob("*.txt"):
    for line in txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            classes[parts[0]] += 1

print("Distribution des class_id dans le test set:")
for cls_id, count in sorted(classes.items()):
    print(f"  class {cls_id}: {count} boxes")

yaml_path = TEST_DIR / "data.yaml"

if yaml_path.exists():
    print("\n--- data.yaml ---")
    print(yaml_path.read_text())
else:
    for p in TEST_DIR.parent.rglob("data.yaml"):
        print(f"\n--- {p} ---")
        print(p.read_text())
        break