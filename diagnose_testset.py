from pathlib import Path

TEST_DIR = Path(r"C:\Users\Marcello Fonseca\OneDrive\Bureau\floorplan_frcnn\test_set")
img_dir = TEST_DIR / "images"
lbl_dir = TEST_DIR / "labels"

missing = 0
empty = 0
with_gt = 0
total_boxes = 0

for img in sorted(img_dir.iterdir()):
    if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        continue
    txt = lbl_dir / f"{img.stem}.txt"
    if not txt.exists():
        missing += 1
        print(f"  MISSING  {img.name}")
        continue
    content = txt.read_text().strip()
    if not content:
        empty += 1
    else:
        n = len([l for l in content.splitlines() if l.strip()])
        with_gt += 1
        total_boxes += n

print(f"\nTotal images:     {missing + empty + with_gt}")
print(f"  .txt manquant:  {missing}")
print(f"  .txt vide:      {empty}")
print(f"  avec GT:        {with_gt}  ({total_boxes} boxes)")