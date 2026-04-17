import shutil
from pathlib import Path

PART2_KEEP_MAP = {1: 0}

def copy_and_remap(src_img_dir: Path, src_lbl_dir: Path,
                dst_img_dir: Path, dst_lbl_dir: Path,
                prefix: str, class_map: dict | None = None):
    """Copie images + labels avec préfixe unique et remap optionnel de classes.
    class_map=None → aucun remap (garde tout).
    class_map={old:new} → garde seulement les classes du dict, remap selon mapping."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists():
        print(f"[skip] {src_img_dir} n'existe pas")
        return 0

    count = 0
    for img in src_img_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        new_stem = f"{prefix}_{img.stem}"
        shutil.copy2(img, dst_img_dir / f"{new_stem}{img.suffix}")

        src_lbl = src_lbl_dir / f"{img.stem}.txt"
        dst_lbl = dst_lbl_dir / f"{new_stem}.txt"

        if src_lbl.exists():
            if class_map is None:
                shutil.copy2(src_lbl, dst_lbl)
            else:
            
                new_lines = []
                for line in src_lbl.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    old_id = int(parts[0])
                    if old_id in class_map:
                        parts[0] = str(class_map[old_id])
                        new_lines.append(" ".join(parts))
                dst_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
        else:
        
            dst_lbl.write_text("")
        count += 1
    return count


def merge_yolo(ds1_path: str, ds2_path: str, output_path: str):
    ds1 = Path(ds1_path)
    ds2 = Path(ds2_path)
    out = Path(output_path)

    n1 = copy_and_remap(
        ds1 / "images", ds1 / "labels",
        out / "train" / "images", out / "train" / "labels",
        prefix="p1", class_map=None  # déjà door=0, pas de remap
    )
    print(f"trainPart1 → {n1} images copiées")

    for split in ["train", "valid", "test"]:
        n2 = copy_and_remap(
            ds2 / split / "images", ds2 / split / "labels",
            out / split / "images", out / split / "labels",
            prefix=f"p2_{split}", class_map=PART2_KEEP_MAP
        )
        if n2 > 0:
            print(f"trainPart2/{split} → {n2} images copiées (classes remappées)")

merge_yolo(
    ds1_path="C:/Users/Marcello Fonseca/OneDrive/Bureau/trainPart1",
    ds2_path="C:/Users/Marcello Fonseca/OneDrive/Bureau/trainPart2",
    output_path="C:/Users/Marcello Fonseca/OneDrive/Bureau/TrainMerged",
)