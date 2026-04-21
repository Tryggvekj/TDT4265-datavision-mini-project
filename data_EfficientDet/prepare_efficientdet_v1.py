import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def parse_yolo(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels = []

    if not txt_path.exists():
        return labels

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            labels.append((cls, cx, cy, w, h))

    return labels


def yolo_to_yxyx(
    yolo: Tuple[int, float, float, float, float],
    img_w: int,
    img_h: int,
):
    """
    Convert YOLO normalized format:
        (cls, cx, cy, w, h)
    to EfficientDet/rwightman target-style box format:
        (cls_1_indexed, y1, x1, y2, x2)
    """
    cls, cx, cy, w, h = yolo

    x_center = cx * img_w
    y_center = cy * img_h
    width = w * img_w
    height = h * img_h

    x1 = max(0.0, x_center - width / 2.0)
    y1 = max(0.0, y_center - height / 2.0)
    x2 = min(float(img_w), x_center + width / 2.0)
    y2 = min(float(img_h), y_center + height / 2.0)

    # EfficientDet training pipeline in rwightman repo expects classes as 1..N
    cls = cls + 1

    return cls, y1, x1, y2, x2


def is_valid_box(y1: float, x1: float, y2: float, x2: float, min_size: float = 1.0) -> bool:
    return (y2 - y1) >= min_size and (x2 - x1) >= min_size


def write_csv(rows: List[List], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "y1", "x1", "y2", "x2"])
        writer.writerows(rows)


def gather_split(split_dir: Path, output_path: Path, include_empty: bool = False):
    label_dir = split_dir.parent / "labels"
    imgs = sorted([p for p in split_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    rows = []
    empty_images = 0
    invalid_boxes = 0
    valid_boxes = 0

    for img_path in imgs:
        if label_dir.exists():
            label_path = label_dir / img_path.with_suffix(".txt").name
        else:
            label_path = img_path.with_suffix(".txt")

        with Image.open(img_path) as image:
            iw, ih = image.size

        yolo_labels = parse_yolo(label_path)

        if len(yolo_labels) == 0:
            empty_images += 1
            if include_empty:
                rows.append([str(img_path), "", "", "", "", ""])
            continue

        for yolo in yolo_labels:
            cls, y1, x1, y2, x2 = yolo_to_yxyx(yolo, iw, ih)

            if not is_valid_box(y1, x1, y2, x2):
                invalid_boxes += 1
                continue

            valid_boxes += 1
            rows.append([
                str(img_path),
                cls,
                round(y1, 2),
                round(x1, 2),
                round(y2, 2),
                round(x2, 2),
            ])

    write_csv(rows, output_path)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Images found: {len(imgs)}")
    print(f"Images without labels: {empty_images}")
    print(f"Valid boxes written: {valid_boxes}")
    print(f"Invalid boxes skipped: {invalid_boxes}")


def main():
    parser = argparse.ArgumentParser(description="YOLO -> EfficientDet CSV converter")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Root dataset folder, e.g. /path/to/dataset",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default=Path("./data_EfficientDet/output"),
        type=Path,
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include images without annotations as empty CSV rows",
    )
    opt = parser.parse_args()

    dataset_dir = opt.dataset_dir

    for split in opt.splits:
        split_images = dataset_dir / split / "images"
        if not split_images.exists():
            print(f"Skipping split {split} because {split_images} does not exist")
            continue

        output_csv = opt.output_dir / f"{split}_efficientdet.csv"
        gather_split(split_images, output_csv, include_empty=opt.include_empty)


if __name__ == "__main__":
    main()