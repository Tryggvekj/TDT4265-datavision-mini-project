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


def yolo_to_yxyx(yolo, img_w: int, img_h: int):
    cls, cx, cy, w, h = yolo

    x_center = cx * img_w
    y_center = cy * img_h
    width = w * img_w
    height = h * img_h

    x1 = max(0.0, x_center - width / 2.0)
    y1 = max(0.0, y_center - height / 2.0)
    x2 = min(float(img_w), x_center + width / 2.0)
    y2 = min(float(img_h), y_center + height / 2.0)

    cls = cls + 1
    return cls, y1, x1, y2, x2


def is_valid_box(y1: float, x1: float, y2: float, x2: float, min_size: float = 1.0) -> bool:
    return (y2 - y1) >= min_size and (x2 - x1) >= min_size


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "y1", "x1", "y2", "x2"])
        writer.writerows(rows)


def gather_split(image_dir: Path, label_dir: Path, output_path: Path, include_empty: bool = False):
    imgs = sorted([p for p in image_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    rows = []
    empty_images = 0
    invalid_boxes = 0
    valid_boxes = 0

    for img_path in imgs:
        label_path = label_dir / img_path.with_suffix(".txt").name

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
    parser = argparse.ArgumentParser(description="YOLO -> EfficientDet CSV converter for Road_poles_iPhone")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("./data_EfficientDet/output_iphone"), type=Path)
    parser.add_argument("--include-empty", action="store_true")
    opt = parser.parse_args()

    dataset_dir = opt.dataset_dir

    split_map = {
        "train": ("images/Train/train", "labels/Train/train"),
        "valid": ("images/Validation/val", "labels/Validation/val"),
    }

    for split_name, (image_subdir, label_subdir) in split_map.items():
        image_dir = dataset_dir / image_subdir
        label_dir = dataset_dir / label_subdir

        if not image_dir.exists():
            print(f"Skipping {split_name}: missing image dir {image_dir}")
            continue
        if not label_dir.exists():
            print(f"Skipping {split_name}: missing label dir {label_dir}")
            continue

        output_csv = opt.output_dir / f"{split_name}_efficientdet.csv"
        gather_split(image_dir, label_dir, output_csv, include_empty=opt.include_empty)

    def write_empty_test_csv(image_dir: Path, output_path: Path):
        imgs = sorted([p for p in image_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        rows = []
        for img_path in imgs:
            rows.append([str(img_path), "", "", "", "", ""])

        write_csv(rows, output_path)

        print(f"Wrote {len(rows)} empty test rows to {output_path}")
        print(f"Test images found: {len(imgs)}")

    test_image_dir = dataset_dir / "images/Test/test"
    if test_image_dir.exists():
        output_csv = opt.output_dir / "test_efficientdet.csv"
        write_empty_test_csv(test_image_dir, output_csv)
    else:
        print(f"Skipping test: missing image dir {test_image_dir}")

if __name__ == "__main__":
    main()