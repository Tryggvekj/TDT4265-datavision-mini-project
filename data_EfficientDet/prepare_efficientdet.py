import argparse
import csv
from pathlib import Path
from typing import List, Tuple


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


def yolo_to_corners(yolo: Tuple[int, float, float, float, float], img_w: int, img_h: int):
    cls, cx, cy, w, h = yolo
    x_center = cx * img_w
    y_center = cy * img_h
    width = w * img_w
    height = h * img_h
    x1 = max(0, x_center - width / 2)
    y1 = max(0, y_center - height / 2)
    x2 = min(img_w, x_center + width / 2)
    y2 = min(img_h, y_center + height / 2)
    return cls, x1, y1, x2, y2


def write_csv(rows: List[List], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)


def gather_split(split_dir: Path, output_path: Path):
    # images live in split_dir and labels in sibling folder split_dir.parent / 'labels'
    label_dir = split_dir.parent / "labels"
    imgs = sorted([p for p in split_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    # image extensions for YOLO dataset
    rows = []
    for img_path in imgs:
        if label_dir.exists():
            label_path = label_dir / img_path.with_suffix(".txt").name
        else:
            label_path = img_path.with_suffix(".txt")
        from PIL import Image
        with Image.open(img_path) as image:
            iw, ih = image.size
        yolo_labels = parse_yolo(label_path)
        for yolo in yolo_labels:
            cls, x1, y1, x2, y2 = yolo_to_corners(yolo, iw, ih)
            rows.append([str(img_path), cls, x1, y1, x2, y2])

    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} box rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO -> EfficientDet CSV converter")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Root dataset folder, e.g. /cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1")
    parser.add_argument("--output-dir", required=False, default=Path("./data_EfficientDet/output"), type=Path)
    parser.add_argument("--splits", nargs="*", default=["train", "valid", "test"],
                        help="Splits to process")

    opt = parser.parse_args()
    dataset_dir = opt.dataset_dir
    for split in opt.splits:
        split_images = dataset_dir / split / "images"
        if not split_images.exists():
            print(f"Skipping split {split} because {split_images} does not exist")
            continue
        output_csv = opt.output_dir / f"{split}_efficientdet.csv"
        gather_split(split_images, output_csv)


if __name__ == "__main__":
    main()
