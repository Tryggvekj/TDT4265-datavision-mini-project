from pathlib import Path
import yaml
from PIL import Image, ImageDraw

DATASET_DIR = Path("/cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1")


def get_image_files(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def image_to_label_path(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / image_path.with_suffix(".txt").name


def parse_yolo_labels(label_path: Path):
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def draw_boxes(image_path: Path, label_path: Path, output_path: Path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    labels = parse_yolo_labels(label_path)

    for class_id, x_center, y_center, width, height in labels:
        x_center_px = x_center * img_w
        y_center_px = y_center * img_h
        width_px = width * img_w
        height_px = height * img_h

        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), str(class_id), fill="red")

    img.save(output_path)


def inspect_split(split_name: str, images_dir: Path, labels_dir: Path):
    print(f"{split_name}:")
    print("  images:", images_dir)
    print("  labels:", labels_dir)
    print("  images finnes?", images_dir.exists())
    print("  labels finnes?", labels_dir.exists())

    if not images_dir.exists():
        print()
        return None

    image_files = get_image_files(images_dir)
    print(f"  antall bilder: {len(image_files)}")

    if not image_files:
        print()
        return None

    sample_image = image_files[0]
    sample_label = image_to_label_path(sample_image, labels_dir)

    print("  sample-bilde:", sample_image)
    print("  sample-label:", sample_label)
    print("  label finnes?", sample_label.exists())

    labels = parse_yolo_labels(sample_label)
    print(f"  antall objekter i sample-label: {len(labels)}")
    if labels:
        print("  første label:", labels[0])

    print()
    return sample_image, sample_label


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Fant ikke datasettmappen: {DATASET_DIR}")

    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    print("Innhold i data.yaml:")
    print(config)
    print()

    print("Mapper i datasettet:")
    for item in sorted(DATASET_DIR.iterdir()):
        print(" -", item.name)
    print()

    # Bruk faktiske mapper i roadpoles_v1, ikke stiene i yaml
    train_images_dir = DATASET_DIR / "train" / "images"
    train_labels_dir = DATASET_DIR / "train" / "labels"

    val_images_dir = DATASET_DIR / "valid" / "images"
    val_labels_dir = DATASET_DIR / "valid" / "labels"

    test_images_dir = DATASET_DIR / "test" / "images"
    test_labels_dir = DATASET_DIR / "test" / "labels"

    train_sample = inspect_split("Train", train_images_dir, train_labels_dir)
    inspect_split("Validation", val_images_dir, val_labels_dir)
    inspect_split("Test", test_images_dir, test_labels_dir)

    if train_sample is None:
        raise ValueError("Fant ingen treningsbilder.")

    sample_image, sample_label = train_sample

    if sample_label.exists():
        output_path = Path("sample_with_boxes.jpg")
        draw_boxes(sample_image, sample_label, output_path)
        print(f"Lagret eksempelbilde med bounding boxes: {output_path}")
    else:
        print("Kunne ikke tegne sample, fordi label-fil mangler.")


if __name__ == "__main__":
    main()