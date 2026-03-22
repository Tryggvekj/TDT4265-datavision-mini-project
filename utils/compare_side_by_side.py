import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import cv2
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side images: original | prediction (optional ground truth)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/weights/yolo/v1.pt"),
        help="Path to trained weights (.pt)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=config.DATA_YOLO / "images" / "val",
        help="Directory with images to evaluate",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional directory with YOLO-format labels for ground truth overlay",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/compare/run1"),
        help="Directory to save side-by-side comparison images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to process",
    )
    parser.add_argument("--imgsz", type=int, default=768, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Inference device, e.g. "0" for GPU or "cpu"',
    )
    return parser.parse_args()


def load_yolo_labels(
    label_path: Path, image_width: int, image_height: int
) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes.append((cls_id, x1, y1, x2, y2))

    return boxes


def draw_ground_truth(
    image: "cv2.typing.MatLike",
    boxes: List[Tuple[int, float, float, float, float]],
) -> "cv2.typing.MatLike":
    output = image.copy()
    for cls_id, x1, y1, x2, y2 in boxes:
        pt1 = (int(max(0, x1)), int(max(0, y1)))
        pt2 = (int(min(output.shape[1] - 1, x2)), int(min(output.shape[0] - 1, y2)))
        cv2.rectangle(output, pt1, pt2, (0, 220, 0), 2)
        cv2.putText(
            output,
            f"GT:{cls_id}",
            (pt1[0], max(18, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def draw_predictions(image: "cv2.typing.MatLike", result) -> "cv2.typing.MatLike":
    output = image.copy()
    if result.boxes is None:
        return output

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()

    for box, score, cls_id in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = box
        pt1 = (int(max(0, x1)), int(max(0, y1)))
        pt2 = (int(min(output.shape[1] - 1, x2)), int(min(output.shape[0] - 1, y2)))
        cv2.rectangle(output, pt1, pt2, (0, 0, 230), 2)
        cv2.putText(
            output,
            f"P:{int(cls_id)} {score:.2f}",
            (pt1[0], max(18, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 230),
            2,
            cv2.LINE_AA,
        )

    return output


def add_panel_title(image: "cv2.typing.MatLike", title: str) -> "cv2.typing.MatLike":
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        panel,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            p
            for p in args.images_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    image_paths = image_paths[: args.max_images]
    model = YOLO(str(args.weights))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        h, w = image.shape[:2]
        gt_boxes = []
        has_labels = args.labels_dir is not None
        if has_labels:
            label_path = args.labels_dir / f"{image_path.stem}.txt"
            gt_boxes = load_yolo_labels(label_path, w, h)

        pred_result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        panel_original = add_panel_title(image, "Original")
        panel_pred = add_panel_title(draw_predictions(image, pred_result), "Prediction")
        if has_labels:
            panel_gt = add_panel_title(draw_ground_truth(image, gt_boxes), "Ground Truth")
            comparison = cv2.hconcat([panel_original, panel_gt, panel_pred])
        else:
            comparison = cv2.hconcat([panel_original, panel_pred])
        output_path = args.output_dir / f"{image_path.stem}_compare.jpg"
        cv2.imwrite(str(output_path), comparison)
        print(f"Saved: {output_path}")

    print(f"Done. Wrote {len(image_paths)} comparison image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
