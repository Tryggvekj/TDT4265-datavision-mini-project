"""
Generate YOLO-format predictions for submission.
Outputs one .txt per image in YOLO format:

class cx cy w h confidence

Assumptions:
- Model was trained with rwightman/efficientdet-pytorch
- Predict bench returns detections as [x1, y1, x2, y2, score, class]
- Class ids are already aligned with YOLO ids (0..N-1)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from effdet import DetBenchPredict, create_model
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def build_transform(img_size: int):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        ToTensorV2(),
    ])

def load_model(model_path: str, model_name: str, num_classes: int, device: torch.device):
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    model.load_state_dict(state_dict)

    model = DetBenchPredict(model)
    model = model.to(device)
    model.eval()
    return model


def nms_xyxy(boxes, scores, iou_threshold=0.5):
    """Simple class-agnostic NMS on XYXY boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-7)

        remaining = np.where(iou < iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def pixel_xyxy_to_yolo(box_xyxy, img_w, img_h):
    """Convert absolute XYXY pixel box to normalized YOLO CXCYWH."""
    x1, y1, x2, y2 = box_xyxy

    x1 = float(np.clip(x1, 0, img_w))
    y1 = float(np.clip(y1, 0, img_h))
    x2 = float(np.clip(x2, 0, img_w))
    y2 = float(np.clip(y2, 0, img_h))

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w /= img_w
    h /= img_h

    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))

    return cx, cy, w, h


def list_images(test_dir: Path):
    return sorted(
        [p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

def predict_one_image(model, image_path: Path, transform, device, img_size: int,
                      conf_threshold: float, nms_threshold: float):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w = img_np.shape[:2]

    transformed = transform(image=img_np)
    img_tensor = transformed["image"].unsqueeze(0).to(device).float()

    scale_x = orig_w / float(img_size)
    scale_y = orig_h / float(img_size)

    with torch.no_grad():
        outputs = model(img_tensor)

        if isinstance(outputs, dict):
            if "detections" not in outputs:
                raise KeyError(f"Expected 'detections' in model output, got keys: {outputs.keys()}")
            detections = outputs["detections"]
        else:
            detections = outputs

        detections = detections[0].detach().cpu().numpy()

    pred_boxes = []
    pred_scores = []
    pred_labels = []

    for det in detections:
        # Predict bench outputs XYXY boxes.
        x1, y1, x2, y2, score, cls = det[:6]

        score = float(score)
        if score < conf_threshold:
            continue

        x1 = float(x1) * scale_x
        y1 = float(y1) * scale_y
        x2 = float(x2) * scale_x
        y2 = float(y2) * scale_y

        if x2 <= x1 or y2 <= y1:
            continue

        cls_id = int(round(float(cls))) - 1
        if cls_id < 0:
            cls_id = 0

        pred_boxes.append([x1, y1, x2, y2])
        pred_scores.append(score)
        pred_labels.append(cls_id)

    # Apply per-class NMS
    final_boxes = []
    final_scores = []
    final_labels = []

    if pred_boxes:
        unique_classes = sorted(set(pred_labels))
        for c in unique_classes:
            cls_indices = [i for i, lab in enumerate(pred_labels) if lab == c]
            cls_boxes = [pred_boxes[i] for i in cls_indices]
            cls_scores = [pred_scores[i] for i in cls_indices]

            keep_local = nms_xyxy(cls_boxes, cls_scores, iou_threshold=nms_threshold)

            for k in keep_local:
                final_boxes.append(cls_boxes[k])
                final_scores.append(cls_scores[k])
                final_labels.append(c)

    return final_boxes, final_scores, final_labels, orig_w, orig_h

def predict_on_directory(
    model,
    test_dir: str,
    device: torch.device,
    output_dir: str,
    conf_threshold: float = 0.5,
    img_size: int = 512,
    nms_threshold: float = 0.5,
    write_empty_files: bool = True,
):
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_txt in output_dir.glob("*.txt"):
        old_txt.unlink()

    transform = build_transform(img_size)
    image_files = list_images(test_dir)

    if not image_files:
        raise FileNotFoundError(f"No images found in {test_dir}")

    print(f"Found {len(image_files)} images in {test_dir}")
    print("Generating predictions...")

    model.eval()

    for img_path in tqdm(image_files, desc="Predicting"):
        boxes, scores, labels, orig_w, orig_h = predict_one_image(
            model=model,
            image_path=img_path,
            transform=transform,
            device=device,
            img_size=img_size,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )

        if len(scores) > 0:
            sorted_idx = np.argsort(np.array(scores))[::-1]
            boxes = [boxes[i] for i in sorted_idx]
            sorted_scores = [scores[i] for i in sorted_idx]
            labels = [labels[i] for i in sorted_idx]
            scores = sorted_scores

        out_path = output_dir / f"{img_path.stem}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            wrote_any = False
            for box_xyxy, score, cls_id in zip(boxes, scores, labels):
                yolo_box = pixel_xyxy_to_yolo(box_xyxy, orig_w, orig_h)
                if yolo_box is None:
                    continue
                cx, cy, w, h = yolo_box
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}\n")
                wrote_any = True

            if not wrote_any and not write_empty_files:
                out_path.unlink(missing_ok=True)

    print(f"Predictions saved to {output_dir}")
    return output_dir


def create_submission_zip(predictions_dir, output_zip="submission.zip"):
    predictions_dir = Path(predictions_dir).resolve()
    output_zip = str(Path(output_zip).with_suffix(""))
    shutil.make_archive(output_zip, "zip", root_dir=predictions_dir.parent, base_dir=predictions_dir.name)
    print(f"Submission ready: {output_zip}.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                        default=str(Path(__file__).resolve().parent / 'weights' / 'best_model.pth'))
    parser.add_argument("--model-name", type=str, default="efficientdet_d0")
    parser.add_argument("--test-dir", type=str, 
                        default=str(Path(__file__).resolve().parents[2] / 'data_YOLO' / 'test' / 'images'),
                        help='Test images directory')
    parser.add_argument("--output-dir", type=str, 
                        default=str(Path(__file__).resolve().parent / 'submissions'),
                        help='Output submissions directory')
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--create-zip", action="store_true")
    parser.add_argument("--no-empty-files", action="store_true",
                        help="Delete txt files for images with no detections instead of writing empty files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    print(f"Loading model from {args.model_path}...")
    model = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        device=device,
    )

    pred_dir = predict_on_directory(
        model=model,
        test_dir=args.test_dir,
        device=device,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        img_size=args.img_size,
        nms_threshold=args.nms_threshold,
        write_empty_files=not args.no_empty_files,
    )

    if args.create_zip:
        create_submission_zip(pred_dir, output_zip=f"{args.output_dir}.zip")


if __name__ == "__main__":
    main()