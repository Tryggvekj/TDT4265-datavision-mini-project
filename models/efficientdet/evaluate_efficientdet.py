"""
Evaluation script for EfficientDet on labeled train/valid splits.

Metrics:
- Precision
- Recall
- mAP@50
- mAP@0.5:0.95

Also saves prediction visualizations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from effdet import DetBenchPredict, create_model

# Navigate to project root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_EfficientDet.dataset import create_dataloaders


def load_model(model_path: str, model_name: str = "efficientdet_d0", num_classes: int = 1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(model_name, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model = DetBenchPredict(model)
    model = model.to(device)
    model.eval()
    return model, device


def yxyx_to_xyxy(boxes):
    boxes = np.array(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    return boxes[:, [1, 0, 3, 2]]


def clip_boxes(boxes, width, height):
    boxes = np.array(boxes, dtype=np.float32).copy()
    if len(boxes) == 0:
        return boxes.reshape(0, 4)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)
    return boxes


def scale_boxes(boxes, from_size, to_size):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.array(boxes, dtype=np.float32).copy()
    from_w, from_h = from_size
    to_w, to_h = to_size

    boxes[:, [0, 2]] *= (to_w / from_w)
    boxes[:, [1, 3]] *= (to_h / from_h)
    return boxes


def remove_invalid_boxes(boxes, scores, labels, min_size=1.0):
    keep_boxes = []
    keep_scores = []
    keep_labels = []

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
            keep_boxes.append([x1, y1, x2, y2])
            keep_scores.append(float(score))
            keep_labels.append(int(label))

    return keep_boxes, keep_scores, keep_labels


def filter_predictions_by_score(predictions, conf_threshold):
    filtered = []

    for pred in predictions:
        keep_boxes = []
        keep_scores = []
        keep_labels = []

        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if score >= conf_threshold:
                keep_boxes.append(box)
                keep_scores.append(score)
                keep_labels.append(label)

        filtered.append({
            "boxes": keep_boxes,
            "scores": keep_scores,
            "labels": keep_labels,
            "image_path": pred["image_path"],
        })

    return filtered


def evaluate_on_split(
    model,
    data_loader,
    device,
    split_name="valid",
    img_size=512,
    pred_label_offset=0,
    debug_labels=False,
):
    model.eval()
    all_predictions = []
    all_targets = []

    print(f"\nEvaluating on {split_name} split...")

    debug_printed = False

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Eval {split_name}"):
            images = batch["images"].to(device).float()
            batch_size = images.shape[0]

            outputs = model(images)

            if isinstance(outputs, dict):
                if "detections" not in outputs:
                    raise KeyError(f"Expected 'detections' in model output, got keys: {outputs.keys()}")
                detections = outputs["detections"]
            else:
                detections = outputs

            for i in range(batch_size):
                pred_boxes = []
                pred_scores = []
                pred_labels = []

                dets = detections[i].detach().cpu().numpy()

                for det in dets:
                    # effdet predict output: [x1, y1, x2, y2, score, class]
                    x1, y1, x2, y2, score, cls = det[:6]

                    # Keep all positive-score predictions for AP/mAP computation
                    if score <= 0:
                        continue

                    pred_boxes.append([x1, y1, x2, y2])
                    pred_scores.append(float(score))
                    pred_labels.append(int(cls) + pred_label_offset)

                pred_boxes = clip_boxes(pred_boxes, img_size, img_size)
                pred_boxes, pred_scores, pred_labels = remove_invalid_boxes(
                    pred_boxes, pred_scores, pred_labels, min_size=1.0
                )

                all_predictions.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                    "image_path": batch["image_path"][i],
                })

                target_boxes = batch["bbox"][i].cpu().numpy()   # yxyx
                target_cls = batch["cls"][i].cpu().numpy()

                all_targets.append({
                    "image_path": batch["image_path"][i],
                    "bbox": target_boxes,
                    "cls": target_cls,
                })

                if debug_labels and not debug_printed:
                    valid_gt = target_cls[target_cls >= 0]
                    print("\n[DEBUG] Sample predicted labels:", pred_labels[:10])
                    print("[DEBUG] Sample GT labels:", valid_gt[:10].tolist())
                    print("[DEBUG] If pred labels are 0 and GT labels are 1, run with --pred-label-offset 1\n")
                    debug_printed = True

    return all_predictions, all_targets


def compute_ap_for_iou(predictions, targets, iou_threshold=0.5):
    all_preds = []
    num_gt = 0
    gt_by_image = {}

    for pred, target in zip(predictions, targets):
        image_path = target["image_path"]

        gt_boxes = np.array(target["bbox"], dtype=np.float32)
        gt_cls = np.array(target["cls"], dtype=np.int64)

        valid = gt_cls >= 0
        gt_boxes = gt_boxes[valid]
        gt_cls = gt_cls[valid]

        # Convert GT from yxyx -> xyxy to match prediction format
        gt_boxes = yxyx_to_xyxy(gt_boxes)

        gt_by_image[image_path] = {
            "boxes": gt_boxes,
            "labels": gt_cls,
            "matched": np.zeros(len(gt_boxes), dtype=bool),
        }
        num_gt += len(gt_boxes)

        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            all_preds.append({
                "image_path": pred["image_path"],
                "box": np.array(box, dtype=np.float32),
                "score": float(score),
                "label": int(label),
            })

    if num_gt == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    if len(all_preds) == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": int(num_gt),
        }

    all_preds.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(all_preds), dtype=np.float32)
    fp = np.zeros(len(all_preds), dtype=np.float32)

    for i, pred in enumerate(all_preds):
        gt_info = gt_by_image[pred["image_path"]]
        gt_boxes = gt_info["boxes"]
        gt_labels = gt_info["labels"]
        matched = gt_info["matched"]

        best_iou = 0.0
        best_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            if matched[j]:
                continue

            # Single-class setup, but keep label check for safety
            if len(gt_labels) > 0 and pred["label"] != int(gt_labels[j]):
                continue

            x1 = max(pred["box"][0], gt_box[0])
            y1 = max(pred["box"][1], gt_box[1])
            x2 = min(pred["box"][2], gt_box[2])
            y2 = min(pred["box"][3], gt_box[3])

            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            inter = inter_w * inter_h

            pred_area = max(0.0, pred["box"][2] - pred["box"][0]) * max(0.0, pred["box"][3] - pred["box"][1])
            gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            union = pred_area + gt_area - inter

            iou = inter / (union + 1e-7)

            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= iou_threshold and best_idx >= 0:
            tp[i] = 1.0
            gt_info["matched"][best_idx] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall_curve = tp_cum / (num_gt + 1e-7)
    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-7)

    # Integral AP
    mrec = np.concatenate(([0.0], recall_curve, [1.0]))
    mpre = np.concatenate(([0.0], precision_curve, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    total_tp = int(tp.sum())
    total_fp = int(fp.sum())
    total_fn = int(num_gt - total_tp)

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (num_gt + 1e-7)

    return {
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def compute_coco_map(predictions, targets):
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    aps = {}
    ap_values = []

    for iou in iou_thresholds:
        result = compute_ap_for_iou(predictions, targets, iou_threshold=float(iou))
        aps[f"{iou:.2f}"] = float(result["ap"])
        ap_values.append(result["ap"])

    return float(np.mean(ap_values)), aps


def visualize_predictions(predictions, targets, output_dir="data_EfficientDet/predictions", max_images=10, img_size=512):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nVisualizing predictions (max {max_images} images)...")

    for idx, (pred, target) in enumerate(zip(predictions[:max_images], targets[:max_images])):
        img_path = target["image_path"]
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        draw = ImageDraw.Draw(img)

        # Ground truth: stored as yxyx, convert to xyxy, then scale to original size
        gt_boxes = np.array(target["bbox"], dtype=np.float32)
        gt_cls = np.array(target["cls"], dtype=np.int64)
        valid = gt_cls >= 0
        gt_boxes = gt_boxes[valid]
        gt_boxes = yxyx_to_xyxy(gt_boxes)
        gt_boxes = scale_boxes(gt_boxes, from_size=(img_size, img_size), to_size=(orig_w, orig_h))

        for box in gt_boxes:
            draw.rectangle(box.tolist(), outline="green", width=2)

        # Predictions are already xyxy in resized image coordinates
        pred_boxes = np.array(pred["boxes"], dtype=np.float32)
        if len(pred_boxes) > 0:
            pred_boxes = scale_boxes(pred_boxes, from_size=(img_size, img_size), to_size=(orig_w, orig_h))

            for box, score in zip(pred_boxes, pred["scores"]):
                draw.rectangle(box.tolist(), outline="red", width=2)
                draw.text((box[0], max(0, box[1] - 10)), f"{score:.2f}", fill="red")

        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"pred_{idx}_{base_name}")
        img.save(output_path)
        print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "weights" / "best_model.pth"),
        help="Path to checkpoint",
    )
    parser.add_argument("--model", type=str, default="tf_efficientdet_d3")
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data_EfficientDet" / "output" / "train_efficientdet.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data_EfficientDet" / "output" / "valid_efficientdet.csv"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid"],
        help="Data split to evaluate. Use 'valid' for fair evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for Precision/Recall")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=896)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pred-label-offset",
        type=int,
        default=0,
        help="Add offset to predicted class labels. Use 1 if model predicts class 0 but GT uses class 1.",
    )
    parser.add_argument(
        "--debug-labels",
        action="store_true",
        help="Print one sample of predicted labels and GT labels for debugging class indexing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "saved_models"),
        help="Output directory for results and prediction images",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model from {args.model_path}...")
    model, device = load_model(
        args.model_path,
        model_name=args.model,
        num_classes=args.num_classes,
        device=device,
    )

    print(f"\nLoading {args.split} data...")
    train_loader, val_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    data_loader = val_loader if args.split == "valid" else train_loader

    predictions, targets = evaluate_on_split(
        model,
        data_loader,
        device,
        split_name=args.split,
        img_size=args.img_size,
        pred_label_offset=args.pred_label_offset,
        debug_labels=args.debug_labels,
    )

    # Precision/Recall at chosen score threshold
    predictions_thr = filter_predictions_by_score(predictions, args.conf_threshold)
    pr_result = compute_ap_for_iou(predictions_thr, targets, iou_threshold=0.5)

    # AP metrics should use all predictions, sorted by confidence
    ap50_result = compute_ap_for_iou(predictions, targets, iou_threshold=0.5)
    coco_map, ap_by_iou = compute_coco_map(predictions, targets)

    print(f"\n{'=' * 50}")
    print(f"Results on {args.split} split:")
    print(f"{'=' * 50}")
    print(f"Precision     : {pr_result['precision']:.4f}")
    print(f"Recall        : {pr_result['recall']:.4f}")
    print(f"mAP@50        : {ap50_result['ap']:.4f}")
    print(f"mAP@0.5:0.95  : {coco_map:.4f}")
    print(f"{'=' * 50}")

    results = {
        "split": args.split,
        "model_path": args.model_path,
        "precision": pr_result["precision"],
        "recall": pr_result["recall"],
        "mAP@50": ap50_result["ap"],
        "mAP@0.5:0.95": coco_map,
        "tp": pr_result["tp"],
        "fp": pr_result["fp"],
        "fn": pr_result["fn"],
        "conf_threshold": args.conf_threshold,
        "num_images": len(predictions),
        "pred_label_offset": args.pred_label_offset,
        "ap_by_iou": ap_by_iou,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"results_{args.split}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Visualize thresholded predictions since those are easier to inspect
    visualize_predictions(
        predictions_thr,
        targets,
        output_dir=args.output_dir,
        max_images=10,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()