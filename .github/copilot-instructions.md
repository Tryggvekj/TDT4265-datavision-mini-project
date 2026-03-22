# Copilot Instructions for TDT4265-datavision-mini-project

## Project structure at a glance
- `inspect_data.py`: main entrypoint for dataset sanity checks and YOLO-label visualization (`data.yaml` + `train/valid/test` dirs).
- `config.py`: external dataset path constants used by helper code; contains local path expectations (`~/cluster/projects/vc/courses/TDT17/ad/Poles2025`).
- `data_processing_pipeline/`: intended data preparation pipeline; current code is scaffolding with `data_processing_pipeline.py` empty and `helper_functions/data_getters.py` setting up `sys.path` + config import.
- `data_YOLO/`: raw content placeholder (mirrored dataset commitments, no logic).

## Big picture architecture
- This repo is primarily a dataset inspection and labeling pipeline starter for road pole detection (YOLO format).
- Core data flow: dataset on shared storage -> `inspect_data.py` reads `data.yaml`, enumerates image/label splits, parses YOLO txt labels -> draws bounding boxes into `sample_with_boxes.jpg`.
- No model training or evaluation code present yet, so tasks usually add data ingestion + pre-processing, then integrate with external training scripts.

## Important conventions
- YOLO label parser in `inspect_data.py` expects 5 columns per row: `class_id x_center y_center width height` (normalized coordinates).
- File extension filter uses `.jpg`, `.jpeg`, `.png` and aligns label names as image name with `.txt` extension (function `image_to_label_path`).
- Paths are hardcoded in `inspect_data.py` as `DATASET_DIR`; adjust before running in different environment.

## Developer workflows
- Run main data sanity check:
  - `python inspect_data.py`
- If using local path override: edit `DATASET_DIR` in `inspect_data.py` or use a temporary wrapper script.
- No tests folder or CI currently included; quick validation is via manual invocation.

## Integration points
- `data_processing_pipeline/helper_functions/data_getters.py` adds repo root to `sys.path` and imports `config`; follow this pattern when writing scripts that need repo-wide path constants.
- `config.py` path constants are meant for dataset PK (iPhone vs v1 datasets). Keep in sync with `inspect_data.py` dataset root.

## EfficientDet data conversion (new)
- `data_EfficientDet/prepare_efficientdet.py` converts YOLO labels to corner format and writes CSV per split.
- Input path format expected: `dataset_dir/<split>/images/*.jpg` and matching `*.txt` in same folder.
- YOLO -> corners transformation (from file):
  - `class,cx,cy,w,h` (normalized) -> `class,x1,y1,x2,y2` (pixels)
  - `x1 = cx*W - w*W/2`, `y1 = cy*H - h*H/2`, `x2 = cx*W + w*W/2`, `y2 = cy*H + h*H/2`
- Example usage:
  - `python data_EfficientDet/prepare_efficientdet.py --dataset-dir /cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1 --output-dir data_EfficientDet/output --splits train valid test`

## Notes on model-ready format
- For EfficientDet training in PyTorch/TF:
  - use converted CSV rows to create dataset entries
  - class IDs are 0-indexed and passthrough unchanged
  - ensure boxes are clipped in image bounds from conversion in script

## EfficientDet training workflow
- `data_EfficientDet/dataset.py`: PyTorch Dataset class that reads CSV, applies augmentation (albumentations), and batches with custom collate.
- `data_EfficientDet/train.py`: Full finetuning script using `efficientdet-pytorch` (rwightman repo). Includes AdamW optimizer, cosine LR schedule, AMP, gradient clipping.
- `data_EfficientDet/README.md`: Quick-start guide for training.
- Train command:
  - `python data_EfficientDet/train.py --model efficientdet_d0 --epochs 100 --batch-size 8 --lr 1e-4 --num-classes 1`
- Outputs: `data_EfficientDet/checkpoints/{best_model.pth, checkpoint_epoch_*.pth}`
- Key hyperparameters for finetuning:
  - `--lr 1e-4`: Low learning rate (don't overfit pretrained backbone)
  - `--num-classes 1`: Road poles (single class)
  - `--batch-size 8`: Adjust based on GPU memory (reduce for OOM)
  - `--img-size 512`: Input resolution; larger uses more memory but may improve accuracy

## Approach for AI edits
- Preserve the existing `inspect_data.py` behavior while adding robust path injection (argument parsing for dataset root), and keep YOLO parsing logic unchanged.
- If you add new feature code in `data_processing_pipeline/`, maintain the existing lightweight dependency chain: no heavy 3rd-party requirements beyond `PIL` and `pyyaml` currently.

## Notes for reviewers
- There is no `.github/copilot-instructions.md` prior to this file; this is the canonical place for agents to capture the project-specific operational context.
- This repo uses explicit dataset paths on shared cluster; avoid assuming local filesystem data names.

