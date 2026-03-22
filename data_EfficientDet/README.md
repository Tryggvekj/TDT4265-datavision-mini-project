# EfficientDet Training Pipeline

This folder contains scripts to prepare YOLO-format datasets for EfficientDet training and the training loop itself.

## Files

- `prepare_efficientdet.py`: Converts YOLO format (`class cx cy w h` normalized) to EfficientDet format (corner boxes `x1 y1 x2 y2` in pixels). Outputs CSV files per split.
- `dataset.py`: PyTorch Dataset and DataLoader factory that reads prepared CSV files, handles augmentation (via albumentations), and batching.
- `train.py`: Complete training script using `efficientdet-pytorch` (rwightman) with standard finetuning patterns.

## Quick Start

### 1. Prepare data (YOLO → CSV)
```bash
python data_EfficientDet/prepare_efficientdet.py \
  --dataset-dir /cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1 \
  --output-dir ./data_EfficientDet/output \
  --splits train valid test
```

Output: `train_efficientdet.csv`, `valid_efficientdet.csv`, `test_efficientdet.csv`

### 2. Train EfficientDet
```bash
# First, activate conda environment
module load Miniconda3/24.7.1-0
conda activate efficientdet

# Then run training
python data_EfficientDet/train.py \
  --model efficientdet_d0 \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-classes 1 \
  --img-size 512
```

Key arguments:
- `--model`: Choose D0 (small, fast), D1, D2, etc. based on resources
- `--num-classes`: 1 for road pole detection
- `--batch-size`: Reduce if GPU memory is limited
- `--img-size`: Input resolution (512, 640, 768, etc.)
- `--amp`: Automatic mixed precision (default: enabled)

### 3. Outputs
- `data_EfficientDet/checkpoints/best_model.pth`: Best validation model
- `data_EfficientDet/checkpoints/checkpoint_epoch_*.pth`: Periodic checkpoints

## Dependencies

```bash
pip install albumentations torchvision
# efficientdet-pytorch already installed in conda env
```

## Notes

- Images are resized to `--img-size` during training (aspect ratio not preserved).
- Augmentation includes: horizontal/vertical flips, brightness/contrast, rotation, Gaussian noise.
- Uses Focal Loss (built into `DetBenchTrain`).
- Cosine LR schedule with warmup via `CosineAnnealingLR`.
- Gradient clipping enabled (clip_grad=10.0) to avoid instability.

## Finetuning Tips

- Start with `efficientdet_d0` for fastest iteration.
- Use `--lr 1e-4` or lower for pretrained finetuning (don't overshoot).
- If loss is NaN, reduce `--lr` or increase `--batch-size`.
- Monitor val loss; if it starts increasing (overfitting), reduce epochs or add more augmentation.
- For best results, tune on your validation set and inspect predictions.
