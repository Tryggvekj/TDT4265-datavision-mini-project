"""
Training script for EfficientDet on custom road poles dataset.
Based on rwightman/efficientdet-pytorch patterns.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from effdet import EfficientDet, DetBenchTrain, create_model
from effdet.config import get_efficientdet_config
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # Parent of data_EfficientDet
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_EfficientDet.dataset import create_dataloaders


class AverageMeter:
    """Track metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, optimizer, scaler, train_loader, device, args):
    """Train one epoch"""
    model.train()
    metric_logger = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['images'].to(device).float()
        boxes = batch['boxes'].to(device).float()
        labels = batch['labels'].to(device).long()
        
        # Prepare targets in the format expected by efficientdet
        # DetBenchTrain expects a single dict with batched tensors, not a list
        all_boxes = []
        all_labels = []
        for i in range(images.shape[0]):
            valid_mask = (boxes[i].sum(dim=1) > 0)
            valid_boxes = boxes[i][valid_mask]
            valid_labels = labels[i][valid_mask]
            
            all_boxes.append(valid_boxes)
            all_labels.append(valid_labels)
        
        # Pad to same length for batching
        max_boxes = max(len(b) for b in all_boxes) if all_boxes else 0
        if max_boxes > 0:
            padded_boxes = []
            padded_labels = []
            for boxes_batch, labels_batch in zip(all_boxes, all_labels):
                if len(boxes_batch) < max_boxes:
                    pad_size = max_boxes - len(boxes_batch)
                    boxes_batch = torch.cat([boxes_batch, torch.zeros(pad_size, 4, device=device)])
                    labels_batch = torch.cat([labels_batch, torch.zeros(pad_size, dtype=torch.long, device=device)])
                padded_boxes.append(boxes_batch)
                padded_labels.append(labels_batch)
            
            targets = {
                'bbox': torch.stack(padded_boxes),
                'cls': torch.stack(padded_labels),
            }
        else:
            targets = {
                'bbox': torch.zeros(images.shape[0], 0, 4, device=device),
                'cls': torch.zeros(images.shape[0], 0, dtype=torch.long, device=device),
            }
        
        # Forward
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss_dict = model(images, targets)
            loss = loss_dict['loss']
        
        if not torch.isfinite(loss):
            print(f"Loss is {loss}, stopping training")
            sys.exit(1)
        
        # Backward
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        
        metric_logger.update(loss.item())
        pbar.set_postfix({'loss': f'{metric_logger.avg:.4f}'})
    
    return metric_logger.avg


@torch.no_grad()
def evaluate(model, val_loader, device, args):
    """Validate"""
    model.train()
    metric_logger = AverageMeter()
    
    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        images = batch['images'].to(device).float()
        boxes = batch['boxes'].to(device).float()
        labels = batch['labels'].to(device).long()
        
        # Prepare batched targets in the format expected by efficientdet
        all_boxes = []
        all_labels = []
        for i in range(images.shape[0]):
            valid_mask = (boxes[i].sum(dim=1) > 0)
            valid_boxes = boxes[i][valid_mask]
            valid_labels = labels[i][valid_mask]

            all_boxes.append(valid_boxes)
            all_labels.append(valid_labels)

        max_boxes = max(len(b) for b in all_boxes) if all_boxes else 0
        if max_boxes > 0:
            padded_boxes = []
            padded_labels = []
            for boxes_batch, labels_batch in zip(all_boxes, all_labels):
                if len(boxes_batch) < max_boxes:
                    pad_size = max_boxes - len(boxes_batch)
                    boxes_batch = torch.cat([boxes_batch, torch.zeros(pad_size, 4, device=device)])
                    labels_batch = torch.cat([labels_batch, torch.zeros(pad_size, dtype=torch.long, device=device)])
                padded_boxes.append(boxes_batch)
                padded_labels.append(labels_batch)

            targets = {
                'bbox': torch.stack(padded_boxes),
                'cls': torch.stack(padded_labels),
            }
        else:
            targets = {
                'bbox': torch.zeros(images.shape[0], 0, 4, device=device),
                'cls': torch.zeros(images.shape[0], 0, dtype=torch.long, device=device),
            }
        
        loss_dict = model(images, targets)
        loss = loss_dict['loss']
        metric_logger.update(loss.item())
        pbar.set_postfix({'loss': f'{metric_logger.avg:.4f}'})
    
    return metric_logger.avg


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if device.type == 'cpu':
        print("Warning: Using CPU - training will be slow. GPU required for reasonable performance.")
        args.amp = False  # AMP not supported on CPU
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        args.train_csv,
        args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print(f"Creating model: {args.model}...")
    if args.pretrained:
        model = create_model(args.model, pretrained=True, num_classes=args.num_classes)
        print(f"Loaded pretrained weights for {args.model}")
    else:
        config = get_efficientdet_config(args.model)
        config.num_classes = args.num_classes
        model = EfficientDet(config)
        print(f"Created model {args.model} from scratch")
    
    model = model.to(device)
    model_bench = DetBenchTrain(model).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # Training loop
    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model_bench, optimizer, scaler, train_loader, device, args)
        val_loss = evaluate(model_bench, val_loader, device, args)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientDet training on custom dataset')
    
    # Dataset
    parser.add_argument('--train-csv', type=str, default='data_EfficientDet/output/train_efficientdet.csv',
                        help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, default='data_EfficientDet/output/valid_efficientdet.csv',
                        help='Path to validation CSV')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of object classes')
    
    # Model
    parser.add_argument('--model', type=str, default='efficientdet_d0',
                        help='EfficientDet model variant (d0, d1, d2, ...)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Load pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip-grad', type=float, default=10.0, help='Gradient clipping value')
    
    # Hardware
    parser.add_argument('--num-workers', type=int, default=1, help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true', default=True, help='Use automatic mixed precision')
    
    # Image
    parser.add_argument('--img-size', type=int, default=512, help='Input image size')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='data_EfficientDet/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint frequency (epochs)')
    
    args = parser.parse_args()
    main(args)
