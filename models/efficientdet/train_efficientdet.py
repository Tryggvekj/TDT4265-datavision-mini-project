"""
Training script for EfficientDet on custom road poles dataset.
Based on rwightman/efficientdet-pytorch patterns.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from effdet import EfficientDet, DetBenchTrain, create_model
from effdet.config import get_efficientdet_config
import sys

# Navigate to project root (3 levels up from models/efficientdet/)
ROOT_DIR = Path(__file__).resolve().parents[2]
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
    model.train()
    metric_logger = AverageMeter()

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['images'].to(device).float()
        targets = {
            'bbox': batch['bbox'].to(device).float(),
            'cls': batch['cls'].to(device).long(),
            'img_size': batch['img_size'].to(device).float(),
            'img_scale': batch['img_scale'].to(device).float(),
        }

        with torch.cuda.amp.autocast(enabled=args.amp):
            loss_dict = model(images, targets)
            loss = loss_dict['loss']

        if not torch.isfinite(loss):
            print(f"Loss is {loss}, stopping training")
            sys.exit(1)

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
    model.eval()
    metric_logger = AverageMeter()

    pbar = tqdm(val_loader, desc='Validation')
    for batch in pbar:
        images = batch['images'].to(device).float()
        targets = {
            'bbox': batch['bbox'].to(device).float(),
            'cls': batch['cls'].to(device).long(),
            'img_size': batch['img_size'].to(device).float(),
            'img_scale': batch['img_scale'].to(device).float(),
        }

        with torch.cuda.amp.autocast(enabled=args.amp):
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
        args.amp = False
    
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

    # Quick sanity check
    batch = next(iter(train_loader))
    print("images shape:", batch["images"].shape)
    print("bbox shape:", batch["bbox"].shape)
    print("cls shape:", batch["cls"].shape)
    print("first image path:", batch["image_path"][0])
    print("first image boxes:", batch["bbox"][0][:5])
    print("first image labels:", batch["cls"][0][:5])
    
    # Create model
    print(f"Creating model: {args.model}...")
    if not args.no_pretrained:
        model = create_model(args.model, pretrained=True, num_classes=args.num_classes)
        print(f"Loaded pretrained weights for {args.model}")
    else:
        config = get_efficientdet_config(args.model)
        config.num_classes = args.num_classes
        model = EfficientDet(config)
        print(f"Created model {args.model} from scratch")
        
    if args.resume is not None:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model_bench = DetBenchTrain(model).to(device)
    
    optimizer = optim.AdamW(model_bench.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    best_val_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model_bench, optimizer, scaler, train_loader, device, args)
        val_loss = evaluate(model_bench, val_loader, device, args)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientDet training on custom dataset')
    
    # Dataset
    parser.add_argument('--train-csv', type=str, 
                        default=str(Path(__file__).resolve().parents[2] / 'data_EfficientDet' / 'output' / 'train_efficientdet.csv'),
                        help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, 
                        default=str(Path(__file__).resolve().parents[2] / 'data_EfficientDet' / 'output' / 'valid_efficientdet.csv'),
                        help='Path to validation CSV')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of object classes')
    
    # Model
    parser.add_argument('--model', type=str, default='efficientdet_d0',
                        help='EfficientDet model variant (d0, d1, d2, ...)')
    parser.add_argument('--no-pretrained', action='store_true',
                    help='Disable pretrained weights')
    
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
    parser.add_argument('--output-dir', type=str, 
                        default=str(Path(__file__).resolve().parent / 'weights'),
                        help='Output directory for checkpoints')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint frequency (epochs)')
    
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
