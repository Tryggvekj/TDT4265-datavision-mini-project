import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class EfficientDetDataset(Dataset):
    """PyTorch Dataset for EfficientDet training from CSV prepared by prepare_efficientdet.py"""
    
    def __init__(self, csv_path: str, img_size: int = 512, augment: bool = False):
        """
        Args:
            csv_path: Path to CSV file with columns [image, class, x1, y1, x2, y2]
            img_size: Target image size for training
            augment: Whether to apply augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.augment = augment
        
        # Group by image for efficient loading
        self.image_groups = self.df.groupby('image')
        self.image_paths = list(self.image_groups.groups.keys())
        
        # Augmentation transforms
        if self.augment:
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.GaussNoise(p=0.2),
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
        else:
            self.transforms = A.Compose([
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img_data = self.image_groups.get_group(img_path)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Extract boxes and labels
        boxes = img_data[['x1', 'y1', 'x2', 'y2']].values.tolist()
        labels = img_data['class'].values.tolist()
        
        # Apply transforms (includes resize)
        if self.transforms:
            transformed = self.transforms(image=img_np, bboxes=boxes, class_labels=labels)
            img_tensor = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Convert to tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        
        return {
            'image': img_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor(idx, dtype=torch.int64),
        }


def create_dataloaders(train_csv: str, val_csv: str, batch_size: int = 8, 
                       num_workers: int = 4, img_size: int = 512):
    """Create train and validation dataloaders"""
    train_dataset = EfficientDetDataset(train_csv, img_size=img_size, augment=True)
    val_dataset = EfficientDetDataset(val_csv, img_size=img_size, augment=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for batching variable-sized bounding boxes"""
    images = torch.stack([item['image'] for item in batch])
    
    # Pad boxes and labels to same length per batch
    max_boxes = max(len(item['boxes']) for item in batch)
    
    boxes_list = []
    labels_list = []
    image_ids = torch.stack([item['image_id'] for item in batch])
    
    for item in batch:
        boxes = item['boxes']
        labels = item['labels']
        
        # Pad with zeros
        if len(boxes) < max_boxes:
            pad_size = max_boxes - len(boxes)
            boxes = torch.cat([boxes, torch.zeros(pad_size, 4)])
            labels = torch.cat([labels, torch.zeros(pad_size, dtype=torch.int64)])
        
        boxes_list.append(boxes)
        labels_list.append(labels)
    
    boxes_batch = torch.stack(boxes_list)
    labels_batch = torch.stack(labels_list)
    
    return {
        'images': images,
        'boxes': boxes_batch,
        'labels': labels_batch,
        'image_ids': image_ids,
    }


if __name__ == '__main__':
    import numpy as np
    # Test
    train_loader, val_loader = create_dataloaders(
        'data_EfficientDet/output/train_efficientdet.csv',
        'data_EfficientDet/output/valid_efficientdet.csv',
        batch_size=4,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"Images shape: {batch['images'].shape}")
    print(f"Boxes shape: {batch['boxes'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
