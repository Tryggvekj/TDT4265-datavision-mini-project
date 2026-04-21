import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class EfficientDetDataset(Dataset):
    """
    Dataset for EfficientDet training from CSV with columns:
    [image, class, y1, x1, y2, x2]

    Intern flyt:
    - CSV lagres som yxyx for effdet/rwightman
    - Albumentations bruker pascal_voc = xyxy
    - Etter transforms konverteres tilbake til yxyx
    """

    def __init__(self, csv_path: str, img_size: int = 512, augment: bool = False):
        if img_size % 128 != 0:
            raise ValueError(f"img_size must be divisible by 128, got {img_size}")

        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.augment = augment
        
        self.image_groups = self.df.groupby("image", sort=False)
        self.image_paths = list(self.image_groups.groups.keys())

        bbox_params = A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.3
        )

        if self.augment:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=0,
                    value=(0, 0, 0)
                ),

                A.HorizontalFlip(p=0.5),

                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-7, 7),
                    shear=(-3, 3),
                    p=0.6
                ),

                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),

                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.5),

                A.OneOf([
                    A.RandomShadow(p=1.0),
                    A.RandomFog(p=1.0),
                    A.RandomRain(p=1.0),
                ], p=0.15),

                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(20, 80),
                    hole_width_range=(20, 80),
                    fill=0,
                    p=0.2
                ),

                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.3
            ))
        else:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=0,
                    value=(0, 0, 0)
                ),
                ToTensorV2(),
            ], bbox_params=bbox_params)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img_data = self.image_groups.get_group(img_path)

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        orig_h, orig_w = img_np.shape[:2]

        valid_rows = img_data.dropna(subset=["class", "y1", "x1", "y2", "x2"])

        boxes_xyxy = []
        labels = []

        for _, row in valid_rows.iterrows():
            y1 = float(row["y1"])
            x1 = float(row["x1"])
            y2 = float(row["y2"])
            x2 = float(row["x2"])
            cls = int(row["class"])

            boxes_xyxy.append([x1, y1, x2, y2])
            labels.append(cls)

        transformed = self.transforms(
            image=img_np,
            bboxes=boxes_xyxy,
            class_labels=labels
        )

        img_tensor = transformed["image"]
        boxes_xyxy = transformed["bboxes"]
        labels = transformed["class_labels"]

        boxes_yxyx = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            boxes_yxyx.append([y1, x1, y2, x2])

        boxes = np.array(boxes_yxyx, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64).reshape(-1)

        boxes_tensor = torch.from_numpy(boxes)
        labels_tensor = torch.from_numpy(labels)

        scale = min(self.img_size / orig_h, self.img_size / orig_w)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "img_size": torch.tensor([self.img_size, self.img_size], dtype=torch.float32),
            "img_scale": torch.tensor([scale], dtype=torch.float32),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
        }

        return img_tensor, target, img_path
    
class RepeatDataset(Dataset):
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

def collate_fn(batch):
    images, targets, image_paths = zip(*batch)

    images = torch.stack(images)

    max_boxes = max(target["boxes"].shape[0] for target in targets)

    batch_boxes = []
    batch_labels = []
    batch_img_size = []
    batch_img_scale = []
    batch_image_id = []
    batch_orig_size = []

    for target in targets:
        boxes = target["boxes"]
        labels = target["labels"]

        n = boxes.shape[0]
        if n < max_boxes:
            pad = max_boxes - n
            boxes = torch.cat([boxes, torch.zeros((pad, 4), dtype=torch.float32)], dim=0)
            labels = torch.cat([labels, torch.full((pad,), -1, dtype=torch.int64)], dim=0)

        batch_boxes.append(boxes)
        batch_labels.append(labels)
        batch_img_size.append(target["img_size"])
        batch_img_scale.append(target["img_scale"])
        batch_image_id.append(target["image_id"])
        batch_orig_size.append(target["orig_size"])

    return {
        "images": images,
        "bbox": torch.stack(batch_boxes),   # [B, N, 4] in yxyx
        "cls": torch.stack(batch_labels),   # [B, N]
        "img_size": torch.stack(batch_img_size),
        "img_scale": torch.stack(batch_img_scale),
        "image_id": torch.stack(batch_image_id).squeeze(1),
        "image_path": list(image_paths),
        "orig_size": torch.stack(batch_orig_size),
    }


def create_dataloaders(train_csv: str, val_csv: str, batch_size: int = 8,
                       num_workers: int = 4, img_size: int = 512):
    train_dataset = EfficientDetDataset(train_csv, img_size=img_size, augment=True)
    train_dataset = RepeatDataset(train_dataset, repeats=3)

    val_dataset = EfficientDetDataset(val_csv, img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader

if __name__ == "__main__":
    train_csv = "/cluster/home/svanhigb/TDT4265-datavision-mini-project/data_EfficientDet/output/train_efficientdet.csv"
    val_csv = "/cluster/home/svanhigb/TDT4265-datavision-mini-project/data_EfficientDet/output/valid_efficientdet.csv"

    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=2,
        num_workers=0,
        img_size=1024,
    )