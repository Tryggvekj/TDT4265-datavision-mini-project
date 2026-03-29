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
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.augment = augment

        if "class" not in self.df.columns:
            raise ValueError("CSV must contain a 'class' column")

        self.df["class"] = pd.to_numeric(self.df["class"], errors="coerce")

        required_cols = ["image", "class", "y1", "x1", "y2", "x2"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        self.image_groups = self.df.groupby("image", sort=False)
        self.image_paths = list(self.image_groups.groups.keys())

        if self.augment:
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                #A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=10, p=0.3),
                A.GaussNoise(p=0.2),
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.3
            ))
        else:
            self.transforms = A.Compose([
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.3
            ))

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

        if self.transforms:
            transformed = self.transforms(
                image=img_np,
                bboxes=boxes_xyxy,
                class_labels=labels
            )
            img_tensor = transformed["image"]
            boxes_xyxy = transformed["bboxes"]
            labels = transformed["class_labels"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Konverter tilbake fra xyxy -> yxyx for effdet
        boxes_yxyx = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            boxes_yxyx.append([y1, x1, y2, x2])

        boxes = np.array(boxes_yxyx, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64).reshape(-1)

        boxes_tensor = torch.from_numpy(boxes)
        labels_tensor = torch.from_numpy(labels)

        target = {
            "boxes": boxes_tensor,   # y1, x1, y2, x2
            "labels": labels_tensor, # 1-indexed classes
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "img_size": torch.tensor([self.img_size, self.img_size], dtype=torch.float32),
            "img_scale": torch.tensor([1.0], dtype=torch.float32),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
        }

        return img_tensor, target, img_path


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
