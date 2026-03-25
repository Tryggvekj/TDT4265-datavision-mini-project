from pathlib import Path

# ===========================
# Internal folder paths
# ===========================
DATA_YOLO_IPHONE = Path("datasets/dataset_iphone_yolo")
DATA_YOLO_V1 = Path("datasets/dataset_v1_yolo")

# ===========================
# External folder and file paths
# ===========================
DATA_FOLDER = Path("/cluster/projects/vc/courses/TDT17/ad/Poles2025")

# Iphone dataset
FOLDER_IPHONE = Path("Road_poles_iPhone")

TRAIN_FOLDER_IPHONE = Path("images/Train/train")
TRAIN_LABEL_FOLDER_IPHONE = Path("labels/Train/train")

VALIDATION_IMAGES_IPHONE = Path("images/Validation/val")
VALIDATION_LABEL_FOLDER_IPHONE = Path("labels/Validation/val")

# V1 dataset
FOLDER_V1 = Path("roadpoles_v1")

TRAIN_FOLDER_V1 = Path("train/images")
TRAIN_LABEL_FOLDER_V1 = Path("train/labels")

VALIDATION_IMAGES_V1 = Path("valid/images")
VALIDATION_LABEL_FOLDER_V1 = Path("valid/labels")
