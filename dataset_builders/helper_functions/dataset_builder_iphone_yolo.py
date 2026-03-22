import sys
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

SOURCE_DATA_FOLDER_IPHONE: Path = config.DATA_FOLDER / config.FOLDER_IPHONE

SOURCE_DATA_TRAIN: Path = SOURCE_DATA_FOLDER_IPHONE / config.TRAIN_FOLDER_IPHONE
SOURCE_DATA_VAL: Path = SOURCE_DATA_FOLDER_IPHONE / config.VALIDATION_IMAGES_IPHONE
SOURCE_LABEL_TRAIN: Path = SOURCE_DATA_FOLDER_IPHONE / config.TRAIN_LABEL_FOLDER_IPHONE
SOURCE_LABEL_VAL: Path = SOURCE_DATA_FOLDER_IPHONE / config.VALIDATION_LABEL_FOLDER_IPHONE

# needed data structure for yolo model
DESTINATION_DATA_TRAIN_YOLO: Path = config.DATA_YOLO / "images/train"
DESTINATION_DATA_VAL_YOLO: Path = config.DATA_YOLO / "images/val"
DESTINATION_LABEL_TRAIN_YOLO: Path = config.DATA_YOLO / "labels/train"
DESTINATION_LABEL_VAL_YOLO: Path = config.DATA_YOLO / "labels/val"


def _render_progress(completed: int, total: int, width: int = 30) -> None:
    if total == 0:
        print("No files found to copy.")
        return

    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rCopying files: [{bar}] {completed}/{total}", end="", flush=True)

    if completed == total:
        print()


def _copy_files_from_source() -> None:
    transfer_jobs = []

    for file_path in SOURCE_DATA_TRAIN.iterdir():
        if file_path.is_file():
            transfer_jobs.append((file_path, DESTINATION_DATA_TRAIN_YOLO))
    for file_path in SOURCE_DATA_VAL.iterdir():
        if file_path.is_file():
            transfer_jobs.append((file_path, DESTINATION_DATA_VAL_YOLO))
    for file_path in SOURCE_LABEL_TRAIN.iterdir():
        if file_path.is_file():
            transfer_jobs.append((file_path, DESTINATION_LABEL_TRAIN_YOLO))
    for file_path in SOURCE_LABEL_VAL.iterdir():
        if file_path.is_file():
            transfer_jobs.append((file_path, DESTINATION_LABEL_VAL_YOLO))

    total_files = len(transfer_jobs)
    _render_progress(0, total_files)

    for copied_count, (source_path, destination_path) in enumerate(transfer_jobs, start=1):
        shutil.copy2(source_path, destination_path)
        _render_progress(copied_count, total_files)


def load_data_iphone() -> None:
    destination_dirs = (
        DESTINATION_DATA_TRAIN_YOLO,
        DESTINATION_DATA_VAL_YOLO,
        DESTINATION_LABEL_TRAIN_YOLO,
        DESTINATION_LABEL_VAL_YOLO,
    )

    for destination_dir in destination_dirs:
        destination_dir.mkdir(parents=True, exist_ok=True)

    _copy_files_from_source()

if __name__ == "__main__":
    load_data_iphone()