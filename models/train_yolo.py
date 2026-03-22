import importlib
import sys
import ultralytics
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

DATA_DIR = config.DATA_YOLO
DATA_YAML = config.DATA_YOLO / "data.yaml"


def train_yolo(
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    run_name: str,
    device: str,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    degrees: float,
    translate: float,
    scale: float,
    shear: float,
    perspective: float,
    flipud: float,
    fliplr: float,
    mosaic: float,
    mixup: float,
    copy_paste: float,
) -> None:
    YOLO = ultralytics.YOLO

    model = YOLO(model_name)
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(ROOT_DIR / "runs" / "yolo"),
        name=run_name,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
    )


def main() -> None:
    train_yolo(
        model_name="models/yolo11n.pt",
        epochs=50,
        imgsz=768,
        batch=16,
        run_name="run1",
        device="0",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )


if __name__ == "__main__":
    main()