import argparse
import sys
import ultralytics
from pathlib import Path
from typing import Any
import optuna

ROOT_DIR = Path(__file__).resolve().parents[2]
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
) -> Any:
    YOLO = ultralytics.YOLO

    model = YOLO(model_name)
    return model.train(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with optional Optuna tuning")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=12, help="Number of Optuna trials")
    parser.add_argument("--optuna-epochs", type=int, default=25, help="Epochs per Optuna trial")
    parser.add_argument("--run-name", type=str, default="run", help="Run name for non-Optuna training")
    return parser.parse_args()


def _extract_map50_95(results: Any) -> float:
    if results is None:
        return 0.0
    results_dict = getattr(results, "results_dict", None)
    if not isinstance(results_dict, dict):
        return 0.0

    keys = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "metrics/mAP50-95(M)",
    ]
    for key in keys:
        if key in results_dict:
            return float(results_dict[key])
    return 0.0


def run_optuna(
    model_name: str,
    imgsz: int,
    batch: int,
    device: str,
    n_trials: int,
    trial_epochs: int,
) -> None:
    def objective(trial: "optuna.trial.Trial") -> float:
        run_name = f"optuna_trial_{trial.number}"
        results = train_yolo(
            model_name=model_name,
            epochs=trial_epochs,
            imgsz=imgsz,
            batch=batch,
            run_name=run_name,
            device=device,
            hsv_h=trial.suggest_float("hsv_h", 0.0, 0.03),
            hsv_s=trial.suggest_float("hsv_s", 0.4, 0.9),
            hsv_v=trial.suggest_float("hsv_v", 0.2, 0.7),
            degrees=trial.suggest_float("degrees", 0.0, 20.0),
            translate=trial.suggest_float("translate", 0.0, 0.2),
            scale=trial.suggest_float("scale", 0.2, 0.8),
            shear=trial.suggest_float("shear", 0.0, 5.0),
            perspective=trial.suggest_float("perspective", 0.0, 0.0008),
            flipud=trial.suggest_float("flipud", 0.0, 0.1),
            fliplr=trial.suggest_float("fliplr", 0.2, 0.7),
            mosaic=trial.suggest_float("mosaic", 0.5, 1.0),
            mixup=trial.suggest_float("mixup", 0.0, 0.3),
            copy_paste=trial.suggest_float("copy_paste", 0.0, 0.3),
        )
        return _extract_map50_95(results)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Optuna complete")
    print(f"Best mAP50-95: {study.best_value:.5f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()

    model_name = "yolo11s.pt"
    epochs = 100
    imgsz = 1080
    batch = 12
    device = "0"

    if args.optuna:
        run_optuna(
            model_name=model_name,
            imgsz=imgsz,
            batch=batch,
            device=device,
            n_trials=args.n_trials,
            trial_epochs=args.optuna_epochs,
        )
        return
    else:
        train_yolo(
            model_name=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            run_name=args.run_name,
            device=device,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,
            translate=0.1,
            scale=0.7,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
        )

if __name__ == "__main__":
    main()