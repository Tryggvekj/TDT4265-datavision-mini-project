from ultralytics import YOLO
import config

WEIGHTS_PATH = "best_optuna_weights.pt"
SOURCE_PATH = str(config.DATA_FOLDER / config.FOLDER_V1 / "test/images")
SAVE_FOLDER = "v1"
SUB_SAVE_FOLDER = "predict"
# WEIGHTS_PATH = "models/yolo/weights/iphone/v5.pt"
# SOURCE_PATH = str(config.DATA_FOLDER / config.FOLDER_IPHONE / "images/Test/test")
# SAVE_FOLDER = "iphone"
# SUB_SAVE_FOLDER = "predict"
DEVICE = "cpu"


def main() -> None:
	
    # YOLO prediction	
	model = YOLO(WEIGHTS_PATH)
	model.predict(
		source=SOURCE_PATH,
		project=SAVE_FOLDER,
		name=SUB_SAVE_FOLDER,
		device=DEVICE,
		save=True,
		save_txt=True,
		save_conf=True,
		iou=0.2, # Set NMS to lower such that we don't get overlapping boxes if they are overlapping with more than 20%
	)

	print(f"Saved predictions to: {SAVE_FOLDER}/{SUB_SAVE_FOLDER}")


if __name__ == "__main__":
	main()