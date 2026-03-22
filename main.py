from ultralytics import YOLO
import config

WEIGHTS_PATH = "models/weights/yolo/v2.pt"
SOURCE_PATH = str(config.DATA_FOLDER / config.FOLDER_IPHONE / "images/Test/test")
SAVE_FOLDER = "results_iphone"
SUB_SAVE_FOLDER = "predict"
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
	)

	print(f"Saved predictions to: {SAVE_FOLDER}/{SUB_SAVE_FOLDER}")


if __name__ == "__main__":
	main()