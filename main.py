from PIL import Image

# Single image
img = Image.open("data/data_iphone_yolo/images/train/919135364.jpg")
width, height = img.size
print(f"Image size: {width}x{height}")