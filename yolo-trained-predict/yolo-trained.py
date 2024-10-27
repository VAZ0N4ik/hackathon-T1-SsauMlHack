from ultralytics import YOLO

PATH = 'best-unprocessed.pt'
model = YOLO(PATH)

model.predict(source = "../ur_path.jpg", save = True, conf = 0.3)