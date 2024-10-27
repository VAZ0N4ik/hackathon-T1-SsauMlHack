from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="dataset.yaml", epochs=1000, imgsz=640)