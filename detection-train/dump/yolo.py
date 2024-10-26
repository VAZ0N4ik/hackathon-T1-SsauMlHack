from ultralytics import YOLO
import torch
# Load a pretrained YOLO11 segment model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="dataset.yaml", epochs=1, imgsz=640)

#torch.save(model.state_dict(), 'yolov8n_mine.pt')