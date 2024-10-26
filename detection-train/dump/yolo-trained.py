import torch
from ultralytics import YOLO

PATH = 'best-processed.pt'
model = YOLO(PATH)

model.predict(source = "imgs\processed_4383372_doc1_A79AD893-D6AF-4A4F-B4C6-93FC07F962F9.jpg", save = True, conf = 0.3)