from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(data="config.yaml", epochs=10 , imgsz=640)