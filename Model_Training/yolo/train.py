from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data=str(BASE_DIR / "dataset" / "detection" / "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=4,
    device='cpu',
    workers=2,
    cache=False,
    project="runs/detect",
    name="bike_parts_detection"
)

# Save/export model
model.export(format="onnx")