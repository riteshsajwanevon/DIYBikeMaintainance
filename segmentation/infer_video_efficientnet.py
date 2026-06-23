import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models, transforms
from pathlib import Path
import numpy as np

# =========================================================
# PATH SETUP
# =========================================================
# This script is located in DIYBikeMaintainance/segmentation
BASE_DIR = Path(__file__).resolve().parent.parent 
SEG_DIR = Path(__file__).resolve().parent

# =========================================================
# LOAD MODELS
# =========================================================

# 1. Load YOLO Segmentation Model
print("Loading YOLO Segmentation Model...")
yolo_model = YOLO(SEG_DIR / 'best.pt')

# 2. Load EfficientNet Classifier
print("Loading EfficientNet Classifier...")
classes = ['high', 'low', 'medium']
classifier = models.efficientnet_b0(weights=None)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, len(classes))
classifier.load_state_dict(torch.load(BASE_DIR / "Model_Training" / "efficientNet" / "best_oil_classifier.pth", map_location="cpu"))
classifier.eval()

# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================
# INPUT VIDEO
# =========================================================
video_path = BASE_DIR / "Test1.mp4" 
cap = cv2.VideoCapture(str(video_path))

print(f"Starting inference on: {video_path}")
print("Press 'q' to stop.")

CLASS_NAMES = yolo_model.names

# =========================================================
# MAIN LOOP
# =========================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO Segmentation Inference
    results = yolo_model(frame, conf=0.50, verbose=False)
    result = results[0]

    # Draw masks on frame
    annotated_frame = result.plot(boxes=False)

    # Process each detection
    if result.masks is not None:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = CLASS_NAMES[cls_id]

            # We only care about the oil window for classification
            if class_name != "engine-oil-window":
                continue

            # =================================================
            # EXTRACT BOUNDING BOX
            # =================================================
            # YOLO provides the bounding box even when doing segmentation
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

            # Add padding so EfficientNet sees the whole window
            padding = 15
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            # Draw detection box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # =================================================
            # CROP ROI FOR CLASSIFIER
            # =================================================
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # =================================================
            # RUN EFFICIENTNET
            # =================================================
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            input_tensor = transform(roi_rgb).unsqueeze(0)

            with torch.no_grad():
                outputs = classifier(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            oil_level = classes[predicted.item()]
            confidence_score = confidence.item()

            # =================================================
            # DISPLAY TEXT ON FRAME
            # =================================================
            label = f"Oil Level: {oil_level.upper()} ({confidence_score:.2f})"
            
            # Shadow effect for text
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            
            # Determine color based on oil level
            color = (0, 255, 0) # Green for medium
            if oil_level == 'low':
                color = (0, 0, 255) # Red for low
            elif oil_level == 'high':
                color = (0, 255, 255) # Yellow for high
                
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # =================================================
            # SHOW ROI (Separate Window)
            # =================================================
            roi_display = cv2.resize(roi, (250, 250))
            cv2.putText(roi_display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Oil Window ROI", roi_display)

    # Show final annotated video
    cv2.imshow("YOLOv11 Seg + EfficientNet", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Inference completed.")
