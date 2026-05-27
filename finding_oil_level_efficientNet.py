from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms


# =========================================================
# LOAD YOLO MODEL
# =========================================================

yolo_model = YOLO(
    r"E:\Project2\Code\AR-DIYcode\runs\detect\runs\detect\engine_parts_detection\weights\best.pt"
)

# =========================================================
# LOAD EFFICIENTNET CLASSIFIER
# =========================================================

classes = ['high', 'low', 'medium']

classifier = models.efficientnet_b0(weights=None)

classifier.classifier[1] = nn.Linear(
    classifier.classifier[1].in_features,
    len(classes)
)

classifier.load_state_dict(
    torch.load(
        r"E:\Project2\Code\AR-DIYcode\Model_Training\efficientNet\best_oil_classifier.pth",
        map_location="cpu"
    )
)

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

video_path = r"E:\Project2\Code\AR-DIYcode\TestVideo\12.mp4"

cap = cv2.VideoCapture(video_path)

CLASS_NAMES = yolo_model.names

# =========================================================
# MAIN LOOP
# =========================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    annotated_frame = frame.copy()

    # =====================================================
    # YOLO INFERENCE
    # =====================================================

    results = yolo_model(
        frame,
        conf=0.7,
        verbose=False
    )

    # =====================================================
    # PROCESS DETECTIONS
    # =====================================================

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = CLASS_NAMES[cls_id]

        # Only process oil_marker
        if class_name != "oil_marker":
            continue

        # =================================================
        # GET BOUNDING BOX
        # =================================================

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Add small padding
        padding = 10

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        # Draw detection box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # =================================================
        # CROP ROI
        # =================================================

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # =================================================
        # PREPARE IMAGE FOR EFFICIENTNET
        # =================================================

        roi_rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        input_tensor = transform(roi_rgb)

        input_tensor = input_tensor.unsqueeze(0)

        # =================================================
        # CLASSIFICATION
        # =================================================

        with torch.no_grad():

            outputs = classifier(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        oil_level = classes[predicted.item()]

        confidence_score = confidence.item()

        # =================================================
        # DISPLAY RESULT
        # =================================================

        label = (
            f"{oil_level} "
            f"{confidence_score:.2f}"
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # =================================================
        # SHOW ROI
        # =================================================

        roi_display = cv2.resize(
            roi,
            (250, 250)
        )

        cv2.putText(
            roi_display,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Oil ROI Classification",
            roi_display
        )

    # =====================================================
    # SHOW FINAL OUTPUT
    # =====================================================

    cv2.imshow(
        "Oil Level Detection",
        annotated_frame
    )

    # Press q to quit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()