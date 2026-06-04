from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

#Finding oil Level using open cv
BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(
    BASE_DIR / "runs" / "detect" / "runs" / "detect" / "engine_parts_detection" / "weights" / "best.pt"
)

# ==========================================
# INPUT VIDEO
# ==========================================

video_path = BASE_DIR / "TestVideo" / "12.mp4"

cap = cv2.VideoCapture(str(video_path))

CLASS_NAMES = model.names

# ==========================================
# MAIN LOOP
# ==========================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    annotated_frame = frame.copy()

    # ==========================================
    # YOLO INFERENCE
    # ==========================================

    results = model(frame, conf=0.7)

    # ==========================================
    # PROCESS DETECTIONS
    # ==========================================

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]

        # Only process oil_marker
        if class_name != "oil_marker":
            continue

        # ==========================================
        # GET BOUNDING BOX
        # ==========================================

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # ==========================================
        # CROP ROI
        # ==========================================

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        roi_resized = cv2.resize(roi, (250, 250))

        # ==========================================
        # PREPROCESSING
        # ==========================================

        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        blurred = cv2.medianBlur(blurred, 5)
        # ==========================================
        # DETECT INNER OIL WINDOW CIRCLE
        # ==========================================

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=40,
            maxRadius=100
        )

        oil_level = "Low"

        circle_mask = np.zeros_like(gray)

        if circles is not None:

            circles = np.round(circles[0, :]).astype("int")

            # Use largest circle
            largest_circle = max(circles, key=lambda c: c[2])

            cx, cy, radius = largest_circle

            # ==========================================
            # DRAW INNER CIRCLE
            # ==========================================

            cv2.circle(
                roi_resized,
                (cx, cy),
                radius,
                (0, 255, 0),
                2
            )

            # ==========================================
            # CREATE CIRCULAR MASK
            # ==========================================

            INNER_OFFSET = 30

            cv2.circle(
                circle_mask,
                (cx, cy),
                radius - INNER_OFFSET,
                255,
                -1
            )

            # Keep only inside circle
            masked_gray = cv2.bitwise_and(
                gray,
                gray,
                mask=circle_mask
            )

            # ==========================================
            # THRESHOLD DARK OIL REGION
            # ==========================================

            _, thresh = cv2.threshold(
                masked_gray,
                70,
                255,
                cv2.THRESH_BINARY_INV
            )
            # thresh = cv2.adaptiveThreshold(
            #     masked_gray,
            #     255,
            #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #     cv2.THRESH_BINARY_INV,
            #     21,
            #     5
            # )q

            # Remove small noise
            kernel = np.ones((3, 3), np.uint8)

            thresh = cv2.morphologyEx(
                thresh,
                cv2.MORPH_OPEN,
                kernel
            )

            # Apply circle mask again
            thresh = cv2.bitwise_and(
                thresh,
                thresh,
                mask=circle_mask
            )

            # ==========================================
            # FIND OIL CONTOURS
            # ==========================================

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:

                # Largest contour
                largest_contour = max(
                    contours,
                    key=cv2.contourArea
                )

                # Ignore tiny noise contours
                area = cv2.contourArea(largest_contour)

                if area > 100:

                    # ==========================================
                    # OIL REGION HEIGHT
                    # ==========================================

                    ox, oy, ow, oh = cv2.boundingRect(
                        largest_contour
                    )

                    # Draw contour
                    cv2.drawContours(
                        roi_resized,
                        [largest_contour],
                        -1,
                        (0, 255, 255),
                        2
                    )

                    # ==========================================
                    # FILL RATIO
                    # ==========================================

                    circle_diameter = radius * 2
                    print("circle_diameter",circle_diameter)
                    fill_ratio = oh / circle_diameter
                    print("oh,fill_ratio",oh,fill_ratio)

                    # ==========================================
                    # OIL LEVEL CLASSIFICATION
                    # ==========================================

                    if fill_ratio < 0.30:
                        oil_level = "LOW"

                    elif fill_ratio < 0.5:
                        oil_level = "NORMAL"

                    else:
                        oil_level = "HIGH"

                    # Draw oil height box
                    cv2.rectangle(
                        roi_resized,
                        (ox, oy),
                        (ox + ow, oy + oh),
                        (255, 0, 0),
                        2
                    )

            # ==========================================
            # SHOW THRESHOLD WINDOW
            # ==========================================

            cv2.imshow("Threshold", thresh)

        # ==========================================
        # DISPLAY OIL LEVEL
        # ==========================================

        cv2.putText(
            annotated_frame,
            f"Oil Level: {oil_level}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ==========================================
        # SHOW ROI
        # ==========================================

        cv2.imshow("Oil ROI", roi_resized)

    # ==========================================
    # SHOW FINAL FRAME
    # ==========================================

    cv2.imshow(
        "Oil Level Detection",
        annotated_frame
    )

    # Press q to quit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()