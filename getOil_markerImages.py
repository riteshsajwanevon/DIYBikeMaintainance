from ultralytics import YOLO
import cv2
import os
from pathlib import Path
# detects oil_marker using yolo and save it in a folder

# =========================================================
# LOAD YOLO MODEL
# =========================================================
model = YOLO(
    r"E:\Project2\Code\AR-DIYcode\runs\detect\runs\detect\engine_parts_detection\weights\best.pt"
)

# =========================================================
# INPUT / OUTPUT PATHS
# =========================================================
input_folder = r"E:\Project2\Code\AR-DIYcode\videoforclassification"

# Folder where cropped oil marker images will be saved
output_folder = r"E:\Project2\Code\AR-DIYcode\dataset\classification"

os.makedirs(output_folder, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================

# YOLO class name
TARGET_CLASS = "oil_marker"

# Confidence threshold
CONFIDENCE = 0.2

# Extract one frame every N seconds
EXTRACT_EVERY_SECONDS = 0.5

# Resize ROI image size
OUTPUT_SIZE = (224, 224)

# Optional padding around bounding box
PADDING = 10

# Supported video formats
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

# =========================================================
# GET VIDEO FILES
# =========================================================
video_files = [
    f for f in os.listdir(input_folder)
    if Path(f).suffix.lower() in video_extensions
]

print(f"\nFound {len(video_files)} videos")

# =========================================================
# PROCESS VIDEOS
# =========================================================
total_saved = 0

for video_name in video_files:

    video_path = os.path.join(input_folder, video_name)

    print(f"\nProcessing video: {video_name}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("Could not read FPS. Skipping video.")
        continue

    # Calculate frame interval
    frame_interval = int(fps * EXTRACT_EVERY_SECONDS)

    print(f"FPS: {fps}")
    print(f"Extracting every {frame_interval} frames")

    frame_count = 0
    saved_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # =========================================
        # FRAME SKIP LOGIC
        # =========================================
        if frame_count % frame_interval != 0:
            continue

        # =========================================
        # YOLO INFERENCE
        # =========================================
        results = model(
            frame,
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        # =========================================
        # CHECK DETECTIONS
        # =========================================
        if result.boxes is None:
            continue

        # =========================================
        # LOOP THROUGH DETECTIONS
        # =========================================
        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            # Detect ONLY oil_marker
            if class_name != TARGET_CLASS:
                continue

            # =========================================
            # GET BOUNDING BOX
            # =========================================
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add padding
            x1 -= PADDING
            y1 -= PADDING
            x2 += PADDING
            y2 += PADDING

            # Clip coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # =========================================
            # CROP ROI
            # =========================================
            roi = frame[y1:y2, x1:x2]

            # Skip invalid crops
            if roi.size == 0:
                continue

            # =========================================
            # RESIZE ROI
            # =========================================
            roi_resized = cv2.resize(roi, OUTPUT_SIZE)

            # =========================================
            # SAVE IMAGE
            # =========================================
            save_name = (
                f"{Path(video_name).stem}"
                f"_frame{frame_count}"
                f"_img{saved_count}.jpg"
            )

            save_path = os.path.join(output_folder, save_name)

            cv2.imwrite(save_path, roi_resized)

            saved_count += 1
            total_saved += 1

            # =========================================
            # VISUALIZATION
            # =========================================
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                TARGET_CLASS,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # =========================================
        # SHOW LIVE PREVIEW
        # =========================================
        cv2.imshow("Oil Marker ROI Extraction", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video
    cap.release()

    print(f"Saved {saved_count} images from {video_name}")

# =========================================================
# CLEANUP
# =========================================================
cv2.destroyAllWindows()

print("\n=========================================")
print("PROCESS COMPLETED")
print(f"Total ROI images saved: {total_saved}")
print(f"Saved to folder:\n{output_folder}")
print("=========================================")