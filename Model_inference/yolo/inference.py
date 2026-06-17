from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load trained model
model = YOLO(BASE_DIR / "runs" / "detect" / "runs" / "detect" / "engine_parts_detection" / "weights" / "best.pt")

# Input video
video_path = BASE_DIR / "Test3.mp4"

# Output video
output_path = BASE_DIR / "output" / "Test2.mp4"

# Open video
cap = cv2.VideoCapture(str(video_path))

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO inference
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Write frame
    out.write(annotated_frame)

    # Show live inference
    cv2.imshow("YOLO Inference", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved to: {output_path}")