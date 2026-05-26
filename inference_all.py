from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Load trained model
model = YOLO(
    r"E:\Project2\Code\AR-DIYcode\runs\detect\runs\train\bike_parts_detection-7\weights\best.pt"
)

# Input videos folder
input_folder = r"E:\Project2\Code\AR-DIYcode\TestVideo"

# Output folder
output_folder = r"E:\Project2\Code\AR-DIYcode\output"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Supported video formats
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

# Get all video files
video_files = [
    f for f in os.listdir(input_folder)
    if Path(f).suffix.lower() in video_extensions
]

print(f"Found {len(video_files)} videos")

# Process each video
for video_name in video_files:

    video_path = os.path.join(input_folder, video_name)

    # Output path
    output_path = os.path.join(output_folder, f"output_{video_name}")

    print(f"\nProcessing: {video_name}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO inference
        results = model(frame, imgsz=640, conf=0.5)

        # Draw detections
        annotated_frame = results[0].plot()

        # Write frame
        out.write(annotated_frame)

        # Show live inference
        cv2.imshow("YOLO Video Inference", annotated_frame)

        frame_count += 1

        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"{video_name} -> Processed {frame_count} frames")

        # Press q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()

    print(f"Saved: {output_path}")

cv2.destroyAllWindows()

print("\nAll videos processed successfully!")