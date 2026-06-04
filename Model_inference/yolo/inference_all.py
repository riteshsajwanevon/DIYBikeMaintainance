from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load trained model
model = YOLO(
    BASE_DIR / "runs" / "detect" / "runs" / "detect" / "engine_parts_detection" / "weights" / "best.pt"
)

# Input videos folder
input_folder = BASE_DIR / "TestVideo"

# Output folder
output_folder = BASE_DIR / "output"

# Create output folder if not exists
output_folder.mkdir(parents=True, exist_ok=True)

# Supported video formats
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

# Get all video files
video_files = [
    f for f in input_folder.iterdir()
    if f.suffix.lower() in video_extensions
]

print(f"Found {len(video_files)} videos")

# Process each video
for video_name in video_files:

    video_path = video_name

    # Output path
    output_path = output_folder / f"output_{video_name.name}"

    print(f"\nProcessing: {video_name.name}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

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
            print(f"{video_name.name} -> Processed {frame_count} frames")

        # Press q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()

    print(f"Saved: {output_path}")

cv2.destroyAllWindows()

print("\nAll videos processed successfully!")