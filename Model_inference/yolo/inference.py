from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r"E:\Project2\Code\AR-DIYcode\runs\detect\runs\detect\engine_parts_detection\weights\best.pt")

# Input video
video_path = r"E:\Project2\Code\AR-DIYcode\TestVideo\9.mp4"

# Output video
output_path = r"E:\Project2\Code\AR-DIYcode\output\9.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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