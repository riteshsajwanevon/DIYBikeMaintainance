import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# =========================
# INPUTS
# =========================

# Folder containing all videos
video_folder = BASE_DIR / "EngineTopUPVideo"

# Output dataset folder
output_folder = BASE_DIR / "dataset"

# Time interval (seconds)
save_interval_sec = 0.5

# Supported video formats
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']

# =========================
# CREATE OUTPUT FOLDER
# =========================

output_folder.mkdir(parents=True, exist_ok=True)

# =========================
# GET ALL VIDEO FILES
# =========================

video_files = []

for ext in video_extensions:
    video_files.extend(Path(video_folder).glob(ext))

print(f"Found {len(video_files)} video(s)")

global_saved_count = 0

# =========================
# PROCESS EACH VIDEO
# =========================

for video_path in video_files:

    print(f"\nProcessing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open {video_path.name}")
        continue

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Invalid FPS in {video_path.name}")
        continue

    # Frames interval
    frame_interval = int(fps * save_interval_sec)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame every interval
        if frame_count % frame_interval == 0:

            filename = f"frame_{global_saved_count:06d}.jpg"

            save_path = output_folder / filename

            cv2.imwrite(str(save_path), frame)

            saved_count += 1
            global_saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Saved {saved_count} frames from {video_path.name}")

print("\nExtraction completed")
print(f"Total frames saved: {global_saved_count}")