from pathlib import Path
from moviepy import VideoFileClip

BASE_DIR = Path(__file__).resolve().parent

# Folder containing videos
input_folder = BASE_DIR / "Camera"

# Output folder for muted videos
output_folder = BASE_DIR / "muted_videos"
output_folder.mkdir(parents=True, exist_ok=True)

# Supported video extensions
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# Loop through all files in folder
for video_file in input_folder.iterdir():

    if video_file.suffix.lower() in video_extensions:

        output_path = output_folder / f"muted_{video_file.name}"

        print(f"Processing: {video_file.name}")

        # Load video
        video = VideoFileClip(str(video_file))

        # Remove audio
        muted_video = video.without_audio()

        # Save muted video
        muted_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio=False
        )

        print(f"Saved: {output_path}")

print("All videos muted successfully!")