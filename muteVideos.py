import os
from moviepy import VideoFileClip

# Folder containing videos
input_folder = "Camera"

# Output folder for muted videos
output_folder = "muted_videos"
os.makedirs(output_folder, exist_ok=True)

# Supported video extensions
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# Loop through all files in folder
for filename in os.listdir(input_folder):

    if filename.lower().endswith(video_extensions):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"muted_{filename}")

        print(f"Processing: {filename}")

        # Load video
        video = VideoFileClip(input_path)

        # Remove audio
        muted_video = video.without_audio()

        # Save muted video
        muted_video.write_videofile(
            output_path,
            codec="libx264",
            audio=False
        )

        print(f"Saved: {output_path}")

print("All videos muted successfully!")