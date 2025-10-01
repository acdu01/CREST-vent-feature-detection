"""Convert videos to images at a specified frame rate using ffmpeg."""

import os
import subprocess
import shutil

# Set your input and output directories
input_folder = "1_video_processing/input_vid"
output_folder = "1_video_processing/output_img"
fps = 10

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported video extensions (customize as needed)
video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm")

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(video_extensions):
        video_path = os.path.join(input_folder, filename)
        name, _ = os.path.splitext(filename)

        # Output pattern: output_folder/videoName_frame_%04d.png
        output_pattern = os.path.join(output_folder, f"{name}_frame_%04d.png")

        # FFmpeg crop filter: crop=width:height:x:y
        # width=-1 (keep original width)
        # height=2/3*ih (keep top 2/3 of the image)
        # x=0, y=0 (start from top-left)
        crop_filter = "crop=iw:2*ih/3:0:0"

        # Clear folder
        shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)

        # ffmpeg command
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps},{crop_filter}", output_pattern]

        print(f"Processing {filename}...")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Conversion complete.")
