import os
import subprocess

# Set your input and output directories
input_folder = '1_video_processing/input_vid'
output_folder = '1_video_processing/output_img'
fps = 6

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported video extensions (customize as needed)
video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm')

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(video_extensions):
        video_path = os.path.join(input_folder, filename)
        name, _ = os.path.splitext(filename)
        
        # Output pattern: output_folder/videoName_frame_%04d.png
        output_pattern = os.path.join(output_folder, f"{name}_frame_%04d.png")

        # ffmpeg command to extract PNGs at 6 fps
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps}',
            output_pattern
        ]

        print(f"Processing {filename}...")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Conversion complete.")
