"""Clips videos into smaller segments using ffmpeg."""

import subprocess
import os

def split_video(input_path, output_dir, segment_length=60):
    os.makedirs(output_dir, exist_ok=True)
    
    output_template = os.path.join(output_dir, "clip_%03d.mp4")
    
    command = [
        "ffmpeg",
        "-i", input_path,          # input file
        "-c", "copy",              # copy codec (no re-encoding)
        "-map", "0",               # map all streams
        "-segment_time", str(segment_length),
        "-f", "segment",
        "-reset_timestamps", "1",  # reset timestamps for each segment
        output_template
    ]
    
    subprocess.run(command, check=True)

# split_video("input.mp4", "output_clips")
