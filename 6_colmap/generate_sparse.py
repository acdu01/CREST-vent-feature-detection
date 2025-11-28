import argparse
from pathlib import Path
import pycolmap
import tqdm

# Input image directory
image_dir = Path("1_video_processing/output_img/AT5007_AL5144_4K-arm_2023_01_17_20_31_26_03")

# Output directories
output_path = Path("6_colmap/colmap_output")
output_path.mkdir(parents=True, exist_ok=True)

database_path = output_path / "database.db"

print("==> Extracting features...")
pycolmap.extract_features(database_path, image_dir)

print("==> Matching images (exhaustive)...")
pycolmap.match_exhaustive(database_path)

print("==> Running sparse reconstruction (incremental mapper)...")
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

print("==> Saving sparse model...")
maps[0].write(output_path)

print("==> Sparse reconstruction complete!")
print(f"Sparse model saved to: {output_path}")
