import argparse
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap
import tqdm



image_dir = Path("1_video_processing/output_img/AT5007_AL5144_4K-arm_2023_01_17_20_31_26_03")

output_path = Path("6_colmap/colmap_output")
if not output_path.exists():
    output_path.mkdir(parents=True)

mvs_path = output_path / "mvs"
if not mvs_path.exists():
    mvs_path.mkdir(parents=True)

database_path = output_path / "database.db"
pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# dense reconstruction
pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path,{
                            "max_image_size": 1200,
                            "window_radius": 3,
                            "geom_consistency": True    
                            })  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)