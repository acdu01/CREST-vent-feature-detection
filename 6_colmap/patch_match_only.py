import pycolmap
from pathlib import Path

output_path = Path("6_colmap/colmap_output")
mvs_path = output_path / "mvs"
image_dir = Path("1_video_processing/output_img")

# ---- Resume PatchMatchStereo ----
opts = pycolmap.PatchMatchOptions()
opts.max_image_size = 1200
opts.window_radius = 3
opts.geom_consistency = True

# This will skip computing depth maps that already exist
pycolmap.patch_match_stereo(
    workspace_path=mvs_path,
    options=opts
)

# Then run fusion
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
