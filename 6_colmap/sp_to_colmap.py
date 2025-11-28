import os
import sqlite3
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import subprocess

###############################################################################
# CONFIGURATION
###############################################################################

IMAGE_ROOT = Path("1_video_processing/output_img")
SP_ROOT = Path("2_sp_implementation/output")

# Output
COLMAP_PROJECT = Path("colmap_project")
IMAGES_DIR = COLMAP_PROJECT / "images"
FEATURES_DIR = COLMAP_PROJECT / "features"
DATABASE_PATH = COLMAP_PROJECT / "database.db"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


###############################################################################
# Helper functions
###############################################################################

def scale_keypoints(backproj_kp, orig_size, sp_input_size):
    """
    Convert SuperPoint keypoints (from resized input) back to original resolution.
    backproj_kp: Nx2 array of (y, x)
    orig_size: (orig_h, orig_w)
    sp_input_size: (sp_h, sp_w)
    """
    sp_h, sp_w = sp_input_size
    orig_h, orig_w = orig_size

    ys = backproj_kp[:, 0] * (orig_h / sp_h)
    xs = backproj_kp[:, 1] * (orig_w / sp_w)

    return np.stack([xs, ys], axis=1)   # COLMAP wants (x, y)


def write_colmap_keypoints(txt_path, keypoints_xy):
    """
    Write COLMAP keypoint format used by feature_importer:
        NUM_FEATURES DIM
        x y scale angle d1 ... dDIM
    """
    raise NotImplementedError("Use write_colmap_features instead.")


def write_colmap_features(out_path, keypoints_xy, descriptors):
    """
    Write COLMAP feature_importer text format:
        NUM_FEATURES DIM
        x y scale angle d1 d2 ... dDIM
    """
    dim = descriptors.shape[1]
    assert len(keypoints_xy) == len(descriptors), "Keypoints and descriptors count mismatch"

    with open(out_path, "w") as f:
        f.write(f"{len(keypoints_xy)} {dim}\n")
        for (x, y), desc in zip(keypoints_xy, descriptors):
            # Use unit scale and zero orientation since SuperPoint is orientation-less
            desc_str = " ".join(f"{v:.8f}" for v in desc)
            f.write(f"{x:.6f} {y:.6f} 1.0 0.0 {desc_str}\n")


###############################################################################
# MAIN PROCESSING
###############################################################################

def process_folder(image_folder):
    """
    Convert all SuperPoint CSV + descriptor files inside:
        SP_ROOT/<folder>/csv
        SP_ROOT/<folder>/descriptors
    to COLMAP format.
    """
    rel = image_folder.relative_to(IMAGE_ROOT)
    sp_folder = SP_ROOT / rel

    csv_dir = sp_folder / "csv"
    desc_dir = sp_folder / "descriptors"

    if not csv_dir.exists():
        print(f"[WARN] No CSV directory for {image_folder}")
        return

    print(f"\n=== Processing folder: {rel} ===")

    # Copy images to COLMAP project images/
    for img_path in sorted(image_folder.glob("*.png")):
        out_img = IMAGES_DIR / img_path.name
        if not out_img.exists():
            out_img.write_bytes(img_path.read_bytes())

    # For each image, create COLMAP feature files
    for csv_file in sorted(csv_dir.glob("*_keypoints.csv")):
        stem = csv_file.stem.replace("_keypoints", "")

        desc_file = desc_dir / f"{stem}_descriptors.npy"
        if not desc_file.exists():
            print(f"[WARN] Missing descriptor for {stem}")
            continue

        # Load image
        img_path = image_folder / f"{stem}.png"
        if not img_path.exists():
            print(f"[WARN] Missing image file for {stem}")
            continue

        img = cv2.imread(str(img_path))
        orig_h, orig_w = img.shape[:2]

        # SuperPoint input size (your code uses img_size=(640,480) → W,H)
        sp_w, sp_h = 640, 480

        # Load CSV keypoints
        df = pd.read_csv(csv_file)
        # CSV has columns y,x,confidence with an unnamed index column from saving
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        kp_y = df["y"].values
        kp_x = df["x"].values
        keypoints_yx = np.stack([kp_y, kp_x], axis=1)

        # Resize back to original image space
        keypoints_xy = scale_keypoints(keypoints_yx, (orig_h, orig_w), (sp_h, sp_w))

        # Load descriptors
        descriptors = np.load(desc_file)

        # Write COLMAP feature_importer file
        feat_out = FEATURES_DIR / f"{stem}.txt"
        write_colmap_features(feat_out, keypoints_xy, descriptors)

        print(f"[OK] {stem}: {len(keypoints_xy)} keypoints")


###############################################################################
# BUILD DATABASE + IMPORT FEATURES
###############################################################################

def create_empty_database(db_path):
    """Create empty COLMAP database by calling COLMAP's internal schema generator."""
    if db_path.exists():
        db_path.unlink()

    # colmap creates the schema automatically when you insert features
    print("[INFO] Database will be created automatically during feature_importer.")


def import_features():
    """Call COLMAP's feature_importer."""
    cmd = [
        "colmap", "feature_importer",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGES_DIR),
        "--import_path", str(FEATURES_DIR),
        "--ImageReader.single_camera", "1"
    ]
    print("\n[INFO] Running COLMAP feature_importer...")
    subprocess.run(cmd, check=True)


###############################################################################
# RUN EVERYTHING
###############################################################################

if __name__ == "__main__":
    print("\n=== Starting SuperPoint → COLMAP conversion ===")

    # Create empty DB
    create_empty_database(DATABASE_PATH)

    # Process recursively every folder under IMAGE_ROOT
    for folder in IMAGE_ROOT.rglob("*"):
        if folder.is_dir():
            # If folder contains images
            imgs = list(folder.glob("*.jpg"))
            if len(imgs) > 0:
                process_folder(folder)

    # Import into database
    import_features()

    print("\n=== Conversion complete! ===")
    print("COLMAP project structure:")
    print(COLMAP_PROJECT)
    print("\nNext, run:")
    print("  colmap exhaustive_matcher --database_path colmap_project/database.db")
    print("  colmap mapper --database_path colmap_project/database.db --image_path colmap_project/images --output_path colmap_project/sparse")
