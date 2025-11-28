# SuperPoint Implementation Directory

This directory contains scripts for SuperPoint feature detection and descriptor conversion.

## Quick Start

### Installation

```bash
# Full installation (feature detection + conversion)
pip3 install -r requirements.txt

# Minimal installation (conversion only)
pip3 install -r requirements-conversion-only.txt
```

### Run Feature Detection

```bash
python3 detect_features.py
```

Processes images from `1_video_processing/output_img/` and saves:
- Keypoint CSVs (y, x, confidence)
- 256-dim descriptors (.npy)
- Visualization images

### Convert to SIFT Format

```bash
# PCA-based conversion (recommended, requires scikit-learn)
python3 convert_to_sift128.py

# Simple conversion (no extra dependencies)
python3 convert_to_sift128_simple.py
```

Output: `sift128_*_descriptors.npy` (128-dim uint8)

## Files in This Directory

### Scripts

- **`detect_features.py`** - Main SuperPoint feature detection pipeline
- **`convert_to_sift128.py`** - PCA-based 256→128 dimension conversion
- **`convert_to_sift128_simple.py`** - Simple truncation/subsample conversion
- **`rename_sift128_prefix.py`** - Utility to rename sift128 files (already applied)

### Dependencies

- **`requirements.txt`** - Full dependencies (TensorFlow, OpenCV, etc.)
- **`requirements-conversion-only.txt`** - Minimal deps for conversion only

### Documentation

- **`README_CONVERSION.md`** - Detailed guide for descriptor conversion

### Data Directories

- **`saved_models/sp_v6/`** - SuperPoint model weights (TensorFlow SavedModel)
- **`output/`** - Output directory with subdirectories:
  - `csv/` - Keypoint coordinates
  - `descriptors/` - 256-dim SuperPoint descriptors
  - `visualizations/` - Images with keypoints overlaid
  - Files: `sift128_*_descriptors.npy` - 128-dim SIFT-format descriptors

## Key Points

- SuperPoint model requires TensorFlow 1.x
- Original descriptors: 256-dim float32 L2-normalized
- Converted descriptors: 128-dim uint8 [0-255] SIFT-compatible
- All descriptors align row-by-row with keypoint CSVs
- `sift128_` prefix identifies converted descriptor files

## For More Information

See the main repository CLAUDE.md for architecture details and workflow.
