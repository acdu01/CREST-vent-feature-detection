# SuperPoint to SIFT Format Conversion

This directory contains scripts to convert 256-dimensional SuperPoint descriptors to 128-dimensional SIFT-compatible format.

## Overview

SuperPoint generates 256-dim float32 L2-normalized descriptors, while SIFT uses 128-dim uint8 descriptors in the range [0-255]. These conversion scripts bridge the gap for compatibility with SIFT-based matching pipelines.

## Available Scripts

### 1. `convert_to_sift128.py` (PCA-based)

**Requires:** `scikit-learn`

**Method:** Uses PCA for optimal dimensionality reduction

**Advantages:**
- Preserves maximum variance in the data
- Better representation of the original descriptor space
- Learns optimal projection from sample data

**Usage:**
```bash
python3 2_sp_implementation/convert_to_sift128.py
```

**How it works:**
1. Collects sample descriptors from first 10 files
2. Fits PCA model to reduce from 256 to 128 dimensions
3. Saves PCA model to `pca_model_256to128.pkl` for reuse
4. Transforms all descriptors using the fitted model
5. Normalizes per-descriptor and scales to uint8 [0-255]

### 2. `convert_to_sift128_simple.py` (No dependencies)

**Requires:** Only numpy (already required for SuperPoint)

**Methods:**
- `truncate`: Takes first 128 dimensions
- `subsample`: Takes every other dimension (indices 0, 2, 4, ..., 254)

**Advantages:**
- No additional dependencies
- Fast and simple
- Deterministic (no fitting required)

**Usage:**
```bash
python3 2_sp_implementation/convert_to_sift128_simple.py
```

**Configuration:**
Edit the script to change the `METHOD` variable:
```python
METHOD = "truncate"    # or "subsample"
OUTPUT_PREFIX = "sift128_"  # Prefix for output files
```

## Output

Both scripts create new files without modifying originals:

**Input:** `AT5007_..._frame_0001_descriptors.npy` (256-dim float32)
**Output:** `sift128_AT5007_..._frame_0001_descriptors.npy` (128-dim uint8)

Files are created in the same directory as the original descriptors with the `sift128_` prefix.

## Conversion Details

### Dimensionality Reduction

1. **PCA** (recommended): Projects 256-dim vectors to 128-dim while maximizing variance preservation
2. **Truncate**: Simple slicing `desc[:, :128]` - fast but discards half the information
3. **Subsample**: Takes every other dimension `desc[:, ::2]` - preserves spatial spread

### Normalization to SIFT Format

After dimensionality reduction, all methods apply per-descriptor normalization:

```python
# Normalize each descriptor to [0, 1]
desc_norm = (desc - desc.min()) / (desc.max() - desc.min())

# Scale to uint8 range
desc_uint8 = (desc_norm * 255).astype(np.uint8)
```

This ensures compatibility with SIFT-based matchers that expect uint8 descriptors.

## File Statistics

Currently in `2_sp_implementation/output/`:
- Total descriptor files: ~1,100
- Organized in 3 video sequence folders
- Each file: ~1,000 keypoints (256-dim each)

After conversion:
- New files: ~1,100 additional files
- Format: 128-dim uint8
- Naming: `sift128_` prefix + original name

## Verification

Both scripts include automatic verification that prints:
- Original vs converted shapes
- Dtype and value ranges
- Keypoint count preservation

Example output:
```
File 1: AT5007_..._frame_0001_descriptors.npy
  Original: shape=(998, 256), dtype=float32, range=[0.000, 1.000]
  Converted (sift128_AT5007_..._frame_0001_descriptors.npy): shape=(998, 128), dtype=uint8, range=[0, 255]
  ✓ Keypoint count preserved: 998
```

## Usage Notes

- Converted files are compatible with OpenCV SIFT/ORB matchers
- Original 256-dim files remain unchanged for future use
- PCA model is saved and can be reused for new descriptors
- All keypoints remain aligned with CSV files (row-to-row correspondence preserved)

## Which Script to Use?

**Use PCA version (`convert_to_sift128.py`) if:**
- You have scikit-learn installed
- You want optimal descriptor quality
- You're doing cross-dataset matching

**Use simple version (`convert_to_sift128_simple.py`) if:**
- You want minimal dependencies
- You need fast, deterministic conversion
- You're doing within-dataset matching only
