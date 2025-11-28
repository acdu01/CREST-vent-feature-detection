import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_superpoint_to_sift128_simple(input_dir, output_prefix="sift128_", method="truncate"):
    """
    Convert 256-dim SuperPoint descriptors to 128-dim SIFT format (simple version, no sklearn required).

    Args:
        input_dir: Root directory containing descriptor folders
        output_prefix: Prefix to add to output files (default: "sift128_")
        method: "truncate" (first 128 dims) or "subsample" (every other dim)
    """
    input_dir = Path(input_dir)

    # Find all descriptor .npy files
    descriptor_files = list(input_dir.rglob("*_descriptors.npy"))
    # Exclude already converted files
    descriptor_files = [f for f in descriptor_files if not f.name.startswith(output_prefix)]

    if not descriptor_files:
        raise ValueError(f"No descriptor files found in {input_dir}")

    print(f"Found {len(descriptor_files)} descriptor files")
    print(f"Conversion method: {method}")

    # Process all descriptor files
    stats = {
        'total_files': 0,
        'total_keypoints': 0,
        'empty_files': 0
    }

    for desc_file in tqdm(descriptor_files, desc="Converting descriptors"):
        # Load 256-dim descriptors
        desc_256 = np.load(desc_file)

        if len(desc_256) == 0:
            stats['empty_files'] += 1
            # Still create empty file
            desc_128_uint8 = np.empty((0, 128), dtype=np.uint8)
        else:
            # Reduce dimensions: 256 -> 128
            if method == "truncate":
                # Take first 128 dimensions
                desc_128 = desc_256[:, :128]
            elif method == "subsample":
                # Take every other dimension
                desc_128 = desc_256[:, ::2]
            else:
                raise ValueError(f"Unknown method: {method}")

            # Convert to SIFT format (uint8, range 0-255)
            # Method 1: Per-descriptor normalization (preserves relative magnitudes within each descriptor)
            desc_min = desc_128.min(axis=1, keepdims=True)
            desc_max = desc_128.max(axis=1, keepdims=True)
            desc_range = desc_max - desc_min

            # Avoid division by zero
            desc_range[desc_range == 0] = 1.0

            desc_normalized = (desc_128 - desc_min) / desc_range

            # Scale to [0, 255] and convert to uint8
            desc_128_uint8 = (desc_normalized * 255).astype(np.uint8)

            stats['total_keypoints'] += len(desc_128_uint8)

        # Create output path (same location, different name)
        # Example: AT5007_..._frame_0001_descriptors.npy -> sift128_AT5007_..._frame_0001_descriptors.npy
        output_file = desc_file.parent / (output_prefix + desc_file.name)

        # Save converted descriptors
        np.save(output_file, desc_128_uint8)
        stats['total_files'] += 1

    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Total files converted: {stats['total_files']}")
    print(f"Empty files: {stats['empty_files']}")
    print(f"Total keypoints processed: {stats['total_keypoints']}")
    if stats['total_files'] > stats['empty_files']:
        print(f"Average keypoints per file: {stats['total_keypoints'] / (stats['total_files'] - stats['empty_files']):.1f}")

    return stats

def verify_conversion(input_dir, output_prefix="sift128_"):
    """Verify that conversion was successful by checking a few files."""
    input_dir = Path(input_dir)

    # Find original and converted files
    original_files = list(input_dir.rglob("*_descriptors.npy"))
    original_files = [f for f in original_files if not f.name.startswith(output_prefix)]

    if not original_files:
        print("No original files found for verification")
        return

    # Check first 3 files
    print("\n=== Verification ===")
    for i, orig_file in enumerate(original_files[:3]):
        converted_file = orig_file.parent / (output_prefix + orig_file.name)

        if not converted_file.exists():
            print(f"❌ Missing converted file: {converted_file.name}")
            continue

        orig_desc = np.load(orig_file)
        conv_desc = np.load(converted_file)

        print(f"\nFile {i+1}: {orig_file.name}")
        print(f"  Original: shape={orig_desc.shape}, dtype={orig_desc.dtype}, range=[{orig_desc.min():.3f}, {orig_desc.max():.3f}]")
        print(f"  Converted: shape={conv_desc.shape}, dtype={conv_desc.dtype}, range=[{conv_desc.min()}, {conv_desc.max()}]")

        if len(orig_desc) == len(conv_desc):
            print(f"  ✓ Keypoint count preserved: {len(orig_desc)}")
        else:
            print(f"  ❌ Keypoint count mismatch!")

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = Path("2_sp_implementation/output")
    OUTPUT_PREFIX = "sift128_"
    METHOD = "truncate"  # Options: "truncate" or "subsample"

    print("SuperPoint (256-dim) to SIFT-format (128-dim) Converter (Simple)")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output prefix: {OUTPUT_PREFIX}")
    print(f"Method: {METHOD}")
    print("=" * 60)

    # Run conversion
    stats = convert_superpoint_to_sift128_simple(
        input_dir=INPUT_DIR,
        output_prefix=OUTPUT_PREFIX,
        method=METHOD
    )

    # Verify conversion
    verify_conversion(INPUT_DIR, OUTPUT_PREFIX)

    print("\n✓ Conversion complete!")
    print(f"New files saved with prefix: {OUTPUT_PREFIX}")
    print(f"\nOriginal files (256-dim float32): *_descriptors.npy")
    print(f"Converted files (128-dim uint8): {OUTPUT_PREFIX}*_descriptors.npy")
