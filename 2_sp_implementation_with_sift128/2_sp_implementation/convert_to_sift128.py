import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle

def convert_superpoint_to_sift128(input_dir, output_prefix="sift128_", use_pca=True):
    """
    Convert 256-dim SuperPoint descriptors to 128-dim SIFT format.

    Args:
        input_dir: Root directory containing descriptor folders
        output_prefix: Prefix to add to output files (default: "sift128_")
        use_pca: If True, use PCA for dimensionality reduction.
                 If False, simply truncate to first 128 dimensions.
    """
    input_dir = Path(input_dir)

    # Find all descriptor .npy files
    descriptor_files = list(input_dir.rglob("*_descriptors.npy"))

    if not descriptor_files:
        raise ValueError(f"No descriptor files found in {input_dir}")

    print(f"Found {len(descriptor_files)} descriptor files")

    # If using PCA, we need to fit it on a sample of the data first
    pca = None
    if use_pca:
        print("\nFitting PCA on sample data...")
        # Collect sample descriptors from first 10 files
        sample_descriptors = []
        for desc_file in tqdm(descriptor_files[:10], desc="Loading samples"):
            desc = np.load(desc_file)
            if len(desc) > 0:
                sample_descriptors.append(desc)

        if sample_descriptors:
            sample_data = np.vstack(sample_descriptors)
            print(f"Sample data shape: {sample_data.shape}")

            # Fit PCA
            pca = PCA(n_components=128, random_state=42)
            pca.fit(sample_data)
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

            # Save PCA model for future use
            pca_path = input_dir / "pca_model_256to128.pkl"
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
            print(f"Saved PCA model to {pca_path}")

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
            if use_pca and pca is not None:
                desc_128 = pca.transform(desc_256)
            else:
                # Simple truncation to first 128 dimensions
                desc_128 = desc_256[:, :128]

            # Convert to SIFT format (uint8, range 0-255)
            # First, normalize to [0, 1] range
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
    USE_PCA = True  # Set to False for simple truncation

    print("SuperPoint (256-dim) to SIFT-format (128-dim) Converter")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output prefix: {OUTPUT_PREFIX}")
    print(f"Method: {'PCA dimensionality reduction' if USE_PCA else 'Simple truncation'}")
    print("=" * 60)

    # Run conversion
    stats = convert_superpoint_to_sift128(
        input_dir=INPUT_DIR,
        output_prefix=OUTPUT_PREFIX,
        use_pca=USE_PCA
    )

    # Verify conversion
    verify_conversion(INPUT_DIR, OUTPUT_PREFIX)

    print("\n✓ Conversion complete!")
    print(f"New files saved with prefix: {OUTPUT_PREFIX}")
