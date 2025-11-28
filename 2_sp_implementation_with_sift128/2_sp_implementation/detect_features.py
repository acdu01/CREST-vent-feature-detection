import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from tqdm import tqdm  # For progress bar
import shutil
import os
import glob

def preprocess_image(img_file, img_size):
    """Prepare image for input to SuperPoint (sp_v6) network."""
    img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_file}")
    img = cv2.resize(img, img_size)
    img_orig = img.copy()

    # Convert to grayscale and float32
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.  # Normalize to [0,1]

    return img_preprocessed, img_orig

def extract_superpoint_keypoints(keypoint_map, keep_k_points=1000):
    """Extract keypoints from SuperPoint (sp_v6) keypoint map."""
    # Get points that are detected (prob > 0)
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    
    # Sort by confidence and keep top k
    sorted_indices = (-keypoints[:, 2]).argsort()[:keep_k_points]
    keypoints = keypoints[sorted_indices]
    
    return keypoints.astype(int)

def filter_dark_keypoints(keypoints, img_gray, threshold=20):
    """
    Remove keypoints that are too dark in the grayscale image.
    """
    filtered = []
    for kp in keypoints:
        y, x = kp[0], kp[1]
        if img_gray[y, x] >= threshold:
            filtered.append(kp)
    return np.array(filtered)

def draw_keypoints(image, keypoints):
    """Draw SuperPoint keypoints on the image."""
    img_with_kp = image.copy()
    # Draw each keypoint as a small circle
    for kp in keypoints:
        y, x = kp[0], kp[1]  # keypoints are in (y,x) format
        conf = kp[2]
        # Draw circle with size based on confidence
        radius = 2
        cv2.circle(img_with_kp, (x, y), radius, (0, 255, 0), -1)
    return img_with_kp

def save_keypoints_to_csv(keypoints, output_path):
    """Save keypoints to a CSV file."""
    # Create DataFrame with column names
    df = pd.DataFrame(keypoints, columns=['y', 'x', 'confidence'])
    
    # Sort by confidence (highest to lowest)
    df = df.sort_values('confidence', ascending=False)
    
    # Add index starting from 1
    df.index = range(1, len(df) + 1)
    
    # Save to CSV
    df.to_csv(output_path)
    
    return df

def extract_descriptors_at_keypoints(keypoints, descriptor_map):
    """Grab descriptor vectors aligned to the filtered keypoints."""
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, descriptor_map.shape[-1]), dtype=np.float32)

    coords = keypoints[:, :2].astype(int)
    descriptors = descriptor_map[coords[:, 0], coords[:, 1]]

    # L2 normalize for downstream matching
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8
    return descriptors / norms

def save_descriptors(descriptors, output_path):
    """Persist descriptors as a .npy array (N x 256)."""
    np.save(output_path, descriptors)
    return output_path

def process_image(image_path, sess, tensors, img_size, keep_k_points):
    """Process a single image with SuperPoint."""
    try:
        # Preprocess image
        img_preprocessed, img_orig = preprocess_image(image_path, img_size)
        
        # Run SuperPoint inference
        prob_nms, descriptor_map = sess.run(
            [tensors['output'], tensors['descriptors']],
            feed_dict={tensors['input']: np.expand_dims(img_preprocessed, 0)}
        )
        
        # Extract keypoints from SuperPoint output
        keypoint_map = np.squeeze(prob_nms)
        descriptor_map = np.squeeze(descriptor_map)
        keypoints = extract_superpoint_keypoints(keypoint_map, keep_k_points)
    
        # Convert original image to grayscale for brightness check
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

        # Filter out keypoints on dark pixels
        keypoints = filter_dark_keypoints(keypoints, img_gray, 20)

        # Grab descriptors corresponding to the surviving keypoints
        descriptors = extract_descriptors_at_keypoints(keypoints, descriptor_map)

        # Create visualization
        img_with_kp = draw_keypoints(img_orig, keypoints)
        
        return keypoints, descriptors, img_with_kp
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None, None

def run_superpoint_on_folder(input_folder, weights_path, output_dir, img_size=(640, 480), keep_k_points=1000):
    """Run SuperPoint (sp_v6) model on all images in a folder."""
    
    # Setup paths
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    weights_dir = Path(weights_path)
    
     # Clear output folder if it exists
    # Do NOT delete output directories during recursive runs.
# Just create subdirectories if missing.
    output_dir.mkdir(parents=True, exist_ok=True)


    # Create output directories
    csv_dir = output_dir / "csv"
    vis_dir = output_dir / "visualizations"
    desc_dir = output_dir / "descriptors"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    desc_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify weights path contains sp_v6
    assert weights_dir.name == "sp_v6", "Must use sp_v6 weights!"
    
    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_folder.glob(f'*{ext}'))
        image_paths.extend(input_folder.glob(f'*{ext.upper()}'))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_folder}")
    
    print(f"Found {len(image_paths)} images to process")
    
    # Initialize TensorFlow graph
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # Load the SuperPoint model
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            str(weights_path)
        )

        # Get SuperPoint specific tensors
        tensors = {
            'input': graph.get_tensor_by_name('superpoint/image:0'),
            'output': graph.get_tensor_by_name('superpoint/prob_nms:0'),
            'descriptors': graph.get_tensor_by_name('superpoint/descriptors:0')
        }

        # Process each image
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            # Process image
            keypoints, descriptors, img_with_kp = process_image(
                image_path, 
                sess, 
                tensors, 
                img_size, 
                keep_k_points
            )
            
            if keypoints is None:
                continue
                
            # Save results
            image_name = image_path.stem
            
            # Save CSV
            csv_path = csv_dir / f"{image_name}_keypoints.csv"
            df = save_keypoints_to_csv(keypoints, csv_path)
            
            # Save visualization
            vis_path = vis_dir / f"{image_name}_keypoints.jpg"
            cv2.imwrite(str(vis_path), img_with_kp)

            # Save descriptors (aligned to rows in the CSV)
            desc_path = desc_dir / f"{image_name}_descriptors.npy"
            save_descriptors(descriptors, desc_path)
            
            results.append({
                'image_name': image_name,
                'num_keypoints': len(keypoints),
                'csv_path': csv_path,
                'visualization_path': vis_path,
                'descriptors_path': desc_path
            })
            
        # Create summary CSV
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nProcessing complete! Summary saved to {summary_path}")
        
        return summary_df

if __name__ == "__main__":
    

    # Set paths
    INPUT_FOLDER = Path("1_video_processing/output_img")   # Root folder containing image subfolders
    WEIGHTS_PATH = "2_sp_implementation/saved_models/sp_v6"
    OUTPUT_DIR = Path("2_sp_implementation/output")

    all_summaries = []

    # Recursively walk all subfolders containing images
    for subfolder in INPUT_FOLDER.rglob("*"):
        if subfolder.is_dir():
            # Check if this folder contains any images
            image_files = list(subfolder.glob("*.jpg")) + \
                          list(subfolder.glob("*.png")) + \
                          list(subfolder.glob("*.jpeg"))

            if len(image_files) == 0:
                continue  # skip empty folders

            # Create matching output subfolder
            relative_path = subfolder.relative_to(INPUT_FOLDER)
            out_subfolder = OUTPUT_DIR / relative_path
            out_subfolder.mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing folder: {subfolder}")
            print(f"Saving results to: {out_subfolder}")

            # Run SuperPoint
            summary = run_superpoint_on_folder(
                str(subfolder),
                WEIGHTS_PATH,
                str(out_subfolder)
            )

            # Store summary with context
            summary["folder"] = str(relative_path)
            all_summaries.append(summary)

    # Print final summary
    total_images = sum(len(s["num_keypoints"]) for s in all_summaries)
    avg_keypoints = sum(s["num_keypoints"].mean() for s in all_summaries) / len(all_summaries)

    print("\n=== Processing Summary (Recursive) ===")
    print(f"Total folders processed: {len(all_summaries)}")
    print(f"Total images processed: {total_images}")
    print(f"Average keypoints per image (overall): {avg_keypoints:.1f}")
