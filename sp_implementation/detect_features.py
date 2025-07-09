import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from tqdm import tqdm  # For progress bar

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

def process_image(image_path, sess, tensors, img_size, keep_k_points):
    """Process a single image with SuperPoint."""
    try:
        # Preprocess image
        img_preprocessed, img_orig = preprocess_image(image_path, img_size)
        
        # Run SuperPoint inference
        prob_nms = sess.run(
            tensors['output'],
            feed_dict={tensors['input']: np.expand_dims(img_preprocessed, 0)}
        )
        
        # Extract keypoints from SuperPoint output
        keypoint_map = np.squeeze(prob_nms)
        keypoints = extract_superpoint_keypoints(keypoint_map, keep_k_points)
        
        # Create visualization
        img_with_kp = draw_keypoints(img_orig, keypoints)
        
        return keypoints, img_with_kp
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

def run_superpoint_on_folder(input_folder, weights_path, output_dir, img_size=(640, 480), keep_k_points=1000):
    """Run SuperPoint (sp_v6) model on all images in a folder."""
    
    # Setup paths
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    weights_dir = Path(weights_path)
    
    # Create output directories
    csv_dir = output_dir / "csv"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
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
            'output': graph.get_tensor_by_name('superpoint/prob_nms:0')
        }

        # Process each image
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            # Process image
            keypoints, img_with_kp = process_image(
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
            
            results.append({
                'image_name': image_name,
                'num_keypoints': len(keypoints),
                'csv_path': csv_path,
                'visualization_path': vis_path
            })
            
        # Create summary CSV
        summary_df = pd.DataFrame(results)
        summary_path = output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nProcessing complete! Summary saved to {summary_path}")
        
        return summary_df

if __name__ == "__main__":
    # Set paths
    INPUT_FOLDER = "/home/adu/crest/SuperPoint/superpoint/DATA_DIR"  # Folder containing images
    WEIGHTS_PATH = "saved_models/sp_v6"  # Path to sp_v6 weights
    OUTPUT_DIR = "/home/adu/crest/SuperPoint/superpoint/OUTPUT_DIR"  # Output directory for results
    
    # Run SuperPoint on all images
    summary = run_superpoint_on_folder(
        INPUT_FOLDER, 
        WEIGHTS_PATH,
        OUTPUT_DIR
    )
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total images processed: {len(summary)}")
    print(f"Average keypoints per image: {summary['num_keypoints'].mean():.1f}")
