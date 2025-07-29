#!/usr/bin/env python3
"""
Test script to verify horizon detection is working correctly
"""
import cv2
import numpy as np
import pickle
from pathlib import Path

# Import functions from paper_script
from paper_script import extract_block_features, pixel_kmeans, fuse_and_postprocess, horizon_from_binary

def test_detection(img_path, model_path):
    """Test the detection pipeline and print debug information"""
    print(f"Testing: {img_path}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    clf = model['classifier']
    
    # Load and process image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {img.shape}")
    
    # Extract features and get sky probability
    feats, grid = extract_block_features(img)
    sky_prob_blocks = clf.predict_proba(feats)[:, 1]
    sky_prob_map = sky_prob_blocks.reshape(grid)
    sky_prob_img = cv2.resize(sky_prob_map, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
    
    print(f"Sky probability range: {sky_prob_img.min():.3f} - {sky_prob_img.max():.3f}")
    print(f"Average sky probability: {sky_prob_img.mean():.3f}")
    
    # Apply clustering
    cluster_map = pixel_kmeans(img)
    print(f"Number of clusters: {len(np.unique(cluster_map))}")
    
    # Fuse and post-process
    sky_bin = fuse_and_postprocess(cluster_map, sky_prob_img)
    sky_pixels = np.sum(sky_bin)
    total_pixels = sky_bin.shape[0] * sky_bin.shape[1]
    sky_ratio = sky_pixels / total_pixels
    
    print(f"Detected sky pixels: {sky_pixels} / {total_pixels} ({sky_ratio:.3f})")
    
    if sky_pixels == 0:
        print("WARNING: No sky detected!")
        return
    
    # Get horizon line
    horizon = horizon_from_binary(sky_bin)
    
    # Analyze horizon line
    print(f"Horizon line stats:")
    print(f"  Min Y: {horizon.min()}")
    print(f"  Max Y: {horizon.max()}")
    print(f"  Mean Y: {horizon.mean():.1f}")
    print(f"  Std Y: {horizon.std():.1f}")
    
    # Check if horizon is reasonable (not all at top)
    if horizon.max() <= 10:
        print("WARNING: Horizon line is at the very top of the image!")
    elif horizon.min() >= img.shape[0] - 10:
        print("WARNING: Horizon line is at the very bottom of the image!")
    else:
        print("Horizon line looks reasonable.")
    
    print("=" * 50)

if __name__ == "__main__":
    model_path = "corrected_model.pkl"
    
    # Test with different images
    test_images = [
        "./data/test/images/MVI_0792_VIS_OB_frame_000000.jpg",
        "./images/ship1.jpg",
        "./images/ship2.jpg"
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            test_detection(img_path, model_path)
        else:
            print(f"Image not found: {img_path}")
