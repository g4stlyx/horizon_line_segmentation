#!/usr/bin/env python3
"""
Batch test script for horizon detection on all images in the images folder.
Compares the new algorithm performance.
"""

import os
import cv2
import numpy as np
from horizon_detect import detect_horizon

def test_all_images():
    """Test horizon detection on all images in the images folder."""
    
    images_dir = "images"
    results_dir = "test_results"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No image files found in the images directory.")
        return
    
    print(f"Testing horizon detection on {len(image_files)} images...")
    print("-" * 60)
    
    successful_detections = 0
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_file)
        
        print(f"[{i}/{len(image_files)}] Processing {image_file}...")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ❌ Could not load image: {image_file}")
            continue
        
        # Detect horizon
        line, mask = detect_horizon(img)
        
        if line is not None:
            x1, y1, x2, y2 = line
            
            # Draw horizon line
            img_with_line = img.copy()
            cv2.line(img_with_line, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Create segmented image
            segmented_img = cv2.bitwise_and(img, img, mask=mask)
            
            # Save results
            base_name = os.path.splitext(image_file)[0]
            line_output = os.path.join(results_dir, f"{base_name}_horizon_line.jpg")
            segment_output = os.path.join(results_dir, f"{base_name}_segmented.jpg")
            
            cv2.imwrite(line_output, img_with_line)
            cv2.imwrite(segment_output, segmented_img)
            
            print(f"  ✅ Horizon detected at y={y1}")
            print(f"     💾 Saved: {line_output}")
            print(f"     💾 Saved: {segment_output}")
            
            successful_detections += 1
        else:
            print(f"  ❌ No horizon detected in {image_file}")
    
    print("-" * 60)
    print(f"Results: {successful_detections}/{len(image_files)} images processed successfully")
    print(f"Success rate: {(successful_detections/len(image_files)*100):.1f}%")
    print(f"Results saved in: {results_dir}/")

if __name__ == "__main__":
    test_all_images()