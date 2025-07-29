#!/usr/bin/env python3
"""
Comparison tool for different horizon detection methods:
1. Original paper method (AI model only)
2. Color-based method (from horizon_detect.py)
3. Hybrid method (AI + color-based)

This script helps evaluate which method works best for different types of images.
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Import detection methods
from paper_script import detect_pipeline as paper_detect
from hybrid_horizon_detection import HybridHorizonDetector


def detect_paper_method(img_path: str, model_path: str) -> tuple:
    """
    Run the original paper method detection.
    """
    try:
        # Use the paper script's detection pipeline
        from paper_script import (extract_block_features, pixel_kmeans, 
                                fuse_and_postprocess, horizon_from_binary)
        import pickle
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        clf = model['classifier']
        
        # Load and process image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, 0
        
        start_time = time.time()
        
        # Extract features and classify
        feats, grid = extract_block_features(img)
        sky_prob_blocks = clf.predict_proba(feats)[:, 1]
        sky_prob_map = sky_prob_blocks.reshape(grid)
        sky_prob_img = cv2.resize(sky_prob_map, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Apply clustering and fusion
        cluster_map = pixel_kmeans(img)
        sky_bin = fuse_and_postprocess(cluster_map, sky_prob_img)
        
        # Get horizon line
        horizon = horizon_from_binary(sky_bin)
        
        processing_time = time.time() - start_time
        
        # Convert to line format
        if len(horizon) > 0:
            avg_y = int(np.mean(horizon))
            horizon_line = (0, avg_y, img.shape[1]-1, avg_y)
            
            # Create mask
            mask = np.zeros(img.shape, dtype=np.uint8)
            for x in range(len(horizon)):
                mask[horizon[x]:, x] = 255
        else:
            horizon_line = None
            mask = None
        
        return horizon_line, mask, processing_time
        
    except Exception as e:
        print(f"Paper method error: {e}")
        return None, None, 0


def detect_color_method(img_path: str) -> tuple:
    """
    Run the color-based detection method.
    """
    try:
        # Import from horizon_detect.py
        from horizon_detect import detect_horizon
        
        img = cv2.imread(img_path)
        if img is None:
            return None, None, 0
        
        start_time = time.time()
        
        # Use the color-based detection
        line, mask = detect_horizon(img)
        
        processing_time = time.time() - start_time
        
        return line, mask, processing_time
        
    except Exception as e:
        print(f"Color method error: {e}")
        return None, None, 0


def detect_hybrid_method(img_path: str, model_path: str) -> tuple:
    """
    Run the hybrid detection method.
    """
    try:
        detector = HybridHorizonDetector(model_path)
        
        img = cv2.imread(img_path)
        if img is None:
            return None, None, 0
        
        start_time = time.time()
        
        horizon_line, mask = detector.detect_horizon_hybrid(img)
        
        processing_time = time.time() - start_time
        
        return horizon_line, mask, processing_time
        
    except Exception as e:
        print(f"Hybrid method error: {e}")
        return None, None, 0


def draw_horizon_on_image(img: np.ndarray, horizon_line: tuple, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Draw horizon line on image.
    """
    result = img.copy()
    if horizon_line is not None:
        x1, y1, x2, y2 = horizon_line
        cv2.line(result, (x1, y1), (x2, y2), color, 2)
        
        # Add method label
        cv2.putText(result, f"Y: {y1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(result, "No detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result


def evaluate_detection_quality(img: np.ndarray, horizon_line: tuple, mask: np.ndarray) -> dict:
    """
    Evaluate the quality of horizon detection.
    """
    if horizon_line is None or mask is None:
        return {
            'position_score': 0.0,
            'smoothness_score': 0.0,
            'sky_ratio': 0.0,
            'overall_score': 0.0
        }
    
    h, w = img.shape[:2]
    x1, y1, x2, y2 = horizon_line
    horizon_y = (y1 + y2) // 2
    
    # Position score (prefer middle region)
    ideal_range = (h * 0.2, h * 0.8)
    if ideal_range[0] <= horizon_y <= ideal_range[1]:
        position_score = 1.0
    else:
        position_score = max(0.0, 1.0 - abs(horizon_y - h/2) / (h/2))
    
    # Sky ratio score
    sky_pixels = np.sum(mask == 0) if mask is not None else 0
    sky_ratio = sky_pixels / (h * w)
    if 0.15 <= sky_ratio <= 0.75:
        sky_ratio_score = 1.0
    else:
        sky_ratio_score = max(0.0, 1.0 - abs(sky_ratio - 0.4) / 0.4)
    
    # Smoothness score (check if horizon line is reasonably straight)
    if mask is not None:
        horizon_points = []
        for x in range(0, w, 10):
            col = mask[:, x]
            transitions = np.where(np.diff(col.astype(int)) > 0)[0]
            if len(transitions) > 0:
                horizon_points.append(transitions[0])
        
        if len(horizon_points) > 2:
            smoothness_score = max(0.0, 1.0 - np.std(horizon_points) / (h * 0.1))
        else:
            smoothness_score = 0.5
    else:
        smoothness_score = 0.0
    
    overall_score = (position_score * 0.4 + sky_ratio_score * 0.3 + smoothness_score * 0.3)
    
    return {
        'position_score': position_score,
        'smoothness_score': smoothness_score,
        'sky_ratio': sky_ratio,
        'sky_ratio_score': sky_ratio_score,
        'overall_score': overall_score
    }


def compare_methods(img_path: str, model_path: str, output_path: str = None, display: bool = False):
    """
    Compare all three detection methods on a single image.
    """
    print(f"\nAnalyzing: {Path(img_path).name}")
    print("=" * 50)
    
    # Load original image
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Could not load image {img_path}")
        return
    
    h, w = original.shape[:2]
    
    # Run all three methods
    methods = {
        'Paper Method': detect_paper_method(img_path, model_path),
        'Color Method': detect_color_method(img_path),
        'Hybrid Method': detect_hybrid_method(img_path, model_path)
    }
    
    # Create comparison visualization
    results = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
    
    for i, (method_name, (horizon_line, mask, proc_time)) in enumerate(methods.items()):
        # Draw result
        result_img = draw_horizon_on_image(original, horizon_line, colors[i])
        
        # Add method name
        cv2.putText(result_img, method_name, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
        
        # Evaluate quality
        quality = evaluate_detection_quality(original, horizon_line, mask)
        
        results.append(result_img)
        
        # Print results
        print(f"{method_name:15s} | Time: {proc_time*1000:6.1f}ms | "
              f"Quality: {quality['overall_score']:4.2f} | "
              f"Position: {horizon_line[1] if horizon_line else 'N/A':>4} | "
              f"Sky Ratio: {quality['sky_ratio']:4.2f}")
    
    # Create combined visualization
    # Resize images for display
    display_width = 400
    display_height = int(h * display_width / w)
    
    resized_results = []
    for result in results:
        resized = cv2.resize(result, (display_width, display_height))
        resized_results.append(resized)
    
    # Original image
    original_resized = cv2.resize(original, (display_width, display_height))
    cv2.putText(original_resized, "Original", (10, display_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Arrange in 2x2 grid
    top_row = np.hstack([original_resized, resized_results[0]])
    bottom_row = np.hstack([resized_results[1], resized_results[2]])
    combined = np.vstack([top_row, bottom_row])
    
    # Add title
    title_height = 50
    title_img = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_img, f"Horizon Detection Comparison - {Path(img_path).name}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    final_result = np.vstack([title_img, combined])
    
    # Save result
    if output_path:
        cv2.imwrite(output_path, final_result)
        print(f"Comparison saved to: {output_path}")
    
    # Display result
    if display:
        cv2.imshow('Horizon Detection Comparison', final_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return final_result


def batch_comparison(image_dir: str, model_path: str, output_dir: str = None):
    """
    Run comparison on all images in a directory.
    """
    image_dir = Path(image_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {img_path.name}")
        
        output_path = None
        if output_dir:
            output_path = output_dir / f"comparison_{img_path.stem}.jpg"
        
        try:
            compare_methods(str(img_path), model_path, str(output_path) if output_path else None)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Compare horizon detection methods')
    parser.add_argument('--image', type=str, help='Single image to analyze')
    parser.add_argument('--image-dir', type=str, help='Directory of images to analyze')
    parser.add_argument('--model', required=True, help='Path to trained AI model')
    parser.add_argument('--output', type=str, help='Output image/directory path')
    parser.add_argument('--display', action='store_true', help='Display results')
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        print("Error: Specify either --image or --image-dir")
        return
    
    if args.image and args.image_dir:
        print("Error: Specify either --image or --image-dir, not both")
        return
    
    try:
        if args.image:
            # Single image comparison
            compare_methods(args.image, args.model, args.output, args.display)
        else:
            # Batch comparison
            batch_comparison(args.image_dir, args.model, args.output)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
