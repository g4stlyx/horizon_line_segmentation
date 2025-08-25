#!/usr/bin/env python3
"""
Hybrid Horizon Line Detection combining AI model with color-based methods.

This script combines:
1. The trained AI model from paper_script.py for sky probability
2. Advanced color-based detection from horizon_detect.py
3. Improved post-processing and temporal stability

The AI model provides sky probability as a prior, while color-based methods
provide the actual horizon line detection
"""

import cv2
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Tuple, Optional
import time

# Import functions from paper_script
from paper_script import extract_block_features, pixel_kmeans

class HybridHorizonDetector:
    def __init__(self, model_path: str):
        """Initialize with trained AI model."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.classifier = model['classifier']
        
    def get_ai_sky_probability(self, img: np.ndarray) -> np.ndarray:
        """Get sky probability map from the trained AI model."""
        feats, grid = extract_block_features(img)
        sky_prob_blocks = self.classifier.predict_proba(feats)[:, 1]
        sky_prob_map = sky_prob_blocks.reshape(grid)
        sky_prob_img = cv2.resize(sky_prob_map, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return sky_prob_img
    
    def detect_horizon_hybrid(self, img_bgr: np.ndarray) -> Tuple[Optional[Tuple], Optional[np.ndarray]]:
        """
        Hybrid horizon detection combining AI model with color-based methods.
        """
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        
        # Step 1: Get AI model's sky probability
        ai_sky_prob = self.get_ai_sky_probability(img_gray)
        
        # Step 2: Use color-based detection with AI guidance
        horizon_y = self._detect_horizon_color_guided(img_bgr, ai_sky_prob)
        
        if horizon_y is None:
            # Fallback to pure AI method with improved post-processing
            horizon_y = self._detect_horizon_ai_fallback(ai_sky_prob)
        
        if horizon_y is None:
            return None, None
            
        # Step 3: Create continuous horizon line with ship avoidance
        horizon_line, mask = self._create_continuous_horizon(img_bgr, horizon_y, ai_sky_prob)
        
        return horizon_line, mask
    
    def _detect_horizon_color_guided(self, img_bgr: np.ndarray, ai_sky_prob: np.ndarray) -> Optional[int]:
        """
        Color-based horizon detection guided by AI sky probability.
        """
        h, w = img_bgr.shape[:2]
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Define search region based on AI prediction
        ai_sky_mask = ai_sky_prob > 0.3  # Lower threshold for guidance
        
        # Find approximate sky region bounds
        sky_rows = np.where(np.any(ai_sky_mask, axis=1))[0]
        if len(sky_rows) == 0:
            return None
            
        sky_top = sky_rows[0]
        sky_bottom = sky_rows[-1]
        
        # Search in a focused region around the AI prediction
        search_start = max(0, sky_top - 50)
        search_end = min(h, sky_bottom + 50)
        
        horizon_candidates = []
        
        # Method 1: Color transition detection with AI weighting
        for y in range(search_start, search_end, 5):
            if y + 20 >= h:
                continue
                
            # Analyze regions above and below this potential horizon
            above_region = img_bgr[max(0, y-15):y, :]
            below_region = img_bgr[y:min(h, y+15), :]
            
            if above_region.size == 0 or below_region.size == 0:
                continue
            
            # Calculate color differences
            above_mean = np.mean(above_region.reshape(-1, 3), axis=0)
            below_mean = np.mean(below_region.reshape(-1, 3), axis=0)
            color_diff = np.linalg.norm(above_mean - below_mean)
            
            # Calculate texture differences
            above_gray = cv2.cvtColor(above_region, cv2.COLOR_BGR2GRAY)
            below_gray = cv2.cvtColor(below_region, cv2.COLOR_BGR2GRAY)
            above_texture = np.std(above_gray)
            below_texture = np.std(below_gray)
            texture_diff = abs(above_texture - below_texture)
            
            # Get AI confidence at this height
            ai_weight = np.mean(ai_sky_prob[max(0, y-10):min(h, y+10), :])
            
            # Combined score
            score = color_diff * 0.4 + texture_diff * 0.3 + ai_weight * 0.3
            
            # Prefer positions where AI predicts sky above and non-sky below
            above_ai = np.mean(ai_sky_prob[max(0, y-10):y, :]) if y > 10 else 0
            below_ai = np.mean(ai_sky_prob[y:min(h, y+10), :]) if y < h-10 else 0
            transition_bonus = max(0, above_ai - below_ai) * 0.5
            
            final_score = score + transition_bonus
            horizon_candidates.append((y, final_score))
        
        # Method 2: Edge-based detection with AI guidance
        edges = cv2.Canny(img_gray, 50, 150)
        for y in range(search_start, search_end, 3):
            if y >= h:
                continue
                
            # Count horizontal edges
            edge_strength = np.sum(edges[y, :]) / w
            
            # Weight by AI prediction
            ai_weight = ai_sky_prob[y, :].mean() if y < h else 0
            
            # Look for strong horizontal features near AI prediction boundary
            edge_score = edge_strength * (1 + ai_weight)
            
            horizon_candidates.append((y, edge_score))
        
        if not horizon_candidates:
            return None
        
        # Select best candidate
        best_candidate = max(horizon_candidates, key=lambda x: x[1])
        return best_candidate[0]
    
    def _detect_horizon_ai_fallback(self, ai_sky_prob: np.ndarray) -> Optional[int]:
        """
        Fallback method using AI sky probability directly.
        """
        h, w = ai_sky_prob.shape
        
        # Find the bottom boundary of sky region
        sky_mask = ai_sky_prob > 0.5
        
        horizon_candidates = []
        for y in range(h):
            # Count sky pixels in this row
            sky_pixels = np.sum(sky_mask[y, :])
            sky_ratio = sky_pixels / w
            
            # Look for transition from high sky ratio to low sky ratio
            if y < h - 10:
                below_sky_ratio = np.mean([np.sum(sky_mask[y+i, :]) / w for i in range(1, 6)])
                transition_score = max(0, sky_ratio - below_sky_ratio)
                
                if transition_score > 0.2:  # Significant transition
                    horizon_candidates.append((y, transition_score))
        
        if not horizon_candidates:
            # Find bottom of largest sky region
            sky_rows = np.where(np.any(sky_mask, axis=1))[0]
            if len(sky_rows) > 0:
                return sky_rows[-1]
            return None
        
        # Select best transition point
        best_candidate = max(horizon_candidates, key=lambda x: x[1])
        return best_candidate[0]
    
    def _create_continuous_horizon(self, img_bgr: np.ndarray, base_horizon_y: int, ai_sky_prob: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        """
        Create a continuous horizon line that avoids ships and objects.
        """
        h, w = img_bgr.shape[:2]
        horizon_points = []
        
        # Detect potential ships/objects that should be avoided
        ship_mask = self._detect_ships_simple(img_bgr, base_horizon_y)
        
        # Create horizon line column by column
        window_size = 20  # Local smoothing window
        
        for x in range(0, w, 5):  # Sample every 5 pixels for efficiency
            local_horizon_y = base_horizon_y
            
            # Local adjustment based on AI sky probability
            x_start = max(0, x - window_size)
            x_end = min(w, x + window_size)
            
            # Find best local horizon position
            best_y = base_horizon_y
            best_score = 0
            
            for y in range(max(0, base_horizon_y - 30), min(h, base_horizon_y + 30)):
                # Check if this position conflicts with detected ships
                if ship_mask[y, x] > 0:
                    continue
                
                # Score based on AI sky probability transition
                above_score = np.mean(ai_sky_prob[max(0, y-5):y, x_start:x_end]) if y > 5 else 0
                below_score = np.mean(ai_sky_prob[y:min(h, y+5), x_start:x_end]) if y < h-5 else 0
                transition_score = max(0, above_score - below_score)
                
                if transition_score > best_score:
                    best_score = transition_score
                    best_y = y
            
            horizon_points.append((x, best_y))
        
        # Smooth the horizon line
        if len(horizon_points) > 3:
            horizon_points = self._smooth_horizon_line(horizon_points)
        
        # Create full-resolution horizon line
        full_horizon = np.zeros(w, dtype=np.int32)
        for i in range(len(horizon_points) - 1):
            x1, y1 = horizon_points[i]
            x2, y2 = horizon_points[i + 1]
            
            # Linear interpolation between points
            for x in range(x1, min(x2, w)):
                if x2 != x1:
                    t = (x - x1) / (x2 - x1)
                    y = int(y1 + t * (y2 - y1))
                    full_horizon[x] = max(0, min(h-1, y))
                else:
                    full_horizon[x] = y1
        
        # Fill any remaining points
        for x in range(w):
            if full_horizon[x] == 0 and len(horizon_points) > 0:
                # Use nearest point
                nearest_point = min(horizon_points, key=lambda p: abs(p[0] - x))
                full_horizon[x] = nearest_point[1]
        
        # Create horizon line tuple and mask
        horizon_line = (0, int(np.mean(full_horizon)), w-1, int(np.mean(full_horizon)))
        
        # Create segmentation mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for x in range(w):
            mask[full_horizon[x]:, x] = 255
        
        return horizon_line, mask
    
    def _detect_ships_simple(self, img_bgr: np.ndarray, horizon_y: int) -> np.ndarray:
        """
        Simple ship detection to avoid crossing ships with horizon line.
        """
        h, w = img_bgr.shape[:2]
        ship_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Look for dark, large objects near the horizon
        search_region = slice(max(0, horizon_y - 50), min(h, horizon_y + 50))
        region_gray = gray[search_region, :]
        
        # Adaptive thresholding to find dark objects
        if region_gray.size > 0:
            thresh = cv2.adaptiveThreshold(region_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 21, 10)
            
            # Find dark regions (ships are typically dark silhouettes)
            dark_regions = 255 - thresh
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
            
            # Filter by size (remove noise)
            contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum ship size
                    cv2.fillPoly(ship_mask[search_region, :], [contour], 255)
        
        return ship_mask
    
    def _smooth_horizon_line(self, horizon_points: list) -> list:
        """
        Apply smoothing to horizon line points.
        """
        if len(horizon_points) < 3:
            return horizon_points
        
        # Simple moving average smoothing
        smoothed = []
        window = 3
        
        for i in range(len(horizon_points)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(horizon_points), i + window // 2 + 1)
            
            x = horizon_points[i][0]
            y_values = [point[1] for point in horizon_points[start_idx:end_idx]]
            y_smooth = int(np.mean(y_values))
            
            smoothed.append((x, y_smooth))
        
        return smoothed


def draw_horizon_line(img: np.ndarray, horizon_line: Tuple, mask: np.ndarray = None) -> np.ndarray:
    """
    Draw horizon line on image.
    """
    result = img.copy()
    
    if horizon_line is not None:
        x1, y1, x2, y2 = horizon_line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw additional points for better visibility
        for x in range(0, img.shape[1], 50):
            if mask is not None and x < mask.shape[1]:
                # Find actual horizon point at this x coordinate
                horizon_col = mask[:, x]
                transition_points = np.where(np.diff(horizon_col) > 0)[0]
                if len(transition_points) > 0:
                    y = transition_points[0]
                    cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Hybrid Horizon Line Detection')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--model', required=True, help='Path to trained AI model')
    parser.add_argument('--output', help='Output image path')
    parser.add_argument('--display', action='store_true', help='Display result')
    
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    # Initialize detector
    try:
        detector = HybridHorizonDetector(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Detect horizon
    print("Detecting horizon...")
    start_time = time.time()
    
    horizon_line, mask = detector.detect_horizon_hybrid(img)
    
    detection_time = time.time() - start_time
    print(f"Detection completed in {detection_time:.3f} seconds")
    
    # Draw result
    if horizon_line is not None:
        result = draw_horizon_line(img, horizon_line, mask)
        print(f"Horizon detected at approximately y={horizon_line[1]}")
    else:
        result = img.copy()
        print("No horizon detected")
    
    # Save result
    if args.output:
        cv2.imwrite(args.output, result)
        print(f"Result saved to {args.output}")
    
    # Display result
    if args.display:
        cv2.imshow('Original', img)
        cv2.imshow('Horizon Detection', result)
        if mask is not None:
            cv2.imshow('Sky Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
