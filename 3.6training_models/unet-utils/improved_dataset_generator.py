# ==============================================================================
# Improved Dataset Generator with Ship-Aware Ground Truth
#! this is added to the training script for ship-aware dataset generation
#! no need to run this separately, it is integrated into the training script
#! keeping this for reference
# ==============================================================================
import cv2
import numpy as np
import os
from scipy.io import loadmat
import random

def detect_ships_in_frame(frame, horizon_y):
    """
    Detect ships and objects that should be classified as non-sky.
    Returns a mask where ships are marked.
    """
    h, w = frame.shape[:2]
    ship_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to different color spaces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Method 1: Detect dark objects (ship hulls) above horizon
    # Look for objects in the region above water but below sky
    search_top = max(0, horizon_y - 100)  # Search above horizon
    search_bottom = min(h, horizon_y + 20)  # Include some water region
    
    roi_gray = gray[search_top:search_bottom, :]
    
    if roi_gray.size > 0:
        # Adaptive threshold to find dark objects
        thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 10)
        
        # Find dark regions (ships are typically dark silhouettes)
        dark_regions = 255 - thresh
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
        
        # Filter by size and aspect ratio
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Minimum ship size
                x, y, cw, ch = cv2.boundingRect(contour)
                # Check aspect ratio (ships are usually wider than tall)
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio > 1.2 and area > 500:
                    # Mark this region as ship in the full mask
                    cv2.rectangle(ship_mask, 
                                (x, search_top + y), 
                                (x + cw, search_top + y + ch), 255, -1)
    
    # Method 2: Color-based ship detection
    # Look for typical ship colors in the region above horizon
    search_region = hsv[search_top:search_bottom, :]
    
    if search_region.size > 0:
        # White/light structures (superstructures)
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(search_region, white_lower, white_upper)
        
        # Dark structures (hulls)
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 80])
        dark_mask = cv2.inRange(search_region, dark_lower, dark_upper)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(white_mask, dark_mask)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Filter by contour size
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Mark this region as ship in the full mask
                ship_mask[search_top + y:search_top + y + ch, x:x + cw] = 255
    
    return ship_mask

def create_improved_mask(frame, horizon_y):
    """
    Create an improved ground truth mask that properly handles ships.
    
    Returns:
    - Class 0: Non-sky (water + ships + objects)  
    - Class 1: Sky (pure sky region only)
    """
    h, w, _ = frame.shape
    
    # Create initial sky mask based on horizon line
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Everything above horizon is initially sky
    mask[:horizon_y, :] = 1
    
    # Detect ships and objects
    ship_mask = detect_ships_in_frame(frame, horizon_y)
    
    # Remove ships from sky region (set ships to non-sky class)
    mask[ship_mask == 255] = 0
    
    return mask

def preprocess_smd_improved():
    """
    Create improved SMD dataset with ship-aware ground truth.
    """
    print("Creating improved SMD dataset with ship-aware ground truth...")
    
    # Update these paths according to your setup
    SMD_VIDEOS_PATH = '/path/to/your/SMD/VIS_Onshore/Videos'
    SMD_GT_PATH = '/path/to/your/SMD/VIS_Onshore/HorizonGT'
    OUTPUT_IMAGES_PATH = '/path/to/output/improved_images'
    OUTPUT_MASKS_PATH = '/path/to/output/improved_masks'
    
    os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_PATH, exist_ok=True)
    
    video_files = sorted([f for f in os.listdir(SMD_VIDEOS_PATH) if f.endswith('.avi')])
    processed_count = 0
    
    for i, video_file in enumerate(video_files):
        print(f"Processing video {i+1}/{len(video_files)}: {video_file}")
        video_name_without_ext = os.path.splitext(video_file)[0]
        gt_filename = f"{video_name_without_ext}_HorizonGT.mat"
        video_path = os.path.join(SMD_VIDEOS_PATH, video_file)
        gt_path = os.path.join(SMD_GT_PATH, gt_filename)
        
        if not os.path.exists(gt_path):
            continue
            
        # Load ground truth
        gt_data = loadmat(gt_path)
        horizon_key = None
        for key in gt_data.keys():
            if not key.startswith('__'):
                horizon_key = key
                break
                
        if horizon_key is None:
            continue
            
        struct_array = gt_data[horizon_key]
        if struct_array.size == 0:
            continue
            
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= struct_array.shape[1]:
                break
                
            try:
                # Extract horizon line parameters
                frame_struct = struct_array[0, frame_idx]
                x_point = float(frame_struct['X'][0,0])
                y_point = float(frame_struct['Y'][0,0])
                nx = float(frame_struct['Nx'][0,0])
                ny = float(frame_struct['Ny'][0,0])
                
                h, w, _ = frame.shape
                
                # Calculate horizon y-coordinate
                if abs(ny) < 1e-6:
                    horizon_y = int(y_point)
                else:
                    c = -(nx * x_point + ny * y_point)
                    horizon_y = int(-c / ny)
                
                # Clamp horizon to frame bounds
                horizon_y = max(0, min(h-1, horizon_y))
                
                # Create improved mask with ship awareness
                mask = create_improved_mask(frame, horizon_y)
                
                # Skip frames with invalid masks
                if mask.min() == mask.max():
                    frame_idx += 1
                    continue
                    
                # Save frame and mask
                frame_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
                mask_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.png"
                
                cv2.imwrite(os.path.join(OUTPUT_IMAGES_PATH, frame_filename), frame)
                cv2.imwrite(os.path.join(OUTPUT_MASKS_PATH, mask_filename), mask)
                
                processed_count += 1
                
                # Optional: Save visualization for debugging
                if processed_count % 100 == 0:
                    # Create visualization
                    vis_frame = frame.copy()
                    overlay = np.zeros_like(frame)
                    overlay[mask == 1] = [0, 0, 255]  # Sky in blue
                    vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
                    cv2.line(vis_frame, (0, horizon_y), (w, horizon_y), (0, 255, 0), 2)
                    
                    debug_filename = f"debug_{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_IMAGES_PATH, debug_filename), vis_frame)
                
            except (IndexError, TypeError, KeyError, ValueError) as e:
                print(f"Error processing frame {frame_idx}: {e}")
                
            frame_idx += 1
            
        cap.release()
        
    print(f"Improved preprocessing complete. Total frames processed: {processed_count}")

if __name__ == "__main__":
    preprocess_smd_improved()