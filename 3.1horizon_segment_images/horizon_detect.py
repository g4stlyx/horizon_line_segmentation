"""
! usage: (for a single image only, for now.)
    py horizon_detect.py --image photo.jpg --display

! datasets: 
    ! Singapore Maritime Dataset (SMD): https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset (onshore)

! papers:
    https://www.ijcaonline.org/archives/volume121/number10/21574-4625/
    https://arxiv.org/abs/1805.08105
    https://link.springer.com/article/10.1007/s00371-024-03767-8
"""

import cv2, numpy as np, argparse, sys
from sklearn.cluster import KMeans

def detect_horizon_color_based(img_bgr):
    """
    Advanced horizon detection using color-based segmentation and clustering.
    Handles various lighting conditions including day, sunset, and night scenes.
    """
    h, w = img_bgr.shape[:2]
    
    # Convert to different color spaces for analysis
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Strategy 1: Color-based sky-water separation
    horizon_y = detect_horizon_by_color_transition(img_bgr, img_hsv, img_lab)
    
    if horizon_y is None:
        # Strategy 2: Use edge detection with color filtering as fallback
        horizon_y = detect_horizon_by_edges_with_color(img_bgr, img_hsv)
    
    if horizon_y is None:
        return None, None
    
    # Create horizon line coordinates (horizontal line)
    x1, y1 = 0, horizon_y
    x2, y2 = w - 1, horizon_y
    
    # Create basic mask without ship preservation
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[horizon_y:, :] = 255
    
    return (x1, y1, x2, y2), mask

def detect_horizon_by_color_transition(img_bgr, img_hsv, img_lab):
    """
    Detect horizon by analyzing color transitions between sky and water.
    Enhanced version that handles different lighting conditions and preserves objects.
    """
    h, w = img_bgr.shape[:2]
    
    # Check if this is a low-light/night scene
    avg_brightness = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    is_low_light = avg_brightness < 60
    
    # Check for sunset/sunrise scenes (bright but with high contrast)
    is_sunset = detect_sunset_scene(img_bgr, img_hsv)
    
    if is_low_light and not is_sunset:
        return detect_horizon_night_mode(img_bgr, img_hsv, img_lab)
    elif is_sunset:
        return detect_horizon_sunset_mode(img_bgr, img_hsv, img_lab)
    
    # Standard daylight processing with object preservation
    return detect_horizon_with_object_preservation(img_bgr, img_hsv, img_lab)

def detect_sunset_scene(img_bgr, img_hsv):
    """
    Detect if this is a sunset/sunrise scene based on color characteristics.
    """
    # Look for warm colors (orange/red hues) in the image
    warm_mask1 = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([20, 255, 255]))    # Red-orange
    warm_mask2 = cv2.inRange(img_hsv, np.array([160, 50, 50]), np.array([179, 255, 255])) # Red
    warm_mask = cv2.bitwise_or(warm_mask1, warm_mask2)
    
    warm_ratio = np.sum(warm_mask > 0) / (img_hsv.shape[0] * img_hsv.shape[1])
    
    # Check for high brightness variation (characteristic of sunset)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness_std = np.std(gray)
    
    # Sunset conditions: significant warm colors and high brightness variation
    return warm_ratio > 0.1 and brightness_std > 50

def detect_horizon_sunset_mode(img_bgr, img_hsv, img_lab):
    """
    Special detection for sunset/sunrise scenes with high contrast.
    Uses brightness consistency and position weighting for optimal results.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    horizon_candidates = []
    
    # Primary strategy: Find bright, consistent horizontal lines in upper half
    # This works best for sunset scenes where the horizon is clearly lit
    search_start = int(h * 0.2)
    search_end = int(h * 0.5)
    
    for y in range(search_start, search_end, 5):
        row = gray[y, :]
        brightness_consistency = 1.0 / (1.0 + np.std(row))
        avg_brightness = np.mean(row)
        
        # Weight by position (prefer higher up) and quality metrics
        height_weight = (search_end - y) / (search_end - search_start)
        
        # Combined score prioritizing bright, consistent, higher lines
        score = brightness_consistency * avg_brightness * height_weight * 100
        
        # Additional bonus for very consistent lines in sunset scenes
        if brightness_consistency > 0.025 and avg_brightness > 160:
            score += 200
        
        horizon_candidates.append((y, score))
    
    # Secondary strategy: Look for gentle brightness transitions
    for y in range(search_start, int(h * 0.6), 10):
        sky_region = gray[max(0, y-10):y, :]
        water_region = gray[y:min(h, y+10), :]
        
        if sky_region.size > 0 and water_region.size > 0:
            sky_brightness = np.mean(sky_region)
            water_brightness = np.mean(water_region)
            brightness_drop = sky_brightness - water_brightness
            
            # In sunset scenes, even small drops can indicate horizon
            if brightness_drop > 3:
                row_colors = img_hsv[y, :]
                color_variance = np.std(row_colors, axis=0)
                color_consistency = 1.0 / (1.0 + np.mean(color_variance))
                
                score = brightness_drop * color_consistency * 50
                horizon_candidates.append((y, score))
    
    if not horizon_candidates:
        return None
    
    # Choose the highest-scoring candidate
    best_candidate = max(horizon_candidates, key=lambda x: x[1])
    horizon_y = best_candidate[0]
    
    # Ensure reasonable bounds
    horizon_y = max(int(h * 0.15), min(int(h * 0.6), horizon_y))
    
    return horizon_y

def detect_horizon_with_object_preservation(img_bgr, img_hsv, img_lab):
    """
    Detection using color transitions without object preservation.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    horizon_candidates = []
    strip_height = 15
    
    search_start = int(h * 0.1)
    search_end = int(h * 0.6)
    
    for y in range(search_start, search_end, strip_height):
        upper_region = gray[max(0, y - strip_height):y, :]
        lower_region = gray[y:min(h, y + strip_height), :]
        
        if upper_region.size > 0 and lower_region.size > 0:
            upper_brightness = np.mean(upper_region)
            lower_brightness = np.mean(lower_region)
            brightness_drop = upper_brightness - lower_brightness
            
            upper_strip = img_hsv[max(0, y - strip_height):y, :]
            lower_strip = img_hsv[y:min(h, y + strip_height), :]
            
            if upper_strip.size > 0 and lower_strip.size > 0:
                upper_mean = np.mean(upper_strip.reshape(-1, 3), axis=0)
                lower_mean = np.mean(lower_strip.reshape(-1, 3), axis=0)
                
                hue_diff = min(abs(upper_mean[0] - lower_mean[0]), 
                              180 - abs(upper_mean[0] - lower_mean[0]))
                sat_diff = abs(upper_mean[1] - lower_mean[1])
                val_diff = abs(upper_mean[2] - lower_mean[2])
                
                color_diff = hue_diff * 1.5 + sat_diff * 1.2 + val_diff * 1
                
                upper_blue_ratio = get_blue_ratio(img_bgr[max(0, y - strip_height):y, :])
                lower_blue_ratio = get_blue_ratio(img_bgr[y:min(h, y + strip_height), :])
                blue_transition = abs(upper_blue_ratio - lower_blue_ratio)
                
                score = color_diff + blue_transition * 30 + max(0, brightness_drop) * 2
                
                if score > 20:
                    horizon_candidates.append((y, score))
    
    if not horizon_candidates:
        return None
    
    # Prefer higher scores
    best_candidate = max(horizon_candidates, key=lambda x: x[1])
    horizon_y = best_candidate[0]
    
    return horizon_y

def detect_horizon_night_mode(img_bgr, img_hsv, img_lab):
    """
    Special detection method for night/low-light scenes.
    Uses contrast and texture analysis instead of color.
    """
    h, w = img_bgr.shape[:2]
    
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate local standard deviation (texture measure)
    kernel_size = 15
    mean_filtered = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    sqr_diff = (gray.astype(np.float32) - mean_filtered) ** 2
    texture = cv2.blur(sqr_diff, (kernel_size, kernel_size))
    
    # Find horizontal transitions in texture
    strip_height = max(h // 15, 5)
    horizon_candidates = []
    
    for y in range(strip_height, h - strip_height, strip_height):
        upper_texture = np.mean(texture[max(0, y - strip_height):y, :])
        lower_texture = np.mean(texture[y:min(h, y + strip_height), :])
        
        texture_diff = abs(upper_texture - lower_texture)
        
        # Also check brightness transition
        upper_brightness = np.mean(gray[max(0, y - strip_height):y, :])
        lower_brightness = np.mean(gray[y:min(h, y + strip_height), :])
        brightness_diff = abs(upper_brightness - lower_brightness)
        
        combined_score = texture_diff + brightness_diff * 2
        
        if combined_score > 50:  # Threshold for night scenes
            horizon_candidates.append((y, combined_score))
    
    if not horizon_candidates:
        return None
    
    # Prefer middle region
    middle_start = int(h * 0.25)
    middle_end = int(h * 0.75)
    
    middle_candidates = [(y, score) for y, score in horizon_candidates 
                        if middle_start <= y <= middle_end]
    
    if middle_candidates:
        return max(middle_candidates, key=lambda x: x[1])[0]
    else:
        return max(horizon_candidates, key=lambda x: x[1])[0]

def detect_horizon_by_edges_with_color(img_bgr, img_hsv):
    """
    Fallback method using edge detection with color filtering.
    """
    h, w = img_bgr.shape[:2]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Focus on horizontal edges in the middle region of the image
    middle_start = int(h * 0.2)
    middle_end = int(h * 0.8)
    
    # Count horizontal edges for each row in the middle region
    horizon_scores = []
    for y in range(middle_start, middle_end):
        # Count edge pixels in this row
        edge_count = np.sum(edges[y, :] > 0)
        
        # Weight by position (prefer middle of the search region)
        middle_y = (middle_start + middle_end) // 2
        distance_weight = 1.0 / (1.0 + abs(y - middle_y) / 100.0)
        
        score = edge_count * distance_weight
        horizon_scores.append((y, score))
    
    if not horizon_scores:
        return None
    
    # Find the row with the highest score
    horizon_y = max(horizon_scores, key=lambda x: x[1])[0]
    
    return horizon_y

def detect_horizon_advanced_clustering(img_bgr):
    """
    Advanced method using K-means clustering to separate sky and water regions.
    """
    h, w = img_bgr.shape[:2]
    
    # Resize for faster processing
    scale = 0.5
    small_img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    small_h, small_w = small_img.shape[:2]
    
    # Reshape for clustering
    pixel_data = small_img.reshape(-1, 3).astype(np.float32)
    
    # Apply K-means clustering (k=3: sky, water, objects)
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape labels back to image shape
    label_img = labels.reshape(small_h, small_w)
    
    # Find the dominant clusters in top and bottom regions
    top_region = label_img[:small_h//3, :]
    bottom_region = label_img[2*small_h//3:, :]
    
    top_cluster = np.bincount(top_region.flatten()).argmax()
    bottom_cluster = np.bincount(bottom_region.flatten()).argmax()
    
    if top_cluster == bottom_cluster:
        return None
    
    # Find the transition line between these clusters
    for y in range(small_h//4, 3*small_h//4):
        row = label_img[y, :]
        if np.sum(row == top_cluster) < np.sum(row == bottom_cluster):
            # Scale back to original image size
            horizon_y = int(y / scale)
            return horizon_y
    
    return None

def detect_horizon(img_bgr):
    """
    Main horizon detection function that tries multiple strategies.
    """
    # Try color-based detection first
    result = detect_horizon_color_based(img_bgr)
    if result[0] is not None:
        return result
    
    # Fallback to clustering method
    horizon_y = detect_horizon_advanced_clustering(img_bgr)
    if horizon_y is not None:
        h, w = img_bgr.shape[:2]
        x1, y1 = 0, horizon_y
        x2, y2 = w - 1, horizon_y
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[horizon_y:, :] = 255
        
        return (x1, y1, x2, y2), mask
    
    return None, None

def get_blue_ratio(img_region):
    """
    Calculate the ratio of blue-ish pixels in a region.
    Useful for detecting water areas.
    """
    if img_region.size == 0:
        return 0
    
    # Convert to HSV for better color analysis
    hsv_region = cv2.cvtColor(img_region, cv2.COLOR_BGR2HSV)
    
    # Define blue range in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue pixels
    blue_mask = cv2.inRange(hsv_region, lower_blue, upper_blue)
    
    # Calculate ratio
    blue_pixels = np.sum(blue_mask > 0)
    total_pixels = img_region.shape[0] * img_region.shape[1]
    
    return blue_pixels / total_pixels if total_pixels > 0 else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="horizon_out.jpg")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Image not found at {args.image}")
        
    img_with_line = img.copy()
    segmented_img = None

    line, mask = detect_horizon(img)
    
    if line:
        x1, y1, x2, y2 = line
        cv2.line(img_with_line, (x1, y1), (x2, y2), (0, 0, 255), 3)
        print(f"Horizon line detected at y={y1}")
        
        if mask is not None:
            segmented_img = cv2.bitwise_and(img, img, mask=mask)
            out_path_parts = args.out.rsplit('.', 1)
            if len(out_path_parts) == 2:
                out_segmented_path = f"{out_path_parts[0]}_segmented.{out_path_parts[1]}"
            else:
                out_segmented_path = f"{args.out}_segmented"
            cv2.imwrite(out_segmented_path, segmented_img)
            print(f"Segmented image saved to {out_segmented_path}")

    else:
        print("No horizon detected :(")
        
    cv2.imwrite(args.out, img_with_line)
    print(f"Image with horizon line saved to {args.out}")

    
    if args.display:
        cv2.imshow("Original Image", img)
        cv2.imshow("Image with Horizon Line", img_with_line)
        if segmented_img is not None:
            cv2.imshow("Segmented Below Horizon", segmented_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()