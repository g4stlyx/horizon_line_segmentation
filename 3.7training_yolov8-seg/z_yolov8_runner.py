# ==============================================================================
# Local YOLOv8-seg Horizon Segmentation Inference Script
# ==============================================================================
#
# How to Run:
# 1. Make sure you have the required libraries:
#    pip install ultralytics opencv-python numpy pillow
#
# 2. Save this script as `z_yolov8_runner.py`.
#
# 3. Place the `best_yolov8n_seg_horizon.pt` file you downloaded from Colab
#    in the same directory as this script.
#
# 4. Run from your terminal with one of the following options:
#
#    - To process a single image:
#      python z_yolov8_runner.py --image /path/to/your/image.jpg
#
#    - To process a folder of images (batch processing):
#      python z_yolov8_runner.py --folder /path/to/your/image_folder
#
#    - To process a video file:
#      python z_yolov8_runner.py --video /path/to/your/video.mp4
#
#    - To use a live camera feed:
#      python z_yolov8_runner.py --camera
#
#    - To save the output (add the --save flag):
#      python z_yolov8_runner.py --image image.jpg --save
#      (Saves to image_yolo_segmented.jpg)
#      python z_yolov8_runner.py --folder ./images --save
#      (Saves segmented images in a new 'output_yolo' folder)
#
#    - To specify model path:
#      python z_yolov8_runner.py --model best_yolov8s_seg_horizon.pt --image test.jpg
#
#    - To adjust confidence threshold:
#      python z_yolov8_runner.py --image test.jpg --conf 0.3
#
# ==============================================================================

import cv2
import numpy as np
import argparse
import time
import os
import glob
from pathlib import Path
from ultralytics import YOLO
import torch

# ==============================================================================
# Helper Functions
# ==============================================================================

def create_overlay(image, combined_mask, class_colors=None):
    """
    Creates a transparent overlay for visualization.
    
    Args:
        image: Original image (BGR format)
        combined_mask: Combined mask from all detections
        class_colors: Dictionary mapping class IDs to colors
    """
    if class_colors is None:
        class_colors = {
            0: [255, 0, 0],    # Red for non-sky
            1: [0, 0, 255]     # Blue for sky
        }
    
    # Create colored overlay
    overlay = np.zeros_like(image, dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        mask_region = (combined_mask == class_id)
        overlay[mask_region] = color
    
    # Blend the overlay with the original image
    viz_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    return viz_image

def detect_ships_for_correction(image, mask):
    """
    Advanced ship detection to correct false sky classifications.
    Similar to U-Net approach but adapted for YOLO output.
    """
    h, w = image.shape[:2]
    corrected_mask = mask.copy()
    
    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Find the approximate horizon line from the current mask
    sky_rows = np.where(np.any(mask == 1, axis=1))[0]
    if len(sky_rows) > 0:
        horizon_y = np.max(sky_rows)  # Bottom edge of sky region
    else:
        horizon_y = h // 2  # Fallback
    
    # Search for ships in the sky region (incorrectly classified)
    search_top = max(0, horizon_y - 100)
    search_bottom = min(h, horizon_y + 20)
    
    # Method 1: Detect dark objects (ship hulls)
    roi_gray = gray[search_top:search_bottom, :]
    if roi_gray.size > 0:
        # Adaptive threshold to find dark objects
        thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 10)
        
        # Find dark regions (ships are typically dark silhouettes)
        dark_regions = 255 - thresh
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and filter by size
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Minimum ship size
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio > 1.2 and area > 500:
                    # Correct this region to non-sky (0)
                    corrected_mask[search_top + y:search_top + y + ch, x:x + cw] = 0
    
    # Method 2: Color-based ship detection
    roi_hsv = hsv[search_top:search_bottom, :]
    if roi_hsv.size > 0:
        # White/light structures (superstructures)
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(roi_hsv, white_lower, white_upper)
        
        # Dark structures (hulls)  
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 80])
        dark_mask_color = cv2.inRange(roi_hsv, dark_lower, dark_upper)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(white_mask, dark_mask_color)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and correct mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Correct this region to non-sky (0)
                corrected_mask[search_top + y:search_top + y + ch, x:x + cw] = 0
    
    return corrected_mask

def predict_yolo(model, image, conf_threshold=0.25, apply_ship_correction=True):
    """
    Run YOLOv8-seg prediction on an image.
    
    Args:
        model: Loaded YOLO model
        image: Input image (BGR format)
        conf_threshold: Confidence threshold for detections
        apply_ship_correction: Whether to apply ship-aware correction
    
    Returns:
        combined_mask: Combined segmentation mask
        detection_info: Dictionary with detection information
    """
    h, w = image.shape[:2]
    
    # Run YOLO prediction with device specification
    results = model.predict(
        image, 
        conf=conf_threshold, 
        save=False, 
        show=False, 
        verbose=False,
        device='cpu'  # Force CPU to avoid CUDA torchvision issues
    )
    
    # Initialize combined mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    detection_info = {
        'num_detections': 0,
        'classes_detected': [],
        'confidences': [],
        'areas': []
    }
    
    # Process results
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        
        if boxes is not None:
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            
            detection_info['num_detections'] = len(masks)
            detection_info['classes_detected'] = classes.tolist()
            detection_info['confidences'] = confidences.tolist()
            
            # Combine all masks by class
            for i, (mask, class_id, conf) in enumerate(zip(masks, classes, confidences)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Calculate area
                area = np.sum(mask_binary)
                detection_info['areas'].append(area)
                
                # Add to combined mask with class ID
                class_id = int(class_id)
                combined_mask[mask_binary == 1] = class_id
    
    # Apply ship-aware correction if enabled
    if apply_ship_correction:
        combined_mask = detect_ships_for_correction(image, combined_mask)
    
    return combined_mask, detection_info

def create_detailed_overlay(image, mask, detection_info, show_stats=True):
    """
    Create detailed visualization with detection statistics.
    """
    # Create basic overlay
    viz_image = create_overlay(image, mask)
    
    if show_stats and detection_info['num_detections'] > 0:
        # Add detection statistics
        stats_text = [
            f"Detections: {detection_info['num_detections']}",
            f"Sky regions: {detection_info['classes_detected'].count(1)}",
            f"Non-sky regions: {detection_info['classes_detected'].count(0)}",
            f"Avg confidence: {np.mean(detection_info['confidences']):.2f}"
        ]
        
        # Draw semi-transparent background for text
        overlay_text = viz_image.copy()
        cv2.rectangle(overlay_text, (10, 10), (400, 120), (0, 0, 0), -1)
        viz_image = cv2.addWeighted(viz_image, 0.7, overlay_text, 0.3, 0)
        
        # Add text
        for i, text in enumerate(stats_text):
            cv2.putText(viz_image, text, (20, 40 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return viz_image

# ==============================================================================
# Main Script
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLOv8-seg Horizon Segmentation Inference")
    parser.add_argument('--model', type=str, default='yolo8seg_best.pt', 
                       help='Path to YOLOv8-seg model file.')
    parser.add_argument('--image', type=str, help='Path to a single image file.')
    parser.add_argument('--folder', type=str, help='Path to a folder with images.')
    parser.add_argument('--video', type=str, help='Path to a video file.')
    parser.add_argument('--camera', action='store_true', help='Use live camera feed.')
    parser.add_argument('--save', action='store_true', help='Save the output.')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold for detections (0.0-1.0).')
    parser.add_argument('--no-ship-correction', action='store_true', 
                       help='Disable ship-aware correction.')
    parser.add_argument('--show-stats', action='store_true', 
                       help='Show detection statistics on output.')
    args = parser.parse_args()

    # Check for device - Force CPU due to torchvision CUDA compatibility issues on Windows
    device = 'cpu'  # Force CPU to avoid torchvision::nms CUDA backend issues
    print(f"Using device: {device} (CPU forced to avoid CUDA torchvision compatibility issues)")

    # Load YOLO model
    try:
        model = YOLO(args.model)
        # Force model to CPU
        model.to(device)
        print(f"Model loaded successfully from: {args.model}")
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure the model file '{args.model}' exists in the current directory.")
        return

    # Process based on input type
    if args.image:
        # --- Process a single image ---
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at '{args.image}'")
            return

        try:
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Could not load image '{args.image}'")
                return

            print(f"Processing image: {args.image}")
            start_time = time.time()
            
            mask, detection_info = predict_yolo(
                model, image, args.conf, 
                apply_ship_correction=not args.no_ship_correction
            )
            
            processing_time = time.time() - start_time
            
            result_image = create_detailed_overlay(image, mask, detection_info, args.show_stats)
            
            # Add processing time
            cv2.putText(result_image, f"Time: {processing_time:.2f}s", 
                       (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if args.save:
                save_path = os.path.splitext(args.image)[0] + "_yolo_segmented.jpg"
                cv2.imwrite(save_path, result_image)
                print(f"Saved segmented image to: {save_path}")

            cv2.imshow('YOLOv8-seg Horizon Segmentation', result_image)
            print("Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing image: {e}")

    elif args.folder:
        # --- Batch process a folder of images ---
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found at '{args.folder}'")
            return

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.folder, ext)))
            image_paths.extend(glob.glob(os.path.join(args.folder, ext.upper())))
        
        if not image_paths:
            print(f"No images found in folder: {args.folder}")
            return

        if args.save:
            output_folder = os.path.join(os.path.dirname(args.folder), "output_yolo_segmented")
            os.makedirs(output_folder, exist_ok=True)
            print(f"Saving results to: {output_folder}")

        print(f"Processing {len(image_paths)} images...")
        total_time = 0

        for i, image_path in enumerate(image_paths):
            try:
                print(f"Processing ({i+1}/{len(image_paths)}): {os.path.basename(image_path)}")
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load: {image_path}")
                    continue

                start_time = time.time()
                mask, detection_info = predict_yolo(
                    model, image, args.conf,
                    apply_ship_correction=not args.no_ship_correction
                )
                processing_time = time.time() - start_time
                total_time += processing_time

                result_image = create_detailed_overlay(image, mask, detection_info, args.show_stats)

                if args.save:
                    save_name = os.path.splitext(os.path.basename(image_path))[0] + "_yolo_segmented.jpg"
                    save_path = os.path.join(output_folder, save_name)
                    cv2.imwrite(save_path, result_image)

                cv2.imshow('YOLOv8-seg Horizon Segmentation', result_image)
                # Display each for 500ms, press 'q' to stop early
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        cv2.destroyAllWindows()
        print(f"Batch processing complete. Average time per image: {total_time/len(image_paths):.2f}s")

    elif args.video or args.camera:
        # --- Process a video file or camera feed ---
        if args.video:
            if not os.path.exists(args.video):
                print(f"Error: Video file not found at '{args.video}'")
                return
            cap = cv2.VideoCapture(args.video)
            source_name = f"Video: {args.video}"
        else:
            cap = cv2.VideoCapture(0)
            source_name = "Camera"

        if not cap.isOpened():
            print(f"Error: Could not open {source_name.lower()}")
            return
            
        video_writer = None
        if args.save:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            save_path = "yolo_output_segmented.mp4"
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"Saving output video to: {save_path}")

        print(f"Processing {source_name}. Press 'q' to quit.")
        
        frame_count = 0
        total_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("End of video reached.")
                break

            start_time = time.time()
            
            mask, detection_info = predict_yolo(
                model, frame, args.conf,
                apply_ship_correction=not args.no_ship_correction
            )
            
            result_frame = create_detailed_overlay(frame, mask, detection_info, args.show_stats)

            processing_time = time.time() - start_time
            fps = 1 / processing_time if processing_time > 0 else 0
            total_fps += fps
            frame_count += 1

            # Add FPS counter
            cv2.putText(result_frame, f"FPS: {fps:.1f} (Avg: {total_fps/frame_count:.1f})", 
                       (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if video_writer:
                video_writer.write(result_frame)

            cv2.imshow('YOLOv8-seg Horizon Segmentation', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            print(f"Processed {frame_count} frames. Average FPS: {total_fps/frame_count:.1f}")

    else:
        print("No input source specified. Please use --image, --folder, --video, or --camera.")
        parser.print_help()

if __name__ == "__main__":
    main()
