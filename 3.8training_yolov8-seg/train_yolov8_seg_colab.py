# ==============================================================================
# YOLOv8-seg Training for Maritime Horizon Detection on Google Colab
# ==============================================================================
"""
YOLOv8-seg Training for Maritime Horizon Detection

This training script adapts the semantic segmentation task to YOLOv8's instance 
segmentation framework by treating 'sky' and 'non-sky' as two large instances.

Key differences from U-Net approach:
- Converts binary masks to polygon annotations in YOLO format
- Uses YOLOv8-seg model which is optimized for speed and real-time inference
- Leverages pre-trained COCO weights for better generalization
- Includes data augmentation built into YOLOv8 training pipeline

Expected benefits:
- Faster inference for real-time applications
- Better generalization due to COCO pre-training
- Easier deployment with Ultralytics framework
- Built-in optimization for maritime scenes

Configurable parameters:
- MODEL_SIZE: 'n' (fastest), 's', 'm', 'l', 'x' (most accurate)
- EPOCHS: Number of training epochs (default: 100)
- BATCH_SIZE: Training batch size (default: 16)
- IMG_SIZE: Image size for training (default: 640)
"""

# Install necessary libraries
!pip install ultralytics opencv-python-headless scipy roboflow

import os
import cv2
import numpy as np
import torch
import random
import collections
from scipy.io import loadmat
from google.colab import drive
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import gc  # For garbage collection

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# Step 2: Connect to Google Drive and Set Up Paths
# ==============================================================================
# Mount your Google Drive to the Colab environment
drive.mount('/content/drive')

# --- IMPORTANT: SET YOUR PATHS HERE ---
BASE_DRIVE_PATH = '/content/drive/My Drive/SMD_Dataset'

# Source data paths (same as U-Net script)
SMD_VIDEOS_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/Videos')
SMD_GT_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/HorizonGT')

# YOLOv8-seg specific paths
GDRIVE_YOLO_DATA_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_yolov8_seg_dataset')
GDRIVE_MODEL_SAVE_PATH = os.path.join(BASE_DRIVE_PATH, 'models')

# Local paths for fast access during training
LOCAL_YOLO_DATA_PATH = '/content/yolov8_seg_dataset'
LOCAL_MODEL_SAVE_PATH = '/content/models'

# Create directories
for path in [GDRIVE_YOLO_DATA_PATH, GDRIVE_MODEL_SAVE_PATH, LOCAL_YOLO_DATA_PATH, LOCAL_MODEL_SAVE_PATH]:
    os.makedirs(path, exist_ok=True)

# YOLO dataset structure
YOLO_TRAIN_IMAGES = os.path.join(LOCAL_YOLO_DATA_PATH, 'train/images')
YOLO_TRAIN_LABELS = os.path.join(LOCAL_YOLO_DATA_PATH, 'train/labels')
YOLO_VAL_IMAGES = os.path.join(LOCAL_YOLO_DATA_PATH, 'val/images')
YOLO_VAL_LABELS = os.path.join(LOCAL_YOLO_DATA_PATH, 'val/labels')
YOLO_TEST_IMAGES = os.path.join(LOCAL_YOLO_DATA_PATH, 'test/images')
YOLO_TEST_LABELS = os.path.join(LOCAL_YOLO_DATA_PATH, 'test/labels')

for path in [YOLO_TRAIN_IMAGES, YOLO_TRAIN_LABELS, YOLO_VAL_IMAGES, 
             YOLO_VAL_LABELS, YOLO_TEST_IMAGES, YOLO_TEST_LABELS]:
    os.makedirs(path, exist_ok=True)

print(f"Google Drive YOLO data path: {GDRIVE_YOLO_DATA_PATH}")
print(f"Local YOLO data path: {LOCAL_YOLO_DATA_PATH}")

# ==============================================================================
# Step 3: Mask to Polygon Conversion Functions
# ==============================================================================
def mask_to_polygons(mask, min_area=500):
    """
    Convert binary mask to polygon coordinates in YOLO format.
    
    Args:
        mask: Binary mask (0s and 1s)
        min_area: Minimum contour area to consider
    
    Returns:
        List of polygons in YOLO format (normalized coordinates)
    """
    h, w = mask.shape
    polygons = []
    
    # Find contours for each class
    for class_id in [0, 1]:  # non-sky (0) and sky (1)
        class_mask = (mask == class_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < min_area:
                continue
            
            # Simplify contour to reduce polygon complexity
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to YOLO format (normalized coordinates)
            if len(approx) >= 3:  # Need at least 3 points for a polygon
                polygon = []
                for point in approx:
                    x, y = point[0]
                    # Normalize coordinates
                    x_norm = x / w
                    y_norm = y / h
                    polygon.extend([x_norm, y_norm])
                
                polygons.append({
                    'class_id': class_id,
                    'polygon': polygon
                })
    
    return polygons

def create_yolo_annotation(polygons):
    """
    Create YOLO annotation string from polygons.
    
    Args:
        polygons: List of polygon dictionaries
    
    Returns:
        String in YOLO annotation format
    """
    lines = []
    for poly_data in polygons:
        class_id = poly_data['class_id']
        polygon = poly_data['polygon']
        
        # Format: class_id x1 y1 x2 y2 x3 y3 ...
        line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in polygon])
        lines.append(line)
    
    return "\n".join(lines)

def detect_ships_in_frame(frame, horizon_y):
    """
    Detect ships and objects that should be classified as non-sky.
    Returns a mask where ships are marked.
    (Same as U-Net script)
    """
    h, w = frame.shape[:2]
    ship_mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to different color spaces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Method 1: Detect dark objects (ship hulls) above horizon
    search_top = max(0, horizon_y - 100)
    search_bottom = min(h, horizon_y + 20)

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
            if area > 200:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio > 1.2 and area > 500:
                    ship_mask[search_top + y:search_top + y + ch, x:x + cw] = 255

    # Method 2: Color-based ship detection
    search_region = hsv[search_top:search_bottom, :]

    if search_region.size > 0:
        # White/light structures
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(search_region, white_lower, white_upper)

        # Dark structures
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
                ship_mask[search_top + y:search_top + y + ch, x:x + cw] = 255

    return ship_mask

def create_ship_aware_mask(frame, horizon_y):
    """
    Create a ship-aware ground truth mask that properly handles ships.
    (Same as U-Net script)
    """
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Everything above horizon is initially sky
    mask[:horizon_y, :] = 1

    # Detect ships and objects
    ship_mask = detect_ships_in_frame(frame, horizon_y)

    # Remove ships from sky region (set ships to non-sky class)
    mask[ship_mask == 255] = 0

    return mask

# ==============================================================================
# Step 4: SMD Dataset Processing for YOLOv8-seg (Memory Efficient Version)
# ==============================================================================
def preprocess_smd_for_yolov8(max_frames_per_video=500):
    """
    Extract frames from SMD videos and create YOLO-format annotations.
    Memory efficient version that saves data directly instead of storing in RAM.
    
    Args:
        max_frames_per_video: Maximum frames to process per video to limit memory usage
    """
    print("Starting SMD preprocessing for YOLOv8-seg...")
    print(f"Processing max {max_frames_per_video} frames per video to conserve memory...")
    video_files = sorted([f for f in os.listdir(SMD_VIDEOS_PATH) if f.endswith('.avi')])

    processed_count = 0
    ships_detected_count = 0
    video_frame_counts = {}  # Track frame counts per video for splitting

    for i, video_file in enumerate(video_files):
        print(f"Processing video {i+1}/{len(video_files)}: {video_file}")
        video_name_without_ext = os.path.splitext(video_file)[0]
        gt_filename = f"{video_name_without_ext}_HorizonGT.mat"
        video_path = os.path.join(SMD_VIDEOS_PATH, video_file)
        gt_path = os.path.join(SMD_GT_PATH, gt_filename)

        if not os.path.exists(gt_path):
            print(f"  Warning: Ground truth file not found for {video_file}")
            continue

        try:
            gt_data = loadmat(gt_path)
        except Exception as e:
            print(f"  Error loading ground truth for {video_file}: {e}")
            continue
            
        horizon_key = None
        for key in gt_data.keys():
            if not key.startswith('__'):
                horizon_key = key
                break

        if horizon_key is None:
            print(f"  Warning: No valid horizon data found in {gt_filename}")
            continue

        struct_array = gt_data[horizon_key]
        if struct_array.size == 0:
            print(f"  Warning: Empty horizon data in {gt_filename}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Error: Could not open video {video_file}")
            continue
            
        frame_idx = 0
        video_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= struct_array.shape[1]:
                break
            
            # Limit frames per video to prevent memory overflow
            if video_frame_count >= max_frames_per_video:
                print(f"  Reached max frames ({max_frames_per_video}) for {video_file}")
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
                    horizon_y = int(y_point - nx * x_point / ny)

                # Clamp horizon to frame bounds
                horizon_y = max(0, min(h-1, horizon_y))

                # Create ship-aware mask
                mask = create_ship_aware_mask(frame, horizon_y)

                # Skip frames with invalid masks
                if mask.min() == mask.max():
                    frame_idx += 1
                    continue

                # Count ships detected
                ship_mask = detect_ships_in_frame(frame, horizon_y)
                if np.any(ship_mask == 255):
                    ships_detected_count += 1

            except (IndexError, TypeError, KeyError, ValueError) as e:
                frame_idx += 1
                continue

            # Save frame and mask directly to temporary directory
            frame_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
            
            # Save to temporary directory first (we'll organize later)
            temp_image_path = os.path.join('/tmp', frame_filename)
            cv2.imwrite(temp_image_path, frame)
            
            # Convert mask to polygons and save annotation
            polygons = mask_to_polygons(mask)
            if polygons:
                annotation = create_yolo_annotation(polygons)
                temp_label_path = os.path.join('/tmp', frame_filename.replace('.jpg', '.txt'))
                with open(temp_label_path, 'w') as f:
                    f.write(annotation)
            
            # Track video frame count for splitting
            if video_name_without_ext not in video_frame_counts:
                video_frame_counts[video_name_without_ext] = []
            video_frame_counts[video_name_without_ext].append(frame_filename)

            frame_idx += 1
            processed_count += 1
            video_frame_count += 1
            
            # Print progress every 100 frames to avoid too much output
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} frames...")
                # Force garbage collection every 100 frames to free memory
                gc.collect()

        cap.release()
        print(f"  Video {video_file}: {video_frame_count} frames processed")
        
        # Clear large variables after each video
        del frame, mask
        if 'ship_mask' in locals():
            del ship_mask
        gc.collect()

    print(f"Preprocessing complete. Total frames processed: {processed_count}")
    print(f"Frames with ships detected: {ships_detected_count} ({ships_detected_count/processed_count*100:.1f}%)")
    
    return video_frame_counts

def split_and_save_yolo_dataset_efficient(video_frame_counts):
    """
    Split data by video and organize into YOLO format efficiently.
    """
    print("Splitting dataset and organizing YOLO format...")
    
    unique_videos = list(video_frame_counts.keys())
    random.seed(42)  # for reproducibility
    random.shuffle(unique_videos)

    # Split videos (80% train, 10% val, 10% test)
    split_idx_1 = int(0.8 * len(unique_videos))
    split_idx_2 = int(0.9 * len(unique_videos))
    train_videos = unique_videos[:split_idx_1]
    val_videos = unique_videos[split_idx_1:split_idx_2]
    test_videos = unique_videos[split_idx_2:]

    print(f"Video split - Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")

    # Organize files into splits
    splits = {
        'train': (train_videos, YOLO_TRAIN_IMAGES, YOLO_TRAIN_LABELS),
        'val': (val_videos, YOLO_VAL_IMAGES, YOLO_VAL_LABELS),
        'test': (test_videos, YOLO_TEST_IMAGES, YOLO_TEST_LABELS)
    }

    split_stats = {}
    
    for split_name, (videos, images_dir, labels_dir) in splits.items():
        frame_count = 0
        
        for video_name in videos:
            for frame_filename in video_frame_counts[video_name]:
                # Move image file
                temp_image_path = os.path.join('/tmp', frame_filename)
                final_image_path = os.path.join(images_dir, frame_filename)
                
                if os.path.exists(temp_image_path):
                    shutil.move(temp_image_path, final_image_path)
                    
                    # Move corresponding label file
                    label_filename = frame_filename.replace('.jpg', '.txt')
                    temp_label_path = os.path.join('/tmp', label_filename)
                    final_label_path = os.path.join(labels_dir, label_filename)
                    
                    if os.path.exists(temp_label_path):
                        shutil.move(temp_label_path, final_label_path)
                    
                    frame_count += 1
        
        split_stats[split_name] = frame_count
        print(f"{split_name.capitalize()} set: {frame_count} frames")
    
    return split_stats

# ==============================================================================
# Step 5: Create YOLO Configuration File
# ==============================================================================
def create_yolo_config():
    """
    Create YOLO dataset configuration file.
    """
    config = {
        'path': LOCAL_YOLO_DATA_PATH,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 2,  # number of classes
        'names': ['non-sky', 'sky']  # class names
    }
    
    config_path = os.path.join(LOCAL_YOLO_DATA_PATH, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLO config saved to: {config_path}")
    return config_path

# ==============================================================================
# Step 6: Check if Data Exists and Process if Needed
# ==============================================================================
def check_and_process_data():
    """
    Check if processed data exists, if not, process the SMD dataset.
    """
    # Check if we have processed data
    if (os.path.exists(YOLO_TRAIN_IMAGES) and 
        len(os.listdir(YOLO_TRAIN_IMAGES)) > 0):
        print("Processed YOLO data found.")
        return True
    
    print("No processed data found. Starting SMD dataset processing...")
    
    # Process SMD dataset efficiently with frame limit to prevent memory overflow
    # Reduce max_frames_per_video if you still get memory errors
    video_frame_counts = preprocess_smd_for_yolov8(max_frames_per_video=300)
    
    if not video_frame_counts:
        print("ERROR: No frame data processed!")
        return False
    
    # Split and save in YOLO format efficiently
    split_stats = split_and_save_yolo_dataset_efficient(video_frame_counts)
    
    # Create config file
    create_yolo_config()
    
    print("Dataset processing complete!")
    return True

# ==============================================================================
# Step 7: YOLOv8-seg Training Functions
# ==============================================================================
def train_yolov8_seg(epochs=100, imgsz=640, batch_size=16, model_size='n'):
    """
    Train YOLOv8-seg model.
    
    Args:
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size
        model_size: Model size ('n', 's', 'm', 'l', 'x')
    """
    print(f"\n--- Starting YOLOv8{model_size}-seg Training ---")
    
    # Initialize model
    model = YOLO(f'yolov8{model_size}-seg.pt')  # Load pre-trained model
    
    # Training configuration
    config_path = os.path.join(LOCAL_YOLO_DATA_PATH, 'dataset.yaml')
    
    # Train the model
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name='horizon_seg',
        project=LOCAL_MODEL_SAVE_PATH,
        save_period=10,  # Save checkpoint every 10 epochs
        patience=20,     # Early stopping patience
        device=device,
        # Data augmentation settings
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation  
        hsv_v=0.4,       # Value augmentation
        degrees=10,      # Rotation augmentation
        translate=0.1,   # Translation augmentation
        scale=0.5,       # Scale augmentation
        shear=0.0,       # Shear augmentation
        perspective=0.0, # Perspective augmentation
        flipud=0.0,      # Vertical flip (disable for horizon detection)
        fliplr=0.5,      # Horizontal flip
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.1,       # Mixup augmentation
        copy_paste=0.1,  # Copy-paste augmentation
        # Other settings
        optimizer='AdamW',
        lr0=0.01,        # Initial learning rate
        lrf=0.01,        # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,         # Box loss gain
        cls=0.5,         # Class loss gain
        dfl=1.5,         # DFL loss gain
        pose=12.0,       # Pose loss gain (unused)
        kobj=2.0,        # Keypoint objective loss gain (unused)
        label_smoothing=0.0,
        nbs=64,          # Nominal batch size
        overlap_mask=True,  # Overlap masks for training
        mask_ratio=4,    # Mask downsample ratio
        dropout=0.0,     # Dropout rate
        val=True,        # Validate during training
        plots=True,      # Save training plots
        verbose=True     # Verbose output
    )
    
    print("Training completed!")
    
    # Save best model to Google Drive with better error handling
    best_model_path = os.path.join(LOCAL_MODEL_SAVE_PATH, 'horizon_seg', 'weights', 'best.pt')
    last_model_path = os.path.join(LOCAL_MODEL_SAVE_PATH, 'horizon_seg', 'weights', 'last.pt')
    
    print(f"Looking for models in: {os.path.join(LOCAL_MODEL_SAVE_PATH, 'horizon_seg', 'weights')}")
    
    # List all available model files
    weights_dir = os.path.join(LOCAL_MODEL_SAVE_PATH, 'horizon_seg', 'weights')
    if os.path.exists(weights_dir):
        print("Available model files:")
        for f in os.listdir(weights_dir):
            file_path = os.path.join(weights_dir, f)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  {f} ({size_mb:.1f} MB)")
    
    # Save best model to Google Drive
    if os.path.exists(best_model_path):
        gdrive_model_path = os.path.join(GDRIVE_MODEL_SAVE_PATH, f'best_yolov8{model_size}_seg_horizon.pt')
        print(f"Copying best model to Google Drive: {gdrive_model_path}")
        try:
            shutil.copy2(best_model_path, gdrive_model_path)
            print(f"✓ Best model saved to Google Drive: {gdrive_model_path}")
        except Exception as e:
            print(f"✗ Error saving to Google Drive: {e}")
            print("Model is still available locally for download")
    else:
        print(f"✗ Best model not found at: {best_model_path}")
    
    # Also save last model
    if os.path.exists(last_model_path):
        gdrive_last_path = os.path.join(GDRIVE_MODEL_SAVE_PATH, f'last_yolov8{model_size}_seg_horizon.pt')
        try:
            shutil.copy2(last_model_path, gdrive_last_path)
            print(f"✓ Last model also saved to Google Drive: {gdrive_last_path}")
        except Exception as e:
            print(f"Note: Could not save last model to Drive: {e}")
    
    # Provide download instructions
    print("\n" + "="*50)
    print("MODEL DOWNLOAD INSTRUCTIONS:")
    print("="*50)
    print("Run this code to download your trained model:")
    print(f"""
from google.colab import files
import shutil

# Download best model
best_model = '{best_model_path}'
download_name = 'best_yolov8{model_size}_seg_horizon.pt'
shutil.copy2(best_model, download_name)
files.download(download_name)
""")
    print("="*50)

# ==============================================================================
# Step 8: Visualization Functions
# ==============================================================================
def visualize_predictions(model, num_images=5):
    """
    Visualize model predictions on validation set.
    """
    print("Visualizing predictions...")
    
    # Get some validation images
    val_images = os.listdir(YOLO_VAL_IMAGES)[:num_images]
    
    fig, axes = plt.subplots(num_images, 2, figsize=(15, num_images * 5))
    if num_images == 1:
        axes = np.array([axes])
    
    for i, img_name in enumerate(val_images):
        img_path = os.path.join(YOLO_VAL_IMAGES, img_name)
        
        # Load original image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        results = model.predict(img_path, conf=0.1, save=False, show=False)
        
        # Display original image
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original: {img_name}")
        axes[i, 0].axis('off')
        
        # Display prediction
        if results[0].masks is not None:
            # Create visualization with masks
            pred_img = img_rgb.copy()
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            
            colors = [(255, 0, 0), (0, 0, 255)]  # Red for non-sky, Blue for sky
            
            for j, mask in enumerate(masks):
                class_id = int(classes[j]) if j < len(classes) else 0
                color = colors[class_id]
                
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
                colored_mask = np.zeros_like(img_rgb)
                colored_mask[mask_resized > 0.5] = color
                
                # Blend with original image
                pred_img = cv2.addWeighted(pred_img, 0.7, colored_mask, 0.3, 0)
            
            axes[i, 1].imshow(pred_img)
        else:
            axes[i, 1].imshow(img_rgb)
            axes[i, 1].text(0.5, 0.5, 'No detections', 
                           transform=axes[i, 1].transAxes, 
                           ha='center', va='center', fontsize=12)
        
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model):
    """
    Evaluate the trained model on the test set.
    """
    print("Evaluating model on test set...")
    
    # Run validation on test set
    config_path = os.path.join(LOCAL_YOLO_DATA_PATH, 'dataset.yaml')
    
    # Temporarily modify config to use test set
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create temporary config for test evaluation
    test_config = config.copy()
    test_config['val'] = 'test/images'
    
    test_config_path = os.path.join(LOCAL_YOLO_DATA_PATH, 'test_dataset.yaml')
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    # Run evaluation
    results = model.val(data=test_config_path)
    
    print("Test evaluation completed!")
    return results

# ==============================================================================
# Step 9: Main Execution
# ==============================================================================
def main():
    """
    Main execution function.
    """
    print("=== YOLOv8-seg Maritime Horizon Detection Training ===")
    
    # Check and process data
    if not check_and_process_data():
        print("ERROR: Data processing failed!")
        return
    
    # Train model
    print("\nStarting YOLOv8-seg training...")
    
    # You can adjust these parameters
    EPOCHS = 100
    # For 4K videos (3840x2160), recommended image sizes:
    # - 1280: Good speed/memory balance, faster training
    # - 1536: Better accuracy, moderate memory usage
    # - 1920: High accuracy, higher memory usage
    # - 2560: Maximum detail, requires lots of GPU memory
    IMG_SIZE = 1280  # Increased from 640 for 4K video quality
    BATCH_SIZE = 8   # Reduced from 16 due to larger image size
    MODEL_SIZE = 'n'  # 'n' for nano (fastest), 's', 'm', 'l', 'x' for larger models
    
    results, model = train_yolov8_seg(
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        model_size=MODEL_SIZE
    )
    
    # Visualize results
    visualize_predictions(model)
    
    # Evaluate on test set
    test_results = evaluate_model(model)
    
    print("\n=== Training Complete ===")
    print(f"Best model saved to Google Drive")
    print(f"Training results: {results}")

# Run the main function
if __name__ == "__main__":
    main()
