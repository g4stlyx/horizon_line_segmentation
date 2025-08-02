# ==============================================================================
# Step 1: Setup and Environment - Ship-Aware U-Net Training
# ==============================================================================
# Install necessary libraries. SciPy is needed for loading .mat files.
!pip install opencv-python-headless scipy

"""
Ship-Aware U-Net Training for Maritime Horizon Detection

This training script creates ship-aware ground truth masks where:
- Ships and maritime objects are correctly labeled as "non-sky"
- Only pure sky regions are labeled as "sky"
- Uses computer vision ship detection to improve ground truth quality

Expected improved results:
- Ships will be correctly segmented as non-sky objects
- Better semantic understanding of maritime scenes
- Reduced false positives in sky classification

Previous results (with basic geometric masks):
Epoch 1/50 -> Train Loss: 0.0971, Val Loss: 0.0366, Val IoU: 0.9812
  -> Model saved with new best validation IoU: 0.9812
Epoch 2/50 -> Train Loss: 0.0070, Val Loss: 0.0082, Val IoU: 0.9934
  -> Model saved with new best validation IoU: 0.9934
! still reaches the best model after 2 epochs, shows overfitting, thats a problem.
! needs a better, more varied dataset to train on
Epoch 3/50 -> Train Loss: 0.0049, Val Loss: 0.0100, Val IoU: 0.9923
Epoch 4/50 -> Train Loss: 0.0034, Val Loss: 0.0090, Val IoU: 0.9930
Epoch 5/50 -> Train Loss: 0.0029, Val Loss: 0.0093, Val IoU: 0.9934

Expected with ship-aware training:
- Lower initial IoU (as ship regions are now correctly labeled)
- Better generalization to real maritime scenes
- Ships correctly classified as non-sky
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # --- CHANGE: Added for learning rate scheduling
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
from google.colab import drive
from scipy.io import loadmat
import collections
import random

# Check if a GPU is available and set the device accordingly.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# Step 2: Connect to Google Drive and Set Up Paths
# ==============================================================================
# This will mount your Google Drive to the Colab environment.
drive.mount('/content/drive')

# --- IMPORTANT: SET YOUR PATHS HERE ---
BASE_DRIVE_PATH = '/content/drive/My Drive/SMD_Dataset'

# --- FIX: ADD THESE TWO LINES ---
SMD_VIDEOS_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/Videos')
SMD_GT_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/HorizonGT')
GDRIVE_MODEL_SAVE_PATH = os.path.join(BASE_DRIVE_PATH, 'models')
GDRIVE_PROCESSED_IMAGES_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_unet_ship_aware/images')
GDRIVE_PROCESSED_MASKS_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_unet_ship_aware/masks')

# --- The rest of your path definitions follow ---
GDRIVE_PROCESSED_DATA_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_unet_ship_aware')
os.makedirs(GDRIVE_MODEL_SAVE_PATH, exist_ok=True)


# This is where the data will be stored LOCALLY on the Colab machine for fast access.
# --- CHANGE: Renamed local path for ship-aware data ---
LOCAL_DATA_PATH = '/content/processed_unet_ship_aware'
LOCAL_IMAGES_PATH = os.path.join(LOCAL_DATA_PATH, 'images')
LOCAL_MASKS_PATH = os.path.join(LOCAL_DATA_PATH, 'masks')

os.makedirs(GDRIVE_PROCESSED_IMAGES_PATH, exist_ok=True)
os.makedirs(GDRIVE_PROCESSED_MASKS_PATH, exist_ok=True)

print(f"Google Drive processed data path: {GDRIVE_PROCESSED_DATA_PATH}")
print(f"Google Drive model save path: {GDRIVE_MODEL_SAVE_PATH}")
print(f"Local runtime data path: {LOCAL_DATA_PATH}")

# ==============================================================================
# Step 3: Ship-Aware Dataset Generation Functions
# ==============================================================================
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

def create_ship_aware_mask(frame, horizon_y):
    """
    Create a ship-aware ground truth mask that properly handles ships.

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

# ==============================================================================
# Step 4: Ship-Aware Preprocess Data for U-Net (Run only ONCE)
# ==============================================================================
def preprocess_smd_for_segmentation():
    """
    Extracts frames and creates ship-aware ground truth segmentation masks.
    Saves the output to your Google Drive.
    """
    print("Starting Ship-Aware SMD preprocessing for Segmentation...")
    video_files = sorted([f for f in os.listdir(SMD_VIDEOS_PATH) if f.endswith('.avi')])

    processed_count = 0
    total_videos = len(video_files)
    ships_detected_count = 0

    for i, video_file in enumerate(video_files):
        print(f"Processing video {i+1}/{total_videos}: {video_file}")
        video_name_without_ext = os.path.splitext(video_file)[0]
        gt_filename = f"{video_name_without_ext}_HorizonGT.mat"
        video_path = os.path.join(SMD_VIDEOS_PATH, video_file)
        gt_path = os.path.join(SMD_GT_PATH, gt_filename)

        if not os.path.exists(gt_path):
            continue

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
            if not ret:
                break
            if frame_idx >= struct_array.shape[1]:
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

            # Save frame and mask
            frame_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
            mask_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.png"

            cv2.imwrite(os.path.join(GDRIVE_PROCESSED_IMAGES_PATH, frame_filename), frame)
            cv2.imwrite(os.path.join(GDRIVE_PROCESSED_MASKS_PATH, mask_filename), mask)

            # Optional: Save debug visualization every 100 frames
            if processed_count % 100 == 0:
                vis_frame = frame.copy()
                overlay = np.zeros_like(frame)
                overlay[mask == 1] = [0, 0, 255]  # Sky in blue
                vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
                cv2.line(vis_frame, (0, horizon_y), (w, horizon_y), (0, 255, 0), 2)

                debug_filename = f"debug_{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(os.path.join(GDRIVE_PROCESSED_IMAGES_PATH, debug_filename), vis_frame)

            frame_idx += 1
            processed_count += 1

        cap.release()

    print(f"Ship-Aware preprocessing complete. Total frames processed: {processed_count}")
    print(f"Frames with ships detected: {ships_detected_count} ({ships_detected_count/processed_count*100:.1f}%)")

# Check if the processed folder on Drive is empty. If so, run preprocessing.
if not os.listdir(GDRIVE_PROCESSED_IMAGES_PATH):
    print("Processed data not found on Google Drive. Running preprocessing. This may take a while.")
    preprocess_smd_for_segmentation()
else:
    print("Processed data found on Google Drive.")

# --- Copy the entire folder from Drive to the local runtime ---
if not os.path.exists(LOCAL_DATA_PATH):
    print(f"Copying data from '{GDRIVE_PROCESSED_DATA_PATH}' to local runtime '{LOCAL_DATA_PATH}'...")
    !cp -r "{GDRIVE_PROCESSED_DATA_PATH}" "{LOCAL_DATA_PATH}"
    print("Data successfully copied to local runtime.")
else:
    print("Data already exists in local runtime. Skipping copy.")

# ==============================================================================
# Step 4: Create the Dataset with Aggressive Data Augmentation
# ==============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

class SegmentationDataset(Dataset):
    def __init__(self, image_files, masks_dir, size, is_train=False):
        self.image_files = image_files
        self.masks_dir = masks_dir
        self.size = size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Check if mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L") # --- CHANGE: Ensure mask is single channel
        except Exception as e:
            raise IOError(f"Error loading image {img_path} or mask {mask_path}: {e}")

        # Resize first
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random affine transformation (rotation, translation, scale, shear)
            angle = random.uniform(-15, 15)
            translate = (random.uniform(0, 0.1 * self.size[0]), random.uniform(0, 0.1 * self.size[1]))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)
            image = TF.affine(image, angle, translate, scale, shear, fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear, fill=0)

            # Color jitter (only on the image)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.7, 1.3))
                image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.7, 1.3))
                image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.3))
                image = TF.adjust_hue(image, hue_factor=random.uniform(-0.1, 0.1))

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        # --- CHANGE: Convert mask to tensor before normalization
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long()

        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask_tensor

# --- Implement Video-Based Splitting using LOCAL paths ---
all_image_files = sorted([os.path.join(LOCAL_IMAGES_PATH, f) for f in os.listdir(LOCAL_IMAGES_PATH)
                         if f.endswith('.jpg') and not f.startswith('debug_')])  # Filter out debug files

# Validate that all images have corresponding masks
def validate_dataset(image_files, masks_dir):
    """
    Validate that all image files have corresponding mask files.
    Returns list of valid image files.
    """
    valid_files = []
    missing_masks = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(masks_dir, mask_name)

        if os.path.exists(mask_path):
            valid_files.append(img_path)
        else:
            missing_masks.append(mask_name)

    if missing_masks:
        print(f"Warning: {len(missing_masks)} image files don't have corresponding masks:")
        for mask in missing_masks[:5]:  # Show first 5 missing masks
            print(f"  - {mask}")
        if len(missing_masks) > 5:
            print(f"  ... and {len(missing_masks) - 5} more")
        print(f"Using {len(valid_files)} valid image-mask pairs out of {len(image_files)} total images.")

    return valid_files

# Validate dataset before splitting
all_image_files = validate_dataset(all_image_files, LOCAL_MASKS_PATH)

if all_image_files:
    video_groups = collections.defaultdict(list)
    for f in all_image_files:
        prefix = '_'.join(os.path.basename(f).split('_')[:-2])
        video_groups[prefix].append(f)

    unique_videos = list(video_groups.keys())
    random.seed(42) # for reproducibility
    random.shuffle(unique_videos)

    split_idx_1 = int(0.8 * len(unique_videos))
    # --- CHANGE: Create a test set for final evaluation
    split_idx_2 = int(0.9 * len(unique_videos))
    train_videos = unique_videos[:split_idx_1]
    val_videos = unique_videos[split_idx_1:split_idx_2]
    test_videos = unique_videos[split_idx_2:]


    train_files = [f for video in train_videos for f in video_groups[video]]
    val_files = [f for video in val_videos for f in video_groups[video]]
    test_files = [f for video in test_videos for f in video_groups[video]]

    train_dataset = SegmentationDataset(train_files, LOCAL_MASKS_PATH, IMG_SIZE, is_train=True)
    val_dataset = SegmentationDataset(val_files, LOCAL_MASKS_PATH, IMG_SIZE, is_train=False)
    test_dataset = SegmentationDataset(test_files, LOCAL_MASKS_PATH, IMG_SIZE, is_train=False)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


    print(f"\nDataset loaded from local runtime and split by video.")
    print(f"Total videos: {len(unique_videos)} -> Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")
    print(f"Total frames: {len(all_image_files)} -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
else:
    print("\n--- ERROR: Dataset is empty ---")


# ==============================================================================
# Step 5: Define U-Net Model, New Loss, and Training Loop
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, n_classes=2, dropout_rate=0.3): # --- CHANGE: Added dropout
        super().__init__()
        self.n_classes = n_classes

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(resnet.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size: 128
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size: 64
        self.layer2 = self.base_layers[5] # size: 32
        self.layer3 = self.base_layers[6] # size: 16
        self.layer4 = self.base_layers[7] # size: 8

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256 + 256, 256, dropout_rate) # --- CHANGE: Added dropout
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128 + 128, 128, dropout_rate) # --- CHANGE: Added dropout
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64 + 64, 64, dropout_rate)   # --- CHANGE: Added dropout
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # --- CHANGE: Corrected skip connection channel size from layer0 ---
        self.dec1 = self.conv_block(64 + 64, 32) # No dropout on last block
        self.final_upconv = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    # --- CHANGE: Added dropout_rate parameter ---
    def conv_block(self, in_channels, out_channels, dropout_rate=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # --- CHANGE: Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # --- CHANGE: Added BatchNorm
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate)) # --- CHANGE: Added Dropout layer
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x0_pool = self.layer0(x)
        x1 = self.layer1(x0_pool)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder
        d4 = self.upconv4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        # --- CHANGE: Corrected skip connection from x0_pool ---
        d1 = torch.cat([d1, x0_pool], dim=1)
        d1 = self.dec1(d1)

        d0 = self.final_upconv(d1)
        return self.final_conv(d0)


# --- FIX: Define Dice Loss and IoU Metric ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        # We are interested in the 'sky' class (class 1)
        probs_sky = probs[:, 1, :, :]
        targets_sky = (targets == 1).float()

        intersection = (probs_sky * targets_sky).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs_sky.sum() + targets_sky.sum() + self.smooth)

        return 1 - dice_coeff

def iou_score(outputs, masks):
    # --- CHANGE: Ensure masks are on the same device as outputs ---
    masks = masks.to(outputs.device)
    preds = torch.argmax(outputs, dim=1)
    intersection = ((preds == 1) & (masks == 1)).float().sum()
    union = ((preds == 1) | (masks == 1)).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

model = UNet(n_classes=2, dropout_rate=0.3).to(device)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 # Increased epochs, but will use early stopping
# --- CHANGE: Added weight decay for regularization ---
WEIGHT_DECAY = 1e-5

criterion_ce = nn.CrossEntropyLoss()
criterion_dice = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# --- CHANGE: Added a learning rate scheduler ---
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs):
    print("\n--- Starting Ship-Aware U-Net Training with Regularization & Early Stopping ---")
    best_iou = 0.0
    # --- CHANGE: Added early stopping mechanism ---
    patience = 10
    epochs_no_improve = 0
    best_val_loss = float('inf')
    # --- CHANGE: Updated model save path for ship-aware version ---
    model_save_path = os.path.join(GDRIVE_MODEL_SAVE_PATH, 'best_unet_ship_aware_smd.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)

            # Combined Loss
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = 0.5 * loss_ce + 0.5 * loss_dice # --- CHANGE: Balanced the two losses

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device, dtype=torch.long)
                outputs = model(images)
                loss_ce = criterion_ce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                val_loss += loss.item() * images.size(0)
                # --- CHANGE: Calculate IoU per batch for a more accurate average ---
                for i in range(images.size(0)):
                    val_iou += iou_score(outputs[i].unsqueeze(0), masks[i].unsqueeze(0))


        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset) # Average IoU across all validation images
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # --- CHANGE: Update scheduler and check for early stopping ---
        scheduler.step(val_loss)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Model saved with new best validation IoU: {best_iou:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"--- Early stopping triggered after {patience} epochs with no improvement. ---")
            break

    print(f"--- Finished Training. Best validation IoU: {best_iou:.4f} ---")
    print(f"Best model saved to: {model_save_path}")

# ==============================================================================
# Step 6: Visualization for Segmentation
# ==============================================================================
def visualize_segmentation(model, loader, num_images=5):
    model.eval()
    try:
        images, masks = next(iter(loader))
    except StopIteration:
        print("Cannot visualize. The data loader is empty.")
        return
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)

    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    images = images.cpu()

    sky_color = np.array([0, 0, 255]) # Blue for sky

    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 4))
    if num_images == 1: axes = np.array([axes]) # Ensure axes is iterable for single image
    fig.suptitle("Ground Truth vs. Model Prediction", fontsize=16)

    for i in range(min(num_images, len(images))):
        img_tensor = images[i]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        gt_mask = masks[i].cpu().numpy()
        gt_overlay = np.zeros_like(img_display)
        gt_overlay[gt_mask == 1] = sky_color / 255.0
        gt_viz = cv2.addWeighted(img_display, 0.7, gt_overlay, 0.3, 0)
        axes[i, 0].imshow(gt_viz)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].axis('off')

        pred_mask = preds[i]
        pred_overlay = np.zeros_like(img_display)
        pred_overlay[pred_mask == 1] = sky_color / 255.0
        pred_viz = cv2.addWeighted(img_display, 0.7, pred_overlay, 0.3, 0)
        axes[i, 1].imshow(pred_viz)
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Main Execution Block ---
if 'train_loader' in locals() and 'val_loader' in locals():
    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, NUM_EPOCHS)

    # Load the best model and visualize results on the validation set
    best_model_path = os.path.join(GDRIVE_MODEL_SAVE_PATH, 'best_unet_ship_aware_smd.pth')
    if os.path.exists(best_model_path):
        print("\n--- Visualizing results with the best ship-aware model on the Validation Set ---")
        model.load_state_dict(torch.load(best_model_path))
        visualize_segmentation(model, val_loader)
    else:
        print("\n--- No saved ship-aware model found to visualize. ---")
else:
    print("\n--- ERROR: Training skipped because dataset was not loaded ---")