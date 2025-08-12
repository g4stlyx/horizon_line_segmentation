# ============================================================================== 
# Step 1: Setup and Environment - Ship-Aware U-Net Training with HF RT-DETR
# ==============================================================================
# Install necessary libraries (Hugging Face Transformers RT-DETR)
!pip install -q opencv-python-headless scipy transformers timm accelerate safetensors

"""
Ship-Aware 3-Class U-Net Training for Maritime Horizon Detection with RT-DETR-based Object Detection

This script trains a U-Net for 3-class semantic segmentation:
- Class 0: Water (includes land/shore as water for horizon extraction purposes)
- Class 1: Sky
- Class 2: Object (ships, buoys, vessels, etc.)

Key ideas:
- Horizon GT provides the sky (above) vs water (below) split. Objects detected by RT-DETR override these labels to class 2.
- This ensures the model learns the true horizon line and does not "hug" ship silhouettes.
- RT-DETR is only used to build better masks; the model itself predicts all three classes.

Benefits over color heuristics:
- Better object localization around the horizon
- Robust in varying lighting and partial occlusions
- Prevents ships above the horizon from being labeled as sky
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

from transformers import AutoImageProcessor, AutoModelForObjectDetection
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

SMD_VIDEOS_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/Videos')
SMD_GT_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/HorizonGT')
GDRIVE_MODEL_SAVE_PATH = os.path.join(BASE_DRIVE_PATH, 'models')
GDRIVE_PROCESSED_IMAGES_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_unet_rtdetr_aware_3_class/images')
GDRIVE_PROCESSED_MASKS_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_unet_rtdetr_aware_3_class/masks')

GDRIVE_PROCESSED_DATA_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_unet_rtdetr_aware_3_class')
os.makedirs(GDRIVE_MODEL_SAVE_PATH, exist_ok=True)

# Local paths for fast access during training
LOCAL_DATA_PATH = '/content/processed_unet_rtdetr_aware_3_class'
LOCAL_IMAGES_PATH = os.path.join(LOCAL_DATA_PATH, 'images')
LOCAL_MASKS_PATH = os.path.join(LOCAL_DATA_PATH, 'masks')

os.makedirs(GDRIVE_PROCESSED_IMAGES_PATH, exist_ok=True)
os.makedirs(GDRIVE_PROCESSED_MASKS_PATH, exist_ok=True)

print(f"Google Drive processed data path: {GDRIVE_PROCESSED_DATA_PATH}")
print(f"Google Drive model save path: {GDRIVE_MODEL_SAVE_PATH}")
print(f"Local runtime data path: {LOCAL_DATA_PATH}")

# ==============================================================================
# Step 3: Initialize RT-DETR Model for Ship Detection
# ==============================================================================
print("Loading HF RT-DETR model for ship detection...")

# Path to your RT-DETR checkpoint folder on Drive (must contain config.json and model.safetensors)
# EXAMPLE: '/content/drive/My Drive/SMD_Dataset/models/rtdetr/final_best_model'
GDRIVE_RTDETR_MODEL_DIR = '/content/drive/My Drive/SMD_Dataset/models/rtdetr/final_best_model'

# Ship-relevant class ids (adapt to your checkpoint). Example from provided config:
# 0 Boat, 1 Buoy, 2 Ferry, 3 Flying bird-plane, 4 Kayak, 5 Other, 6 Sail boat, 7 Speed boat, 8 Vessel-ship
SHIP_CLASSES = [0, 1, 2, 4, 6, 7, 8]

try:
    rtdetr_processor = AutoImageProcessor.from_pretrained(GDRIVE_RTDETR_MODEL_DIR)
    rtdetr_model = AutoModelForObjectDetection.from_pretrained(GDRIVE_RTDETR_MODEL_DIR)
    rtdetr_model.eval()
    rtdetr_model.to(device)
    print("HF RT-DETR model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load HF RT-DETR from '{GDRIVE_RTDETR_MODEL_DIR}': {e}")
    rtdetr_processor = None
    rtdetr_model = None

# ==============================================================================
# Step 4: RT-DETR-based Ship Detection Function
# ==============================================================================
def detect_ships_with_rtdetr(frame, horizon_y, hf_processor, hf_model, confidence_threshold=0.3, ship_classes=SHIP_CLASSES):
    """
    Detect ships/objects using a Hugging Face RT-DETR checkpoint.
    Returns a binary mask where detected object pixels are 255.

    horizon_y is used to optionally filter detections to a band near the horizon,
    which reduces false positives far from the horizon (dataset-specific choice).
    """
    h, w = frame.shape[:2]
    ship_mask = np.zeros((h, w), dtype=np.uint8)

    if hf_processor is None or hf_model is None:
        return ship_mask

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = hf_processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = hf_model(**{k: v.to(hf_model.device) for k, v in inputs.items()})
    target_sizes = torch.tensor([(h, w)], device=hf_model.device)
    results = hf_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]

    boxes = results.get("boxes")
    labels = results.get("labels")
    if boxes is None:
        return ship_mask

    boxes = boxes.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype(int)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        class_id = int(labels[i])
        if ship_classes and class_id not in ship_classes:
            continue
        ship_bottom = int(y2)
        ship_top = int(y1)
        # keep boxes intersecting a band around horizon
        if ship_bottom > horizon_y - 50 or (ship_top < horizon_y + 50 and ship_bottom > horizon_y - 100):
            pad = 5
            x1_pad = max(0, int(x1) - pad)
            y1_pad = max(0, int(y1) - pad)
            x2_pad = min(w, int(x2) + pad)
            y2_pad = min(h, int(y2) + pad)
            ship_mask[y1_pad:y2_pad, x1_pad:x2_pad] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ship_mask = cv2.morphologyEx(ship_mask, cv2.MORPH_CLOSE, kernel)
    return ship_mask

def create_rtdetr_aware_mask(frame, horizon_y, hf_processor, hf_model):
    """
    Create a RT-DETR-aware 3-class ground truth mask.

    Classes:
    - 0: Water (and land/shore treated as water)
    - 1: Sky
    - 2: Object (ships/buoys etc.)
    """
    h, w, _ = frame.shape
    
    # Initialize with sky/water by horizon line
    mask = np.zeros((h, w), dtype=np.uint8)  # default water = 0
    mask[:horizon_y, :] = 1  # sky = 1
    
    # Detect objects using RT-DETR and override to class 2
    ship_mask = detect_ships_with_rtdetr(frame, horizon_y, hf_processor, hf_model)
    mask[ship_mask == 255] = 2
    
    return mask

# ==============================================================================
# Step 5: RT-DETR-Aware Preprocess Data for U-Net
# ==============================================================================
def preprocess_smd_with_rtdetr():
    """
    Extracts frames and creates RTDETR-aware ground truth segmentation masks.
    Saves the output to your Google Drive.
    """
    print("Starting RT-DETR-Aware SMD preprocessing for 3-Class Segmentation (water/sky/object)...")
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
                
                # Create RT-DETR-aware 3-class mask (0=water,1=sky,2=object)
                mask = create_rtdetr_aware_mask(frame, horizon_y, rtdetr_processor, rtdetr_model)
                
                # Skip frames with invalid masks
                if mask.min() == mask.max():
                    frame_idx += 1
                    continue
                
                # Count ships detected
                ship_mask = detect_ships_with_rtdetr(frame, horizon_y, rtdetr_processor, rtdetr_model)
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
                # BGR overlay: sky (blue), water (green), object (orange)
                overlay[mask == 1] = [255, 0, 0]    # sky -> blue
                overlay[mask == 0] = [0, 255, 0]    # water -> green
                overlay[mask == 2] = [0, 165, 255]  # object -> orange
                vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
                cv2.line(vis_frame, (0, horizon_y), (w, horizon_y), (255, 255, 0), 2)
                
                # Draw RT-DETR bounding boxes
                try:
                    inputs = rtdetr_processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt")
                    with torch.no_grad():
                        outputs = rtdetr_model(**{k: v.to(rtdetr_model.device) for k, v in inputs.items()})
                    target_sizes = torch.tensor([(h, w)], device=rtdetr_model.device)
                    det = rtdetr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
                    if det and 'boxes' in det:
                        b = det['boxes'].cpu().numpy().astype(int)
                        s = det['scores'].cpu().numpy()
                        l = det['labels'].cpu().numpy().astype(int)
                        for (x1, y1, x2, y2), conf, cls in zip(b, s, l):
                            if SHIP_CLASSES and cls not in SHIP_CLASSES:
                                continue
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(vis_frame, f"Ship {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                except Exception:
                    pass
                
                debug_filename = f"debug_{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(os.path.join(GDRIVE_PROCESSED_IMAGES_PATH, debug_filename), vis_frame)
            
            frame_idx += 1
            processed_count += 1
            
            # Process every 10th frame to speed up (optional)
            # frame_idx += 10
        
        cap.release()
    
    print(f"RT-DETR-Aware preprocessing complete. Total frames processed: {processed_count}")
    print(f"Frames with ships detected: {ships_detected_count} ({ships_detected_count/processed_count*100:.1f}%)")

# Check if the processed folder on Drive is empty. If so, run preprocessing.
if not os.listdir(GDRIVE_PROCESSED_IMAGES_PATH):
    print("Processed data not found on Google Drive. Running RT-DETR preprocessing. This may take a while.")
    preprocess_smd_with_rtdetr()
else:
    print("Processed data found on Google Drive.")

# Copy the entire folder from Drive to the local runtime
if not os.path.exists(LOCAL_DATA_PATH):
    print(f"Copying data from '{GDRIVE_PROCESSED_DATA_PATH}' to local runtime '{LOCAL_DATA_PATH}'...")
    !cp -r "{GDRIVE_PROCESSED_DATA_PATH}" "{LOCAL_DATA_PATH}"
    print("Data successfully copied to local runtime.")
else:
    print("Data already exists in local runtime. Skipping copy.")

# ==============================================================================
# The rest of the code remains the same (Dataset, Model, Training)
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

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
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

            # Random affine transformation
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
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long()

        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask_tensor

# Implement Video-Based Splitting
all_image_files = sorted([os.path.join(LOCAL_IMAGES_PATH, f) for f in os.listdir(LOCAL_IMAGES_PATH)
                         if f.endswith('.jpg') and not f.startswith('debug_')])

# Validate dataset
def validate_dataset(image_files, masks_dir):
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
        for mask in missing_masks[:5]:
            print(f"  - {mask}")
        if len(missing_masks) > 5:
            print(f"  ... and {len(missing_masks) - 5} more")
        print(f"Using {len(valid_files)} valid image-mask pairs out of {len(image_files)} total images.")

    return valid_files

all_image_files = validate_dataset(all_image_files, LOCAL_MASKS_PATH)

if all_image_files:
    video_groups = collections.defaultdict(list)
    for f in all_image_files:
        prefix = '_'.join(os.path.basename(f).split('_')[:-2])
        video_groups[prefix].append(f)

    unique_videos = list(video_groups.keys())
    random.seed(42)
    random.shuffle(unique_videos)

    split_idx_1 = int(0.8 * len(unique_videos))
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

# U-Net Model Definition (3-class)
class UNet(nn.Module):
    def __init__(self, n_classes=3, dropout_rate=0.3):
        super().__init__()
        self.n_classes = n_classes

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(resnet.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256 + 256, 256, dropout_rate)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128 + 128, 128, dropout_rate)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64 + 64, 64, dropout_rate)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64 + 64, 32)
        self.final_upconv = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_rate=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
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
        d1 = torch.cat([d1, x0_pool], dim=1)
        d1 = self.dec1(d1)

        d0 = self.final_upconv(d1)
        return self.final_conv(d0)

# Losses and Metrics for 3-class
class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        total_dice = 0.0
        for c in range(self.num_classes):
            p_c = probs[:, c, :, :]
            t_c = (targets == c).float()
            intersection = (p_c * t_c).sum()
            denom = p_c.sum() + t_c.sum()
            dice_c = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            total_dice += dice_c
        return 1.0 - (total_dice / self.num_classes)

def mean_iou(outputs, masks, num_classes=3):
    masks = masks.to(outputs.device)
    preds = torch.argmax(outputs, dim=1)
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (masks == c)).float().sum()
        union = ((preds == c) | (masks == c)).float().sum()
        iou_c = (inter + 1e-6) / (union + 1e-6)
        ious.append(iou_c)
    return torch.stack(ious).mean().item()

model = UNet(n_classes=3, dropout_rate=0.3).to(device)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5

class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)  # objects are rarer
criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
criterion_dice = MultiClassDiceLoss(num_classes=3)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs):
    print("\n--- Starting RT-DETR-Aware U-Net Training ---")
    best_iou = 0.0
    patience = 10
    epochs_no_improve = 0
    best_val_loss = float('inf')
    model_save_path = os.path.join(GDRIVE_MODEL_SAVE_PATH, 'best_unet_rtdetr_aware_smd_3cls.pth')

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
            loss = 0.5 * loss_ce + 0.5 * loss_dice

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
                val_iou += mean_iou(outputs, masks, num_classes=3) * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

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

# Visualization Function
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

    # RGB colors for visualization
    sky_rgb = np.array([0, 153, 255])     # light blue
    water_rgb = np.array([0, 255, 102])   # greenish
    obj_rgb = np.array([255, 140, 0])     # orange

    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 4))
    if num_images == 1: axes = np.array([axes])
    fig.suptitle("Ground Truth vs. Model Prediction (3-Class RT-DETR-Aware)", fontsize=16)

    for i in range(min(num_images, len(images))):
        img_tensor = images[i]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        gt_mask = masks[i].cpu().numpy()
        gt_overlay = np.zeros_like(img_display)
        gt_overlay[gt_mask == 1] = sky_rgb / 255.0
        gt_overlay[gt_mask == 0] = water_rgb / 255.0
        gt_overlay[gt_mask == 2] = obj_rgb / 255.0
        gt_viz = cv2.addWeighted(img_display, 0.7, gt_overlay, 0.3, 0)
        axes[i, 0].imshow(gt_viz)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].axis('off')

        pred_mask = preds[i]
        pred_overlay = np.zeros_like(img_display)
        pred_overlay[pred_mask == 1] = sky_rgb / 255.0
        pred_overlay[pred_mask == 0] = water_rgb / 255.0
        pred_overlay[pred_mask == 2] = obj_rgb / 255.0
        pred_viz = cv2.addWeighted(img_display, 0.7, pred_overlay, 0.3, 0)
        axes[i, 1].imshow(pred_viz)
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Main Execution Block
if 'train_loader' in locals() and 'val_loader' in locals():
    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, NUM_EPOCHS)

    # Load the best model and visualize results
    best_model_path = os.path.join(GDRIVE_MODEL_SAVE_PATH, 'best_unet_rtdetr_aware_smd_3cls.pth')
    if os.path.exists(best_model_path):
        print("\n--- Visualizing results with the best 3-class RT-DETR-aware model ---")
        model.load_state_dict(torch.load(best_model_path))
        visualize_segmentation(model, val_loader)
    else:
        print("\n--- No saved RTDETR-aware model found to visualize. ---")
else:
    print("\n--- ERROR: Training skipped because dataset was not loaded ---")
