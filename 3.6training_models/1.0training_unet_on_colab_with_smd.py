# ==============================================================================
# Step 1: Setup and Environment
# ==============================================================================
# Install necessary libraries. SciPy is needed for loading .mat files.
!pip install opencv-python-headless scipy

#! does nothing, will look at it again, this is the problem:
"""
Preprocessing complete. Total frames processed: 17222

Dataset loaded: 17222 images.
Training set: 13777, Validation set: 3445

--- Starting U-Net Training ---
Epoch 1/25 -> Train Loss: 0.0133, Val Loss: 0.0000
  -> Model saved with new best validation loss: 0.0000
!Epoch 2/25 -> Train Loss: 0.0000, Val Loss: 0.0000
  -> Model saved with new best validation loss: 0.0000
!Epoch 3/25 -> Train Loss: 0.0000, Val Loss: 0.0000
  -> Model saved with new best validation loss: 0.0000

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
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
BASE_DRIVE_PATH = '/content/drive/My Drive/SMD_Dataset' # <-- CHANGE THIS
SMD_VIDEOS_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/Videos')
SMD_GT_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/HorizonGT')

# This is where the processed images and labels (now as masks) will be saved.
PROCESSED_DATA_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_smd_unet')
PROCESSED_IMAGES_PATH = os.path.join(PROCESSED_DATA_PATH, 'images')
PROCESSED_MASKS_PATH = os.path.join(PROCESSED_DATA_PATH, 'masks')

os.makedirs(PROCESSED_IMAGES_PATH, exist_ok=True)
os.makedirs(PROCESSED_MASKS_PATH, exist_ok=True)

print(f"Paths set. Looking for data in: {BASE_DRIVE_PATH}")

# ==============================================================================
# Step 3: Preprocess Data for U-Net (Creating Segmentation Masks)
# ==============================================================================

# --- DEBUGGING FLAG ---
# Set to True to visualize the first few generated masks and stop.
# Set to False to run the full preprocessing.
# FIX: Set to False to process the entire dataset for proper training.
DEBUG_PREPROCESSING = False

def preprocess_smd_for_segmentation():
    """
    Extracts frames and creates ground truth segmentation masks using the
    precise normal vector and point data from the .mat files.
    """
    print("Starting SMD preprocessing for Segmentation...")
    video_files = sorted([f for f in os.listdir(SMD_VIDEOS_PATH) if f.endswith('.avi')])

    processed_count = 0
    total_videos = len(video_files)

    # Lists to hold debug data
    debug_images = []
    debug_masks = []

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

        if horizon_key is None: continue
        struct_array = gt_data[horizon_key]
        if struct_array.size == 0: continue

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx >= struct_array.shape[1]: break

            try:
                frame_struct = struct_array[0, frame_idx]

                x_point = float(frame_struct['X'][0,0])
                y_point = float(frame_struct['Y'][0,0])
                nx = float(frame_struct['Nx'][0,0])
                ny = float(frame_struct['Ny'][0,0])

                h, w, _ = frame.shape

                if abs(ny) < 1e-6:
                    y_left = y_point
                    y_right = y_point
                else:
                    c = -(nx * x_point + ny * y_point)
                    y_left = -c / ny
                    y_right = -(c + nx * (w - 1)) / ny

                mask = np.zeros((h, w), dtype=np.uint8)

                poly_points = np.array([
                    [0, 0], [w - 1, 0], [w - 1, y_right], [0, y_left]
                ], dtype=np.int32)

                cv2.fillPoly(mask, [poly_points], 1)

                # --- QUALITY CONTROL ---
                # Check if the mask is all zeros or all ones. If so, discard it.
                if mask.min() == mask.max():
                    # print(f"  - Discarding frame {frame_idx}: Mask is trivial (all {mask.min()}).")
                    frame_idx += 1
                    continue

            except (IndexError, TypeError, KeyError, ValueError) as e:
                frame_idx += 1
                continue

            # If debugging, store the frame and mask for visualization
            if DEBUG_PREPROCESSING:
                if len(debug_images) < 5: # Store up to 5 examples
                    debug_images.append(frame)
                    debug_masks.append(mask)

            # Save the original frame and the generated mask
            frame_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
            mask_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.png"

            cv2.imwrite(os.path.join(PROCESSED_IMAGES_PATH, frame_filename), frame)
            cv2.imwrite(os.path.join(PROCESSED_MASKS_PATH, mask_filename), mask)

            frame_idx += 1
            processed_count += 1

        cap.release()

        # If debugging, stop after the first video
        if DEBUG_PREPROCESSING:
            break

    print(f"Preprocessing complete. Total frames processed: {processed_count}")

    # --- VISUALIZE DEBUG DATA ---
    if DEBUG_PREPROCESSING and debug_images:
        print("\n--- Preprocessing Debug Visualization ---")
        fig, axes = plt.subplots(len(debug_images), 2, figsize=(10, len(debug_images) * 4))
        fig.suptitle("Sample Frames and Generated Masks", fontsize=16)
        for i in range(len(debug_images)):
            # Display original frame
            axes[i, 0].imshow(cv2.cvtColor(debug_images[i], cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f"Original Frame {i+1}")
            axes[i, 0].axis('off')
            # Display generated mask
            axes[i, 1].imshow(debug_masks[i], cmap='gray')
            axes[i, 1].set_title(f"Generated Mask {i+1}")
            axes[i, 1].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Run preprocessing only if the directory is empty
if len(os.listdir(PROCESSED_IMAGES_PATH)) == 0:
    preprocess_smd_for_segmentation()
else:
    print("Processed data already exists. Skipping preprocessing.")

# ==============================================================================
# Step 4: Create the Dataset for Segmentation
# ==============================================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 8 # Smaller batch size for segmentation models due to memory usage

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.size = size
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '.png')

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask.squeeze(0).long()

full_dataset = SegmentationDataset(PROCESSED_IMAGES_PATH, PROCESSED_MASKS_PATH, IMG_SIZE)

if len(full_dataset) > 0:
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\nDataset loaded: {len(full_dataset)} images.")
    print(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
else:
    print("\n--- ERROR: Dataset is empty ---")

# ==============================================================================
# Step 5: Define U-Net Model and Training Loop
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes

        # Encoder (using ResNet-18)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(resnet.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # 64 channels
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # 64 channels
        self.layer2 = self.base_layers[5] # 128 channels
        self.layer3 = self.base_layers[6] # 256 channels
        self.layer4 = self.base_layers[7] # 512 channels

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256 + 256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128 + 128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64 + 64, 32)

        self.final_upconv = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
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
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        out = self.final_upconv(d1)
        return self.final_conv(out)

model = UNet(n_classes=2).to(device)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("\n--- Starting U-Net Training ---")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(BASE_DRIVE_PATH, 'best_unet_model_smd.pth'))
            print(f"  -> Model saved with new best validation loss: {best_loss:.4f}")

    print("--- Finished Training ---")

if 'train_loader' in locals() and len(full_dataset) > 0:
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

# ==============================================================================
# Step 6: Visualization for Segmentation
# ==============================================================================
def visualize_segmentation(model, loader, num_images=5):
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)

    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    images = images.cpu()

    sky_color = np.array([0, 0, 255]) # Blue for sky

    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 4))
    fig.suptitle("Ground Truth vs. Model Prediction", fontsize=16)

    for i in range(num_images):
        # Un-normalize image for display
        img_tensor = images[i]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = (img_tensor * std + mean).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        # --- Ground Truth ---
        gt_mask = masks[i].cpu().numpy()
        gt_overlay = np.zeros_like(img_display)
        gt_overlay[gt_mask == 1] = sky_color / 255.0
        gt_viz = cv2.addWeighted(img_display, 0.7, gt_overlay, 0.3, 0)
        axes[i, 0].imshow(gt_viz)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].axis('off')

        # --- Prediction ---
        pred_mask = preds[i]
        pred_overlay = np.zeros_like(img_display)
        pred_overlay[pred_mask == 1] = sky_color / 255.0
        pred_viz = cv2.addWeighted(img_display, 0.7, pred_overlay, 0.3, 0)
        axes[i, 1].imshow(pred_viz)
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if 'val_loader' in locals() and len(full_dataset) > 0:
    # Load the best model for visualization
    model.load_state_dict(torch.load(os.path.join(BASE_DRIVE_PATH, 'best_unet_model_smd.pth')))
    visualize_segmentation(model, val_loader)
