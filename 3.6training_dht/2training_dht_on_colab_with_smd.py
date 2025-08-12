# ==============================================================================
# Step 1: Setup and Environment
# ==============================================================================
# Install necessary libraries. SciPy is needed for loading .mat files.
!pip install opencv-python-headless scipy

#! the problem: it is limited with a simple line.
#! It's a regression model, not a true segmentation model, and it's not robust enough for your real-world data.
#! Real segmentation requires a model that can classify every single pixel as either "sky" or "sea/land., so i will try u-net now.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import cv2
import os
import zipfile
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
# This will mount your Google Drive to the Colab environment, allowing access
# to your dataset.

drive.mount('/content/drive')

# --- IMPORTANT: SET YOUR PATHS HERE ---
# Update this path to point to the location of your 'VIS_Onshore' folder
# inside your Google Drive.
BASE_DRIVE_PATH = '/content/drive/My Drive/SMD_Dataset' # <-- CHANGE THIS
SMD_VIDEOS_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/Videos')
SMD_GT_PATH = os.path.join(BASE_DRIVE_PATH, 'VIS_Onshore/HorizonGT')

# This is where the processed images and labels will be saved.
PROCESSED_DATA_PATH = os.path.join(BASE_DRIVE_PATH, 'processed_smd')
PROCESSED_IMAGES_PATH = os.path.join(PROCESSED_DATA_PATH, 'images')
PROCESSED_LABELS_PATH = os.path.join(PROCESSED_DATA_PATH, 'labels')

# Create directories if they don't exist
os.makedirs(PROCESSED_IMAGES_PATH, exist_ok=True)
os.makedirs(PROCESSED_LABELS_PATH, exist_ok=True)

print(f"Paths set. Looking for data in: {BASE_DRIVE_PATH}")
if not os.path.exists(SMD_VIDEOS_PATH):
    print("\n--- WARNING ---")
    print(f"The path '{SMD_VIDEOS_PATH}' does not exist.")
    print("Please make sure you have uploaded the 'VIS_Onshore' folder to your Google Drive")
    print("and correctly set the 'BASE_DRIVE_PATH' variable in this script.")
    print("-----------------")


# ==============================================================================
# Step 3: Preprocess the SMD Dataset (Run this cell only once)
# ==============================================================================
# This script extracts frames from the videos and parses the corresponding
# .mat ground truth files to create a clean dataset for training.

def preprocess_smd_data():
    """
    Extracts frames from videos and processes .mat ground truth files.
    """
    print("Starting SMD preprocessing...")
    video_files = sorted([f for f in os.listdir(SMD_VIDEOS_PATH) if f.endswith('.avi')])

    processed_count = 0
    total_videos = len(video_files)
    for i, video_file in enumerate(video_files):
        print(f"\n>>> Processing video {i+1}/{total_videos}: {video_file}")
        video_name_without_ext = os.path.splitext(video_file)[0]
        gt_filename = f"{video_name_without_ext}_HorizonGT.mat"

        video_path = os.path.join(SMD_VIDEOS_PATH, video_file)
        gt_path = os.path.join(SMD_GT_PATH, gt_filename)

        if not os.path.exists(gt_path):
            print(f"  - Skipping: Ground truth file not found.")
            continue

        gt_data = loadmat(gt_path)

        horizon_key = None
        for key in gt_data.keys():
            if not key.startswith('__'):
                horizon_key = key
                break

        if horizon_key is None:
             print(f"  - Skipping: Could not find any valid data key in {gt_filename}.")
             continue

        struct_array = gt_data[horizon_key]

        if struct_array.size == 0:
            print(f"  - Skipping: Annotation struct is empty.")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= struct_array.shape[1]:
                break

            # --- ADDING DEBUGGING to see the exact point of failure ---
            try:
                frame_struct = struct_array[0, frame_idx]

                x_point = frame_struct['X'][0,0]
                y_point = frame_struct['Y'][0,0]
                nx = frame_struct['Nx'][0,0]
                ny = frame_struct['Ny'][0,0]

                # If we get here, extraction was successful for this frame
                # print(f"  - Frame {frame_idx}: Success extracting data.") # Uncomment for extreme verbosity

                theta = np.arctan2(ny, nx)
                rho = x_point * np.cos(theta) + y_point * np.sin(theta)

            except (IndexError, TypeError, KeyError, ValueError) as e:
                # This will now print the *actual* error for the failing frame
                print(f"  - DEBUG: Skipping frame {frame_idx} in {video_file} due to a parsing error: {e}")
                frame_idx += 1
                continue

            frame_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.jpg"
            label_filename = f"{video_name_without_ext}_frame_{frame_idx:04d}.txt"

            cv2.imwrite(os.path.join(PROCESSED_IMAGES_PATH, frame_filename), frame)
            with open(os.path.join(PROCESSED_LABELS_PATH, label_filename), 'w') as f:
                f.write(f"{theta} {rho}")

            frame_idx += 1
            processed_count += 1

        cap.release()

    print(f"\nPreprocessing complete. Total frames processed: {processed_count}")

# Check if preprocessing is needed.
# To force reprocessing, delete the 'processed_smd' folder from your Drive.
if len(os.listdir(PROCESSED_IMAGES_PATH)) == 0:
    preprocess_smd_data()
else:
    print("Processed data already exists. Skipping preprocessing.")


# ==============================================================================
# Step 4: Create the Custom PyTorch Dataset
# ==============================================================================

class SMD_Dataset(Dataset):
    """
    Custom PyTorch dataset for the preprocessed Singapore Maritime Dataset.
    """
    def __init__(self, images_dir, labels_dir, transform=None, img_size=(256, 256)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_size = img_size
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        with open(label_path, 'r') as f:
            theta, rho = map(float, f.read().split())

        w_orig, h_orig = image.size
        h_new, w_new = self.img_size
        rho_scaled = rho * (h_new / h_orig)

        target = torch.tensor([theta, rho_scaled], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target

# --- DataLoaders ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = SMD_Dataset(
    images_dir=PROCESSED_IMAGES_PATH,
    labels_dir=PROCESSED_LABELS_PATH,
    transform=transform,
    img_size=IMG_SIZE
)

if len(full_dataset) > 0:
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\nDataset loaded: {len(full_dataset)} images from the SMD dataset.")
    print(f"Training set: {len(train_dataset)} images, Validation set: {len(val_dataset)} images")
else:
    print("\n--- ERROR ---")
    print("Dataset is empty after preprocessing. Cannot start training.")
    print("Please check your file paths and the contents of your SMD dataset folders.")
    print("To re-run preprocessing, delete the 'processed_smd' folder from your Drive and run this cell again.")
    print("---------------")


# ==============================================================================
# Step 5: Define and Train the Model (Architecture is unchanged)
# ==============================================================================
class DeepHoughTransform(nn.Module):
    def __init__(self, num_theta=180, num_rho=200):
        super(DeepHoughTransform, self).__init__()
        self.backbone = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2])
        self.dht_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        features = self.backbone(x)
        dht_features = self.dht_head(features)
        dht_features = torch.flatten(dht_features, 1)
        return self.regressor(dht_features)

model = DeepHoughTransform().to(device)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("\n--- Starting Training on SMD Dataset ---")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(BASE_DRIVE_PATH, 'best_dht_model_smd.pth'))
            print(f"  -> Model saved with new best validation loss: {best_loss:.6f}")

    print("--- Finished Training ---")

if 'train_loader' in locals() and len(full_dataset) > 0:
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)


# ==============================================================================
# Step 6: Visualization (Unchanged logic)
# ==============================================================================
if 'full_dataset' in locals() and len(full_dataset) > 0 and os.path.exists(os.path.join(BASE_DRIVE_PATH, 'best_dht_model_smd.pth')):
    model.load_state_dict(torch.load(os.path.join(BASE_DRIVE_PATH, 'best_dht_model_smd.pth')))
    model.eval()
    print("\nLoaded best SMD-trained model for inference.")

    def draw_line(image, theta, rho):
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image

else:
    print("\nSkipping visualization because training did not run or model was not saved.")