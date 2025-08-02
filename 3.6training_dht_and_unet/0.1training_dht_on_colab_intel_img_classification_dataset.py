# ==============================================================================
# Step 1: Setup and GPU Verification
# ==============================================================================
# Install necessary libraries. Kaggle for dataset download, OpenCV for image
# manipulation.
!pip install kaggle opencv-python-headless

#! initial attempt of horizon line segmentation, using a simple CNN model. bad results.

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

# Check if a GPU is available and set the device accordingly.
# Google Colab's T4 GPU is perfect for this task.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# Step 2: Download and Prepare the Dataset (Corrected & Improved)
# ==============================================================================
# We will use the "Intel Image Classification" dataset from Kaggle, which has a 'sea' category.
# This is a more stable way to access data than direct download links.

# --- Kaggle API Setup ---
# To use the Kaggle API, you need to upload your 'kaggle.json' file.
# 1. Go to your Kaggle account page: https://www.kaggle.com/account
# 2. Click on "Create New API Token". This will download 'kaggle.json'.
# 3. Upload 'kaggle.json' to your Colab session using the file browser on the left.

if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("Please upload your 'kaggle.json' file to the Colab session.")
    # You can use the file uploader in Colab's file browser (folder icon on the left)
    from google.colab import files
    files.upload()
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
else:
    print("Kaggle API token already found.")


# --- Download and Extract Dataset ---
DATASET_NAME = "puneet6060/intel-image-classification"
DATASET_DIR = "intel-image-classification"
SEA_IMAGES_DIR = os.path.join(DATASET_DIR, 'seg_train/seg_train/sea')

if not os.path.exists(DATASET_DIR):
    print(f"Downloading dataset '{DATASET_NAME}'...")
    !kaggle datasets download -d {DATASET_NAME} -p {DATASET_DIR} --unzip
    print("Dataset ready.")
else:
    print("Dataset already downloaded.")


# --- Custom PyTorch Dataset Class with Improved Annotation ---
class HorizonDataset(Dataset):
    """
    Custom dataset for loading maritime images.
    It now uses a CV-based method to generate more realistic "pseudo-labels"
    for the horizon line, which is a significant improvement over random generation.
    """
    def __init__(self, image_dir, transform=None, img_size=(256, 256)):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # --- Improved Annotation Generation using OpenCV ---
        # This "pseudo-labeling" approach finds the most likely horizon line.
        original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_orig, w_orig = original_image_cv.shape[:2]
        
        gray = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # **FIX:** Crop the edge map to the central vertical region to avoid noise
        # from the sky and foreground water. We'll search for the horizon in the
        # middle 50% of the image.
        top_crop = h_orig // 4
        bottom_crop = h_orig - (h_orig // 4)
        cropped_edges = edges[top_crop:bottom_crop, :]

        lines = cv2.HoughLines(cropped_edges, 1, np.pi / 180, threshold=80)

        best_line = None
        if lines is not None:
            # Find the line that is most horizontal
            best_diff = float('inf')
            for line in lines:
                rho, theta = line[0]
                # We want theta close to pi/2 (90 degrees)
                diff = abs(theta - np.pi / 2)
                # Only consider lines within +/- 15 degrees of horizontal
                if diff < best_diff and diff < np.deg2rad(15):
                    best_diff = diff
                    # **FIX:** Adjust rho by adding the crop offset
                    best_line = (rho + top_crop, theta)

        if best_line is not None:
            # We found a good line, use its parameters
            rho, theta = best_line
            # Adjust rho for the final resized image
            h_new, w_new = self.img_size
            rho = rho * (h_new / h_orig)
        else:
            # Fallback to synthetic generation if no good line is found
            h, w = self.img_size
            theta = (np.pi / 2.0) + np.random.uniform(-0.05, 0.05)
            rho = (h / 2) + np.random.uniform(-h/8, h/8)

        target = torch.tensor([theta, rho], dtype=torch.float32)

        # Apply transformations AFTER generating the label from the original
        if self.transform:
            image = self.transform(image)

        return image, target

# --- DataLoaders ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

# Define transformations: resize images and convert them to PyTorch tensors.
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = HorizonDataset(image_dir=SEA_IMAGES_DIR, transform=transform, img_size=IMG_SIZE)

# Split into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Dataset loaded: {len(full_dataset)} images from the 'sea' category.")
print(f"Training set: {len(train_dataset)} images, Validation set: {len(val_dataset)} images")


# ==============================================================================
# Step 3: Define the Deep Hough Transform (DHT) Model
# ==============================================================================
class DeepHoughTransform(nn.Module):
    """
    A simplified Deep Hough Transform model.
    It uses a pre-trained CNN backbone to extract features and then regresses
    the parameters (theta, rho) of the dominant line.
    """
    def __init__(self, num_theta=180, num_rho=200):
        super(DeepHoughTransform, self).__init__()
        # Use a pre-trained ResNet-18 as the feature extractor
        # We remove the final fully connected layer (the classifier)
        self.backbone = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2])

        # The output of ResNet-18's feature layers is 512 channels.
        # We add a small convolutional head to process these features.
        self.dht_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # Global average pooling
        )

        # A final regressor to predict the two line parameters
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output: [theta, rho]
        )

    def forward(self, x):
        # Pass input through the backbone
        features = self.backbone(x)
        # Pass features through the DHT head
        dht_features = self.dht_head(features)
        # Flatten the features for the regressor
        dht_features = torch.flatten(dht_features, 1)
        # Predict the line parameters
        line_params = self.regressor(dht_features)
        return line_params

# Instantiate the model and move it to the GPU
model = DeepHoughTransform().to(device)
print("Model created and moved to GPU.")
# print(model) # Uncomment to see model architecture

# ==============================================================================
# Step 4: Training the Model
# ==============================================================================
# --- Hyperparameters ---
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25 # Increased epochs for better convergence with real data

# --- Loss Function and Optimizer ---
# We use Mean Squared Error loss as we are regressing continuous values.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Main training loop for the DHT model.
    """
    print("\n--- Starting Training ---")
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.6f}")

        # --- Validation ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.6f}")

        # Save the model if validation loss has decreased
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_dht_model.pth')
            print(f"Model saved with new best validation loss: {best_loss:.6f}")

    print("--- Finished Training ---")

# Start the training process
train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)


# ==============================================================================
# Step 5: Inference and Visualization
# ==============================================================================
# Load the best performing model
model.load_state_dict(torch.load('best_dht_model.pth'))
model.eval()
print("\nLoaded best model for inference.")

def draw_line(image, theta, rho):
    """Draws a line on an image given theta and rho."""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Calculate two points on the line
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    # Draw the line on the image
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red line with thickness 2
    return image

def visualize_predictions(model, dataset, num_images=5):
    """
    Runs inference on a few images and displays the results.
    """
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 4))
    fig.suptitle("Model Predictions vs. Ground Truth", fontsize=16)

    # Get a few random samples from the validation set
    indices = np.random.choice(len(dataset.indices), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        # Get image and ground truth target from the original dataset using the split index
        image_tensor, target_params = dataset.dataset[dataset.indices[idx]]
        
        # Prepare image for model
        input_tensor = image_tensor.unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            predicted_params = model(input_tensor).cpu().squeeze()

        # --- Convert Tensors to NumPy for visualization ---
        # Un-normalize the image for display
        img_display = image_tensor.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        # Convert to uint8 for OpenCV
        img_display_bgr = (img_display * 255).astype(np.uint8)[:, :, ::-1].copy() # RGB to BGR

        # --- Draw Ground Truth Line ---
        gt_theta, gt_rho = target_params.numpy()
        img_with_gt_line = draw_line(img_display_bgr.copy(), gt_theta, gt_rho)
        axes[i, 0].imshow(img_with_gt_line[:, :, ::-1]) # BGR to RGB for plotting
        axes[i, 0].set_title(f"Ground Truth (Pseudo-Label)\nθ={gt_theta:.2f}, ρ={gt_rho:.1f}")
        axes[i, 0].axis('off')

        # --- Draw Predicted Line ---
        pred_theta, pred_rho = predicted_params.numpy()
        img_with_pred_line = draw_line(img_display_bgr.copy(), pred_theta, pred_rho)
        axes[i, 1].imshow(img_with_pred_line[:, :, ::-1]) # BGR to RGB for plotting
        axes[i, 1].set_title(f"Prediction\nθ={pred_theta:.2f}, ρ={pred_rho:.1f}")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Run visualization on the validation dataset
visualize_predictions(model, val_dataset, num_images=5)