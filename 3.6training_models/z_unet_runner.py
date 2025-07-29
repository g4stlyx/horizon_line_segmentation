# ==============================================================================
# Local U-Net Horizon Segmentation Inference Script
# ==============================================================================
#
# How to Run:
# 1. Make sure you have the required libraries:
#    pip install torch torchvision opencv-python numpy pillow
#
# 2. Save this script as `run_unet_inference.py`.
#
# 3. Place the `best_unet_model_smd.pth` file you downloaded from Colab
#    in the same directory as this script.
#
# 4. Run from your terminal with one of the following options:
#
#    - To process a single image:
#      python run_unet_inference.py --image /path/to/your/image.jpg
#
#    - To process a folder of images (batch processing):
#      python run_unet_inference.py --folder /path/to/your/image_folder
#
#    - To process a video file:
#      python run_unet_inference.py --video /path/to/your/video.mp4
#
#    - To use a live camera feed:
#      python run_unet_inference.py --camera
#
#    - To save the output (add the --save flag):
#      python run_unet_inference.py --image image.jpg --save
#      (Saves to image_segmented.jpg)
#      python run_unet_inference.py --folder ./images --save
#      (Saves segmented images in a new 'output' folder)
#
# ==============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
import argparse
import time
import os
import glob

# ==============================================================================
# Step 1: Define the U-Net Model Architecture
# ==============================================================================
# This class definition must be identical to the one used for training.
class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        
        # Encoder (using a pre-trained ResNet-18)
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

# ==============================================================================
# Step 2: Helper Functions and Setup
# ==============================================================================

def create_overlay(image, mask):
    """Creates a transparent overlay for visualization."""
    sky_color = np.array([0, 0, 255], dtype=np.uint8) # Blue for sky
    
    # Create a color overlay from the mask
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask == 1] = sky_color
    
    # Blend the overlay with the original image
    # The alpha channel controls the transparency of the overlay
    viz_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    return viz_image

def predict(model, image, device, transform):
    """Runs a single prediction on a PIL image and returns the mask."""
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Get the predicted class for each pixel
    preds = torch.argmax(outputs, dim=1).cpu().squeeze(0)
    
    # Resize mask back to original image size
    mask_transform = transforms.Resize(original_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)
    mask_resized = mask_transform(preds.unsqueeze(0)).squeeze(0)
    
    return mask_resized.numpy()

# --- Main Setup ---
parser = argparse.ArgumentParser(description="Horizon Segmentation Inference")
parser.add_argument('--image', type=str, help='Path to a single image file.')
parser.add_argument('--folder', type=str, help='Path to a folder with images.')
parser.add_argument('--video', type=str, help='Path to a video file.')
parser.add_argument('--camera', action='store_true', help='Use live camera feed.')
parser.add_argument('--save', action='store_true', help='Save the output.')
args = parser.parse_args()

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model ---
MODEL_PATH = '1.0unet_model_smd.pth'
model = UNet(n_classes=2).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()
model.eval()
print("Model loaded successfully.")

# --- Define Image Transformations ---
IMG_SIZE = (256, 256)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# Step 3: Inference Logic
# ==============================================================================

if args.image:
    # --- Process a single image ---
    try:
        image_pil = Image.open(args.image).convert('RGB')
        mask = predict(model, image_pil, device, transform)
        
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        result_image = create_overlay(image_cv, mask)

        if args.save:
            save_path = os.path.splitext(args.image)[0] + "_segmented.jpg"
            cv2.imwrite(save_path, result_image)
            print(f"Saved segmented image to: {save_path}")

        cv2.imshow('Horizon Segmentation', result_image)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError:
        print(f"Error: Image file not found at '{args.image}'")

elif args.folder:
    # --- Batch process a folder of images ---
    image_paths = glob.glob(os.path.join(args.folder, '*.jpg')) + \
                  glob.glob(os.path.join(args.folder, '*.png'))
    
    if not image_paths:
        print(f"No images found in folder: {args.folder}")
        exit()

    if args.save:
        output_folder = os.path.join(os.path.dirname(args.folder), "output_segmented")
        os.makedirs(output_folder, exist_ok=True)
        print(f"Saving results to: {output_folder}")

    for image_path in image_paths:
        try:
            print(f"Processing: {os.path.basename(image_path)}")
            image_pil = Image.open(image_path).convert('RGB')
            mask = predict(model, image_pil, device, transform)
            
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            result_image = create_overlay(image_cv, mask)

            if args.save:
                save_path = os.path.join(output_folder, os.path.basename(image_path))
                cv2.imwrite(save_path, result_image)

            cv2.imshow('Horizon Segmentation', result_image)
            # Display each for 100ms, press 'q' to stop early
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Could not process {image_path}. Error: {e}")
    cv2.destroyAllWindows()


elif args.video or args.camera:
    # --- Process a video file or camera feed ---
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else: # args.camera
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()
        
    video_writer = None
    if args.save:
        # Get video properties for the writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
        save_path = "output_segmented.mp4"
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"Saving output video to {save_path}")

    print("Processing video. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = predict(model, frame_pil, device, transform)
        result_frame = create_overlay(frame, mask)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if video_writer:
            video_writer.write(result_frame)

        cv2.imshow('Horizon Segmentation', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

else:
    print("No input source specified. Please use --image, --folder, --video, or --camera.")
    parser.print_help()
