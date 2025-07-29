# ==============================================================================
# Local Horizon Line Detection Inference Script
# ==============================================================================
#
# How to Run:
# 1. Make sure you have the required libraries:
#    pip install torch torchvision opencv-python numpy
#
# 2. Save this script as `run_inference.py`.
#
# 3. Place the `best_dht_model.pth` file you downloaded from Colab
#    in the same directory as this script.
#
# 4. Run from your terminal with one of the following options:
#
#    - To process a single image:
#      python run_inference.py --image /path/to/your/image.jpg
#
#    - To process a video file:
#      python run_inference.py --video /path/to/your/video.mp4
#
#    - To use a live camera feed (usually the default webcam):
#      python run_inference.py --camera
#
#    - To save the output (works with any input source):
#      python run_inference.py --image /path/to/image.jpg --save /path/to/output.jpg
#      python run_inference.py --video /path/to/video.mp4 --save /path/to/output.mp4
#      python run_inference.py --camera --save /path/to/output.mp4
#
# ==============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import time

# ==============================================================================
# Step 1: Define the Model Architecture
# ==============================================================================
# This class definition must be identical to the one used for training in Colab.
class DeepHoughTransform(nn.Module):
    """
    A simplified Deep Hough Transform model.
    It uses a pre-trained CNN backbone to extract features and then regresses
    the parameters (theta, rho) of the dominant line.
    """
    def __init__(self, num_theta=180, num_rho=200):
        super(DeepHoughTransform, self).__init__()
        # Use a pre-trained ResNet-18 as the feature extractor
        self.backbone = nn.Sequential(*list(models.resnet18(weights=None).children())[:-2])

        # The output of ResNet-18's feature layers is 512 channels.
        self.dht_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # A final regressor to predict the two line parameters
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output: [theta, rho]
        )

    def forward(self, x):
        features = self.backbone(x)
        dht_features = self.dht_head(features)
        dht_features = torch.flatten(dht_features, 1)
        line_params = self.regressor(dht_features)
        return line_params

# ==============================================================================
# Step 2: Helper Functions and Setup
# ==============================================================================

def draw_line(image, theta, rho, color=(0, 0, 255), thickness=2):
    """Draws a line on an image given theta and rho."""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Calculate two points on the line to extend it across the image
    x1 = int(x0 + 1200 * (-b))
    y1 = int(y0 + 1200 * (a))
    x2 = int(x0 - 1200 * (-b))
    y2 = int(y0 - 1200 * (a))
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def predict(model, image, device, transform):
    """Runs a single prediction on a PIL image."""
    # Apply transformations and add a batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get model prediction
        predicted_params = model(image_tensor).cpu().squeeze().numpy()
        
    theta, rho = predicted_params
    return theta, rho

# --- Main Setup ---
# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Horizon Line Detection Inference")
parser.add_argument('--image', type=str, help='Path to a single image file.')
parser.add_argument('--video', type=str, help='Path to a video file.')
parser.add_argument('--camera', action='store_true', help='Use live camera feed.')
parser.add_argument('--save', type=str, help='Path to save the output (image/video).')
args = parser.parse_args()

# Check for device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model ---
MODEL_PATH = '0.2dht_model_smd.pth'
model = DeepHoughTransform().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please download '0.2dht_model_smd.pth' from Colab and place it in the same directory.")
    exit()

model.eval()
print("Model loaded successfully.")

# --- Define Image Transformations ---
# These must be the same as the transformations used during training
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
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h, w = image_cv.shape[:2]

        # Get prediction
        theta, rho = predict(model, image_pil, device, transform)
        
        # Scale rho back to the original image size
        rho_scaled = rho * (h / IMG_SIZE[0])

        # Draw the line
        result_image = draw_line(image_cv, theta, rho_scaled)

        # Save the result if requested
        if args.save:
            cv2.imwrite(args.save, result_image)
            print(f"Result saved to: {args.save}")

        cv2.imshow('Horizon Detection', result_image)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError:
        print(f"Error: Image file not found at '{args.image}'")

elif args.video or args.camera:
    # --- Process a video file or camera feed ---
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{args.video}'")
            exit()
    else: # args.camera
        cap = cv2.VideoCapture(0) # 0 is usually the default webcam
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

    # Set up video writer if saving is requested
    video_writer = None
    if args.save and (args.video or args.camera):
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Video will be saved to: {args.save}")

    print("Processing video. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video or error

        start_time = time.time()
        
        h, w = frame.shape[:2]
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get prediction
        theta, rho = predict(model, frame_pil, device, transform)

        # Scale rho back to the original frame size
        rho_scaled = rho * (h / IMG_SIZE[0])
        
        # Draw the line on the frame
        result_frame = draw_line(frame, theta, rho_scaled)

        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to output video if saving
        if video_writer is not None:
            video_writer.write(result_frame)

        cv2.imshow('Horizon Detection', result_frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {args.save}")
    cv2.destroyAllWindows()

else:
    print("No input source specified. Please use --image, --video, or --camera.")
    parser.print_help()
