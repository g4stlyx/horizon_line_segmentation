# ==============================================================================
# Hybrid Horizon Detection with U-Net + Ship Detection
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

# Import the U-Net model from your existing script
from z_unet_runner import UNet, create_overlay

class HybridHorizonDetector:
    def __init__(self, model_path, device):
        self.device = device
        
        # Load U-Net model
        self.unet_model = UNet(n_classes=2).to(device)
        self.unet_model.load_state_dict(torch.load(model_path, map_location=device))
        self.unet_model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def detect_ships_advanced(self, frame):
        """
        Advanced ship detection using multiple methods.
        """
        h, w = frame.shape[:2]
        ship_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Method 1: Detect large contrasting objects
        # Use adaptive threshold to find objects
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 10)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum ship size
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Check aspect ratio (ships are typically wider than tall)
                aspect_ratio = cw / ch if ch > 0 else 0
                
                if aspect_ratio > 1.2:  # Ship-like aspect ratio
                    # Additional validation
                    roi = gray[y:y+ch, x:x+cw]
                    if roi.size > 0:
                        # Check for edges (ships have structure)
                        edges = cv2.Canny(roi, 50, 150)
                        edge_density = np.sum(edges > 0) / (cw * ch)
                        
                        # Check brightness (avoid water/sky regions)
                        mean_brightness = np.mean(roi)
                        
                        if 0.05 < edge_density < 0.3 and 30 < mean_brightness < 200:
                            cv2.rectangle(ship_mask, (x, y), (x + cw, y + ch), 255, -1)
        
        # Method 2: Color-based ship detection
        # White/light structures (superstructures)
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Dark structures (hulls)
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
        
        # Red colors (common on ships)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(white_mask, dark_mask)
        color_mask = cv2.bitwise_or(color_mask, red_mask)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Filter by size
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                cv2.fillPoly(ship_mask, [contour], 255)
        
        return ship_mask
    
    def predict_hybrid(self, image):
        """
        Hybrid prediction combining U-Net with ship detection.
        """
        # Convert PIL to CV2 format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = image_cv.shape[:2]
        
        # Step 1: Get U-Net prediction
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.unet_model(image_tensor)
        
        # Get the predicted class for each pixel
        preds = torch.argmax(outputs, dim=1).cpu().squeeze(0)
        
        # Resize mask back to original image size
        mask_transform = transforms.Resize(original_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)
        unet_mask = mask_transform(preds.unsqueeze(0)).squeeze(0).numpy()
        
        # Step 2: Detect ships using traditional CV methods
        ship_mask = self.detect_ships_advanced(image_cv)
        
        # Step 3: Combine predictions
        # Rule: If ship detection finds a ship in the sky region, correct it to non-sky
        final_mask = unet_mask.copy()
        
        # Any pixel that is both "sky" in U-Net prediction AND "ship" in ship detection
        # should be corrected to "non-sky"
        ship_in_sky = (unet_mask == 1) & (ship_mask == 255)
        final_mask[ship_in_sky] = 0
        
        # Optional: Also expand ship regions slightly to ensure complete coverage
        if np.any(ship_mask == 255):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            expanded_ship_mask = cv2.morphologyEx(ship_mask, cv2.MORPH_DILATE, kernel)
            ship_expansion = (unet_mask == 1) & (expanded_ship_mask == 255)
            final_mask[ship_expansion] = 0
        
        return final_mask, ship_mask
    
    def create_enhanced_overlay(self, image, mask, ship_mask=None):
        """Create visualization with ship boundaries."""
        sky_color = np.array([0, 0, 255], dtype=np.uint8)  # Blue for sky
        ship_color = np.array([0, 255, 255], dtype=np.uint8)  # Yellow for ships
        
        # Create overlay
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[mask == 1] = sky_color
        
        # Add ship highlights if provided
        if ship_mask is not None:
            # Create ship contours for better visualization
            contours, _ = cv2.findContours(ship_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(overlay, [contour], -1, ship_color, 3)
        
        # Blend with original image
        viz_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        return viz_image

def main():
    parser = argparse.ArgumentParser(description="Hybrid Horizon Segmentation")
    parser.add_argument('--model', type=str, default='1.1best_unet_ship_aware_smd.pth', help='Path to U-Net model')
    parser.add_argument('--image', type=str, help='Path to a single image file.')
    parser.add_argument('--video', type=str, help='Path to a video file.')
    parser.add_argument('--camera', action='store_true', help='Use live camera feed.')
    parser.add_argument('--save', action='store_true', help='Save the output.')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize hybrid detector
    try:
        detector = HybridHorizonDetector(args.model, device)
        print("Hybrid detector loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model}'")
        return
    
    if args.image:
        # Process single image
        try:
            image_pil = Image.open(args.image).convert('RGB')
            mask, ship_mask = detector.predict_hybrid(image_pil)
            
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            result_image = detector.create_enhanced_overlay(image_cv, mask, ship_mask)
            
            if args.save:
                save_path = os.path.splitext(args.image)[0] + "_hybrid_segmented.jpg"
                cv2.imwrite(save_path, result_image)
                print(f"Saved result to: {save_path}")
            
            cv2.imshow('Hybrid Horizon Segmentation', result_image)
            print("Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except FileNotFoundError:
            print(f"Error: Image file not found at '{args.image}'")
    
    elif args.video or args.camera:
        # Process video or camera
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        video_writer = None
        if args.save:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            save_path = "hybrid_output.mp4"
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"Saving output video to {save_path}")
        
        print("Processing video. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mask, ship_mask = detector.predict_hybrid(frame_pil)
            result_frame = detector.create_enhanced_overlay(frame, mask, ship_mask)
            
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, "Hybrid: U-Net + Ship Detection", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if video_writer:
                video_writer.write(result_frame)
            
            cv2.imshow('Hybrid Horizon Segmentation', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
    
    else:
        print("No input source specified. Use --image, --video, or --camera.")
        parser.print_help()

if __name__ == "__main__":
    main()
