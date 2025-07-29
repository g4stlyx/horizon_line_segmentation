"""
Test script for trained horizon detection model
Loads the trained model and performs inference on test images
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from train_gpu import HorizonDatasetOptimized

def load_trained_model(model_path="best_horizon_gpu_model.pth", device="cuda"):
    """Load the trained model"""
    # Create model architecture
    model = models.segmentation.fcn_resnet50(weights=None)
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, 2, kernel_size=1)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"📊 Best accuracy: {checkpoint.get('best_acc', 'N/A'):.4f}")
    print(f"🎯 Best IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")
    print(f"📏 Image size: {checkpoint.get('img_size', 224)}")
    
    return model, checkpoint.get('img_size', 224)

def test_model():
    """Test the trained model on test dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 Device: {device}")
    
    # Load model
    if not os.path.exists("best_horizon_gpu_model.pth"):
        print("❌ No trained model found. Please run training first.")
        return
    
    model, img_size = load_trained_model(device=device)
    
    # Load test dataset
    test_ds = HorizonDatasetOptimized("MaritimeSkyFixed", "test", img_size=img_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"📊 Testing on {len(test_ds)} images...")
    
    # Test metrics
    correct = 0
    total = 0
    iou_sum = 0
    iou_count = 0
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    with torch.no_grad():
        for idx, (img, mask) in enumerate(test_loader):
            img, mask = img.to(device), mask.to(device)
            
            # Inference
            output = model(img)['out']
            pred = output.argmax(1)
            
            # Calculate accuracy
            correct += (pred == mask).float().sum().item()
            total += mask.numel()
            
            # Calculate IoU for sky class
            pred_sky = (pred[0] == 1)
            true_sky = (mask[0] == 1)
            intersection = (pred_sky & true_sky).float().sum()
            union = (pred_sky | true_sky).float().sum()
            if union > 0:
                iou = intersection / union
                iou_sum += iou.item()
                iou_count += 1
            
            # Save visualization for first few samples
            if idx < 5:
                save_prediction_visualization(img[0], mask[0], pred[0], idx)
    
    # Print results
    accuracy = correct / total
    avg_iou = iou_sum / max(iou_count, 1)
    
    print(f"\n📊 Test Results:")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 Average IoU: {avg_iou:.4f}")
    print(f"💾 Visualizations saved to test_output/")

def save_prediction_visualization(img_tensor, true_mask, pred_mask, idx):
    """Save a visualization comparing input, ground truth, and prediction"""
    # Convert tensors to numpy
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np, 0, 1)
    
    true_mask_np = true_mask.cpu().numpy()
    pred_mask_np = pred_mask.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title("Ground Truth\\n(1=Sky, 0=Non-Sky)")
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(pred_mask_np, cmap='gray')
    axes[2].set_title("Prediction\\n(1=Sky, 0=Non-Sky)")
    axes[2].axis('off')
    
    # Overlay prediction on original image
    overlay = img_np.copy()
    sky_pixels = pred_mask_np == 1
    overlay[sky_pixels] = overlay[sky_pixels] * 0.7 + np.array([0.3, 0.6, 1.0]) * 0.3
    
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay\\n(Blue = Sky)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"test_output/test_sample_{idx:03d}.png", dpi=150, bbox_inches='tight')
    plt.close()

def inference_on_single_image(image_path, model_path="best_horizon_gpu_model.pth"):
    """Run inference on a single image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, img_size = load_trained_model(model_path, device)
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)['out']
        pred = output.argmax(1)
    
    # Convert back to PIL image
    pred_np = pred[0].cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(pred_np, cmap='gray')
    axes[1].set_title("Sky Segmentation\\n(1=Sky, 0=Non-Sky)")
    axes[1].axis('off')
    
    # Overlay
    img_np = np.array(img.resize((img_size, img_size)))
    overlay = img_np.copy()
    sky_pixels = pred_np == 1
    overlay[sky_pixels] = overlay[sky_pixels] * 0.7 + np.array([100, 150, 255]) * 0.3
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Sky Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = f"test_output/{os.path.splitext(os.path.basename(image_path))[0]}_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Result saved to {output_path}")
    
    return pred_np

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained horizon detection model")
    parser.add_argument("--mode", choices=["test", "inference"], default="test",
                       help="Mode: 'test' for test dataset evaluation, 'inference' for single image")
    parser.add_argument("--image", type=str, help="Path to image for inference mode")
    parser.add_argument("--model", type=str, default="best_horizon_gpu_model.pth",
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_model()
    elif args.mode == "inference":
        if args.image:
            inference_on_single_image(args.image, args.model)
        else:
            print("❌ Please provide --image path for inference mode")
