"""
Optimized GPU Training for GTX 1650 SUPER (4GB VRAM)
Conservative memory usage with efficient training
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T, models
from PIL import Image
import time

class HorizonDatasetOptimized(Dataset):
    def __init__(self, root, split, img_size=256):  # Reduced size for memory
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        
        # Simpler transforms to save memory
        self.t_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.t_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_p = os.path.join(self.img_dir, name)
        mask_p = os.path.join(self.mask_dir, name.replace(".jpg", ".png"))
        
        img = self.t_img(Image.open(img_p).convert("RGB"))
        mask = self.t_mask(Image.open(mask_p))
        mask = mask.squeeze(0).long()
        return img, mask

def train_optimized_gpu():
    print("🚀 Memory-Optimized GPU Training for Maritime Horizon Detection")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 Device: {device}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 CUDA Version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 Total GPU Memory: {total_memory:.1f}GB")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # Ultra-conservative parameters for 4GB GPU
    batch_size = 2  # Reduced further for safety
    img_size = 224  # Smaller images for better memory usage
    learning_rate = 2e-4  # Slightly higher LR to compensate for smaller batches
    epochs = 15
    
    print(f"⚙️  Configuration:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Image size: {img_size}x{img_size}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Epochs: {epochs}")
    
    # Data loading with minimal workers
    print(f"\n📊 Loading datasets...")
    train_ds = HorizonDatasetOptimized("MaritimeSkyFixed", "train", img_size=img_size)
    val_ds = HorizonDatasetOptimized("MaritimeSkyFixed", "val", img_size=img_size)
    
    # Minimal workers for memory conservation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)  # No workers to save memory
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    print(f"   📈 Training samples: {len(train_ds)}")
    print(f"   📉 Validation samples: {len(val_ds)}")
    print(f"   🔄 Training batches: {len(train_loader)}")
    print(f"   🔄 Validation batches: {len(val_loader)}")
    
    # Model setup - FCN with lighter backbone
    print(f"\n🧠 Setting up model...")
    model = models.segmentation.fcn_resnet50(weights="DEFAULT")
    
    # Replace classifier for binary segmentation (sky/non-sky)
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, 2, kernel_size=1)
    model = model.to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                  weight_decay=1e-4, eps=1e-7)
    
    # Scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                          factor=0.5, patience=3)
    
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print(f"   ✅ Model loaded: FCN-ResNet50")
    print(f"   ✅ Optimizer: AdamW")
    print(f"   ✅ Mixed precision: {'Enabled' if scaler else 'Disabled'}")
    
    print(f"\n🏋️  Starting training...")
    print("=" * 60)
    
    best_acc = 0
    best_iou = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n🔄 Epoch {epoch}/{epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)['out']
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            pred = outputs.argmax(1)
            train_correct += (pred == masks).float().sum().item()
            train_total += masks.numel()
            
            # Memory cleanup every few batches
            if batch_idx % 5 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Progress report
            if (batch_idx + 1) % max(1, len(train_loader) // 2) == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                current_acc = train_correct / train_total
                memory_used = torch.cuda.memory_allocated() / 1024**3 if device.type == 'cuda' else 0
                print(f"   📊 Batch {batch_idx+1}/{len(train_loader)} ({progress:.0f}%) | "
                      f"Loss: {loss.item():.4f} | Acc: {current_acc:.4f} | GPU: {memory_used:.1f}GB")
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_iou_sum = 0
        val_iou_count = 0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)['out']
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(imgs)['out']
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                pred = outputs.argmax(1)
                val_correct += (pred == masks).float().sum().item()
                val_total += masks.numel()
                
                # Calculate IoU for sky class (class 1)
                for b in range(pred.shape[0]):
                    pred_sky = (pred[b] == 1)
                    true_sky = (masks[b] == 1)
                    intersection = (pred_sky & true_sky).float().sum()
                    union = (pred_sky | true_sky).float().sum()
                    if union > 0:
                        iou = intersection / union
                        val_iou_sum += iou.item()
                        val_iou_count += 1
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou_sum / max(val_iou_count, 1)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n   � Train | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"   � Val   | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | IoU: {avg_val_iou:.4f}")
        print(f"   🎯 LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc or avg_val_iou > best_iou:
            if val_acc > best_acc:
                best_acc = val_acc
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_iou': best_iou,
                'img_size': img_size,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_iou': avg_val_iou
            }, "best_horizon_gpu_model2.pth")
            print(f"   💾 ✨ New best model saved! Acc: {val_acc:.4f} | IoU: {avg_val_iou:.4f}")
        
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 Training completed successfully!")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")
    print(f"🎯 Best IoU score: {best_iou:.4f}")
    print(f"⏱️  Total training time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"💾 Best model saved as: best_horizon_gpu_model2.pth")
    
    # Final memory report
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"📊 Peak GPU memory usage: {max_memory:.2f}GB")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_optimized_gpu()
