"""
Simple training test to verify the setup
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from train_gpu import HorizonDatasetOptimized
import time

def simple_train_test():
    print("🔧 Simple Training Test")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Small test parameters
    batch_size = 1
    img_size = 224
    
    # Load small dataset
    train_ds = HorizonDatasetOptimized("MaritimeSkyFixed", "train", img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset size: {len(train_ds)}")
    print(f"Batches: {len(train_loader)}")
    
    # Simple model
    model = models.segmentation.fcn_resnet50(weights="DEFAULT")
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, 2, kernel_size=1)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\\nRunning 5 batches...")
    model.train()
    
    for i, (imgs, masks) in enumerate(train_loader):
        if i >= 5:  # Only test 5 batches
            break
            
        start_time = time.time()
        
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - start_time
        print(f"  Batch {i+1}/5: Loss = {loss.item():.4f}, Time = {batch_time:.2f}s")
        
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"    GPU Memory: {memory_used:.2f}GB")
    
    print("\\n✅ Test completed successfully!")
    
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"📊 Peak GPU memory: {max_memory:.2f}GB")

if __name__ == "__main__":
    simple_train_test()
