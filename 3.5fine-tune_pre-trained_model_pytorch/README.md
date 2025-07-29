# Maritime Horizon Detection & Sky Segmentation Training

## 🎯 Project Overview

This project provides a complete training pipeline for horizon line detection and sky segmentation using maritime image/video datasets. The system converts video datasets with MATLAB ground truth files to image datasets suitable for deep learning training.

## 📁 Dataset Structure

### Input (Original)
```
VIS_Onboard/
├── Videos/                 # .avi video files
│   ├── MVI_0788_VIS_OB.avi
│   ├── MVI_0789_VIS_OB.avi
│   └── ...
├── HorizonGT/             # MATLAB ground truth files
│   ├── MVI_0788_VIS_OB_HorizonGT.mat
│   ├── MVI_0789_VIS_OB_HorizonGT.mat
│   └── ...
├── ObjectGT/              # Object annotations (optional)
└── TrackGT/               # Tracking annotations (optional)
```

### Output (Processed)
```
MaritimeSkyFixed/
├── train/
│   ├── images/            # .jpg training images
│   └── masks/             # .png binary masks (1=sky, 0=non-sky)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## 🛠️ Setup & Installation

### 1. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scipy pillow numpy matplotlib scikit-learn
```

### 2. Verify CUDA Setup
```bash
python check_cuda.py
```

Expected output:
```
PyTorch version: 2.7.1+cu118
CUDA available: True
Device name: NVIDIA GeForce GTX 1650 SUPER
```

## 🚀 Usage

### Step 1: Convert Dataset
Convert your video dataset to image format:

```bash
python prepare_dataset.py --max_frames 30 --input_dir VIS_Onboard --output_dir MaritimeSkyFixed
```

**Parameters:**
- `--max_frames`: Number of frames to extract per video (default: 50)
- `--input_dir`: Directory containing videos and GT files (default: VIS_Onboard)  
- `--output_dir`: Output directory for processed dataset (default: MaritimeSkyFixed)
- `--seed`: Random seed for reproducible dataset splits (default: 42)

**Output:**
```
📊 Total frames extracted: 220
📁 Dataset saved to: MaritimeSkyFixed
  train: 140 images, 140 masks
  val: 40 images, 40 masks  
  test: 40 images, 40 masks
```

### Step 2: Train Model
Start GPU-optimized training:

```bash
python train_gpu.py
```

**Training Configuration (Optimized for GTX 1650 SUPER 4GB):**
- **Batch size**: 2 (conservative for 4GB VRAM)
- **Image size**: 224x224 (reduced for memory efficiency)
- **Learning rate**: 2e-4
- **Epochs**: 15
- **Mixed precision**: Enabled
- **Model**: FCN-ResNet50 (pre-trained backbone)

**Expected Output:**
```
🚀 Memory-Optimized GPU Training for Maritime Horizon Detection
🎮 Device: cuda
📊 GPU: NVIDIA GeForce GTX 1650 SUPER
💾 Total GPU Memory: 4.0GB
📊 Peak GPU memory usage: ~0.8GB
🏆 Best validation accuracy: 0.9xxx
💾 Model saved as: best_horizon_gpu_model.pth
```

### Step 3: Test & Evaluate
Test the trained model:

```bash
python test_model.py --mode test
```

Or run inference on a single image:
```bash
python test_model.py --mode inference --image path/to/image.jpg
```

## 📊 Model Performance

The trained model performs **binary semantic segmentation**:
- **Class 0**: Non-sky (water, ships, land, objects)
- **Class 1**: Sky (everything above the horizon line)

**Key Metrics:**
- **Pixel Accuracy**: Overall pixel classification accuracy
- **IoU (Intersection over Union)**: For sky class segmentation quality

## 🔧 File Descriptions

### Core Scripts

| File | Description |
|------|-------------|
| `prepare_dataset.py` | Converts videos + MAT files to image dataset |
| `train_gpu.py` | GPU-optimized training script |
| `test_model.py` | Model testing and inference |
| `train_sky_fcn.py` | Alternative FCN training script |

### Utility Scripts

| File | Description |
|------|-------------|
| `check_cuda.py` | CUDA availability checker |
| `test_training.py` | Simple training verification |
| `inspect_gt.py` | MAT file structure inspector |

### Key Features

| Feature | Description |
|---------|-------------|
| **Memory Optimization** | Conservative settings for 4GB GPU |
| **Mixed Precision** | Reduces memory usage by ~40% |
| **Flexible GT Parsing** | Handles various MAT file formats |
| **Progress Monitoring** | Real-time training metrics |
| **Automatic Splitting** | Train/Val/Test split (70/20/10) |

## 🎯 Advanced Usage

### Custom Training Parameters
Modify `train_gpu.py` for different configurations:

```python
# For more GPU memory (8GB+)
batch_size = 4
img_size = 256

# For faster training
epochs = 10
learning_rate = 5e-4

# For higher accuracy
epochs = 25
img_size = 512  # Requires more memory
```

### Dataset Customization
Adjust `prepare_dataset.py` for different splits:

```python
# Custom dataset split ratios
converter.split_videos(train_ratio=0.8, val_ratio=0.15)  # 80/15/5 split

# More frames per video
converter.convert_dataset(max_frames_per_video=100)
```

## 💡 Tips & Best Practices

### Memory Management
1. **Monitor GPU usage**: Training uses ~0.8GB, well below 4GB limit
2. **Batch size scaling**: Increase cautiously if you have more VRAM
3. **Image size**: 224x224 is optimal for GTX 1650 SUPER

### Training Optimization
1. **Use mixed precision**: Enabled by default for memory efficiency
2. **Learning rate scheduling**: Automatically reduces LR on plateaus
3. **Early stopping**: Best model saved automatically

### Dataset Quality
1. **Frame sampling**: Evenly distributed across video timeline
2. **GT validation**: Handles various MAT file structures
3. **Mask quality**: Binary masks with proper horizon line interpolation

## 🚨 Troubleshooting

### Common Issues

**CUDA not available:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory:**
- Reduce batch_size from 2 to 1
- Reduce img_size from 224 to 192
- Disable mixed precision

**MAT file errors:**
- Check `inspect_gt.py` output for GT file structure
- Verify filename matching between videos and GT files

**Poor accuracy:**
- Increase training epochs
- Check mask quality in test_output/
- Verify horizon line extraction

## 📈 Results

After training completion, you'll have:
- ✅ Trained model: `best_horizon_gpu_model.pth`
- ✅ Test visualizations: `test_output/`
- ✅ Performance metrics: Accuracy & IoU scores
- ✅ Ready for maritime horizon detection tasks

The trained model can detect horizon lines and segment sky regions in maritime imagery, useful for:
- **Maritime navigation systems**
- **Autonomous vessel guidance**  
- **Weather condition assessment**
- **Object detection preprocessing**
- **Image enhancement applications**

---

**Hardware Requirements:**
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8+
- 8GB+ System RAM
- ~5GB storage for dataset and models
