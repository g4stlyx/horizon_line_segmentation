# Maritime Horizon Line Detection & Segmentation

This repository contains multiple approaches and implementations for detecting horizon lines in maritime environments, ranging from simple image-based detection to advanced real-time video processing with deep learning models.

## 📁 Project Structure

### `0sunum/` - Presentations & Documentation -hidden lol-
Contains presentations and academic papers related to the project:
- **Presentations**: `.pptx` files documenting the project progress and results
- **Papers (`makaleler/`)**: Collection of research papers used as references:
  - *Real-time Horizon Line Detection Based on Fusion of Classification and Clustering*
  - *Comparison of Semantic Segmentation Approaches for Horizon Line Detection*
  - *Deep Hough Transform for Semantic Line Detection*
  - *Segmentation to Horizon Line in Real-Time*

### `3.1horizon_segment_images/` - Basic Image Processing
Simple horizon detection for static images using traditional computer vision techniques:
- **`horizon_detect.py`**: Core algorithm for single image horizon detection
- **`test_all_images.py`**: Batch processing script for multiple images

**Methods Used**: Color-based segmentation, K-means clustering, contour detection

### `3.2horizon_segment_video/` - Enhanced Video Processing
Advanced real-time video horizon detection with ship-aware segmentation:
- **`horizon_detect_video.py`**: Main video processing with intelligent ship detection
- **Features**: 
  - Ship-aware horizon line segmentation (avoids crossing vessels)
  - Temporal stability and frame-to-frame consistency
  - Multiple input sources (video files, live camera, webcam)
  - Performance monitoring and adaptive detection

**Paper Reference**: Based on multiple approaches including classification and clustering methods

### `3.3horizon_segment_real-time_based_on_a_paper/` - Paper Implementation
Implementation based on the research paper:
- **Paper**: *"Real-time Horizon Line Detection Based on Fusion of Classification and Clustering"*
- **Files**:
  - `paper_script.py`: Direct implementation of the paper's algorithm
  - `hybrid_horizon_detection.py`: Enhanced version with improvements
  - `compare_methods.py`: Comparison between different approaches
  - `hld_model.pth` & `corrected_model.pkl`: Trained models

**Key Innovation**: Fusion of classification and clustering for robust real-time detection

### `3.4hld_using_tsa--not_good_enough_+_slow/` - Time Series Analysis Approach
Implementation using Time Series Analysis for dynamic ROI generation:
- **Paper**: *"Dynamic Region of Interest Generation for Maritime Horizon Line Detection using Time Series Analysis"*
- **Method**: ARIMA and GARCH models for ROI prediction
- **Files**:
  - `HL_Detect_TSA.m`: MATLAB implementation
  - `HL_Detect_TSA.py`: Python port
- **Performance**: Marked as "not good enough + slow" - experimental approach

**Datasets Used**: 
- Singapore Maritime Dataset (Onboard)
- Buoy Dataset

### `3.5fine-tune_pre-trained_model_pytorch/` - Deep Learning Training
Complete training pipeline for horizon detection using deep learning:
- **Purpose**: Fine-tuning pre-trained models for maritime horizon detection
- **Features**:
  - Dataset conversion from video to image datasets
  - Sky segmentation training (binary masks: sky vs non-sky)
  - Support for maritime datasets with MATLAB ground truth
- **Files**:
  - `train_sky_fcn.py`: Fully Convolutional Network training
  - `train_gpu.py`: GPU-optimized training
  - `prepare_dataset.py`: Dataset preprocessing and conversion

**Input**: VIS_Onboard dataset with MATLAB ground truth files  
**Output**: Trained models for semantic segmentation

### `3.6training_dht/` - DHT Training & Experiments
Collection of trained models and training scripts:
- **Deep Hough Transform (DHT) Models**:
  - `0.1dht_model_intel.pth`: Trained on Intel image classification dataset
  - `0.2dht_model_smd.pth`: Trained on Singapore Maritime Dataset

**Paper Reference**: *"Deep Hough Transform for Semantic Line Detection"*

### `3.7training_unet/` - 2-Class (Sky/Non-Sky) and 3-Class (Sky/Water/Object) U-Net
- **U-Net (legacy 2-class + ship-aware) Training**:
  - Binary sky vs non-sky training and ship-aware variants
  - Utilities and runners (legacy), e.g. `z_unet_runner.py`, `z_unet_runner_dist_calc_rtdetr_obj_det.py`
- **RT-DETR Checkpoints** (for object detection during preprocessing/inference):
  - `rtdetr_obj_det_model/final_best_model/` (contains `config.json`, `model.safetensors`, `preprocessor_config.json`)
Modern 3-class U-Net training and runners with RT-DETR integration:
- **Training**:
  - `4training_unet_ship_aware_rtdetr_3_class.py`: End-to-end Colab pipeline
    - Preprocessing with horizon GT (sky/water) + RT-DETR overrides to mark objects (class 2)
    - U-Net head with `n_classes=3`; loss = weighted CrossEntropy + multi-class Dice
    - Saves best weights as `best_unet_rtdetr_aware_smd_3cls.pth`
- **Runner / Inference**:
  - `z_unet_runner_dist_calc_rtdetr_obj_det_3_class.py`
    - Horizon estimated from sky/water boundary (objects ignored)
    - RT-DETR used for per-object boxes and signed center-to-horizon distances
    - Supports image/folder/video/camera, optional CSV export and angle column
- **Note**: Land/shore pixels are treated as water to keep the horizon boundary clean.

**Paper References**:
- U-Net (Ronneberger et al., 2015)
- RT-DETR (Real-Time Detection Transformer)

### `3.8training_yolov8-seg/` - YOLOv8 Segmentation Training
End-to-end scripts to train and run YOLOv8 segmentation for maritime scenes:
- `train_yolov8_seg_colab.py`: Colab training script (can be adapted for 3-class sky/water/object)
- `z_yolov8_runner.py`: Inference/runner utilities
- `results/`: Outputs and examples

**Note**: Can mirror the U-Net horizon extraction logic (sky/water boundary) and integrate object distances.

### `3.9training_transformers/` - Transformers (SegFormer)
Utilities, notes, and configs for transformer-based object detection:
- to be continued

## 🔬 Research Papers & Methods

### Primary Papers Referenced:
1. **"Real-time Horizon Line Detection Based on Fusion of Classification and Clustering"**
   - Used in: `3.3horizon_segment_real-time_based_on_a_paper/`
   - Method: Hybrid approach combining multiple detection strategies

2. **"Dynamic Region of Interest Generation for Maritime Horizon Line Detection using Time Series Analysis"**
   - Used in: `3.4hld_using_tsa--not_good_enough_+_slow/`
   - Method: ARIMA/GARCH models for ROI prediction

3. **"Deep Hough Transform for Semantic Line Detection"**
   - Used in: `3.6training_dht_and_unet/`
   - Method: Deep learning approach to line detection

4. **"Comparison of Semantic Segmentation Approaches for Horizon Line Detection"**
   - Referenced throughout for method comparison

5. **"U-Net: Convolutional Networks for Biomedical Image Segmentation"**
   - Used in: `3.7training_unet/`
   - Method: Encoder-decoder segmentation backbone for sky/water/object


## 🗄️ Datasets Used

- **Singapore Maritime Dataset (SMD)**: Onboard and onshore maritime videos
- **Intel Image Classification Dataset**: Used in DHT 1.0
- **VIS_Onboard**: Maritime video dataset with MATLAB ground truth

## 🎯 Key Features Across Implementations

- Real-time video processing
- Ship-aware segmentation (horizon lines avoid crossing vessels)
- Multiple detection algorithms (color-based, clustering, deep learning)
- Temporal stability and frame consistency
- Performance monitoring and adaptive detection
- Support for various input sources (video files, webcam, live camera)

---

*This project represents a comprehensive exploration of horizon line detection methods in maritime environments, from traditional computer vision to modern deep learning approaches.*