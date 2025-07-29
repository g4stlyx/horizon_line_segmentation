# Maritime Horizon Line Detection & Segmentation

***IMPORTANT: IT IS NOT COMPLETED YET***

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

### `3.6training_models/` - Model Training & Experiments
Collection of trained models and training scripts:
- **Deep Hough Transform (DHT) Models**:
  - `0.1dht_model_intel.pth`: Trained on Intel image classification dataset
  - `0.2dht_model_smd.pth`: Trained on Singapore Maritime Dataset
- **U-Net Models**:
  - `1.0unet_model_smd.pth`: U-Net trained on SMD
- **Training Scripts**:
  - Google Colab compatible training scripts
  - Comparison runners for different model architectures

**Paper Reference**: *"Deep Hough Transform for Semantic Line Detection"*

## 🔬 Research Papers & Methods

### Primary Papers Referenced:
1. **"Real-time Horizon Line Detection Based on Fusion of Classification and Clustering"**
   - Used in: `3.3horizon_segment_real-time_based_on_a_paper/`
   - Method: Hybrid approach combining multiple detection strategies

2. **"Dynamic Region of Interest Generation for Maritime Horizon Line Detection using Time Series Analysis"**
   - Used in: `3.4hld_using_tsa--not_good_enough_+_slow/`
   - Method: ARIMA/GARCH models for ROI prediction

3. **"Deep Hough Transform for Semantic Line Detection"**
   - Used in: `3.6training_models/`
   - Method: Deep learning approach to line detection

4. **"Comparison of Semantic Segmentation Approaches for Horizon Line Detection"**
   - Referenced throughout for method comparison

## 🗄️ Datasets Used

- **Singapore Maritime Dataset (SMD)**: Onboard and onshore maritime videos
- **Horizon Lines in the Wild (HLW)**: Natural horizon line dataset
- **KITTI-Horizon**: Automotive dataset adapted for horizon detection
- **Intel Image Classification Dataset**: Used for transfer learning
- **VIS_Onboard**: Maritime video dataset with MATLAB ground truth

## 🚀 Getting Started

1. **For simple image processing**: Start with `3.1horizon_segment_images/`
2. **For video processing**: Use `3.2horizon_segment_video/`
3. **For research implementation**: Check `3.3horizon_segment_real-time_based_on_a_paper/`
4. **For training custom models**: Use `3.5fine-tune_pre-trained_model_pytorch/` or `3.6training_models/`

## 📊 Performance Notes

- **Best Performance**: `3.2horizon_segment_video/` and `3.3horizon_segment_real-time_based_on_a_paper/`
- **Experimental**: `3.4hld_using_tsa--not_good_enough_+_slow/` (slower, less accurate)
- **Training Ready**: `3.5fine-tune_pre-trained_model_pytorch/` and `3.6training_models/`

## 🎯 Key Features Across Implementations

- Real-time video processing
- Ship-aware segmentation (horizon lines avoid crossing vessels)
- Multiple detection algorithms (color-based, clustering, deep learning)
- Temporal stability and frame consistency
- Performance monitoring and adaptive detection
- Support for various input sources (video files, webcam, live camera)

---

*This project represents a comprehensive exploration of horizon line detection methods in maritime environments, from traditional computer vision to modern deep learning approaches.*