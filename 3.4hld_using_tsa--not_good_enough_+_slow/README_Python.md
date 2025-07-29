# Horizon Line Detection using Time Series Analysis - Python Version

This is a Python implementation of the MATLAB horizon line detection algorithm using time series analysis.

## Installation

1. **Install Python 3.7 or later**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy: For numerical computations
- opencv-python: For video processing
- matplotlib: For visualization
- scikit-image: For image processing (edge detection, Radon transform)
- scipy: For scientific computing
- statsmodels: For ARIMA time series modeling
- arch: For GARCH modeling

## Usage

### Basic Usage

```python
from HL_Detect_TSA import hl_detect_tsa

# For Singapore Maritime Dataset video
vid_name = 'VIS_Onboard/Videos/MVI_0788_VIS_OB.avi'
N = 60
hl_est = hl_detect_tsa(vid_name, N)

# For Buoy Dataset video  
vid_name = 'Buoy/buoyGT_2_5_3_4.avi'
N = 20
hl_est = hl_detect_tsa(vid_name, N)

# For any video file
vid_name = 'path/to/your/video.mp4'
N = 30  # Number of frames for listener block
hl_est = hl_detect_tsa(vid_name, N)
```

### Command Line Usage

```python
python HL_Detect_TSA.py
```

**Note:** Edit the `__main__` section in `HL_Detect_TSA.py` to specify your video file path and parameters.

## Parameters

- `vid_name` (str): Path to the video file
- `N` (int): Number of frames to process in the Listener Block
  - Recommended: 60 for maritime videos, 20 for buoy videos

## Output

- **hl_est**: NumPy array where each row represents a frame with:
  - Column 0: Vertical position of horizon line (pixels)
  - Column 1: Orientation angle (degrees)

- **Real-time display**: Video frames with:
  - Yellow line/contour showing the ROI
  - Red line showing the detected horizon

## Key Differences from MATLAB Version

1. **Video Reading**: Uses OpenCV instead of VideoReader
2. **Image Processing**: Uses scikit-image for edge detection and Radon transform
3. **Time Series**: Uses statsmodels for ARIMA modeling
4. **Visualization**: Uses matplotlib instead of MATLAB plotting
5. **Array Indexing**: Python uses 0-based indexing vs MATLAB's 1-based

## Example Output

```python
hl_est = hl_detect_tsa('video.mp4', 30)
print(f"Processed {len(hl_est)} frames")
print("First 5 horizon line estimates:")
print(hl_est[:5])
# Output:
# [[245.3   1.2]
#  [244.8   1.1] 
#  [246.1   1.3]
#  [245.9   1.0]
#  [244.7   1.4]]
```

## Troubleshooting

1. **Video not found**: Ensure the video path is correct and the file exists
2. **Missing dependencies**: Install all required packages using `pip install -r requirements.txt`
3. **Display issues**: If running headless, comment out plotting functions or use a different backend
4. **Memory issues**: For large videos, consider processing in chunks

## Performance Notes

- The Python version may be slower than MATLAB due to different optimization strategies
- For better performance, consider using GPU-accelerated libraries where available
- The time series modeling might differ slightly from MATLAB's implementation
