import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature, transform, measure, morphology
from skimage.filters import gaussian
from scipy import ndimage
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
import os
warnings.filterwarnings('ignore')

#! python conversion of the given matlab code in paper.

# Configure matplotlib for interactive display
plt.ion()  # Turn on interactive mode
plt.show()

def hl_detect_tsa(vid_name, N):
    """
    Detects the horizon line in a video using time series analysis.
    
    Parameters:
    vid_name (str): Name of the video file.
    N (int): Number of frames to process in the Listener Block.
    
    Returns:
    HL_est (numpy.ndarray): Matrix of estimated horizon line states. Rows correspond
                           to the frame indices. The 1st column is the vertical position of the
                           Horizon Line (HL) in pixels whereas the 2nd column consists the orientation
                           angle in degrees.
    """
    
    print(f"Starting horizon line detection for: {vid_name}")
    print(f"Listener block frames: {N}")
    
    # Check if video file exists
    if not os.path.exists(vid_name):
        raise ValueError(f"Video file not found: {vid_name}")
    
    # Read the video file
    cap = cv2.VideoCapture(vid_name)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {vid_name}")
    
    # Extract video parameters
    nof = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {nof} frames, {width}x{height}, {fps:.2f} fps")
    
    # Read all frames
    frames = []
    print("Reading video frames...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Read {frame_count} frames...")
    cap.release()
    
    print(f"Total frames read: {len(frames)}")
    
    if len(frames) == 0:
        raise ValueError("No frames found in video")
    
    # Get first frame dimensions
    h, w = frames[0].shape[:2]
    
    # Set algorithm parameters
    M = 3  # Maximum number of iterations for the control loop
    n_bins = np.arange(256)  # Histogram bins vector for error calculation
    z_score = 1.96  # Standard z-score corresponding to 95% confidence interval
    
    # Define coordinate matrices for the pixels
    x_mat, y_mat = np.meshgrid(np.arange(1, w+1), np.arange(1, h+1))
    
    # Initialize a blank binary image for ROI calculations
    bw_or = np.zeros((h, w), dtype=bool)
    
    # Set the height for the parallelogram used in error calculation
    delta_h = 0.025 * h
    
    # Set the minimum height for the rectangular ROI
    min_h = round(h * 0.075)
    
    # Listener Block
    print("Starting Listener Block...")
    hl_est, err, k = listener_block(frames, N, x_mat, y_mat, n_bins, w, delta_h, min_h, h)
    print(f"Listener Block completed. Processed {k} frames.")
    
    # Create univariate autoregressive integrated moving average (ARIMA) models
    print("Fitting ARIMA models...")
    # For y, ARIMA(2,2,0) and GARCH(1,1)
    y_data = hl_est[max(0, len(hl_est)-N):, 0]
    theta_data = hl_est[max(0, len(hl_est)-N):, 1]
    
    # Fit ARIMA models
    try:
        y_model = ARIMA(y_data, order=(2, 2, 0)).fit()
        theta_model = ARIMA(theta_data, order=(2, 0, 0)).fit()
        print("ARIMA models fitted successfully.")
    except:
        # Fallback to simpler models if fitting fails
        print("Falling back to simpler ARIMA models...")
        y_model = ARIMA(y_data, order=(1, 1, 0)).fit()
        theta_model = ARIMA(theta_data, order=(1, 0, 0)).fit()
    
    # Set AD (Absence Detector variable) false
    AD = 0
    ind_temp = k - 1
    
    print(f"Processing remaining {nof - k} frames...")
    # Process the remaining frames
    while k < nof:
        k += 1
        if k % 50 == 0:
            print(f"Processing frame {k}/{nof}")
        
        rgb = frames[k-1]  # Get the frame (0-indexed)
        
        if AD > 0:  # Check absence of HL
            # if HL is absent, route to Presence Detector Block
            hl_est, err, AD = presence_detector(rgb, hl_est, err, min_h, bw_or, k, 
                                              ind_temp, x_mat, y_mat, n_bins, w, h, delta_h)
            continue
        
        # Main Block
        # Get forecasts for y and theta and their variances
        try:
            y_forecast = y_model.forecast(steps=1)
            theta_forecast = theta_model.forecast(steps=1)
            
            # Get forecast variances (simplified approach)
            y_var = np.var(y_model.resid) if len(y_model.resid) > 0 else 1.0
            theta_var = np.var(theta_model.resid) if len(theta_model.resid) > 0 else 1.0
            
            yf = y_forecast.iloc[0] if hasattr(y_forecast, 'iloc') else y_forecast[0]
            tf = theta_forecast.iloc[0] if hasattr(theta_forecast, 'iloc') else theta_forecast[0]
        except:
            # Fallback to last known values
            yf = hl_est[k-2, 0] if k > 1 else h/2
            tf = hl_est[k-2, 1] if k > 1 else 0
            y_var = 1.0
            theta_var = 1.0
        
        # Calculate the standard deviations
        yf_sd = np.sqrt(y_var)
        tf_sd = np.sqrt(theta_var)
        
        # Determine the ROI based on the forecasted states and their variances
        roi, AD, bw, delh = roi_tsm(rgb, [yf, tf], yf_sd, tf_sd, z_score, w)
        
        if AD < 1:  # Control Loop
            # Call HLDA to get the estimated state of the HL within the ROI
            x = hlda(roi)
            
            # Update the state estimate relative to the image coordinate system
            if len(hl_est) <= k-1:
                hl_est = np.vstack([hl_est, np.zeros((k - len(hl_est), 2))])
            
            hl_est[k-1, :] = [yf - delh + x[0]/np.cos(np.radians(tf)), tf + x[1]]
            
            # Calculate the error metric based on the current state estimate
            if len(err) <= k-1:
                err = np.append(err, 0)
            err[k-1] = get_error(hl_est[k-1, :], rgb, x_mat, y_mat, n_bins, w, delta_h)
            
            # Initialize Control Loop variables
            noi = 0  # Number of iterations
            minh2 = 0  # Rectangular ROI height increment
            
            # Control Loop: Iterate while error is significant or state estimates are out of bounds
            recent_errors = err[max(0, k-5):k-1] if k > 5 else err[:k-1]
            mean_recent_error = np.mean(recent_errors) if len(recent_errors) > 0 else 0
            
            error_condition = (abs((err[k-1] - mean_recent_error) / mean_recent_error) > 0.15 
                             if mean_recent_error != 0 else False)
            bounds_condition = x[0] > 3*h or x[0] < 0
            
            while noi < M and (error_condition or bounds_condition):
                noi += 1
                
                # Adjust ROI to a larger rectangular region
                roi, h_lims = roi_rect(rgb, hl_est[k-2, :] if k > 1 else [h/2, 0], minh2, w, h)
                
                # Update the binary image BW for the new ROI
                bw = bw_or.copy()
                bw[h_lims[0]:h_lims[1], :] = True
                
                # Re-estimate the HL state within the updated ROI
                x = hlda(roi)
                hl_est[k-1, :] = [x[0] + h_lims[0], x[1]]
                
                # Recalculate the error metric
                err[k-1] = get_error(hl_est[k-1, :], rgb, x_mat, y_mat, n_bins, w, delta_h)
                
                # Increment the ROI height adjustment for the next iteration
                minh2 += round(0.075 * h)
                
                # Update conditions for next iteration
                recent_errors = err[max(0, k-5):k-1] if k > 5 else err[:k-1]
                mean_recent_error = np.mean(recent_errors) if len(recent_errors) > 0 else 0
                error_condition = (abs((err[k-1] - mean_recent_error) / mean_recent_error) > 0.15 
                                 if mean_recent_error != 0 else False)
                bounds_condition = x[0] > 3*h or x[0] < 0
            
            # Plot the estimated HL and the ROI
            hl_plot(rgb, hl_est[k-1, :], w)
            roi_plot(bw)
            plt.draw()
            plt.pause(0.01)
            
        else:  # Absence Detector
            # Handle cases where the horizon line is not detected
            ind_temp = k - 1  # Record the last frame index when the HL was present
            
            if len(hl_est) <= k-1:
                hl_est = np.vstack([hl_est, np.zeros((k - len(hl_est), 2))])
            
            # Set the horizon line estimate based on the last known state
            if hl_est[k-2, 0] > h/2:
                hl_est[k-1, :] = [h, 0]
                bw = bw_or.copy()
                bw[h-2*min_h:h, :] = True
            else:
                hl_est[k-1, :] = [1, 0]
                bw = bw_or.copy()
                bw[0:2*min_h, :] = True
            
            # Plot the fallback HL and the ROI
            hl_plot(rgb, hl_est[k-1, :], w)
            roi_plot(bw)
            plt.draw()
            plt.pause(0.01)
    
    print("Horizon line detection completed!")
    return hl_est


def listener_block(frames, N, x_mat, y_mat, n_bins, w, delta_h, min_h, h):
    """
    Processes the first N frames to initialize the system.
    """
    print(f"Processing first {N} frames in Listener Block...")
    k = 1
    hl_est = np.zeros((N, 2))
    err = np.zeros(N)
    
    # Process the first frame using the full frame
    print("Processing frame 1...")
    rgb = frames[0]
    hl_est[0, :] = hlda(rgb)
    err[0] = get_error(hl_est[0, :], rgb, x_mat, y_mat, n_bins, w, delta_h)
    
    # Process the subsequent frames using a rectangular ROI
    for k in range(1, N):
        if (k + 1) % 10 == 0:
            print(f"Processing frame {k + 1}...")
        rgb = frames[k]
        roi, h_lims = roi_rect(rgb, hl_est[k-1, :], min_h, w, h)
        x = hlda(roi)
        x[0] = x[0] + h_lims[0]
        hl_est[k, :] = x
        err[k] = get_error(hl_est[k, :], rgb, x_mat, y_mat, n_bins, w, delta_h)
        hl_plot(rgb, hl_est[k, :], w)
        plt.draw()
        plt.pause(0.001)  # Shorter pause for faster processing
    
    return hl_est, err, k


def presence_detector(rgb, hl_est, err, min_h, bw_or, k, ind_temp, x_mat, y_mat, n_bins, w, h, delta_h):
    """
    Detects the presence of the horizon line and adjusts the state estimates.
    """
    # Ensure hl_est has enough rows
    if len(hl_est) <= k-1:
        hl_est = np.vstack([hl_est, np.zeros((k - len(hl_est), 2))])
    
    # Ensure err has enough elements
    if len(err) <= k-1:
        err = np.append(err, np.zeros(k - len(err)))
    
    # Adjust the ROI based on the previous horizon line estimate
    if hl_est[k-2, 0] > 0:
        roi = rgb[h - 2*min_h:h, :, :]
        bw = bw_or.copy()
        bw[h - 2*min_h:h, :] = True
        x = hlda(roi)
        hl_est[k-1, :] = [x[0] + h - 2*min_h, x[1]]
    else:
        roi = rgb[0:2*min_h, :, :]
        bw = bw_or.copy()
        bw[0:2*min_h, :] = True
        x = hlda(roi)
        hl_est[k-1, :] = x
    
    # Calculate the error metric for the current horizon line estimate
    err[k-1] = get_error(hl_est[k-1, :], rgb, x_mat, y_mat, n_bins, w, delta_h)
    
    # Check if the error exceeds a threshold compared to the reference frame
    if ind_temp < len(err) and abs((err[k-1] - err[ind_temp]) / err[ind_temp]) > 0.05:
        AD = 1
        hl_est[k-1, :] = hl_est[k-2, :]
        hl_plot(rgb, hl_est[k-1, :], w)
        roi_plot(bw)
        plt.draw()
        plt.pause(0.01)
    else:
        AD = 0
        hl_plot(rgb, hl_est[k-1, :], w)
        roi_plot(bw)
        plt.draw()
        plt.pause(0.01)
    
    return hl_est, err, AD


def hl_plot(rgb, x, w):
    """
    Plots the estimated horizon line on the current video frame.
    """
    plt.figure(1, figsize=(10, 6))
    plt.clf()
    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    
    # Calculate the y-coordinates of the horizon line across the frame width
    y = np.round(np.tan(np.radians(x[1])) * np.arange(1, w+1) + x[0] - np.tan(np.radians(x[1])) * w / 2)
    
    # Plot the horizon line in red with a thickness of 3 pixels
    plt.plot([1, w], [y[0], y[-1]], 'r-', linewidth=3, label='Horizon Line')
    plt.title(f'Horizon Detection - Y: {x[0]:.1f}, Angle: {x[1]:.1f}°')
    plt.axis('off')
    plt.tight_layout()


def hlda(rgb):
    """
    Detects the horizon line in a given video frame or a ROI.
    """
    if rgb.size == 0:
        return np.array([0, 0])
    
    # Convert the RGB image to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if len(rgb.shape) == 3 else rgb
    
    # Perform Canny edge detection
    edges = feature.canny(gray, sigma=1, low_threshold=0.1, high_threshold=0.2)
    
    # Remove small objects from the binary image
    min_size = max(1, round(np.sum(edges) * 0.005))
    edges_cleaned = morphology.remove_small_objects(edges, min_size=min_size)
    
    # Define the angles for Radon transform
    theta = np.arange(0, 180, 0.25)
    
    # Perform the Radon transform on the binary image
    try:
        radon_transform = transform.radon(edges_cleaned.astype(float), theta=theta, circle=False)
        
        # Find the peak in the Radon transform which corresponds to the HL
        peak_idx = np.unravel_index(np.argmax(radon_transform), radon_transform.shape)
        
        # Get the distance and angle corresponding to the peak
        center = radon_transform.shape[0] // 2
        dist = peak_idx[0] - center
        th = theta[peak_idx[1]]
        
        # Calculate the orientation of the horizon line (theta_k)
        x_2 = 90 - th
        
        # Calculate the vertical position of the horizon line (y_k)
        if dist != 0:
            uy = np.sign(dist) * np.sin(np.radians(th))
            if uy != 0:
                A = abs(dist) / uy
            else:
                A = 0
        else:
            A = 0
        
        x_1 = -A + rgb.shape[0] / 2
        
        return np.array([x_1, x_2])
    
    except:
        # Fallback if Radon transform fails
        return np.array([rgb.shape[0] / 2, 0])


def get_error(x, rgb, x_mat, y_mat, n_bins, w, delta_h):
    """
    Calculates the error metric between the sky and sea regions.
    """
    # Identify the sky and sea regions based on the estimated HL state
    ind_sky, ind_sea = find_indices_of_regions(x, w, x_mat, y_mat, delta_h)
    
    # Convert the RGB image to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if len(rgb.shape) == 3 else rgb
    
    # Calculate the histograms of the sky and sea regions
    if np.any(ind_sky) and np.any(ind_sea):
        hist_sky, _ = np.histogram(gray[ind_sky], bins=n_bins, density=True)
        hist_sea, _ = np.histogram(gray[ind_sea], bins=n_bins, density=True)
        
        # Compute the error as the negative square root of the mean squared difference
        err = -np.sqrt(np.mean((hist_sky - hist_sea)**2))
    else:
        err = 0
    
    # Handle cases where there is no sky or sea region
    if np.isnan(err):
        err = 0
    
    return err


def find_indices_of_regions(x, w, x_mat, y_mat, delta_h):
    """
    Identifies the pixel indices for the sky and sea regions based on the estimated horizon line.
    """
    # Calculate the reference line (horizon line) based on the estimated state
    y_i = x[0]
    theta = x[1]
    c = y_i - np.tan(np.radians(theta)) * w / 2
    ind_ref = np.tan(np.radians(theta)) * x_mat + c
    
    # Calculate the upper and lower bounds around the horizon line
    ind_lower = np.tan(np.radians(theta)) * x_mat + c - delta_h
    ind_upper = np.tan(np.radians(theta)) * x_mat + c + delta_h
    
    # Identify the sky region as pixels above the reference line but below the upper bound
    ind_sky = (y_mat < ind_ref) & (y_mat > ind_lower)
    
    # Identify the sea region as pixels below the reference line but above the lower bound
    ind_sea = (y_mat > ind_ref) & (y_mat < ind_upper)
    
    return ind_sky, ind_sea


def roi_rect(rgb, x, min_h, w, h):
    """
    Determines a rectangular Region of Interest (ROI) around the estimated horizon line.
    """
    # Calculate the vertical positions along the horizon line across the frame width
    y = np.round(np.tan(np.radians(x[1])) * np.arange(1, w+1) + x[0] - np.tan(np.radians(x[1])) * w / 2)
    
    # Define the height limits of the ROI
    h_lims = [int(max(1, min(y) - min_h)), int(min(h, max(y) + min_h))]
    
    # Ensure valid limits
    h_lims[0] = max(0, h_lims[0])
    h_lims[1] = min(h, h_lims[1])
    if h_lims[1] <= h_lims[0]:
        h_lims[1] = min(h, h_lims[0] + 1)
    
    # Crop the image to create the ROI based on the height limits
    roi = rgb[h_lims[0]:h_lims[1], :, :]
    
    return roi, h_lims


def roi_tsm(rgb, x, yf_sd, tf_sd, z_score, w):
    """
    Generates the parallelogram ROI using Time Series Model forecasts.
    """
    AD = 0
    h = rgb.shape[0]
    
    # Calculate the horizon line using the forecasted theta and y values
    hl = np.tan(np.radians(x[1])) * np.array([1, w]) + x[0] - np.tan(np.radians(x[1])) * w / 2
    
    # Calculate the horizon line for an angled forecast
    hl_angled = np.tan(np.radians(x[1] + tf_sd)) * w + x[0] - np.tan(np.radians(x[1] + tf_sd)) * w / 2
    
    # Round the horizon line positions to integer values
    hl = np.round(hl).astype(int)
    
    # Calculate the vertical distance for the ROI height
    delh = round(z_score * (yf_sd + abs(hl_angled - hl[1])))
    
    # Define the ROI boundaries as a parallelogram
    try:
        from skimage.draw import polygon
        r = np.array([hl[0] - delh, hl[0] + delh, hl[1] + delh, hl[1] - delh])
        c = np.array([0, 0, w-1, w-1])
        
        # Ensure coordinates are within bounds
        r = np.clip(r, 0, h-1)
        c = np.clip(c, 0, w-1)
        
        # Create binary mask
        bw = np.zeros((h, w), dtype=bool)
        rr, cc = polygon(r, c, (h, w))
        bw[rr, cc] = True
        
        # Check if the ROI is within the frame
        if np.sum(bw) / bw.size > 0.0025:
            # Extract the ROI from the RGB image using the binary mask
            roi_indices = np.where(bw)
            if len(roi_indices[0]) > 0:
                roi = rgb[bw]
                roi = roi.reshape(-1, 3) if len(rgb.shape) == 3 else roi.reshape(-1, 1)
            else:
                roi = np.array([])
                AD = 1
        else:
            roi = np.array([])
            AD = 1
            bw = np.array([])
            delh = 0
            
    except:
        # Fallback to rectangular ROI if parallelogram fails
        roi, _ = roi_rect(rgb, x, int(delh) if 'delh' in locals() else 10, w, h)
        bw = np.zeros((h, w), dtype=bool)
        delh = 10
    
    return roi, AD, bw, delh


def roi_plot(bw):
    """
    Plots the Region of Interest (ROI) on the current video frame.
    """
    if bw.size == 0:
        return
    
    try:
        # Find contours in the binary image
        contours = measure.find_contours(bw.astype(float), 0.5)
        
        # Plot the largest contour
        if contours:
            largest_contour = max(contours, key=len)
            plt.plot(largest_contour[:, 1], largest_contour[:, 0], 'y-', linewidth=3, label='ROI')
    except:
        pass


if __name__ == "__main__":
    # Check if video files exist
    videos_to_try = [
        'VIS_Onboard/Videos/MVI_0788_VIS_OB.avi',
        'Buoy/buoyGT_2_5_3_4.avi'
    ]
    
    # Find an existing video file
    vid_name = None
    N = 60
    
    for video_path in videos_to_try:
        if os.path.exists(video_path):
            vid_name = video_path
            if 'Buoy' in video_path:
                N = 20
            break
    
    if vid_name is None:
        print("No video files found. Please check the following paths:")
        for video_path in videos_to_try:
            print(f"  - {video_path}")
        print("\nEither:")
        print("1. Download the datasets as mentioned in README.md")
        print("2. Or modify the vid_name variable below to point to your video file")
        
        # You can manually set your video path here:
        # vid_name = 'path/to/your/video.mp4'
        # N = 30
    
    if vid_name:
        print(f"\nUsing video: {vid_name}")
        print(f"Listener block frames: {N}")
        print("Starting horizon line detection...")
        print("Press Ctrl+C to stop processing early.\n")
        
        try:
            hl_est = hl_detect_tsa(vid_name, N)
            print(f"\nHorizon line detection completed successfully!")
            print(f"Result shape: {hl_est.shape}")
            print("First 5 estimates (Y position, Angle in degrees):")
            print(hl_est[:5])
            print("Last 5 estimates:")
            print(hl_est[-5:])
            
            # Keep the plot window open
            print("\nClose the plot window to exit.")
            plt.show(block=True)
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please ensure you have the required dependencies:")
            print("pip install numpy opencv-python matplotlib scikit-image scipy statsmodels arch")
    else:
        print("Please set a valid video file path in the script.")
