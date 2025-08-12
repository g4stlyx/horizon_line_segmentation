"""
================================================================================
3-Class U-Net Horizon Segmentation + Object Distance Calculator (HF RT-DETR)
================================================================================

This runner uses a 3-class U-Net (0=water, 1=sky, 2=object) to estimate the
horizon line from sky/water, while using RT-DETR for per-object detections and
distance measurement to the horizon.

What it does:
- Computes horizon from the sky/water boundary, ignoring object pixels.
- Optionally detects objects with RT-DETR and measures signed distance from each
  object's center to the horizon line.

Key Arguments:
    --band-up / --band-down : vertical search band around horizon (pixels)
    --min-area              : minimum blob area for segmentation-based objects
    --fov-vertical          : adds vertical angle column (deg)
    --frame-step            : process every N-th frame for video
    --export-csv            : export results to CSV
    --prefer-rtdetr         : prefer using RT-DETR boxes over segmentation objects

Example RT-DETR model:
  rtdetr_obj_det_model/final_best_model (directory with config.json & model.safetensors)

Examples:
  Video: py z_unet_runner_dist_calc_rtdetr_obj_det_3_class.py --video .\0example_data\VIS_Onshore\Videos\MVI_1614_VIS.avi --rtdetr-model rtdetr_obj_det_model/final_best_model --prefer-rtdetr --rtdetr-conf 0.25 --rtdetr-classes 0,1,2,4,6,7,8 --rtdetr-interval 5 --band-up 160 --band-down 140 --min-area 350 --show-horizon --save
  Image: py z_unet_runner_dist_calc_rtdetr_obj_det_3_class.py --image .\0example_data\images\image.jpeg --rtdetr-model rtdetr_obj_det_model/final_best_model --prefer-rtdetr --rtdetr-conf 0.25 --rtdetr-classes 0,1,2,4,6,7,8 --rtdetr-interval 5 \--band-up 160 --band-down 140 --min-area 350 --show-horizon --save
"""

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
import csv
try:
    # Hugging Face Transformers RT-DETR checkpoint (directory with config.json + model.safetensors)
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    HAS_HF_RTDETR = True
except Exception:
    HAS_HF_RTDETR = False

# ==============================================================================
# Step 1: Define the U-Net Model Architecture
# ==============================================================================
# This class definition must be identical to the one used for training.
class UNet(nn.Module):
    def __init__(self, n_classes=3, dropout_rate=0.3):
        super().__init__()
        self.n_classes = n_classes
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        base = list(resnet.children())
        self.layer0 = nn.Sequential(*base[:3])
        self.layer1 = nn.Sequential(*base[3:5])
        self.layer2 = base[5]
        self.layer3 = base[6]
        self.layer4 = base[7]
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = self._block(256 + 256, 256, dropout_rate)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self._block(128 + 128, 128, dropout_rate)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._block(64 + 64, 64, dropout_rate)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = self._block(64 + 64, 32, 0.0)
        self.final_upconv = nn.ConvTranspose2d(32, 32, 2, 2)
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def _block(self, in_c, out_c, dropout_rate=0.0):
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d4 = self.upconv4(x4); d4 = torch.cat([d4, x3], 1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); d3 = torch.cat([d3, x2], 1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); d2 = torch.cat([d2, x1], 1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); d1 = torch.cat([d1, x0], 1); d1 = self.dec1(d1)
        d0 = self.final_upconv(d1)
        return self.final_conv(d0)

# ==============================================================================
# Step 2: Helper Functions and Setup
# ==============================================================================

def create_overlay(image_bgr, mask):
    overlay = np.zeros_like(image_bgr, dtype=np.uint8)
    # BGR colors: sky (blue), water (green), object (orange)
    overlay[mask == 1] = (255, 0, 0)      # sky
    overlay[mask == 0] = (0, 255, 0)      # water
    overlay[mask == 2] = (0, 165, 255)    # object
    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

def detect_ships_for_correction(image_cv, mask):
    """
    Optional heuristic correction for ships near horizon when only segmentation
    is used. In 3-class setup with RT-DETR preferred, this is typically disabled.
    """
    h, w = image_cv.shape[:2]
    corrected_mask = mask.copy()
    
    # Convert to different color spaces for ship detection
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    
    # Find the approximate horizon line from the current mask
    sky_rows = np.where(np.any(mask == 1, axis=1))[0]
    if len(sky_rows) > 0:
        horizon_y = np.max(sky_rows)  # Bottom edge of sky region
    else:
        horizon_y = h // 2  # Fallback
    
    # Search for ships in the sky region (incorrectly classified)
    search_top = max(0, horizon_y - 100)
    search_bottom = horizon_y + 20
    
    # Method 1: Detect dark objects (ship hulls)
    roi_gray = gray[search_top:search_bottom, :]
    if roi_gray.size > 0:
        # Use adaptive thresholding to find dark objects
        thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 10)
        
        # Find dark regions (ships are typically dark silhouettes)
        dark_regions = 255 - thresh
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and filter by size
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Minimum ship size
                x, y, cw, ch = cv2.boundingRect(contour)
                # Check aspect ratio (ships are usually wider than tall)
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio > 1.2 and area > 500:
                    # Mark as object class (2)
                    corrected_mask[search_top + y:search_top + y + ch, x:x + cw] = 2
    
    # Method 2: Color-based ship detection
    roi_hsv = hsv[search_top:search_bottom, :]
    if roi_hsv.size > 0:
        # Detect typical ship colors
        # White/light structures (superstructures)
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(roi_hsv, white_lower, white_upper)
        
        # Dark structures (hulls)  
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 80])
        dark_mask_color = cv2.inRange(roi_hsv, dark_lower, dark_upper)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(white_mask, dark_mask_color)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and correct mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Mark as object class (2)
                corrected_mask[search_top + y:search_top + y + ch, x:x + cw] = 2
    
    return corrected_mask

def predict(model, image_pil, device, transform, apply_ship_correction=True):
    orig_size = image_pil.size
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    pred = torch.argmax(out, 1).cpu().squeeze(0)
    resize_back = transforms.Resize(orig_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)
    mask = resize_back(pred.unsqueeze(0)).squeeze(0).numpy()
    if apply_ship_correction:
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        mask = detect_ships_for_correction(image_bgr, mask)
    return mask

def compute_horizon_line(mask, gap_threshold=15, smooth_window=21):
    """Compute horizon strictly as the sky/water separator, bridging over objects.

    Strategy per column:
    - y_sky_bottom  = last index where mask == 1 (sky)
    - y_water_top   = first index where mask == 0 (water)
    - If both exist, candidate horizon = round((y_sky_bottom + y_water_top) / 2)
      If the gap between them is too large (object occlusion), mark as invalid.
    - If only one exists, use it as a fallback candidate.
    - After building candidates, linearly interpolate invalid columns from neighbors
      and then apply a moving-average smoothing.
    """
    h, w = mask.shape
    horizon = np.full(w, np.nan, dtype=np.float32)

    for col in range(w):
        col_vals = mask[:, col]
        sky_rows = np.where(col_vals == 1)[0]
        water_rows = np.where(col_vals == 0)[0]

        y_sky_bottom = int(sky_rows.max()) if sky_rows.size > 0 else None
        y_water_top = int(water_rows.min()) if water_rows.size > 0 else None

        if y_sky_bottom is not None and y_water_top is not None:
            # If there is an object occluding between sky and water, the gap can be large
            gap = y_water_top - y_sky_bottom
            if gap >= 0:
                if gap <= gap_threshold:
                    horizon[col] = 0.5 * (y_sky_bottom + y_water_top)
                else:
                    # Large occlusion span -> mark invalid and interpolate later
                    horizon[col] = np.nan
            else:
                # Overlap due to noise; choose the average anyway
                horizon[col] = 0.5 * (y_sky_bottom + y_water_top)
        elif y_sky_bottom is not None:
            # Only sky present -> take bottom-most sky as horizon estimate
            horizon[col] = float(y_sky_bottom)
        elif y_water_top is not None:
            # Only water present -> take top-most water as horizon estimate
            horizon[col] = float(y_water_top)
        else:
            # Neither sky nor water (unlikely) -> leave NaN to be interpolated
            horizon[col] = np.nan

    # Interpolate NaNs from nearest valid neighbors
    xs = np.arange(w)
    valid = ~np.isnan(horizon)
    if valid.any():
        # Use edge holds for ends
        left_idx = np.where(valid)[0][0]
        right_idx = np.where(valid)[0][-1]
        # Fill edges
        horizon[:left_idx] = horizon[left_idx]
        horizon[right_idx+1:] = horizon[right_idx]
        # Linear interpolate interior NaNs
        nan_mask = np.isnan(horizon)
        if nan_mask.any():
            horizon[nan_mask] = np.interp(xs[nan_mask], xs[valid], horizon[valid])
    else:
        # Fallback if all NaN
        horizon[:] = h // 2

    # Moving average smoothing
    k = max(3, smooth_window | 1)  # make odd
    pad = k // 2
    smoothed = horizon.copy()
    for i in range(w):
        l = max(0, i - pad)
        r = min(w, i + pad + 1)
        smoothed[i] = np.mean(horizon[l:r])
    return horizon.astype(np.int32), smoothed.astype(np.int32)

def detect_objects_near_horizon(
    mask,
    horizon_smoothed,
    band_up=120,
    band_down=120,
    min_area=150,
    max_width_frac=0.9,
):
    """Detect object-class blobs within a vertical band around the horizon and
    compute their vertical distance to the horizon using object centers.
    """
    h, w = mask.shape

    # Construct a band mask per column: [horizon - band_up, horizon + band_down]
    row_indices = np.arange(h)[:, None]
    top_band = np.clip(horizon_smoothed - band_up, 0, h - 1)
    bot_band = np.clip(horizon_smoothed + band_down, 0, h - 1)
    band_mask = (row_indices >= top_band[None, :]) & (row_indices <= bot_band[None, :])

    # Objects are class 2 within the band
    obj_mask = ((mask == 2) & band_mask).astype(np.uint8)
    if obj_mask.sum() == 0:
        return []

    # Merge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    objects = []
    for i in range(1, num):  # skip background 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w_box = int(stats[i, cv2.CC_STAT_WIDTH])
        h_box = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Filter out the large sea strip spanning most of the width
        if w_box > max_width_frac * w:
            continue

        cx = int(np.clip(int(round(centroids[i][0])), 0, w - 1))
        cy = int(np.clip(int(round(centroids[i][1])), 0, h - 1))
        horizon_y = int(horizon_smoothed[cx])

        # Keep only if the CENTER lies inside the vertical band
        tolerance = 5
        if not (horizon_y - band_up - tolerance <= cy <= horizon_y + band_down + tolerance):
            continue

        objects.append({
            'bbox': (x, y, w_box, h_box),
            'center': (cx, cy),
            'horizon_y': horizon_y,
            'pixel_distance': int(abs(horizon_y - cy)),
            'signed_distance': int(cy - horizon_y),
        })

    return objects

def annotate(image_bgr, mask, horizon_smoothed, objects, show_horizon=True, fps=None):
    vis = create_overlay(image_bgr, mask)
    h, w = mask.shape
    if show_horizon:
        for x in range(0, w-1):
            y1 = horizon_smoothed[x]
            y2 = horizon_smoothed[x+1]
            cv2.line(vis, (x, y1), (x+1, y2), (0,255,255), 2)
    
    # COCO class names for reference
    coco_names = {0: 'person', 2: 'car', 5: 'bus', 7: 'truck', 8: 'boat'}
    
    for idx, obj in enumerate(objects):
        x,y,bw,bh = obj['bbox']
        cx, cy = obj['center']
        horizon_y = obj['horizon_y']
        dist = obj['pixel_distance']
        
        # Choose color based on detection method
        if obj.get('det') == 'rtdetr':
            box_color = (0, 165, 255)  # Orange for RT-DETR detections
            class_id = obj.get('class_id', -1)
            conf = obj.get('confidence', 0.0)
            class_name = coco_names.get(class_id, f'cls{class_id}')
            sdist = obj.get('signed_distance', cy - horizon_y)
            label = f"ID{idx} {class_name} {conf:.2f} d={dist}px s={sdist}"
        else:
            box_color = (0, 255, 0)  # Green for segmentation detections
            sdist = obj.get('signed_distance', cy - horizon_y)
            label = f"ID{idx} d={dist}px s={sdist}"
        
        cv2.rectangle(vis, (x,y), (x+bw, y+bh), box_color, 2)
        cv2.circle(vis, (cx, cy), 4, (0,0,255), -1)
        # line from center vertically to horizon
        cv2.line(vis, (cx, cy), (cx, horizon_y), (255,0,255), 1)
        
        # Put label with background for better visibility
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(vis, (x, y-label_size[1]-4), (x+label_size[0], y), box_color, -1)
        cv2.putText(vis, label, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    
    if fps is not None:
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    
    # Add detection count info
    det_count = sum(1 for obj in objects if obj.get('det') == 'rtdetr')
    seg_count = len(objects) - det_count
    info_text = f"RT-DETR: {det_count}, Seg: {seg_count}"
    cv2.putText(vis, info_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    
    return vis
def detect_objects_rtdetr(
    frame_bgr,
    horizon_smoothed,
    hf_processor,
    hf_model,
    conf=0.25,
    class_filter=None,
    band_up=120,
    band_down=120,
):
    """Detect objects with Hugging Face RT-DETR (AutoModelForObjectDetection).

    Keeps detections whose bbox center lies within the horizon band and measures
    center-to-horizon distance.
    """
    objects = []
    if hf_processor is None or hf_model is None:
        return objects

    h, w = frame_bgr.shape[:2]
    # Convert BGR to RGB for HF processors
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        inputs = hf_processor(images=frame_rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**{k: v.to(hf_model.device) for k, v in inputs.items()})
        target_sizes = torch.tensor([(h, w)], device=hf_model.device)
        results = hf_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf)[0]
    except Exception as e:
        print(f"[DEBUG] HF RT-DETR inference failed: {e}")
        return objects

    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels")
    if boxes is None or scores is None or labels is None:
        return objects

    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy().astype(int)

    print(f"[DEBUG] HF RT-DETR found {len(boxes)} detections over threshold {conf}")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        class_id = int(labels[i])
        confidence = float(scores[i])

        if class_filter is not None and len(class_filter) > 0 and class_id not in class_filter:
            continue

        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        if cx < 0 or cx >= w:
            continue

        horizon_y = int(horizon_smoothed[cx])
        band_top = horizon_y - band_up
        band_bottom = horizon_y + band_down
        if not (band_top <= cy <= band_bottom):
            continue

        pixel_distance = int(abs(horizon_y - cy))

        objects.append({
            'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            'center': (cx, cy),
            'horizon_y': horizon_y,
            'pixel_distance': pixel_distance,
            'confidence': confidence,
            'class_id': class_id,
            'det': 'rtdetr'
        })

    print(f"[DEBUG] Filtered to {len(objects)} objects in band")
    return objects

def _resolve_rtdetr_weights(path_or_dir: str) -> str:
    """Return a usable weights file path for RT-DETR.

    - If a file is given and endswith .pt/.pth -> return it.
    - If a directory is given -> search for first *.pt or *.pth inside (recursive).
    - If a .zip is given -> ask user to unzip and point to .pt file.
    """
    if not path_or_dir:
        return None
    if os.path.isfile(path_or_dir):
        lower = path_or_dir.lower()
        if lower.endswith('.pt') or lower.endswith('.pth'):
            return path_or_dir
        if lower.endswith('.zip'):
            print(f"[WARN] Provided a zip archive: {path_or_dir}. Please unzip and pass the .pt/.pth file.")
            return None
        print(f"[WARN] Unsupported file type for RT-DETR weights: {path_or_dir}")
        return None
    if os.path.isdir(path_or_dir):
        # Try common subdir names first
        candidates = []
        for root, _dirs, files in os.walk(path_or_dir):
            for f in files:
                if f.lower().endswith(('.pt', '.pth')):
                    candidates.append(os.path.join(root, f))
        if candidates:
            # Prefer files with 'best' in name
            candidates.sort(key=lambda p: (0 if 'best' in os.path.basename(p).lower() else 1, len(p)))
            print(f"[INFO] Using RT-DETR weights: {candidates[0]}")
            return candidates[0]
        print(f"[WARN] No .pt/.pth weights found under directory: {path_or_dir}")
    return None

def maybe_write_csv_header(csv_path, include_angle):
    if not csv_path:
        return
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['source','frame','object_id','cx','cy','horizon_y','pixel_distance']
            if include_angle:
                header.append('vertical_angle_deg')
            header.append('signed_distance')
            writer.writerow(header)

def append_csv_rows(csv_path, rows):
    if not csv_path or not rows:
        return
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description='3-Class U-Net Horizon Distance Calculator')
    parser.add_argument('--model-path', type=str, default='4best_unet_rtdetr_aware_smd_3cls.pth', help='Path to trained 3-class U-Net weights')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--folder', type=str, help='Folder of images')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--camera', action='store_true', help='Use live camera')
    parser.add_argument('--save', action='store_true', help='Save annotated outputs')
    parser.add_argument('--export-csv', type=str, help='CSV file to append distance data')
    parser.add_argument('--min-area', type=int, default=150, help='Minimum blob area to count as object')
    parser.add_argument('--band-up', type=int, default=120, help='Pixels above horizon to search for objects')
    parser.add_argument('--band-down', type=int, default=120, help='Pixels below horizon to search for objects')
    parser.add_argument('--no-ship-correction', dest='no_ship_correction', action='store_true', help='Disable ship-aware correction')
    parser.add_argument('--fov-vertical', dest='fov_vertical', type=float, default=None, help='Vertical FOV in degrees (optional)')
    parser.add_argument('--frame-step', type=int, default=1, help='Process every Nth frame for video/camera')
    parser.add_argument('--show-horizon', action='store_true', help='Draw horizon polyline')
    parser.add_argument('--img-size', type=int, default=256, help='Model input square size')
    # RT-DETR integration options
    parser.add_argument('--rtdetr-model', type=str, help='Path to HF RT-DETR checkpoint directory (contains config.json & model.safetensors)')
    parser.add_argument('--rtdetr-conf', type=float, default=0.25, help='RT-DETR confidence threshold')
    parser.add_argument('--rtdetr-classes', type=str, help='Comma-separated class ids to keep (optional)')
    parser.add_argument('--prefer-rtdetr', action='store_true', help='Use RT-DETR detections instead of segmentation blobs if available')
    parser.add_argument('--rtdetr-interval', type=int, default=1, help='Run RT-DETR every N processed frames and reuse between (improves FPS)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load U-Net model
    model = UNet(n_classes=3).to(device)
    # Optional HF RT-DETR model load
    hf_processor = None
    hf_model = None
    rtdetr_class_filter = None
    if args.rtdetr_model:
        if not HAS_HF_RTDETR:
            print('[WARN] transformers not installed; RT-DETR disabled.')
        else:
            try:
                hf_processor = AutoImageProcessor.from_pretrained(args.rtdetr_model)
                hf_model = AutoModelForObjectDetection.from_pretrained(args.rtdetr_model)
                hf_model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                hf_model.to(device)
                print(f"HF RT-DETR loaded from: {args.rtdetr_model}")
            except Exception as e:
                print(f"[WARN] Failed to load HF RT-DETR: {e}")
                hf_processor = None
                hf_model = None
    if args.rtdetr_classes and rtdetr_class_filter is None:
        try:
            rtdetr_class_filter = [int(c.strip()) for c in args.rtdetr_classes.split(',') if c.strip().isdigit()]
        except Exception:
            print('[WARN] Could not parse --rtdetr-classes; ignoring.')
    prefer_rtdetr_mode = args.prefer_rtdetr
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model not found: {args.model_path}")
        return
    model.eval()
    print('Model loaded.')

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    include_angle = args.fov_vertical is not None
    maybe_write_csv_header(args.export_csv, include_angle)

    def process_image(path, source_label, frame_idx=None):
        start_t = time.time()
        image_pil = Image.open(path).convert('RGB')
        mask = predict(model, image_pil, device, transform, apply_ship_correction=not args.no_ship_correction)
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        _, horizon_smoothed = compute_horizon_line(mask)
        # Choose detection source
        if prefer_rtdetr_mode and hf_model is not None and hf_processor is not None:
            objects = detect_objects_rtdetr(
                image_bgr,
                horizon_smoothed,
                hf_processor,
                hf_model,
                conf=args.rtdetr_conf,
                class_filter=rtdetr_class_filter,
                band_up=args.band_up,
                band_down=args.band_down,
            )
            if not objects:  # fallback to segmentation-based blobs
                objects = detect_objects_near_horizon(
                    mask,
                    horizon_smoothed,
                    band_up=args.band_up,
                    band_down=args.band_down,
                    min_area=args.min_area,
                )
        else:
            objects = detect_objects_near_horizon(
                mask,
                horizon_smoothed,
                band_up=args.band_up,
                band_down=args.band_down,
                min_area=args.min_area,
            )
        fps = 1.0 / max(1e-6, (time.time() - start_t))
        annotated = annotate(image_bgr, mask, horizon_smoothed, objects, args.show_horizon, fps=fps)
        rows = []
        for idx, obj in enumerate(objects):
            cx, cy = obj['center']
            horizon_y = obj['horizon_y']
            dist = obj['pixel_distance']
            sdist = obj.get('signed_distance', cy - horizon_y)
            row = [source_label, frame_idx if frame_idx is not None else 0, idx, cx, cy, horizon_y, dist]
            if include_angle:
                h_img = mask.shape[0]
                deg_per_pixel = args.fov_vertical / h_img
                row.append(dist * deg_per_pixel)
            row.append(sdist)
            rows.append(row)
        append_csv_rows(args.export_csv, rows)
        return annotated

    # IMAGE MODE
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        annotated = process_image(args.image, os.path.basename(args.image))
        if args.save:
            out_path = os.path.splitext(args.image)[0] + '_dist.jpg'
            cv2.imwrite(out_path, annotated)
            print(f"Saved: {out_path}")
        cv2.imshow('Horizon Distance', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # FOLDER MODE
    if args.folder:
        exts = ['*.jpg','*.jpeg','*.png','*.bmp']
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(args.folder, e)))
            paths.extend(glob.glob(os.path.join(args.folder, e.upper())))
        if not paths:
            print('No images found.')
            return
        out_dir = None
        if args.save:
            out_dir = os.path.join(args.folder, 'output_dist')
            os.makedirs(out_dir, exist_ok=True)
        for p in paths:
            annotated = process_image(p, os.path.basename(p))
            cv2.imshow('Horizon Distance', annotated)
            if args.save and out_dir:
                out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + '_dist.jpg')
                cv2.imwrite(out_path, annotated)
            if cv2.waitKey(75) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    # VIDEO / CAMERA MODE
    if args.video or args.camera:
        if args.video:
            if not os.path.exists(args.video):
                print(f"Video not found: {args.video}")
                return
            cap = cv2.VideoCapture(args.video)
            source_label = os.path.basename(args.video)
        else:
            cap = cv2.VideoCapture(0)
            source_label = 'camera'
        if not cap.isOpened():
            print('Cannot open video source.')
            return
        writer = None
        if args.save:
            w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = 'video_horizon_dist.mp4'
            writer = cv2.VideoWriter(out_path, fourcc, fps/max(1,args.frame_step), (w_frame, h_frame))
            print(f"Saving video to {out_path}")
        frame_idx = 0
        # Cache for RT-DETR results when using --rtdetr-interval > 1
        last_det_objects = []
        last_det_used_at = -999999
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % args.frame_step == 0:
                frame_start = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                mask = predict(model, image_pil, device, transform, apply_ship_correction=not args.no_ship_correction)
                _, horizon_smoothed = compute_horizon_line(mask)
                if prefer_rtdetr_mode and hf_model is not None and hf_processor is not None:
                    run_det_now = (frame_idx // max(1, args.frame_step)) % max(1, args.rtdetr_interval) == 0
                    if run_det_now:
                        objects = detect_objects_rtdetr(
                            frame,
                            horizon_smoothed,
                            hf_processor,
                            hf_model,
                            conf=args.rtdetr_conf,
                            class_filter=rtdetr_class_filter,
                            band_up=args.band_up,
                            band_down=args.band_down,
                        )
                        last_det_objects = objects
                        last_det_used_at = frame_idx
                    else:
                        # Reuse last detector boxes but recompute distances with current horizon
                        objects = []
                        for o in last_det_objects:
                            cx, cy = o['center']
                            if cx < 0 or cx >= frame.shape[1]:
                                continue
                            horizon_y = int(horizon_smoothed[cx])
                            if not (horizon_y - args.band_up <= cy <= horizon_y + args.band_down):
                                continue
                            objects.append({
                                'bbox': o['bbox'],
                                'center': (cx, cy),
                                'horizon_y': horizon_y,
                                'pixel_distance': int(abs(horizon_y - cy)),
                                'signed_distance': int(cy - horizon_y),
                                'confidence': o.get('confidence', 0.0),
                                'class_id': o.get('class_id', -1),
                                'det': 'rtdetr'
                            })
                    if not objects:
                        objects = detect_objects_near_horizon(
                            mask,
                            horizon_smoothed,
                            band_up=args.band_up,
                            band_down=args.band_down,
                            min_area=args.min_area,
                        )
                else:
                    objects = detect_objects_near_horizon(
                        mask,
                        horizon_smoothed,
                        band_up=args.band_up,
                        band_down=args.band_down,
                        min_area=args.min_area,
                    )
                fps_frame = 1.0 / max(1e-6, (time.time() - frame_start))
                annotated = annotate(frame, mask, horizon_smoothed, objects, args.show_horizon, fps=fps_frame)
                rows = []
                for idx, obj in enumerate(objects):
                    cx, cy = obj['center']
                    horizon_y = obj['horizon_y']
                    dist = obj['pixel_distance']
                    sdist = obj.get('signed_distance', cy - horizon_y)
                    row = [source_label, frame_idx, idx, cx, cy, horizon_y, dist]
                    if include_angle:
                        h_img = mask.shape[0]
                        deg_per_pixel = args.fov_vertical / h_img
                        row.append(dist * deg_per_pixel)
                    row.append(sdist)
                    rows.append(row)
                append_csv_rows(args.export_csv, rows)
                display_frame = annotated
            else:
                display_frame = frame
            if writer:
                writer.write(display_frame)
            cv2.imshow('Horizon Distance', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t0
        print(f"Done. Frames processed: {frame_idx} in {elapsed:.1f}s")
        return

    print('No input specified. Use --image, --folder, --video or --camera.')

if __name__ == '__main__':
    main()
