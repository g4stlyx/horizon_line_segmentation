"""
================================================================================
3-Class U-Net Horizon Segmentation + Object Distance Calculator (YOLO)
================================================================================

This runner uses a 3-class U-Net (0=water, 1=sky, 2=object) to estimate the
horizon line from sky/water, while using YOLO for per-object detections and
distance measurement to the horizon.

What it does:
- Computes horizon from the sky/water boundary, ignoring object pixels.
- Optionally detects objects with YOLO and measures signed distance from each
  object's center to the horizon line.

Key Arguments:
    --band-up / --band-down : vertical search band around horizon (pixels)
    --min-area              : minimum blob area for segmentation-based objects
    --fov-vertical          : adds vertical angle column (deg)
    --frame-step            : process every N-th frame for video
    --export-csv            : export results to CSV
    --prefer-yolo           : prefer using YOLO boxes over segmentation objects

Example YOLO model:
  obj_det_havelsan.pt

Examples:
  Video: py z_unet_runner_dist_calc_yolo_obj_det_3_class.py --video .\0example_data\VIS_Onshore\Videos\MVI_1614_VIS.avi --yolo-model obj_det_havelsan.pt --prefer-yolo --yolo-conf 0.25 --yolo-interval 5 --band-up 160 --band-down 140 --min-area 350 --show-horizon --save
  Image: py z_unet_runner_dist_calc_yolo_obj_det_3_class.py --image .\0example_data\images\image.jpeg --yolo-model obj_det_havelsan.pt --prefer-yolo --yolo-conf 0.25 --yolo-interval 5 --band-up 160 --band-down 140 --min-area 350 --show-horizon --save
"""

#! py 5z_unet_runner_dist_calc_yolo_obj_det_3_class.py --video .\0example_data\VIS_Onshore\Videos\MVI_1614_VIS.avi --yolo-model obj_det_havelsan.pt --prefer-yolo --yolo-conf 0.25 --yolo-interval 5 --band-up 160 --band-down 140 --min-area 350 --camera-height-m 12 --fov-vertical 30 --refraction-k 1.3333 --distance-units m --show-horizon --save
#! py 5z_unet_runner_dist_calc_rtdetr_obj_det_3_class.py --video .\0data\havelsan.mkv --yolo-model .\0data\obj_det_havelsan.pt --prefer-yolo --yolo-conf 0.25 --yolo-interval 5 --band-up 160 --band-down 140 --min-area 350 --show-horizon --save
#* You must provide --fov-vertical and a realistic --camera-height-m for physical distances to be meaningful. With missing FOV, real-world distances are skipped but pixel gap is reported.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import argparse
import time
import os
import glob
import csv
import math
try:
    # YOLO for object detection
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

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

def predict(model, image_pil, device, transform, apply_ship_correction=True, use_fp16=False):
    orig_size = image_pil.size
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        if use_fp16 and device.type == 'cuda':
            # Mixed precision: keep weights in fp32, run ops in fp16 where safe
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(tensor)
        else:
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

    sky = (mask == 1)
    water = (mask == 0)

    # Vectorized bottom-most sky index per column
    sky_any = sky.any(axis=0)
    sky_rev = sky[::-1, :]
    sky_bottom_from_bottom = np.argmax(sky_rev, axis=0)
    y_sky_bottom = (h - 1 - sky_bottom_from_bottom).astype(np.float32)
    y_sky_bottom[~sky_any] = np.nan

    # Vectorized top-most water index per column
    water_any = water.any(axis=0)
    y_water_top = np.argmax(water, axis=0).astype(np.float32)
    y_water_top[~water_any] = np.nan

    # Compute gaps and candidates
    both_valid = ~np.isnan(y_sky_bottom) & ~np.isnan(y_water_top)
    gap = np.full(w, np.nan, dtype=np.float32)
    gap[both_valid] = y_water_top[both_valid] - y_sky_bottom[both_valid]

    # Average where the gap is small and non-negative
    good_gap = (gap >= 0) & (gap <= gap_threshold)
    horizon[good_gap] = 0.5 * (y_sky_bottom[good_gap] + y_water_top[good_gap])

    # Fallbacks where only one side is available
    only_sky = ~np.isnan(y_sky_bottom) & np.isnan(y_water_top)
    only_water = np.isnan(y_sky_bottom) & ~np.isnan(y_water_top)
    horizon[only_sky] = y_sky_bottom[only_sky]
    horizon[only_water] = y_water_top[only_water]

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

    # Moving average smoothing via convolution
    k = int(max(3, smooth_window | 1))  # make odd
    kernel = np.ones(k, dtype=np.float32) / float(k)
    # Pad by edge values to preserve endpoints
    pad = k // 2
    padded = np.pad(horizon, (pad, pad), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return horizon.astype(np.int32), smoothed.astype(np.int32)

def compute_depression_angle_and_range_m(
    signed_pixel_offset: int,
    image_height_px: int,
    fov_vertical_deg: float,
    camera_height_m: float,
    effective_earth_radius_m: float,
) -> tuple:
    """
    Convert vertical pixel offset from the horizon (positive = below) to:
      - depression angle from horizon (deg)
      - surface range to a sea-level target (meters)
      - gap from the geometric horizon point (meters)

    Uses a robust ray–sphere intersection in the vertical plane with effective
    Earth radius (refraction-adjusted). This is stable and physically correct.
    """
    if fov_vertical_deg is None or image_height_px <= 0:
        return None, None, None

    # Angle below horizon for this pixel offset (signed from caller).
    deg_per_pixel = fov_vertical_deg / float(image_height_px)
    depression_deg = signed_pixel_offset * deg_per_pixel

    R = float(effective_earth_radius_m)
    h = max(0.0, float(camera_height_m))

    # Minimum depression to just see the horizon
    theta0 = math.acos(max(-1.0, min(1.0, R / (R + h))))
    s0 = R * theta0
    alpha0 = theta0  # depression from horizontal to horizon ray

    # Total depression from horizontal to the viewing ray
    alpha = alpha0 + math.radians(max(0.0, depression_deg))
    if alpha <= alpha0:
        return depression_deg, None, 0.0

    # Ray-sphere intersection: observer O=(0,0,R+h), dir v=(cos a,0,-sin a)
    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    b = (R + h) * sin_a
    disc = b * b - (2.0 * R * h + h * h)
    if disc < 0.0:
        # Numerical guard
        return depression_deg, None, 0.0
    t = b + math.sqrt(disc)  # forward intersection along the viewing ray
    if t <= 0:
        return depression_deg, None, 0.0

    # Central angle to intersection point
    pz = (R + h) - t * sin_a
    cos_theta = max(-1.0, min(1.0, pz / R))
    theta = math.acos(cos_theta)

    range_m = R * theta
    # Gap beyond the horizon arc along the surface
    gap_m = max(0.0, range_m - s0)
    return depression_deg, range_m, gap_m

def attach_ranges(
    objects: list,
    image_height_px: int,
    fov_vertical_deg: float,
    camera_height_m: float,
    effective_earth_radius_m: float,
    units: str,
    model: str = 'flat',
):
    """Augments each object dict with 'depression_angle_deg', 'range_m', 'gap_m', and 'units'.

    If the object center lies above the horizon (negative signed distance), we
    attempt to use the bbox bottom edge relative to the horizon to recover a
    positive depression angle and compute distance. This covers ships whose
    centers are above the horizon while their hulls extend below it.
    """
    for o in objects:
        # Center-based signed distance
        if 'signed_distance' in o and o['signed_distance'] is not None:
            sdist_center = int(o['signed_distance'])
        else:
            cx, cy = o.get('center', (None, None))
            horizon_y = o.get('horizon_y', None)
            sdist_center = int(cy - horizon_y) if (cy is not None and horizon_y is not None) else 0

        # Alternative: bbox bottom against horizon
        horizon_y = o.get('horizon_y', None)
        bbox = o.get('bbox', None)
        sdist_bottom = None
        if bbox is not None and horizon_y is not None:
            bx, by, bw, bh = bbox
            bottom_y = by + bh
            sdist_bottom = int(bottom_y - int(horizon_y))

        # Choose the offset for range calc:
        # - If s (center) < 0, per request use |s| (positive) instead of skipping.
        # - Otherwise use s if > 0. If still not >0, try bbox-bottom.
        if sdist_center < 0:
            sdist_for_range = abs(sdist_center)
            o['range_source'] = 'center_abs'
        elif sdist_center > 0:
            sdist_for_range = sdist_center
            o['range_source'] = 'center'
        elif sdist_bottom is not None and sdist_bottom > 0:
            sdist_for_range = sdist_bottom
            o['range_source'] = 'bbox_bottom'
        else:
            sdist_for_range = 0
            o['range_source'] = 'none'

        dep_deg, rng_m, gap_m = compute_depression_angle_and_range_m(
            sdist_for_range,
            image_height_px,
            fov_vertical_deg,
            camera_height_m,
            effective_earth_radius_m,
        )
        # Optional flat-plane distance from camera to intersection with sea plane
        flat_m = None
        if fov_vertical_deg is not None and image_height_px > 0 and camera_height_m is not None:
            dep_rad_abs = abs((fov_vertical_deg / float(image_height_px)) * sdist_for_range) * (math.pi / 180.0)
            if dep_rad_abs > 1e-6:
                flat_m = float(camera_height_m) / math.tan(dep_rad_abs)
        o['depression_angle_deg'] = dep_deg
        # Choose which distance to expose as D via 'gap_m'
        if model == 'flat' and flat_m is not None and flat_m >= 0:
            o['range_m'] = None
            o['gap_m'] = flat_m
            o['units'] = units
        elif rng_m is not None and rng_m >= 0:
            o['range_m'] = float(rng_m)
            o['gap_m'] = float(gap_m) if gap_m is not None else None
            o['units'] = units
        else:
            o['range_m'] = None
            o['gap_m'] = None
            o['units'] = units

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
            # Prefer real-world distance if available (gap from horizon)
            rng_m = obj.get('gap_m', None)
            units = obj.get('units', 'm')
            if rng_m is not None:
                if units == 'nm':
                    rng_val = rng_m / 1852.0
                    rng_txt = f"{rng_val:.2f}nm"
                else:
                    rng_val = rng_m
                    rng_txt = f"{rng_val:.0f}m"
                label = f"ID{idx} {class_name} {conf:.2f} D={rng_txt} s={sdist}"
            else:
                # If real-world distance missing (e.g., no FOV), show pixel gap in m equivalent per-pixel angle
                if obj.get('depression_angle_deg') is not None and obj.get('units') == 'm':
                    # derive gap from angle magnitude when R known
                    dep = obj['depression_angle_deg']
                    # these values are redundantly computed in attach_ranges; if None, fallback to pixels
                    label = f"ID{idx} {class_name} {conf:.2f} s={sdist}"
                else:
                    label = f"ID{idx} {class_name} {conf:.2f} s={sdist}"
        else:
            box_color = (0, 255, 0)  # Green for segmentation detections
            sdist = obj.get('signed_distance', cy - horizon_y)
            rng_m = obj.get('gap_m', None)
            units = obj.get('units', 'm')
            if rng_m is not None:
                if units == 'nm':
                    rng_val = rng_m / 1852.0
                    rng_txt = f"{rng_val:.2f}nm"
                else:
                    rng_val = rng_m
                    rng_txt = f"{rng_val:.0f}m"
                label = f"ID{idx} D={rng_txt} s={sdist}"
            else:
                label = f"ID{idx} s={sdist}"
        
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
    det_count = sum(1 for obj in objects if obj.get('det') == 'yolo')
    seg_count = len(objects) - det_count
    info_text = f"YOLO: {det_count}, Seg: {seg_count}"
    cv2.putText(vis, info_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    
    return vis
def detect_objects_yolo(
    frame_bgr,
    horizon_smoothed,
    yolo_model,
    conf=0.25,
    band_up=120,
    band_down=120,
):
    """Detect objects with YOLO model.

    Keeps detections whose bbox center lies within the horizon band and measures
    center-to-horizon distance.
    """
    objects = []
    if yolo_model is None:
        return objects

    h, w = frame_bgr.shape[:2]

    try:
        # Run YOLO inference
        results = yolo_model.predict(frame_bgr, conf=conf, verbose=False)
        
        print(f"[DEBUG] YOLO found {len(results[0].boxes) if results[0].boxes is not None else 0} detections over threshold {conf}")
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                confidence = float(scores[i])

                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                if cx < 0 or cx >= w:
                    continue

                # Accept detection if its bbox intersects the horizon band at any of a few sampled columns.
                x1i = max(0, int(round(x1)))
                x2i = min(w - 1, int(round(x2)))
                if x2i < x1i:
                    continue
                sample_xs = [int((x1i + x2i) / 2)]
                if x2i - x1i > 10:
                    q1 = int(x1i + 0.25 * (x2i - x1i))
                    q3 = int(x1i + 0.75 * (x2i - x1i))
                    sample_xs.extend([q1, q3])
                intersects_band = False
                for sx in sample_xs:
                    hy = int(horizon_smoothed[sx])
                    band_top = hy - band_up
                    band_bottom = hy + band_down
                    # y-intervals [y1,y2] and [band_top,band_bottom] intersect?
                    if max(y1, band_top) <= min(y2, band_bottom):
                        intersects_band = True
                        break
                if not intersects_band:
                    continue

                horizon_y = int(horizon_smoothed[cx])
                pixel_distance = int(abs(horizon_y - cy))

                objects.append({
                    'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    'center': (cx, cy),
                    'horizon_y': horizon_y,
                    'pixel_distance': pixel_distance,
                    'signed_distance': int(cy - horizon_y),
                    'confidence': confidence,
                    'class_id': 0,  # YOLO classes can be added if needed
                    'det': 'yolo'
                })

    except Exception as e:
        print(f"[DEBUG] YOLO inference failed: {e}")
        return objects

    print(f"[DEBUG] Filtered to {len(objects)} objects in band")
    return objects


def maybe_write_csv_header(csv_path, include_angle, distance_units):
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
            # Gap from horizon to object center, in chosen units
            header.append('gap_' + ('nm' if distance_units == 'nm' else 'm'))
            writer.writerow(header)

def append_csv_rows(csv_path, rows):
    if not csv_path or not rows:
        return
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description='3-Class U-Net Horizon Distance Calculator')
    parser.add_argument('--model-path', type=str, default='./3.7training_unet/4best_unet_rtdetr_aware_smd_3cls.pth', help='Path to trained 3-class U-Net weights')
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
    # YOLO integration options
    parser.add_argument('--yolo-model', type=str, help='Path to YOLO model file (.pt)')
    parser.add_argument('--yolo-conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--prefer-yolo', action='store_true', help='Use YOLO detections instead of segmentation blobs if available')
    parser.add_argument('--yolo-interval', type=int, default=1, help='Run YOLO every N processed frames and reuse between (improves FPS)')
    # Real-distance estimation options
    parser.add_argument('--camera-height-m', type=float, default=10.0, help='Camera height above sea level in meters')
    parser.add_argument('--earth-radius-m', type=float, default=6371000.0, help='Earth radius in meters (mean ~6371000)')
    parser.add_argument('--refraction-k', type=float, default=1.3333, help='Refraction factor k (effective Earth radius = k * Earth radius), typical ~4/3')
    parser.add_argument('--distance-units', type=str, choices=['m','nm'], default='m', help='Units for displayed/exported distance (meters or nautical miles)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        cudnn.benchmark = True  # speed up convs for fixed input sizes

    # Load U-Net model
    model = UNet(n_classes=3).to(device)
    # Optional YOLO model load
    yolo_model = None
    if args.yolo_model:
        if not HAS_YOLO:
            print('[WARN] ultralytics not installed; YOLO disabled.')
        else:
            try:
                yolo_model = YOLO(args.yolo_model)
                yolo_model.to(device)
                print(f"YOLO loaded from: {args.yolo_model}")
            except Exception as e:
                print(f"[WARN] Failed to load YOLO: {e}")
                yolo_model = None
    
    prefer_yolo_mode = args.prefer_yolo
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
    # Precompute effective Earth radius
    effective_R = (args.refraction_k if args.refraction_k is not None else 1.0) * (args.earth_radius_m if args.earth_radius_m is not None else 6371000.0)
    distance_units = args.distance_units
    maybe_write_csv_header(args.export_csv, include_angle, distance_units)


    def process_image(path, source_label, frame_idx=None):
        start_t = time.time()
        image_pil = Image.open(path).convert('RGB')
        mask = predict(model, image_pil, device, transform, apply_ship_correction=not args.no_ship_correction, use_fp16=True)
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        _, horizon_smoothed = compute_horizon_line(mask)
        # Choose detection source
        if prefer_yolo_mode and yolo_model is not None:
            objects = detect_objects_yolo(
                image_bgr,
                horizon_smoothed,
                yolo_model,
                conf=args.yolo_conf,
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
        # Attach real-world ranges
        attach_ranges(
            objects,
            image_height_px=mask.shape[0],
            fov_vertical_deg=args.fov_vertical,
            camera_height_m=args.camera_height_m,
            effective_earth_radius_m=effective_R,
            units=distance_units,
            model='flat',
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
            # Export real-world gap from horizon (converted) or empty
            rng_m = obj.get('gap_m', None)
            if rng_m is not None:
                rng_val = (rng_m / 1852.0) if distance_units == 'nm' else rng_m
                row.append(rng_val)
            else:
                row.append("")
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
                mask = predict(model, image_pil, device, transform, apply_ship_correction=not args.no_ship_correction, use_fp16=True)
                _, horizon_smoothed = compute_horizon_line(mask)
                if prefer_yolo_mode and yolo_model is not None:
                    run_det_now = (frame_idx // max(1, args.frame_step)) % max(1, args.yolo_interval) == 0
                    if run_det_now:
                        objects = detect_objects_yolo(
                            frame,
                            horizon_smoothed,
                            yolo_model,
                            conf=args.yolo_conf,
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
                            # Accept if bbox intersects current horizon band at any sampled x across bbox
                            x, y, bw, bh = o['bbox']
                            x1 = x
                            y1 = y
                            x2 = x + bw
                            y2 = y + bh
                            x1i = max(0, int(round(x1)))
                            x2i = min(frame.shape[1] - 1, int(round(x2)))
                            if x2i < x1i:
                                continue
                            sample_xs = [int((x1i + x2i) / 2)]
                            if x2i - x1i > 10:
                                q1 = int(x1i + 0.25 * (x2i - x1i))
                                q3 = int(x1i + 0.75 * (x2i - x1i))
                                sample_xs.extend([q1, q3])
                            intersects_band = False
                            for sx in sample_xs:
                                hy = int(horizon_smoothed[sx])
                                band_top = hy - args.band_up
                                band_bottom = hy + args.band_down
                                if max(y1, band_top) <= min(y2, band_bottom):
                                    intersects_band = True
                                    break
                            if not intersects_band:
                                continue
                            horizon_y = int(horizon_smoothed[cx])
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
                # Attach real-world ranges for current frame
                attach_ranges(
                    objects,
                    image_height_px=mask.shape[0],
                    fov_vertical_deg=args.fov_vertical,
                    camera_height_m=args.camera_height_m,
                    effective_earth_radius_m=effective_R,
                    units=distance_units,
                    model='flat',
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
                    rng_m = obj.get('gap_m', None)
                    if rng_m is not None:
                        rng_val = (rng_m / 1852.0) if distance_units == 'nm' else rng_m
                        row.append(rng_val)
                    else:
                        row.append("")
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
