"""
Real-time horizon detection and sky segmentation for maritime videos.
Preserves all ships and non-sky objects while detecting horizon line.

! py horizon_detect_video.py --video videos/MVI_1626_VIS.avi --show-objects --output results/1626_result.mp4

TODO: need to be tested on more videos
TODO: it sometimes stops working, should be more stable, threshold may be lowered.
TODO: scene-specific horizon detection modes (sunset, night, foggy etc.) using horizon_detect.py functions
TODO: detect_horizon_by_edges_with_color can be used when color-based detection fails
TODO: test for real-time performance on various hardware (only tested in video mode, not camera)
TODO: c/cpp may be used for better performance
TODO: if found a nice dataset, an ai model can be trained to detect horizon line for a more hybrid approach

"""

import cv2
import numpy as np
import argparse
import sys
import time
from datetime import datetime
from sklearn.cluster import KMeans

# Import horizon detection functions from the original script
from horizon_detect import (
    detect_horizon_color_based,
    detect_horizon_by_color_transition,
    detect_sunset_scene,
    detect_horizon_sunset_mode,
    detect_horizon_with_object_preservation,
    detect_horizon_night_mode,
    detect_horizon_by_edges_with_color,
    detect_horizon_advanced_clustering,
    get_blue_ratio
)

class RealTimeHorizonDetector:
    def __init__(self, stability_buffer_size=5, confidence_threshold=0.7, horizon_offset=15):
        """
        Initialize the real-time horizon detector with temporal stability.
        
        Args:
            stability_buffer_size: Number of recent detections to consider for stability
            confidence_threshold: Minimum confidence required for horizon detection
            horizon_offset: Pixels to raise horizon line to preserve distant ships
        """
        self.stability_buffer_size = stability_buffer_size
        self.confidence_threshold = confidence_threshold
        self.horizon_offset = horizon_offset  # New parameter for horizon adjustment
        self.horizon_history = []
        self.frame_count = 0
        self.processing_times = []
        
        # Performance tracking
        self.detection_success_count = 0
        self.total_frames_processed = 0
        
    def detect_horizon_with_stability(self, frame):
        """
        Detect horizon with temporal stability and object preservation.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            tuple: (horizon_line_segments, confidence_score, segmented_frame)
        """
        start_time = time.time()
        
        # Primary horizon detection using the existing advanced method
        line, mask = detect_horizon_color_based(frame)
        
        confidence_score = 0.0
        horizon_segments = None
        
        if line is not None:
            x1, y1, x2, y2 = line
            
            # Calculate confidence based on multiple factors
            confidence_score = self._calculate_confidence(frame, y1, mask)
            
            # Apply temporal stability
            stabilized_y = self._apply_temporal_stability(y1, confidence_score)
            
            if stabilized_y is not None:
                # Create ship-aware horizon line segments
                horizon_segments, _ = self._create_ship_aware_horizon_line(frame, stabilized_y)
                
                # Create enhanced segmented frame preserving objects
                segmented_frame = self._create_enhanced_segmentation(frame, stabilized_y)
                
                self.detection_success_count += 1
            else:
                horizon_segments = None
                segmented_frame = frame.copy()
        else:
            # Fallback: try to use previous stable detection
            if len(self.horizon_history) > 0:
                last_stable = self._get_last_stable_detection()
                if last_stable is not None:
                    horizon_segments, _ = self._create_ship_aware_horizon_line(frame, last_stable)
                    segmented_frame = self._create_enhanced_segmentation(frame, last_stable)
                    confidence_score = 0.5  # Reduced confidence for fallback
                else:
                    segmented_frame = frame.copy()
            else:
                segmented_frame = frame.copy()
        
        # Update performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:  # Keep only recent 100 measurements
            self.processing_times.pop(0)
        
        self.total_frames_processed += 1
        self.frame_count += 1
        
        return horizon_segments, confidence_score, segmented_frame
    
    def _calculate_confidence(self, frame, horizon_y, mask):
        """
        Calculate confidence score for the detected horizon line.
        """
        h, w = frame.shape[:2]
        
        # Factor 1: Position reasonableness (prefer middle region)
        ideal_range = (h * 0.25, h * 0.75)
        if ideal_range[0] <= horizon_y <= ideal_range[1]:
            position_score = 1.0
        else:
            position_score = max(0.1, 1.0 - abs(horizon_y - h/2) / (h/2))
        
        # Factor 2: Color consistency above and below horizon
        strip_height = max(10, h // 20)
        
        if horizon_y - strip_height >= 0 and horizon_y + strip_height < h:
            upper_region = frame[max(0, horizon_y - strip_height):horizon_y, :]
            lower_region = frame[horizon_y:min(h, horizon_y + strip_height), :]
            
            if upper_region.size > 0 and lower_region.size > 0:
                upper_mean = np.mean(upper_region.reshape(-1, 3), axis=0)
                lower_mean = np.mean(lower_region.reshape(-1, 3), axis=0)
                
                color_diff = np.linalg.norm(upper_mean - lower_mean)
                color_score = min(1.0, color_diff / 100.0)
            else:
                color_score = 0.5
        else:
            color_score = 0.3
        
        # Factor 3: Horizontal line strength
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if 0 <= horizon_y < h:
            horizon_row = gray[horizon_y, :]
            row_variance = np.var(horizon_row)
            # Lower variance indicates more consistent (straighter) line
            line_score = max(0.1, 1.0 - row_variance / 10000.0)
        else:
            line_score = 0.1
        
        # Combine factors
        confidence = (position_score * 0.3 + color_score * 0.5 + line_score * 0.2)
        
        return min(1.0, max(0.0, confidence))
    
    def _apply_temporal_stability(self, current_y, confidence):
        """
        Apply temporal stability to reduce jitter in horizon detection.
        """
        # Add current detection to history
        self.horizon_history.append((current_y, confidence, self.frame_count))
        
        # Keep only recent detections
        if len(self.horizon_history) > self.stability_buffer_size:
            self.horizon_history.pop(0)
        
        # Filter detections by confidence
        confident_detections = [(y, conf, frame) for y, conf, frame in self.horizon_history 
                              if conf >= self.confidence_threshold]
        
        if len(confident_detections) == 0:
            return None
        
        # Calculate weighted average of recent confident detections
        total_weight = 0
        weighted_sum = 0
        
        for y, conf, frame_num in confident_detections:
            # Give more weight to recent detections and higher confidence
            recency_weight = 1.0 - (self.frame_count - frame_num) / self.stability_buffer_size
            weight = conf * recency_weight
            
            weighted_sum += y * weight
            total_weight += weight
        
        if total_weight > 0:
            stabilized_y = int(weighted_sum / total_weight)
            return stabilized_y
        
        return None
    
    def _get_last_stable_detection(self):
        """
        Get the last stable horizon detection for fallback.
        """
        if len(self.horizon_history) == 0:
            return None
        
        # Find the most recent confident detection
        for y, conf, frame_num in reversed(self.horizon_history):
            if conf >= self.confidence_threshold * 0.8:  # Slightly lower threshold for fallback
                return y
        
        return None
    
    def _detect_ships_and_objects(self, frame, horizon_y):
        """
        Detect ships and large objects that should not be crossed by horizon line.
        Returns a mask of object regions to avoid.
        Improved version with better false positive filtering.
        """
        h, w = frame.shape[:2]
        
        # Convert to different color spaces for better object detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create object mask
        object_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define horizon detection zone (restrict object detection to near horizon)
        horizon_zone_buffer = 80  # Pixels above and below horizon to search for objects
        search_zone_top = max(0, horizon_y - horizon_zone_buffer)
        search_zone_bottom = min(h, horizon_y + horizon_zone_buffer)
        
        # Method 1: Detect large contrasting objects (ships, structures)
        # Apply adaptive thresholding with more conservative parameters
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 35, 15)  # Larger block size, higher C value
        
        # Find contours of potential objects
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # More restrictive size filtering
        min_object_area = (w * h) * 0.002  # Minimum 0.2% of image area (doubled)
        max_object_area = (w * h) * 0.15   # Maximum 15% of image area (halved)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_object_area < area < max_object_area:
                # Get bounding box
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Restrict to horizon zone only
                if not (search_zone_top <= y <= search_zone_bottom or 
                       search_zone_top <= y + ch <= search_zone_bottom):
                    continue
                
                # Additional position filtering - objects should be near water line
                object_center_y = y + ch // 2
                distance_from_horizon = abs(object_center_y - horizon_y)
                if distance_from_horizon > horizon_zone_buffer:
                    continue
                
                # Check if object has ship-like characteristics
                object_region = frame[y:y+ch, x:x+cw]
                gray_region = gray[y:y+ch, x:x+cw]
                
                if object_region.size > 0 and gray_region.size > 0:
                    if self._is_likely_ship(object_region, gray_region, horizon_y, y):
                        # Add to object mask with reduced padding
                        padding = 10  # Reduced padding
                        cv2.rectangle(object_mask, 
                                    (max(0, x-padding), max(0, y-padding)), 
                                    (min(w, x+cw+padding), min(h, y+ch+padding)), 
                                    255, -1)
        
        # Method 2: More selective color-based ship detection
        ship_color_mask = self._detect_ship_colors_selective(hsv, horizon_y, h, w)
        
        # Combine masks but only within the horizon zone
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        zone_mask[search_zone_top:search_zone_bottom, :] = 255
        
        # Apply zone restriction to color-based detection
        ship_color_mask = cv2.bitwise_and(ship_color_mask, zone_mask)
        
        # Combine with contour-based detection
        combined_mask = cv2.bitwise_or(object_mask, ship_color_mask)
        
        # Apply more conservative morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # Smaller kernel
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def _is_likely_ship(self, color_region, gray_region, horizon_y, object_y):
        """
        Check if a detected region is likely to be a ship based on visual characteristics.
        Enhanced version with stricter filtering.
        """
        if color_region.size == 0 or gray_region.size == 0:
            return False
        
        h, w = gray_region.shape
        
        # Check minimum size (ships should be reasonably sized)
        if h < 10 or w < 20:
            return False
        
        # Check aspect ratio (ships are typically wider than tall)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 1.2:  # More strict: ships should be notably wider than tall
            return False
        
        # Check position relative to horizon (ships should be ON or slightly above horizon)
        object_center_y = object_y + h // 2
        position_score = abs(object_center_y - horizon_y)
        if position_score > 60:  # Too far from horizon
            return False
        
        # Check for structural features (edges, lines) - more conservative
        edges = cv2.Canny(gray_region, 80, 160)  # Higher thresholds
        edge_density = np.sum(edges > 0) / (h * w) if (h * w) > 0 else 0
        
        # Ships typically have moderate edge density (structured but not too busy)
        if edge_density < 0.05 or edge_density > 0.25:  # More restrictive range
            return False
        
        # Check color variance (ships usually have distinct color regions)
        color_std = np.std(color_region.reshape(-1, 3), axis=0)
        avg_color_variance = np.mean(color_std)
        
        # Ships typically have some color structure but not too much
        if avg_color_variance < 15 or avg_color_variance > 80:  # More restrictive range
            return False
        
        # Check for horizontal structure (ships have horizontal lines)
        horizontal_edges = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_strength = np.mean(np.abs(horizontal_edges))
        
        # Ships should have some horizontal structure
        if horizontal_strength < 5:
            return False
        
        # Check brightness consistency (ships usually have consistent lighting)
        mean_brightness = np.mean(gray_region)
        brightness_std = np.std(gray_region)
        
        # Reject regions that are too dark (likely shadows/water) or too bright (likely clouds/sky)
        if mean_brightness < 40 or mean_brightness > 220:
            return False
        
        # Ships should have moderate brightness variation
        if brightness_std < 8 or brightness_std > 50:
            return False
        
        return True
    
    def _detect_ship_colors_selective(self, hsv_frame, horizon_y, h, w):
        """
        Detect typical ship colors in HSV color space with better filtering.
        More selective version to reduce false positives.
        """
        ship_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Only process the area near the horizon
        search_zone_buffer = 60
        search_zone_top = max(0, horizon_y - search_zone_buffer)
        search_zone_bottom = min(h, horizon_y + search_zone_buffer)
        
        # Get the region of interest
        roi = hsv_frame[search_zone_top:search_zone_bottom, :]
        
        if roi.size == 0:
            return ship_mask
        
        # Define more restrictive color ranges for typical ship colors
        
        # White/light structures (superstructures) - more restrictive
        white_lower = np.array([0, 0, 220])  # Higher value threshold
        white_upper = np.array([180, 25, 255])  # Lower saturation threshold
        white_mask = cv2.inRange(roi, white_lower, white_upper)
        
        # Dark hulls - more restrictive to avoid water shadows
        dark_lower = np.array([0, 0, 20])     # Not pure black
        dark_upper = np.array([180, 200, 70])  # Avoid very saturated colors
        dark_mask = cv2.inRange(roi, dark_lower, dark_upper)
        
        # Metallic/gray colors - more restrictive
        gray_lower = np.array([0, 0, 90])      # Brighter than dark
        gray_upper = np.array([180, 40, 180])  # Less saturated
        gray_mask = cv2.inRange(roi, gray_lower, gray_upper)
        
        # Red colors (common on ships) - more restrictive
        red_lower1 = np.array([0, 100, 100])   # More saturated and brighter
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(roi, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(roi, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine ship color masks
        roi_ship_mask = cv2.bitwise_or(white_mask, dark_mask)
        roi_ship_mask = cv2.bitwise_or(roi_ship_mask, gray_mask)
        roi_ship_mask = cv2.bitwise_or(roi_ship_mask, red_mask)
        
        # Apply morphological operations to clean up - smaller kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi_ship_mask = cv2.morphologyEx(roi_ship_mask, cv2.MORPH_CLOSE, kernel)
        roi_ship_mask = cv2.morphologyEx(roi_ship_mask, cv2.MORPH_OPEN, kernel)
        
        # Filter out small regions (noise)
        contours, _ = cv2.findContours(roi_ship_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(roi_ship_mask)
        
        min_area = 100  # Minimum area for color-based detection
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        # Place the filtered mask back into the full image
        ship_mask[search_zone_top:search_zone_bottom, :] = filtered_mask
        
        return ship_mask
    
    def _create_ship_aware_horizon_line(self, frame, horizon_y):
        """
        Create a horizon line that avoids crossing through ships and objects.
        Returns modified horizon line coordinates or segments.
        Applies horizon offset to preserve distant ships.
        """
        h, w = frame.shape[:2]
        
        # Apply horizon offset to preserve distant ships
        adjusted_horizon_y = max(0, horizon_y - self.horizon_offset)
        
        # Detect ships and objects using the original horizon position
        object_mask = self._detect_ships_and_objects(frame, horizon_y)
        
        # Create horizon line segments that avoid objects
        horizon_segments = []
        
        # Scan the horizon line for object intersections
        horizon_buffer = 25  # Reduced vertical buffer
        scan_region = object_mask[max(0, adjusted_horizon_y - horizon_buffer):
                                 min(h, adjusted_horizon_y + horizon_buffer), :]
        
        if scan_region.size == 0:
            # No scan region, return adjusted line
            return [(0, adjusted_horizon_y, w-1, adjusted_horizon_y)], object_mask
        
        # Find object intersections along the horizon
        object_intersections = np.any(scan_region > 0, axis=0)
        
        # Create line segments between objects
        segment_start = 0
        in_object = False
        
        for x in range(w):
            if object_intersections[x] and not in_object:
                # Start of object - end current segment
                if x > segment_start + 10:  # Minimum segment length
                    horizon_segments.append((segment_start, adjusted_horizon_y, x-1, adjusted_horizon_y))
                in_object = True
            elif not object_intersections[x] and in_object:
                # End of object - start new segment
                segment_start = x
                in_object = False
        
        # Add final segment if not in object
        if not in_object and segment_start < w - 10:
            horizon_segments.append((segment_start, adjusted_horizon_y, w-1, adjusted_horizon_y))
        
        # If no valid segments found, create a single line above objects
        if not horizon_segments:
            # Find the highest point of objects and place line above
            object_pixels = np.where(object_mask > 0)
            if len(object_pixels[0]) > 0:
                highest_object_y = np.min(object_pixels[0])
                safe_horizon_y = max(20, highest_object_y - 30)
                horizon_segments.append((0, safe_horizon_y, w-1, safe_horizon_y))
            else:
                # No objects detected, use adjusted line
                horizon_segments.append((0, adjusted_horizon_y, w-1, adjusted_horizon_y))
        
        return horizon_segments, object_mask
    
    def _create_enhanced_segmentation(self, frame, horizon_y):
        """
        Create enhanced segmentation that preserves all objects above horizon.
        This ensures ships and other maritime objects remain fully visible.
        """
        h, w = frame.shape[:2]
        
        # Create a copy of the original frame
        segmented = frame.copy()
        
        # Get ship-aware horizon information
        horizon_segments, object_mask = self._create_ship_aware_horizon_line(frame, horizon_y)
        
        # Create a more sophisticated segmentation mask
        segmentation_mask = np.zeros((h, w), dtype=np.uint8)
        
        # For each horizontal line, determine if it's above or below the effective horizon
        for y in range(h):
            for x in range(w):
                # Check if this pixel is part of a detected object
                if object_mask[y, x] > 0:
                    continue  # Skip object pixels - they remain unchanged
                
                # Find the closest horizon segment to this x-coordinate
                closest_horizon_y = horizon_y  # Default
                for seg_x1, seg_y1, seg_x2, seg_y2 in horizon_segments:
                    if seg_x1 <= x <= seg_x2:
                        closest_horizon_y = seg_y1
                        break
                
                # Mark as water if below horizon
                if y > closest_horizon_y:
                    segmentation_mask[y, x] = 255
        
        # Apply segmentation effects only to non-object areas
        if np.any(segmentation_mask > 0):
            # Slightly darken sky regions (above horizon, not objects)
            sky_mask = (segmentation_mask == 0) & (object_mask == 0)
            segmented[sky_mask] = (segmented[sky_mask] * 0.85).astype(np.uint8)
            
            # Slightly enhance water regions
            water_mask = (segmentation_mask > 0) & (object_mask == 0)
            segmented[water_mask] = np.clip(segmented[water_mask] * 1.15, 0, 255).astype(np.uint8)
        
        return segmented
    
    def get_performance_stats(self):
        """
        Get performance statistics for monitoring.
        """
        if self.total_frames_processed == 0:
            return {}
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        detection_rate = self.detection_success_count / self.total_frames_processed
        
        return {
            'total_frames': self.total_frames_processed,
            'detection_success_rate': detection_rate,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'fps_capability': 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        }

def draw_horizon_info(frame, horizon_segments, confidence, detector, show_stats=True, show_objects=False):
    """
    Draw horizon line segments and information overlay on the frame.
    """
    h, w = frame.shape[:2]
    
    # Draw horizon line segments
    if horizon_segments is not None and len(horizon_segments) > 0:
        # Draw all horizon segments
        for segment in horizon_segments:
            x1, y1, x2, y2 = segment
            # Draw main horizon line segments in red
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw segment endpoints to show breaks
            cv2.circle(frame, (x1, y1), 5, (0, 255, 255), -1)  # Yellow circles at breaks
            cv2.circle(frame, (x2, y2), 5, (0, 255, 255), -1)
        
        # Get average horizon y-coordinate for display
        avg_horizon_y = int(np.mean([seg[1] for seg in horizon_segments]))
        
        # Draw confidence indicator
        conf_color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        cv2.circle(frame, (30, 30), 15, conf_color, -1)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (50, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw horizon y-coordinate and segment count
        cv2.putText(frame, f"Horizon Y: {avg_horizon_y}", (w - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Segments: {len(horizon_segments)}", (w - 250, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Optional: Show detected objects overlay
        if show_objects:
            # Get object mask for visualization
            detector_instance = detector
            if hasattr(detector_instance, '_detect_ships_and_objects'):
                object_mask = detector_instance._detect_ships_and_objects(frame, avg_horizon_y)
                # Create colored overlay for objects
                object_overlay = np.zeros_like(frame)
                object_overlay[object_mask > 0] = [0, 255, 255]  # Yellow overlay for detected objects
                frame = cv2.addWeighted(frame, 0.9, object_overlay, 0.1, 0)
    
    if show_stats:
        # Draw performance statistics
        stats = detector.get_performance_stats()
        if stats:
            y_offset = 80  # Start lower to avoid overlap with horizon info
            cv2.putText(frame, f"Detection Rate: {stats['detection_success_rate']:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(frame, f"Avg Time: {stats['avg_processing_time_ms']:.1f}ms", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
            cv2.putText(frame, f"Max FPS: {stats['fps_capability']:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add ship detection indicator
            if horizon_segments and len(horizon_segments) > 1:
                y_offset += 20
                cv2.putText(frame, "Ship-aware detection active", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Real-time maritime horizon detection')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--camera', type=int, help='Camera device index for live capture')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--record', type=str, help='Path to save recorded output (for camera input)')
    parser.add_argument('--display', action='store_true', help='Display real-time results')
    parser.add_argument('--no-stats', action='store_true', help='Hide performance statistics overlay')
    parser.add_argument('--show-objects', action='store_true', 
                       help='Show detected ship/object overlay')
    parser.add_argument('--stability-buffer', type=int, default=5, 
                       help='Number of frames for temporal stability (default: 5)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Minimum confidence threshold (default: 0.7)')
    parser.add_argument('--horizon-offset', type=int, default=15,
                       help='Pixels to raise horizon line to preserve distant ships (default: 15)')
    
    args = parser.parse_args()
    
    # Input validation
    if not args.video and args.camera is None:
        print("Error: Please specify either --video or --camera")
        return
    
    if args.video and args.camera is not None:
        print("Error: Please specify either --video or --camera, not both")
        return
    
    # Initialize video capture
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.video}")
            return
        print(f"Processing video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            return
        print(f"Using camera: {args.camera}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video else -1
    
    print(f"Video properties: {width}x{height} @ {fps}FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    
    # Initialize video writer for output
    writer = None
    if args.output or args.record:
        output_path = args.output or args.record
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording to: {output_path}")
    
    # Initialize horizon detector
    detector = RealTimeHorizonDetector(
        stability_buffer_size=args.stability_buffer,
        confidence_threshold=args.confidence_threshold,
        horizon_offset=args.horizon_offset
    )
    
    print("\nStarting real-time horizon detection...")
    print("Enhanced with improved object detection and horizon offset for distant ships")
    print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame, 'o' to toggle object overlay")
    
    frame_count = 0
    paused = False
    start_time = time.time()
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.video:
                        print("End of video reached")
                    else:
                        print("Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Detect horizon with temporal stability
                horizon_segments, confidence, segmented_frame = detector.detect_horizon_with_stability(frame)
                
                # Create display frame with overlays
                display_frame = frame.copy()
                display_frame = draw_horizon_info(
                    display_frame, horizon_segments, confidence, detector, 
                    show_stats=not args.no_stats, show_objects=args.show_objects
                )
                
                # Show progress for video files
                if args.video and total_frames > 0:
                    progress = frame_count / total_frames
                    cv2.putText(display_frame, f"Progress: {progress:.1%}", 
                               (width - 150, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write to output video if specified
                if writer is not None:
                    writer.write(display_frame)
                
                # Display results
                if args.display:
                    cv2.imshow('Maritime Horizon Detection', display_frame)
                    cv2.imshow('Segmented View', segmented_frame)
                
                # Print periodic statistics
                if frame_count % (fps * 5) == 0:  # Every 5 seconds
                    elapsed_time = time.time() - start_time
                    processing_fps = frame_count / elapsed_time
                    stats = detector.get_performance_stats()
                    print(f"Frame {frame_count}: Processing at {processing_fps:.1f} FPS, "
                          f"Detection rate: {stats['detection_success_rate']:.1%}")
            
            # Handle keyboard input
            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('o'):
                    # Toggle object detection overlay
                    args.show_objects = not args.show_objects
                    print("Object detection overlay:", "ON" if args.show_objects else "OFF")
                elif key == ord('s') and not paused:
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"horizon_frame_{timestamp}.jpg"
                    cv2.imwrite(save_path, display_frame)
                    print(f"Frame saved to {save_path}")
            else:
                # For non-display mode, allow early termination
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Cleanup and final statistics
        elapsed_time = time.time() - start_time
        processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Average processing FPS: {processing_fps:.1f}")
        
        final_stats = detector.get_performance_stats()
        print(f"Detection success rate: {final_stats['detection_success_rate']:.1%}")
        print(f"Average processing time per frame: {final_stats['avg_processing_time_ms']:.1f}ms")
        
        # Release resources
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
