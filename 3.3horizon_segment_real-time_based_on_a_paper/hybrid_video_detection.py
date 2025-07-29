#!/usr/bin/env python3
"""
Real-time Hybrid Horizon Detection for Video Streams

Combines the trained AI model with advanced color-based detection methods
for improved accuracy and temporal stability in maritime video streams.
"""

import cv2
import numpy as np
import argparse
import time
from typing import Tuple, Optional, List
from collections import deque

from hybrid_horizon_detection import HybridHorizonDetector


class RealTimeHybridDetector:
    def __init__(self, model_path: str, stability_buffer_size: int = 5, confidence_threshold: float = 0.6):
        """
        Initialize real-time hybrid horizon detector.
        
        Args:
            model_path: Path to trained AI model
            stability_buffer_size: Number of frames for temporal stability
            confidence_threshold: Minimum confidence for accepting detection
        """
        self.detector = HybridHorizonDetector(model_path)
        self.stability_buffer_size = stability_buffer_size
        self.confidence_threshold = confidence_threshold
        
        # Temporal stability tracking
        self.horizon_history = deque(maxlen=stability_buffer_size)
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        
        # Performance tracking
        self.total_detections = 0
        self.successful_detections = 0
        
    def detect_with_stability(self, frame: np.ndarray) -> Tuple[Optional[Tuple], Optional[np.ndarray], float]:
        """
        Detect horizon with temporal stability and confidence scoring.
        
        Returns:
            (horizon_line, mask, confidence_score)
        """
        start_time = time.time()
        
        # Primary detection
        horizon_line, mask = self.detector.detect_horizon_hybrid(frame)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        confidence_score = 0.0
        stable_horizon = None
        stable_mask = None
        
        if horizon_line is not None:
            # Calculate confidence based on detection quality
            confidence_score = self._calculate_confidence(frame, horizon_line, mask)
            
            # Apply temporal stability
            if confidence_score >= self.confidence_threshold:
                stable_horizon = self._apply_temporal_smoothing(horizon_line, confidence_score)
                stable_mask = mask
                self.successful_detections += 1
            else:
                # Use previous detection if current is low confidence
                stable_horizon = self._get_last_stable_detection()
                stable_mask = self._create_mask_from_horizon(frame, stable_horizon) if stable_horizon else None
        else:
            # Use previous detection as fallback
            stable_horizon = self._get_last_stable_detection()
            stable_mask = self._create_mask_from_horizon(frame, stable_horizon) if stable_horizon else None
        
        self.total_detections += 1
        self.frame_count += 1
        
        return stable_horizon, stable_mask, confidence_score
    
    def _calculate_confidence(self, frame: np.ndarray, horizon_line: Tuple, mask: np.ndarray) -> float:
        """
        Calculate confidence score for the detected horizon.
        """
        if horizon_line is None or mask is None:
            return 0.0
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = horizon_line
        horizon_y = (y1 + y2) // 2
        
        confidence_factors = []
        
        # Factor 1: Position reasonableness (prefer middle 50% of image)
        ideal_range = (h * 0.25, h * 0.75)
        if ideal_range[0] <= horizon_y <= ideal_range[1]:
            position_score = 1.0
        else:
            # Penalize extreme positions
            distance_from_ideal = min(abs(horizon_y - ideal_range[0]), abs(horizon_y - ideal_range[1]))
            position_score = max(0.0, 1.0 - distance_from_ideal / (h * 0.25))
        confidence_factors.append(position_score * 0.25)
        
        # Factor 2: Sky region size (should be reasonable)
        sky_pixels = np.sum(mask == 0)  # Sky is 0 in our mask
        sky_ratio = sky_pixels / (h * w)
        if 0.1 <= sky_ratio <= 0.8:
            sky_score = 1.0
        else:
            sky_score = max(0.0, 1.0 - abs(sky_ratio - 0.4) / 0.4)
        confidence_factors.append(sky_score * 0.25)
        
        # Factor 3: Horizon line smoothness
        line_y_values = []
        for x in range(0, w, 10):
            horizon_col = mask[:, x]
            transition_points = np.where(np.diff(horizon_col.astype(int)) > 0)[0]
            if len(transition_points) > 0:
                line_y_values.append(transition_points[0])
        
        if len(line_y_values) > 2:
            smoothness_score = max(0.0, 1.0 - np.std(line_y_values) / (h * 0.1))
        else:
            smoothness_score = 0.5
        confidence_factors.append(smoothness_score * 0.25)
        
        # Factor 4: Color consistency above/below horizon
        if horizon_y > 10 and horizon_y < h - 10:
            above_region = frame[max(0, horizon_y-20):horizon_y, :]
            below_region = frame[horizon_y:min(h, horizon_y+20), :]
            
            if above_region.size > 0 and below_region.size > 0:
                above_mean = np.mean(above_region, axis=(0, 1))
                below_mean = np.mean(below_region, axis=(0, 1))
                color_diff = np.linalg.norm(above_mean - below_mean)
                color_score = min(1.0, color_diff / 50.0)  # Normalize color difference
            else:
                color_score = 0.5
        else:
            color_score = 0.5
        confidence_factors.append(color_score * 0.25)
        
        return sum(confidence_factors)
    
    def _apply_temporal_smoothing(self, current_horizon: Tuple, confidence: float) -> Tuple:
        """
        Apply temporal smoothing to reduce jitter.
        """
        x1, y1, x2, y2 = current_horizon
        current_y = (y1 + y2) // 2
        
        # Add to history
        self.horizon_history.append((current_y, confidence, self.frame_count))
        
        if len(self.horizon_history) < 2:
            return current_horizon
        
        # Calculate weighted average with recent detections
        total_weight = 0
        weighted_sum = 0
        
        for y, conf, frame_num in self.horizon_history:
            # More recent detections have higher weight
            age_weight = max(0.1, 1.0 - (self.frame_count - frame_num) * 0.1)
            weight = conf * age_weight
            
            weighted_sum += y * weight
            total_weight += weight
        
        if total_weight > 0:
            smoothed_y = int(weighted_sum / total_weight)
            return (x1, smoothed_y, x2, smoothed_y)
        
        return current_horizon
    
    def _get_last_stable_detection(self) -> Optional[Tuple]:
        """
        Get the most recent stable horizon detection.
        """
        if not self.horizon_history:
            return None
        
        # Find the most recent high-confidence detection
        for y, conf, frame_num in reversed(self.horizon_history):
            if conf >= self.confidence_threshold:
                return (0, y, 1920, y)  # Default width, will be adjusted
        
        # If no high-confidence detection, use the most recent one
        if self.horizon_history:
            y, _, _ = self.horizon_history[-1]
            return (0, y, 1920, y)
        
        return None
    
    def _create_mask_from_horizon(self, frame: np.ndarray, horizon_line: Tuple) -> Optional[np.ndarray]:
        """
        Create a simple mask from horizon line.
        """
        if horizon_line is None:
            return None
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = horizon_line
        horizon_y = (y1 + y2) // 2
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[horizon_y:, :] = 255  # Water/land below horizon
        
        return mask
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.
        """
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        success_rate = self.successful_detections / max(1, self.total_detections)
        
        return {
            'avg_processing_time': avg_processing_time,
            'success_rate': success_rate,
            'total_frames': self.total_detections,
            'successful_detections': self.successful_detections,
            'fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        }


def draw_enhanced_overlay(frame: np.ndarray, horizon_line: Tuple, confidence: float, 
                         stats: dict, show_stats: bool = True) -> np.ndarray:
    """
    Draw enhanced overlay with horizon line and statistics.
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw horizon line
    if horizon_line is not None:
        x1, y1, x2, y2 = horizon_line
        
        # Adjust line endpoints to frame width
        x1, x2 = 0, w - 1
        
        # Color based on confidence
        if confidence >= 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence >= 0.6:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        cv2.line(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw additional markers along the line
        for x in range(0, w, 100):
            cv2.circle(result, (x, y1), 3, color, -1)
    
    # Draw statistics overlay
    if show_stats:
        overlay_height = 120
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background
        
        # Text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        texts = [
            f"Confidence: {confidence:.2f}",
            f"Processing: {stats.get('avg_processing_time', 0)*1000:.1f}ms",
            f"Success Rate: {stats.get('success_rate', 0)*100:.1f}%",
            f"FPS: {stats.get('fps', 0):.1f}"
        ]
        
        y_offset = 25
        for i, text in enumerate(texts):
            cv2.putText(overlay, text, (10, y_offset + i * 25), font, font_scale, (255, 255, 255), thickness)
        
        # Confidence bar
        bar_width = 200
        bar_height = 10
        bar_x = w - bar_width - 10
        bar_y = 10
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Confidence level bar
        conf_width = int(bar_width * confidence)
        if confidence >= 0.8:
            conf_color = (0, 255, 0)
        elif confidence >= 0.6:
            conf_color = (0, 255, 255)
        else:
            conf_color = (0, 0, 255)
        
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), conf_color, -1)
        cv2.putText(overlay, "Confidence", (bar_x, bar_y - 5), font, 0.4, (255, 255, 255), 1)
        
        # Combine overlay with result
        result = np.vstack([overlay, result])
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Real-time Hybrid Horizon Detection for Video')
    parser.add_argument('--video', type=str, help='Input video file path')
    parser.add_argument('--camera', type=int, help='Camera device index (0, 1, 2, ...)')
    parser.add_argument('--model', required=True, help='Path to trained AI model')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--display', action='store_true', help='Display real-time results')
    parser.add_argument('--no-stats', action='store_true', help='Hide statistics overlay')
    parser.add_argument('--buffer-size', type=int, default=5, help='Temporal stability buffer size')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Input validation
    if not args.video and args.camera is None:
        print("Error: Specify either --video or --camera")
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
    
    print(f"Video properties: {width}x{height} @ {fps}FPS")
    
    # Initialize video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_height = height + (120 if not args.no_stats else 0)
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, output_height))
        print(f"Output will be saved to: {args.output}")
    
    # Initialize detector
    try:
        detector = RealTimeHybridDetector(
            args.model, 
            stability_buffer_size=args.buffer_size,
            confidence_threshold=args.confidence_threshold
        )
        print("Hybrid detector initialized successfully")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    print("\n=== Real-time Hybrid Horizon Detection ===")
    print("Press 'q' to quit, 'p' to pause/resume, 's' to toggle stats")
    print("Enhanced with AI model guidance and temporal stability")
    
    frame_count = 0
    paused = False
    show_stats = not args.no_stats
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or camera disconnected")
                    break
                
                frame_count += 1
                
                # Detect horizon
                horizon_line, mask, confidence = detector.detect_with_stability(frame)
                
                # Get performance stats
                stats = detector.get_performance_stats()
                
                # Create visualization
                result = draw_enhanced_overlay(frame, horizon_line, confidence, stats, show_stats)
                
                # Display result
                if args.display:
                    cv2.imshow('Hybrid Horizon Detection', result)
                
                # Save to output video
                if writer is not None:
                    writer.write(result)
                
                # Print progress
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count:5d} | Confidence: {confidence:.2f} | "
                          f"Processing: {stats.get('avg_processing_time', 0)*1000:.1f}ms | "
                          f"Success: {stats.get('success_rate', 0)*100:.1f}%")
            
            # Handle keyboard input
            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('s'):
                    show_stats = not show_stats
                    print("Stats overlay:", "ON" if show_stats else "OFF")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Final statistics
        final_stats = detector.get_performance_stats()
        print(f"\n=== Final Statistics ===")
        print(f"Total frames processed: {final_stats.get('total_frames', 0)}")
        print(f"Successful detections: {final_stats.get('successful_detections', 0)}")
        print(f"Success rate: {final_stats.get('success_rate', 0)*100:.1f}%")
        print(f"Average processing time: {final_stats.get('avg_processing_time', 0)*1000:.1f}ms")
        print(f"Average FPS: {final_stats.get('fps', 0):.1f}")


if __name__ == "__main__":
    main()
