"""
Dataset Preparation Script for Maritime Horizon Detection
Converts videos + MAT ground truth files to image dataset format

Input:
- Videos: VIS_Onboard/Videos/*.avi
- Horizon GT: VIS_Onboard/HorizonGT/*.mat
- Object GT: VIS_Onboard/ObjectGT/*.mat (optional for sky segmentation)

Output:
- MaritimeSkyFixed/
    train/images/  train/masks/
    val/images/    val/masks/
    test/images/   test/masks/
"""
import os
import cv2
import numpy as np
import scipy.io as sio
from PIL import Image
import shutil
from pathlib import Path
import argparse

class DatasetConverter:
    def __init__(self, input_dir="VIS_Onboard", output_dir="MaritimeSky"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        self.videos_dir = self.input_dir / "Videos"
        self.horizon_gt_dir = self.input_dir / "HorizonGT"
        self.object_gt_dir = self.input_dir / "ObjectGT"
        
        # Create output structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    
    def load_horizon_gt(self, mat_file):
        """Load horizon line coordinates from MAT file"""
        try:
            data = sio.loadmat(mat_file)
            # Common field names in horizon GT files
            possible_keys = ['gt', 'horizon', 'horizonLine', 'HorizonGT', 'groundTruth', 'structXML']
            
            horizon_data = None
            for key in possible_keys:
                if key in data:
                    horizon_data = data[key]
                    break
            
            if horizon_data is None:
                # Print available keys for debugging
                print(f"Available keys in {mat_file}: {list(data.keys())}")
                # Try the first non-metadata key
                keys = [k for k in data.keys() if not k.startswith('__')]
                if keys:
                    horizon_data = data[keys[0]]
            
            # Parse the structured XML data format
            if horizon_data is not None and hasattr(horizon_data, 'dtype') and horizon_data.dtype.names:
                # Extract X and Y coordinates from structured array
                x_coords = []
                y_coords = []
                for i in range(len(horizon_data)):
                    frame_data = horizon_data[i]
                    if hasattr(frame_data, 'dtype') and 'X' in frame_data.dtype.names:
                        x = frame_data['X'][0] if len(frame_data['X']) > 0 else 960.5
                        y = frame_data['Y'][0] if len(frame_data['Y']) > 0 else 400
                        x_coords.append(float(x[0]) if hasattr(x, '__len__') else float(x))
                        y_coords.append(float(y[0]) if hasattr(y, '__len__') else float(y))
                
                if x_coords and y_coords:
                    return np.array([x_coords, y_coords])
            
            return horizon_data
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            return None
    
    def load_object_gt(self, mat_file):
        """Load object annotations (optional for better sky segmentation)"""
        try:
            if not os.path.exists(mat_file):
                return None
            data = sio.loadmat(mat_file)
            # Look for object/ship annotations
            possible_keys = ['objects', 'ships', 'ObjectGT', 'gt']
            for key in possible_keys:
                if key in data:
                    return data[key]
            return None
        except:
            return None
    
    def create_sky_mask(self, frame_shape, horizon_line, objects=None):
        """
        Create binary mask: 1=sky, 0=non-sky
        
        Args:
            frame_shape: (height, width, channels)
            horizon_line: array of coordinates
            objects: optional object annotations
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if horizon_line is not None:
            try:
                # Handle different horizon line formats
                if horizon_line.ndim == 2 and horizon_line.shape[0] == 2:
                    # Format: [[x1, x2, ...], [y1, y2, ...]]
                    x_coords = horizon_line[0].astype(int)
                    y_coords = horizon_line[1].astype(int)
                elif horizon_line.ndim == 2 and horizon_line.shape[1] >= 2:
                    # Format: [[x1,y1], [x2,y2], ...]
                    x_coords = horizon_line[:, 0].astype(int)
                    y_coords = horizon_line[:, 1].astype(int)
                elif horizon_line.ndim == 1:
                    # Format: [y1, y2, y3, ...] for each x
                    x_coords = np.arange(len(horizon_line))
                    y_coords = horizon_line.astype(int)
                else:
                    # Fallback: use single horizon point
                    if len(horizon_line) >= 2:
                        horizon_y = int(horizon_line[1])  # Use Y coordinate
                    else:
                        horizon_y = h // 2  # Default to middle
                    mask[:horizon_y, :] = 1
                    return mask
                
                # If we have multiple points, interpolate a horizon line
                if len(x_coords) > 1 and len(y_coords) > 1:
                    # Sort by x coordinate
                    sorted_indices = np.argsort(x_coords)
                    x_coords = x_coords[sorted_indices]
                    y_coords = y_coords[sorted_indices]
                    
                    # Interpolate horizon line across full width
                    x_full = np.arange(w)
                    y_horizon = np.interp(x_full, x_coords, y_coords)
                else:
                    # Single point - use constant horizon
                    y_horizon = np.full(w, y_coords[0] if len(y_coords) > 0 else h//2)
                
                # Clip coordinates to image bounds
                y_horizon = np.clip(y_horizon, 0, h-1).astype(int)
                
                # Create sky mask (everything above horizon line)
                for x in range(w):
                    mask[:y_horizon[x], x] = 1
                    
            except Exception as e:
                print(f"Warning: Error creating mask: {e}")
                # Fallback to simple horizontal split
                mask[:h//2, :] = 1
        
        return mask
    
    def extract_frames_and_masks(self, video_path, gt_file, split, max_frames=50):
        """Extract frames from video and create corresponding masks"""
        print(f"Processing {video_path.name}...")
        
        # Load ground truth
        horizon_data = self.load_horizon_gt(gt_file)
        if horizon_data is None:
            print(f"Warning: No horizon data found for {video_path.name}")
            return 0
        
        # Load optional object data
        object_file = self.object_gt_dir / gt_file.name.replace("HorizonGT", "ObjectGT")
        object_data = self.load_object_gt(object_file)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Total frames: {total_frames}")
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
        
        saved_count = 0
        video_name = video_path.stem
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Get horizon line for this frame
            if horizon_data is not None:
                if horizon_data.ndim >= 2 and frame_idx < horizon_data.shape[1]:
                    # Multiple horizon points per frame
                    x_coord = horizon_data[0, frame_idx] if horizon_data.shape[0] >= 1 else 960
                    y_coord = horizon_data[1, frame_idx] if horizon_data.shape[0] >= 2 else 400
                    horizon_line = np.array([[x_coord], [y_coord]])
                elif horizon_data.ndim == 2 and horizon_data.shape[0] == 2:
                    # Use average coordinates if single line available
                    x_coord = np.mean(horizon_data[0])
                    y_coord = np.mean(horizon_data[1])
                    horizon_line = np.array([[x_coord], [y_coord]])
                else:
                    # Use the full horizon data (for single frame videos)
                    horizon_line = horizon_data
            else:
                # Default horizon line in middle of frame
                horizon_line = np.array([[frame.shape[1]//2], [frame.shape[0]//2]])
            
            # Create mask
            mask = self.create_sky_mask(frame.shape, horizon_line)
            
            # Save frame and mask
            frame_name = f"{video_name}_frame_{frame_idx:06d}"
            
            # Save image
            img_path = self.output_dir / split / "images" / f"{frame_name}.jpg"
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save mask
            mask_path = self.output_dir / split / "masks" / f"{frame_name}.png"
            Image.fromarray(mask).save(mask_path)
            
            saved_count += 1
        
        cap.release()
        print(f"  Saved {saved_count} frames to {split}")
        return saved_count
    
    def split_videos(self, train_ratio=0.7, val_ratio=0.2):
        """Split videos into train/val/test sets"""
        video_files = list(self.videos_dir.glob("*.avi"))
        np.random.shuffle(video_files)
        
        n_total = len(video_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': video_files[:n_train],
            'val': video_files[n_train:n_train+n_val],
            'test': video_files[n_train+n_val:]
        }
        
        print(f"Video split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        return splits
    
    def convert_dataset(self, max_frames_per_video=50):
        """Main conversion function"""
        print("🚀 Starting dataset conversion...")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        
        # Get video splits
        video_splits = self.split_videos()
        
        total_frames = 0
        
        for split_name, video_list in video_splits.items():
            print(f"\n📁 Processing {split_name} split ({len(video_list)} videos)...")
            split_frames = 0
            
            for video_path in video_list:
                # Find corresponding GT file
                gt_filename = video_path.stem + "_HorizonGT.mat"
                gt_file = self.horizon_gt_dir / gt_filename
                
                if not gt_file.exists():
                    # Try alternative naming
                    alt_gt_filename = video_path.stem.replace("_VIS_OB", "_VIS_OBHorizonGT") + ".mat"
                    gt_file = self.horizon_gt_dir / alt_gt_filename
                
                if gt_file.exists():
                    frames = self.extract_frames_and_masks(video_path, gt_file, split_name, max_frames_per_video)
                    split_frames += frames
                else:
                    print(f"Warning: GT file not found for {video_path.name}")
            
            print(f"✅ {split_name}: {split_frames} frames saved")
            total_frames += split_frames
        
        print(f"\n🎉 Dataset conversion complete!")
        print(f"📊 Total frames extracted: {total_frames}")
        print(f"📁 Dataset saved to: {self.output_dir}")
        
        # Print dataset summary
        for split in ['train', 'val', 'test']:
            img_count = len(list((self.output_dir / split / "images").glob("*.jpg")))
            mask_count = len(list((self.output_dir / split / "masks").glob("*.png")))
            print(f"  {split}: {img_count} images, {mask_count} masks")

def main():
    parser = argparse.ArgumentParser(description="Convert video dataset to image dataset")
    parser.add_argument("--input_dir", default="VIS_Onboard", help="Input directory with videos and GT")
    parser.add_argument("--output_dir", default="MaritimeSkyFixed", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=50, help="Max frames per video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    np.random.seed(args.seed)
    
    # Convert dataset
    converter = DatasetConverter(args.input_dir, args.output_dir)
    converter.convert_dataset(args.max_frames)

if __name__ == "__main__":
    main()
