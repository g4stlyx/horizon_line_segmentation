# ==============================================================================
# Debug File Cleanup Script for Ship-Aware U-Net Training
#! add as a code block to the colab (as in .ipynb files) to clean up debug files.
# ==============================================================================

import os
import glob

def cleanup_debug_files(images_dir):
    """
    Remove debug visualization files from the images directory.
    These files don't have corresponding masks and cause training errors.
    """
    print("Cleaning up debug files...")
    
    # Find all debug files
    debug_pattern = os.path.join(images_dir, "debug_*.jpg")
    debug_files = glob.glob(debug_pattern)
    
    if not debug_files:
        print("No debug files found to clean up.")
        return
    
    print(f"Found {len(debug_files)} debug files to remove:")
    
    for debug_file in debug_files:
        try:
            print(f"  Removing: {os.path.basename(debug_file)}")
            os.remove(debug_file)
        except OSError as e:
            print(f"  Error removing {debug_file}: {e}")
    
    print(f"Cleanup complete. Removed {len(debug_files)} debug files.")

def verify_dataset_integrity(images_dir, masks_dir):
    """
    Verify that all images have corresponding masks.
    """
    print("Verifying dataset integrity...")
    
    # Get all image files (excluding debug files)
    image_files = [f for f in os.listdir(images_dir) 
                   if f.endswith('.jpg') and not f.startswith('debug_')]
    
    missing_masks = []
    valid_pairs = 0
    
    for img_file in image_files:
        mask_file = img_file.replace('.jpg', '.png')
        mask_path = os.path.join(masks_dir, mask_file)
        
        if os.path.exists(mask_path):
            valid_pairs += 1
        else:
            missing_masks.append(mask_file)
    
    print(f"Dataset verification results:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Valid image-mask pairs: {valid_pairs}")
    print(f"  Missing masks: {len(missing_masks)}")
    
    if missing_masks:
        print(f"  Files missing masks:")
        for mask in missing_masks[:10]:  # Show first 10
            print(f"    - {mask}")
        if len(missing_masks) > 10:
            print(f"    ... and {len(missing_masks) - 10} more")
    
    return valid_pairs, missing_masks

def main():
    """
    Run cleanup and verification for ship-aware training data.
    """
    # Update these paths based on your setup
    LOCAL_IMAGES_PATH = '/content/processed_unet_ship_aware/images'
    LOCAL_MASKS_PATH = '/content/processed_unet_ship_aware/masks'
    
    print("Ship-Aware Dataset Cleanup and Verification")
    print("=" * 50)
    
    # Check if directories exist
    if not os.path.exists(LOCAL_IMAGES_PATH):
        print(f"Error: Images directory not found: {LOCAL_IMAGES_PATH}")
        return
    
    if not os.path.exists(LOCAL_MASKS_PATH):
        print(f"Error: Masks directory not found: {LOCAL_MASKS_PATH}")
        return
    
    # Clean up debug files
    cleanup_debug_files(LOCAL_IMAGES_PATH)
    print()
    
    # Verify dataset integrity
    valid_pairs, missing_masks = verify_dataset_integrity(LOCAL_IMAGES_PATH, LOCAL_MASKS_PATH)
    
    if missing_masks:
        print(f"\n⚠️  Warning: {len(missing_masks)} images don't have corresponding masks.")
        print("This might indicate an issue with the preprocessing step.")
    else:
        print(f"\n✅ Dataset integrity verified: {valid_pairs} valid image-mask pairs.")
    
    print("\nDataset is ready for training!")

if __name__ == "__main__":
    main()
