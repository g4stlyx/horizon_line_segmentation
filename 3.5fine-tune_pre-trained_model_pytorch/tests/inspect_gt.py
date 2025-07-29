"""
Quick test to inspect the MAT file structure
"""
import scipy.io as sio
import os

# Test loading a MAT file to understand structure
mat_file = "VIS_Onboard/HorizonGT/MVI_0788_VIS_OB_HorizonGT.mat"

if os.path.exists(mat_file):
    try:
        data = sio.loadmat(mat_file)
        print(f"Keys in {mat_file}:")
        for key in data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {type(data[key])} - shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")
                if hasattr(data[key], 'shape') and len(data[key].shape) <= 2:
                    print(f"    Sample data: {data[key][:3] if len(data[key]) > 3 else data[key]}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {mat_file}")
