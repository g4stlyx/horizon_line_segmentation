# horizon_line_detection_python.py
"""
Python translation of the paper:
    "Real-Time Horizon Line Detection based on Fusion of Classification and Clustering"

Main pipeline:
1. Extract GLCM texture features + grayscale intensities from 9x9 non-overlapping blocks.
2. Classify blocks as SKY / NON-SKY with a shallow Neural Network (MLP).
3. Apply K-means to the grayscale image (k=10 by default).
4. Fuse classifier & clustering results according to the paper’s rule.
5. Simple post-processing to clean isolated blobs and ensure minimum sky size.


Usage
-----
>>> python horizon_line_detection_python.py --train path/to/train_images --mask path/to/masks \
        --model out.pkl
>>> python horizon_line_detection_python.py --detect img.jpg --model out.pkl --out result.png

The script is structured so that each stage can be swapped out (e.g. replace the
CPU k-means with cuML, or the scikit-learn MLP with a tiny PyTorch network on
CUDA).
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from scipy import ndimage as ndi

# -----------------------------------------------------
# 1. Feature extraction
# -----------------------------------------------------

def _glcm_features(block: np.ndarray) -> np.ndarray:
    """Compute 5 GLCM statistics for a single 9×9 grayscale block."""
    # quantise to 8 levels for speed (0–7)
    block_q = np.floor(block / 32).astype(np.uint8)
    glcm = graycomatrix(block_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=8, symmetric=True, normed=True)
    # energy, dissimilarity, correlation, cluster shade, cluster prominence
    feats = [
        graycoprops(glcm, prop).mean() for prop in [
            'energy',       # equivalent to Angular Second Moment
            'dissimilarity',
            'correlation']
    ]
    # cluster shade & prominence are not provided; compute manually
    i, j = np.indices((8, 8))
    p = glcm[..., 0]  # shape (8,8)
    mu_i = (i * p).sum()
    mu_j = (j * p).sum()
    sigma_i = np.sqrt(((i - mu_i) ** 2 * p).sum())
    sigma_j = np.sqrt(((j - mu_j) ** 2 * p).sum())
    # avoid divide‑by‑zero for flat textures
    if sigma_i * sigma_j == 0:
        corr = 0.0
    else:
        corr = (((i - mu_i) * (j - mu_j) * p).sum()) / (sigma_i * sigma_j)
    # cluster shade & prominence
    cs = (((i + j - mu_i - mu_j) ** 3) * p).sum()
    cp = (((i + j - mu_i - mu_j) ** 4) * p).sum()
    feats.extend([cs, cp])
    return np.array(feats, dtype=np.float32)


def extract_block_features(img: np.ndarray, block_size: int = 9) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Split image into non‑overlapping blocks and extract features per block.

    Returns
    -------
    feat_matrix : (N_blocks, 6)
        5 texture features + mean pixel intensity.
    grid_shape  : (rows, cols)
        Grid layout so we can reshape classifier output back to image size.
    """
    h, w = img.shape
    rows, cols = h // block_size, w // block_size
    feats = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * block_size, c * block_size
            block = img[y0:y0 + block_size, x0:x0 + block_size]
            tex = _glcm_features(block)
            mean_int = block.mean()
            feats.append(np.concatenate([tex, [mean_int]]))
    return np.vstack(feats), (rows, cols)

# -----------------------------------------------------
# 2. Classifier wrapper
# -----------------------------------------------------

def train_classifier(feature_matrix: np.ndarray, labels: np.ndarray) -> MLPClassifier:
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', solver='adam',
                        learning_rate_init=1e-3, max_iter=500)
    clf.fit(feature_matrix, labels)
    return clf

# -----------------------------------------------------
# 3. Clustering helper
# -----------------------------------------------------

def pixel_kmeans(img: np.ndarray, k: int = 10, seed: int | None = 0) -> np.ndarray:
    """Run K‑means on pixel intensities, return label image (same size)."""
    h, w = img.shape
    pixels = img.reshape(-1, 1).astype(np.float32)
    km = KMeans(k, random_state=seed, n_init='auto').fit(pixels)
    return km.labels_.reshape(h, w)

# -----------------------------------------------------
# 4. Fusion + post‑processing
# -----------------------------------------------------

def fuse_and_postprocess(cluster_map: np.ndarray, sky_prob: np.ndarray, th: float = 0.6,
                         min_sky_ratio: float = 5e-4) -> np.ndarray:
    """Implements the fusion rule & simple morphological cleanup."""
    h, w = cluster_map.shape
    sky_bin = (sky_prob > 0.5).astype(np.uint8)

    # --- fusion ---
    fused = np.zeros_like(sky_bin)
    for label in np.unique(cluster_map):
        mask = (cluster_map == label)
        inter = mask & sky_bin
        if inter.sum() / mask.sum() > th:
            fused[mask] = 1

    # --- post‑processing ---
    # remove very small sky area
    if fused.sum() < min_sky_ratio * h * w:
        return np.zeros_like(fused)

    # fill holes & remove tiny blobs
    fused = ndi.binary_fill_holes(fused).astype(np.uint8)
    fused = ndi.binary_opening(fused, structure=np.ones((5, 5))).astype(np.uint8)
    return fused

# -----------------------------------------------------
# 5. Helper: derive horizon line (row index per column)
# -----------------------------------------------------

def horizon_from_binary(sky_bin: np.ndarray) -> np.ndarray:
    h, w = sky_bin.shape
    horizon = np.zeros(w, dtype=np.int32)
    for c in range(w):
        # Find the transition from sky to non-sky (horizon line)
        sky_column = sky_bin[:, c]
        
        # Find the last sky pixel (bottom of sky region)
        sky_indices = np.where(sky_column)[0]
        if sky_indices.size == 0:
            # No sky detected in this column, set horizon to top
            horizon[c] = 0
        else:
            # The horizon is at the last sky pixel + 1 (first non-sky pixel below)
            last_sky_row = sky_indices.max()
            # But we want the horizon line to be at the boundary, so we use last_sky_row
            # The horizon line is between the last sky pixel and the first non-sky pixel
            horizon[c] = min(last_sky_row + 1, h - 1)
    return horizon

# -----------------------------------------------------
# 6. Training / detection pipelines
# -----------------------------------------------------

def train_pipeline(train_dir: Path, mask_dir: Path, model_out: Path):
    feats_list, lbls_list = [], []
    for p_img in sorted(train_dir.glob('*')):
        img = cv2.imread(str(p_img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image file: {p_img}")
            continue
            
        # Handle different extensions: image files (.jpg) -> mask files (.png)
        mask_name = p_img.stem + '.png'
        mask_path = mask_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask file: {mask_path}")
            continue
        mask = mask > 0  # Changed from > 127 to > 0 since masks have values 0 and 1
        
        # Ensure image and mask have the same dimensions
        if img.shape != mask.shape:
            print(f"Warning: Image and mask dimensions mismatch for {p_img.name}")
            print(f"  Image shape: {img.shape}, Mask shape: {mask.shape}")
            # Resize mask to match image
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            
        feats, (rows, cols) = extract_block_features(img)
        
        # Check if mask can be reshaped to the same block grid
        expected_h, expected_w = rows * 9, cols * 9
        if mask.shape != (expected_h, expected_w):
            # Silently crop or pad the mask to fit the grid (this is expected for non-divisible dimensions)
            # Crop or pad the mask to fit the grid
            h, w = mask.shape
            if h > expected_h:
                mask = mask[:expected_h, :]
            elif h < expected_h:
                mask = np.pad(mask, ((0, expected_h - h), (0, 0)), mode='constant', constant_values=False)
            
            h, w = mask.shape
            if w > expected_w:
                mask = mask[:, :expected_w]
            elif w < expected_w:
                mask = np.pad(mask, ((0, 0), (0, expected_w - w)), mode='constant', constant_values=False)
        
        lbl_blocks = mask.reshape(rows, 9, cols, 9).transpose(0, 2, 1, 3)
        lbl_blocks = lbl_blocks.reshape(rows * cols, 9, 9)
        lbls = lbl_blocks.mean(axis=(1, 2)) > 0.5  # majority rule => sky=1
        feats_list.append(feats)
        lbls_list.append(lbls.astype(np.uint8))
    X = np.vstack(feats_list)
    y = np.concatenate(lbls_list)
    clf = train_classifier(X, y)
    with open(model_out, 'wb') as f:
        pickle.dump({'classifier': clf}, f)
    print(f"Model saved to {model_out}")


def detect_pipeline(img_path: Path, model_path: Path, out_path: Path | None = None):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    clf: MLPClassifier = model['classifier']

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    feats, grid = extract_block_features(img)
    sky_prob_blocks = clf.predict_proba(feats)[:, 1]
    sky_prob_map = sky_prob_blocks.reshape(grid)
    sky_prob_img = cv2.resize(sky_prob_map, img.shape[::-1], interpolation=cv2.INTER_NEAREST)

    cluster_map = pixel_kmeans(img)
    sky_bin = fuse_and_postprocess(cluster_map, sky_prob_img)

    horizon = horizon_from_binary(sky_bin)

    # visualise result
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c, r in enumerate(horizon):
        cv2.circle(rgb, (c, r), 1, (0, 255, 0), -1)
    if out_path:
        cv2.imwrite(str(out_path), rgb)
        print(f"Saved result to {out_path}")
    else:
        cv2.imshow("Horizon", rgb)
        cv2.waitKey(0)

# -----------------------------------------------------
# 7. CLI
# -----------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    tr = sub.add_parser('train')
    tr.add_argument('--train', required=True, type=Path, help='Dir with training images (grayscale)')
    tr.add_argument('--mask', required=True, type=Path, help='Dir with binary masks (sky regions)')
    tr.add_argument('--model', required=True, type=Path)
    # py paper_script.py train --train ./train_images --mask ./masks --model model.pkl

    dt = sub.add_parser('detect')
    dt.add_argument('--img', required=True, type=Path)
    dt.add_argument('--model', required=True, type=Path)
    dt.add_argument('--out', type=Path)
    # py paper_script.py detect --img img.jpg --model model.pkl --out result.png
    return ap.parse_args()


def main():
    args = _parse_args()
    if args.cmd == 'train':
        train_pipeline(args.train, args.mask, args.model)
    else:
        detect_pipeline(args.img, args.model, args.out)


if __name__ == '__main__':
    main()