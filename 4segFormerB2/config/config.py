"""
Configuration for SegFormer-B2 maritime segmentation (LaRS + MaSTr1325).
Unified 4-class: Sky, Water, Land, Obstacle.
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (relative to project root 4segFormerB2)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# LaRS: lars_images/{train,val,test}/images, lars_annotations/{train,val,test}/semantic_masks
LARS_IMAGES = DATASETS_ROOT / "lars" / "lars_images"
LARS_ANNOTATIONS = DATASETS_ROOT / "lars" / "lars_annotations"

# MaSTr1325: flat image folder + mask folder (0001.jpg <-> 0001m.png)
MASTR_IMAGES = DATASETS_ROOT / "mastr1325" / "MaSTr1325_images_512x384"
MASTR_MASKS = DATASETS_ROOT / "mastr1325" / "MaSTr1325_masks_512x384"

# Outputs
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# -----------------------------------------------------------------------------
# Classes and label mappings
# -----------------------------------------------------------------------------
NUM_CLASSES = 4
CLASS_NAMES = ("Sky", "Water", "Land", "Obstacle")
IGNORE_INDEX = 255

# LaRS semantic PNG: 0=Obstacles, 1=Water, 2=Sky, 255=Ignore
# We map to: Sky=0, Water=1, Land=2, Obstacle=3
LARS_TO_UNIFIED = {0: 3, 1: 1, 2: 0}  # Obstacles->3, Water->1, Sky->0

# MaSTr1325: 0=Obstacles&Environment, 1=Water, 2=Sky, 4=Ignore (from docs)
# We map Obstacles&Environment -> Land (2) for consistency with "not sky/water"
MASTR_TO_UNIFIED = {0: 2, 1: 1, 2: 0}  # Env/Obstacle->2, Water->1, Sky->0
MASTR_IGNORE_VALUE = 4

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 60
LR = 6e-5
WEIGHT_DECAY = 0.01
LR_POLICY = "poly"  # poly decay
SAVE_EVERY_N_EPOCHS = 5
VAL_EVERY_N_EPOCHS = 1

# Input size (SegFormer often uses 512; MaSTr is 512x384)
INPUT_HEIGHT = 384
INPUT_WIDTH = 512

# Model
ENCODER = "nvidia/mit-b2"
PRETRAINED = True
