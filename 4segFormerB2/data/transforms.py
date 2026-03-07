"""
Augmentations for maritime segmentation. Normalization matches SegFormer/ImageNet.
"""
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ImageNet mean/std for encoder pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms():
    return A.Compose([
        A.Resize(config.INPUT_HEIGHT, config.INPUT_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ], p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(config.INPUT_HEIGHT, config.INPUT_WIDTH),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ])
