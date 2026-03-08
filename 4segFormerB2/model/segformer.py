"""
SegFormer-B2 for 4-class maritime semantic segmentation.
Uses Hugging Face transformers; decoder head is reset for num_classes.
"""
from pathlib import Path
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_segformer_b2_maritime(pretrained: bool = True) -> nn.Module:
    """
    Load SegFormer-B2 with 3-class decoder (Sky, Water, Obstacle).
    If pretrained, loads nvidia/segformer-b2-finetuned-ade-512-512 and replaces
    the classifier head for num_labels=4.
    """
    if pretrained:
        # Full SegFormer-B2 pretrained on ADE20K (150 classes); override num_labels
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=config.NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    else:
        cfg = SegformerConfig.from_pretrained("nvidia/mit-b2")
        cfg.num_labels = config.NUM_CLASSES
        model = SegformerForSemanticSegmentation(cfg)
    return model
