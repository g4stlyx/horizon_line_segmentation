"""
Maritime segmentation datasets: MaSTr1325 and LaRS with unified 4-class labels.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _map_mastr_mask(mask: np.ndarray) -> np.ndarray:
    """MaSTr: 0=Env/Obstacle, 1=Water, 2=Sky, 4=Ignore -> unified 0=Sky, 1=Water, 2=Land, 3=Obstacle, 255=Ignore."""
    out = np.full_like(mask, config.IGNORE_INDEX, dtype=np.int64)
    for src, dst in config.MASTR_TO_UNIFIED.items():
        out[mask == src] = dst
    out[mask == config.MASTR_IGNORE_VALUE] = config.IGNORE_INDEX
    return out


def _map_lars_mask(mask: np.ndarray) -> np.ndarray:
    """LaRS semantic: 0=Obstacles, 1=Water, 2=Sky, 255=Ignore -> unified 0=Sky, 1=Water, 2=Land, 3=Obstacle."""
    out = np.full_like(mask, config.IGNORE_INDEX, dtype=np.int64)
    for src, dst in config.LARS_TO_UNIFIED.items():
        out[mask == src] = dst
    out[mask == 255] = config.IGNORE_INDEX
    return out


class MaSTr1325Dataset(Dataset):
    """MaSTr1325: images 0001.jpg, masks 0001m.png."""

    def __init__(self, images_dir: Path, masks_dir: Path, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.samples = []
        for p in sorted(self.images_dir.glob("*.jpg")):
            stem = p.stem
            mask_path = self.masks_dir / f"{stem}m.png"
            if mask_path.exists():
                self.samples.append((p, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = _map_mastr_mask(mask)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        return {
            "image": image,
            "mask": mask,
            "source": "mastr",
            "name": img_path.stem,
        }


class LaRSDataset(Dataset):
    """LaRS: uses image_annotations.json for file list; images in .../images, masks in .../semantic_masks."""

    def __init__(self, split: str, images_root: Path, annotations_root: Path, transform=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.images_dir = Path(images_root) / split / "images"
        self.masks_dir = Path(annotations_root) / split / "semantic_masks"
        self.transform = transform
        ann_file = Path(annotations_root) / split / "image_annotations.json"
        with open(ann_file) as f:
            data = json.load(f)
        self.file_names = [a["file_name"] for a in data.get("annotations", [])]
        self.samples = []
        for fn in self.file_names:
            img_path = self.images_dir / fn
            mask_path = self.masks_dir / (Path(fn).stem + ".png")
            if img_path.exists() and mask_path.exists():
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = _map_lars_mask(mask)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        return {
            "image": image,
            "mask": mask,
            "source": "lars",
            "name": img_path.stem,
        }


def CombinedMaritimeDataset(
    split: str,
    transform=None,
    use_mastr: bool = True,
    use_lars: bool = True,
) -> Dataset:
    """Build combined train/val dataset. MaSTr has no official split; we use 90% train / 10% val by index."""
    datasets = []
    if use_mastr and config.MASTR_IMAGES.exists():
        full_mastr = MaSTr1325Dataset(config.MASTR_IMAGES, config.MASTR_MASKS, transform=transform)
        n = len(full_mastr)
        if n > 0:
            # 90% train, 10% val
            val_size = max(1, n // 10)
            train_size = n - val_size
            if split == "train":
                mastr = torch.utils.data.Subset(full_mastr, range(0, train_size))
            else:
                mastr = torch.utils.data.Subset(full_mastr, range(train_size, n))
            datasets.append(mastr)
    if use_lars and config.LARS_IMAGES.exists():
        lars = LaRSDataset(
            split,
            config.LARS_IMAGES,
            config.LARS_ANNOTATIONS,
            transform=transform,
        )
        if len(lars) > 0:
            datasets.append(lars)
    if not datasets:
        raise FileNotFoundError("No dataset found. Check config paths and that LaRS/MaSTr folders exist.")
    return ConcatDataset(datasets)
