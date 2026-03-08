"""
Evaluate SegFormer-B2 maritime model on LaRS test set.

The model was trained on:
  - MaSTr1325: ~90% train, ~10% val (no official split; last 10% held out for val only).
  - LaRS: train split for training, val split for validation.
LaRS test split was never used during training or validation, so it is a clean hold-out set
for reporting test metrics (mIoU, pixel accuracy, per-class IoU).

Expects LaRS layout:
  lars_images/test/images/*.jpg
  lars_annotations/test/image_annotations.json  (list of file_name)
  lars_annotations/test/semantic_masks/*.png     (optional; if missing, runs inference-only, no metrics)

In inference-only mode, prints predicted class distribution. If you see only 2 colors (e.g. red=Obstacle,
blue=Water) and ~0% for Sky/Land, the model has collapsed to 2 classes; see docs/CLASS_COLLAPSE.md for fixes.

Usage:
  py scripts/test_on_lars.py
  py scripts/test_on_lars.py --checkpoint path/to/best.pt
  py scripts/test_on_lars.py --save-dir output/lars_test_vis
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from model import get_segformer_b2_maritime
from data.dataset import LaRSDataset
from data.transforms import get_val_transforms


def load_model(ckpt_path: Path, device: str):
    model = get_segformer_b2_maritime(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()
    return model


def get_test_image_paths_from_json(images_dir: Path, annotations_dir: Path) -> list[Path]:
    """Build list of test image paths from image_annotations.json when semantic_masks are missing."""
    ann_file = annotations_dir / "test" / "image_annotations.json"
    if not ann_file.exists():
        return []
    with open(ann_file) as f:
        data = json.load(f)
    file_names = [a["file_name"] for a in data.get("annotations", [])]
    images_dir_test = images_dir / "test" / "images"
    paths = []
    for fn in file_names:
        p = images_dir_test / fn
        if p.exists():
            paths.append(p)
    return paths


def compute_metrics(pred: np.ndarray, mask: np.ndarray, num_classes: int, ignore_index: int):
    """Per-class IoU, mIoU, pixel accuracy. Only non-ignore pixels count."""
    valid = mask != ignore_index
    if not np.any(valid):
        return None
    pred_valid = pred[valid]
    mask_valid = mask[valid]
    acc = np.mean(pred_valid == mask_valid)

    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        mask_c = mask == c
        inter = np.logical_and(pred_c, mask_c).sum()
        union = np.logical_or(pred_c, mask_c).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    miou = float(np.mean(ious)) if ious else 0.0
    return {"accuracy": acc, "miou": miou, "ious": ious, "num_classes_found": len(ious)}


def main():
    default_ckpt = _PROJECT_ROOT / "output" / "models" / "best_segformer_b2_maritime.pt"
    parser = argparse.ArgumentParser(description="Evaluate on LaRS test set")
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt), help="Path to .pt checkpoint")
    parser.add_argument("--save-dir", type=str, default=None, help="If set, save overlay images here")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not config.LARS_IMAGES.exists() or not config.LARS_ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"LaRS data not found. Expected:\n  {config.LARS_IMAGES}\n  {config.LARS_ANNOTATIONS}"
        )

    device = args.device
    model = load_model(ckpt_path, device)
    transform = get_val_transforms()
    test_ds = LaRSDataset(
        "test",
        config.LARS_IMAGES,
        config.LARS_ANNOTATIONS,
        transform=transform,
    )
    if len(test_ds) == 0:
        # No semantic_masks for test: run inference-only using image list from image_annotations.json
        test_image_paths = get_test_image_paths_from_json(config.LARS_IMAGES, config.LARS_ANNOTATIONS)
        if not test_image_paths:
            raise RuntimeError(
                "LaRS test set is empty. Either provide\n"
                "  lars_annotations/test/semantic_masks/*.png\n"
                "or ensure lars_annotations/test/image_annotations.json exists and lists file_name entries "
                "that exist under lars_images/test/images/."
            )
        save_dir = Path(args.save_dir) if args.save_dir else _PROJECT_ROOT / "output" / "lars_test_vis"
        save_dir.mkdir(parents=True, exist_ok=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference", _PROJECT_ROOT / "scripts" / "inference.py")
        inf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inf)
        print(f"LaRS test semantic_masks not found; running inference-only on {len(test_image_paths)} images from image_annotations.json.")
        print(f"Overlays will be saved to {save_dir}")
        class_pixels = [0] * config.NUM_CLASSES
        total_pixels = 0
        for i, img_path in enumerate(test_image_paths):
            out_path = save_dir / f"{img_path.stem}_seg.png"
            _, pred = inf.run_image(model, img_path, device, out_path, show_horizon=False)
            for c in range(config.NUM_CLASSES):
                class_pixels[c] += int((pred == c).sum())
            total_pixels += pred.size
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_image_paths)}")
        print(f"Done. Saved {len(test_image_paths)} overlays to {save_dir}")
        if total_pixels > 0:
            print("Predicted class distribution (Sky=0, Water=1, Obstacle=2):")
            for c, name in enumerate(config.CLASS_NAMES):
                pct = 100.0 * class_pixels[c] / total_pixels
                print(f"  {name} ({c}): {pct:.1f}%")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # BGR colors for overlay (same as inference.py)
    CLASS_COLORS_BGR = [
        (180, 120, 255),   # Sky      (0)
        (255, 200, 100),   # Water    (1)
        (100, 100, 255),   # Obstacle (2)
    ]

    all_acc = []
    all_miou = []
    per_class_intersection = [0] * config.NUM_CLASSES
    per_class_union = [0] * config.NUM_CLASSES

    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()  # (B, H, W)
        names = batch["name"]

        with torch.no_grad():
            out = model(pixel_values=images)
        logits = torch.nn.functional.interpolate(
            out.logits,
            size=masks.shape[1:3],
            mode="bilinear",
            align_corners=False,
        )
        preds = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)

        for i in range(images.size(0)):
            pred = preds[i]
            mask = masks[i]
            m = compute_metrics(pred, mask, config.NUM_CLASSES, config.IGNORE_INDEX)
            if m is not None:
                all_acc.append(m["accuracy"])
                all_miou.append(m["miou"])
                for c in range(config.NUM_CLASSES):
                    pred_c = pred == c
                    mask_c = mask == c
                    valid = mask != config.IGNORE_INDEX
                    if np.any(mask_c & valid):
                        per_class_intersection[c] += np.logical_and(pred_c, mask_c).sum()
                        per_class_union[c] += np.logical_or(pred_c, mask_c).sum()

            if save_dir:
                # Save overlay: need RGB image from dataset (we have tensor); decode and draw
                img_t = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_np = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                img_bgr = (img_np * 255).astype(np.uint8)
                img_bgr = np.ascontiguousarray(img_bgr[:, :, ::-1])
                overlay = img_bgr.copy()
                for c in range(config.NUM_CLASSES):
                    overlay[pred == c] = CLASS_COLORS_BGR[c]
                overlay = (0.55 * img_bgr + 0.45 * overlay).astype(np.uint8)
                out_path = save_dir / f"{names[i]}_seg.png"
                cv2.imwrite(str(out_path), overlay)

    # Overall metrics
    mean_acc = float(np.mean(all_acc)) if all_acc else 0.0
    mean_miou = float(np.mean(all_miou)) if all_miou else 0.0

    # Per-class IoU from accumulated intersection/union
    class_ious = []
    for c in range(config.NUM_CLASSES):
        if per_class_union[c] > 0:
            class_ious.append(per_class_intersection[c] / per_class_union[c])
        else:
            class_ious.append(float("nan"))

    print(f"LaRS test set: {len(test_ds)} samples")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Pixel accuracy: {mean_acc:.4f}")
    print(f"mIoU: {mean_miou:.4f}")
    print("Per-class IoU:")
    for c, name in enumerate(config.CLASS_NAMES):
        iou = class_ious[c]
        s = f"  {name}: {iou:.4f}" if not np.isnan(iou) else f"  {name}: (no pixels)"
        print(s)
    if save_dir:
        print(f"Overlays saved to {save_dir}")


if __name__ == "__main__":
    main()
