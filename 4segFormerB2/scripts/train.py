"""
Train SegFormer-B2 on combined LaRS + MaSTr1325 with unified 4-class labels.
"""
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data import CombinedMaritimeDataset, get_train_transforms, get_val_transforms
from model import get_segformer_b2_maritime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--no-mastr", action="store_true", help="Disable MaSTr1325")
    parser.add_argument("--no-lars", action="store_true", help="Disable LaRS")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint dir (default: config.CHECKPOINTS_DIR)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    save_dir = Path(args.save_dir or config.CHECKPOINTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    train_ds = CombinedMaritimeDataset(
        "train",
        transform=train_tf,
        use_mastr=not args.no_mastr,
        use_lars=not args.no_lars,
    )
    val_ds = CombinedMaritimeDataset(
        "val",
        transform=val_tf,
        use_mastr=not args.no_mastr,
        use_lars=not args.no_lars,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = get_segformer_b2_maritime(pretrained=config.PRETRAINED)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    # Poly LR
    def lr_lambda(epoch):
        return (1 - epoch / args.epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            optimizer.zero_grad()
            out = model(pixel_values=images)
            logits = out.logits  # (B, C, h, w) at 1/4 resolution
            h, w = masks.shape[1], masks.shape[2]
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        if (epoch + 1) % config.VAL_EVERY_N_EPOCHS == 0:
            model.eval()
            val_loss = 0.0
            correct = total_pixels = 0
            class_correct = [0] * config.NUM_CLASSES
            class_total = [0] * config.NUM_CLASSES
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(args.device)
                    masks = batch["mask"].to(args.device)
                    out = model(pixel_values=images)
                    logits = out.logits
                    h, w = masks.shape[1], masks.shape[2]
                    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                    loss = criterion(logits, masks)
                    val_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    valid = masks != config.IGNORE_INDEX
                    correct += (pred[valid] == masks[valid]).sum().item()
                    total_pixels += valid.sum().item()
                    for c in range(config.NUM_CLASSES):
                        m = (masks == c)
                        if m.any():
                            class_total[c] += m.sum().item()
                            class_correct[c] += (pred[m] == c).sum().item()
            val_loss /= max(len(val_loader), 1)
            acc = correct / max(total_pixels, 1)
            ious = []
            for c in range(config.NUM_CLASSES):
                if class_total[c] > 0:
                    ious.append(class_correct[c] / class_total[c])
            miou = sum(ious) / max(len(ious), 1) if ious else 0.0
            print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  acc={acc:.4f}  mIoU={miou:.4f}")

            if miou > best_miou:
                best_miou = miou
                ckpt = save_dir / "best_segformer_b2_maritime.pt"
                torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "miou": miou}, ckpt)
                print(f"  -> saved best to {ckpt}")

        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_dir / f"segformer_b2_epoch_{epoch+1}.pt")

    print(f"Done. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
