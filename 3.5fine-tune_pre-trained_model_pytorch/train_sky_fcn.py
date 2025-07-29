"""
Fine-tune FCN-ResNet50 for binary SKY / NON-SKY segmentation
on maritime imagery (e.g. Singapore Maritime Dataset, Buoy Dataset).

Dataset layout  (one sub-dir per split):
   MaritimeSky/
       train/
           images/
               frame_0001.jpg
               frame_0002.jpg
               ...
           masks/
               frame_0001.png   # uint8 {0,1}, 1 = sky
               frame_0002.png
       val/
           images/...
           masks/...
       test/
           images/...
           masks/...

Mask files must share **exact** filename stem with the image.
"""
import argparse, os, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T, models
from PIL import Image

# -----------------------  Dataset -----------------------
class MaritimeSky(Dataset):
    def __init__(self, root, split, img_size=512):
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        # ── transforms
        self.t_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
        ])
        self.t_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.PILToTensor(),        # -> (1,H,W) uint8
        ])

    def __len__(self):  return len(self.files)

    def __getitem__(self, idx):
        name   = self.files[idx]
        img_p  = os.path.join(self.img_dir,  name)
        mask_p = os.path.join(self.mask_dir, name).replace(".jpg",".png")
        img  = self.t_img(Image.open(img_p).convert("RGB"))
        mask = self.t_mask(Image.open(mask_p))      # (1,H,W)
        mask = mask.squeeze(0).long()               # (H,W)
        return img, mask

# ----------------------  Model --------------------------
def get_fcn(num_classes=2, pretrained=True):
    """Load torchvision FCN‑ResNet50 and re‑init classifier."""
    model = models.segmentation.fcn_resnet50(weights="DEFAULT" if pretrained else None)
    in_ch  = model.classifier[4].in_channels          # 512
    model.classifier[4] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    return model

# --------------------  Metrics --------------------------
@torch.no_grad()
def pixel_acc(logit, mask):
    """logit: (B,2,H,W), mask: (B,H,W)"""
    pred = logit.argmax(1)
    correct = (pred == mask).float().sum()
    total   = mask.numel()
    return correct / total

# --------------------  Train / Eval ---------------------
def run_epoch(loader, model, optim, device, train=True):
    if train: model.train()
    else:     model.eval()
    tot_loss, tot_acc = 0, 0
    for img, m in loader:
        img, m = img.to(device), m.to(device)
        out    = model(img)['out']          # (B,2,H,W)
        loss   = nn.CrossEntropyLoss()(out, m)
        if train:
            optim.zero_grad(); loss.backward(); optim.step()
        tot_loss += loss.item() * img.size(0)
        tot_acc  += pixel_acc(out, m) * img.size(0)
    n = len(loader.dataset)
    return tot_loss / n, tot_acc.item() / n

# -------------------------  Main ------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Data
    train_ds = MaritimeSky(args.data_root, "train", args.img_size)
    val_ds   = MaritimeSky(args.data_root, "val",   args.img_size)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    # Model
    model = get_fcn(num_classes=2, pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc, best_ep = 0, -1
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_ld, model, optim, device, train=True)
        val_loss,val_acc= run_epoch(val_ld,   model, optim, device, train=False)
        if val_acc > best_acc:
            best_acc, best_ep = val_acc, ep
            torch.save(model.state_dict(), "best_fcn_sky.pth")
        print(f"[{ep:02d}/{args.epochs}] "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} "
              f"(best {best_acc:.3f}@{best_ep})  [{time.time()-t0:.1f}s]")

    print("✓ training finished – best model saved to best_fcn_sky.pth")

# --------------------  CLI ------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="root folder of dataset")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--img_size",  type=int, default=512)
    p.add_argument("--lr",        type=float, default=1e-4)
    args = p.parse_args()
    main(args)