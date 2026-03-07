"""
Inference: image, folder, video, or camera. Optional horizon line from sky/water boundary.
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from model import get_segformer_b2_maritime

# Normalization (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CLASS_COLORS_BGR = [
    (180, 120, 255),   # Sky - light purple
    (255, 200, 100),   # Water - blue-ish
    (100, 200, 100),   # Land - green
    (100, 100, 255),   # Obstacle - red
]


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


def preprocess(image: np.ndarray, target_h: int, target_w: int) -> torch.Tensor:
    """Resize, normalize, to tensor (1, 3, H, W)."""
    img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return x


def horizon_from_mask(mask: np.ndarray, sky_class: int = 0, water_class: int = 1) -> np.ndarray:
    """
    For each column, find the topmost water pixel (first transition from sky to water).
    Returns (W,) array of y-coordinates, or NaN where no horizon.
    """
    h, w = mask.shape
    horizon_y = np.full(w, np.nan, dtype=np.float32)
    for x in range(w):
        col = mask[:, x]
        for y in range(1, h):
            if col[y] == water_class and col[y - 1] == sky_class:
                horizon_y[x] = float(y)
                break
    return horizon_y


def draw_overlay(frame: np.ndarray, mask: np.ndarray, horizon_y: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    out = frame.copy()
    seg_bgr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(config.NUM_CLASSES):
        seg_bgr[mask == c] = CLASS_COLORS_BGR[c]
    out = (out * (1 - alpha) + seg_bgr * alpha).astype(np.uint8)
    if horizon_y is not None:
        valid = ~np.isnan(horizon_y)
        if np.any(valid):
            pts = np.column_stack((np.where(valid)[0], horizon_y[valid].astype(int)))
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(out, [pts], False, (0, 255, 0), 2)
    return out


def run_image(model: torch.nn.Module, image_path: Path, device: str, save_path: Path | None, show_horizon: bool):
    image = cv2.imread(str(image_path))
    if image is None:
        image = np.array(Image.open(image_path).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h_orig, w_orig = image.shape[:2]
    x = preprocess(image, config.INPUT_HEIGHT, config.INPUT_WIDTH).to(device)
    with torch.no_grad():
        out = model(pixel_values=x)
    logits = out.logits
    logits = torch.nn.functional.interpolate(
        logits, size=(h_orig, w_orig), mode="bilinear", align_corners=False
    )
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    horizon_y = horizon_from_mask(pred) if show_horizon else None
    vis = draw_overlay(image, pred, horizon_y)
    if save_path:
        cv2.imwrite(str(save_path), vis)
    return vis, pred


def run_video(model: torch.nn.Module, video_path: Path | int, device: str, save_path: Path | None, show_horizon: bool):
    cap = cv2.VideoCapture(int(video_path) if isinstance(video_path, int) else str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    out_video = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = cv2.VideoWriter(str(save_path), fourcc, cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h_orig, w_orig = frame.shape[:2]
        x = preprocess(frame, config.INPUT_HEIGHT, config.INPUT_WIDTH).to(device)
        with torch.no_grad():
            out = model(pixel_values=x)
        logits = out.logits
        logits = torch.nn.functional.interpolate(
            logits, size=(h_orig, w_orig), mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        horizon_y = horizon_from_mask(pred) if show_horizon else None
        vis = draw_overlay(frame, pred, horizon_y)
        if out_video:
            out_video.write(vis)
        cv2.imshow("SegFormer maritime", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_segformer_b2_maritime.pt")
    parser.add_argument("--input", type=str, required=True, help="Image path, folder path, video path, or camera index (e.g. 0)")
    parser.add_argument("--output", type=str, default=None, help="Output image/video path")
    parser.add_argument("--no-horizon", action="store_true", help="Do not draw horizon line")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    model = load_model(Path(args.checkpoint), device)
    show_horizon = not args.no_horizon

    inp = Path(args.input)
    if inp.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        out_path = Path(args.output) if args.output else inp.parent / (inp.stem + "_seg.png")
        run_image(model, inp, device, out_path, show_horizon)
        print(f"Saved to {out_path}")
    elif inp.is_dir():
        out_dir = Path(args.output) if args.output else inp / "seg_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(inp.glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                run_image(model, f, device, out_dir / (f.stem + "_seg.png"), show_horizon)
        print(f"Saved to {out_dir}")
    else:
        try:
            idx = int(args.input)
            run_video(model, idx, device, Path(args.output) if args.output else None, show_horizon)
        except ValueError:
            run_video(model, inp, device, Path(args.output) if args.output else None, show_horizon)


if __name__ == "__main__":
    main()
