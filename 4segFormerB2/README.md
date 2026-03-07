# SegFormer-B2 Maritime Segmentation (LaRS + MaSTr1325)

4-class semantic segmentation (Sky, Water, Land, Obstacle) for horizon line detection and sea-sky-land segmentation in maritime/water domains. Uses a single SegFormer-B2 model trained on **LaRS** and **MaSTr1325** with unified labels.

## Dataset layout

Place datasets under `4segFormerB2/datasets/`:

- **LaRS**: `datasets/lars/lars_images/` and `datasets/lars/lars_annotations/` (train/val/test with `images/`, `semantic_masks/`, `image_annotations.json`).
- **MaSTr1325**: `datasets/mastr1325/MaSTr1325_images_512x384/` (e.g. `0001.jpg`) and `datasets/mastr1325/MaSTr1325_masks_512x384/` (e.g. `0001m.png`).

Label mapping:

- **Unified classes**: 0=Sky, 1=Water, 2=Land, 3=Obstacle, 255=ignore.
- **LaRS** (semantic PNG): 0→Obstacle, 1→Water, 2→Sky.
- **MaSTr1325**: 0→Land, 1→Water, 2→Sky; 4→ignore.

## Colab (one notebook)

Use **`SegFormer_B2_maritime_colab.ipynb`** on Google Colab. Data is read from Drive:

- **Path**: `MaritimeSegmentation/datasets/` with subfolders `lars/` and `mastr1325/` (same structure as above).
- Checkpoints are saved to `MaritimeSegmentation/checkpoints/`.

Upload the notebook to Colab, set runtime to GPU (T4), mount Drive, and run all cells.

## Setup (local)

```bash
cd 4segFormerB2
pip install -r requirements.txt
```

## Training

```bash
python train.py --save-dir checkpoints
# Optional: --batch-size 8 --epochs 60 --lr 6e-5 --no-mastr or --no-lars
```

Best checkpoint is saved as `checkpoints/best_segformer_b2_maritime.pt`.

## Inference

- **Single image**: `python inference.py --checkpoint checkpoints/best_segformer_b2_maritime.pt --input path/to/image.jpg [--output out.png]`
- **Folder**: `python inference.py --checkpoint ... --input path/to/folder/ [--output path/to/out_dir/]`
- **Video**: `python inference.py --checkpoint ... --input path/to/video.mp4 [--output out.mp4]`
- **Camera**: `python inference.py --checkpoint ... --input 0`

Use `--no-horizon` to skip drawing the horizon line (sky/water boundary).

## Config

Edit `config.py` for paths, input size (`INPUT_HEIGHT`, `INPUT_WIDTH`), batch size, epochs, LR, and encoder name.
