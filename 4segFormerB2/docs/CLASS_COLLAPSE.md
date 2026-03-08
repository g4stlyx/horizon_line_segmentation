# Dataset label inconsistency → Land vs Obstacle confusion

> **This document has been updated after diagnosing the actual failure mode observed in training run (epoch 1-60, mIoU 0.98+).**

## What was observed

Segmentation overlays on LaRS test images showed **ships and port structures as green (Land)** instead of red (Obstacle). On open-ocean LaRS images the model correctly predicted red (Obstacle) for ships. The 0.98+ mIoU was entirely misleading.

## Root cause: contradictory labels between datasets

The original 4-class schema mapped the same visual concept to **different classes** depending on which dataset an image came from:

| Dataset   | Raw class 0 label           | Mapped to (unified) |
|-----------|-----------------------------|---------------------|
| MaSTr1325 | Obstacles & Environment (0) | **Land (class 2)**  |
| LaRS      | Obstacles (0)               | **Obstacle (class 3)** |

A ship was **Land** in every MaSTr image and **Obstacle** in every LaRS image. The model stopped learning semantics and instead learned *dataset style* as a shortcut: harbour/port scene → predict Land; open-ocean scene → predict Obstacle.

The 0.98 mIoU was real — but only because the val set was split per-dataset. Each dataset''s val images were internally consistent, so the model looked correct on both splits, while being wrong on any image that didn''t match a known dataset distribution.

## Fix applied (✅ already done)

Switched to a **3-class model: Sky (0), Water (1), Obstacle (2)**.

Both datasets now map identically:

```python
LARS_TO_UNIFIED  = {0: 2, 1: 1, 2: 0}   # Obstacles->2, Water->1, Sky->0
MASTR_TO_UNIFIED = {0: 2, 1: 1, 2: 0}   # Env/Obstacles->2, Water->1, Sky->0
```

This eliminates all contradictory supervision. The horizon line (sky/water boundary) is still perfectly extractable from the 3-class output.

## Expected outcome after retraining

- mIoU will likely drop to ~0.85–0.92 (honest number, no longer inflated by the dataset-style shortcut)
- Obstacle predictions will be visually correct and consistent across different scene types
- Horizon line detection quality unaffected

## Remaining risk: class imbalance

With consistent labels, Sky pixels may be sparse in heavily obstructed harbour shots and Obstacle pixels sparse in open-ocean shots. Mitigate with class weights in the loss:

```python
# In training loop (already added to the Colab notebook)
class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)  # sky, water, obstacle
criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
```

Tune these once you have a pixel-frequency count from the training set.

## Summary

| Symptom                          | Cause                                              |
|----------------------------------|----------------------------------------------------|
| Ships green in harbour scenes    | MaSTr mapped obstacles to Land (class 2)           |
| Ships red in open-ocean scenes   | LaRS mapped obstacles to Obstacle (class 3)        |
| mIoU 0.98+ but wrong on new data | Model learned dataset style, not visual semantics  |
| Fix                              | 3-class unified schema with identical mappings     |
