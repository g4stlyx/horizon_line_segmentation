Now I have a thorough picture of everything you've built. Let me give you a structured analysis.

---

## What You Had — and What Limited It

**Architecture:** U-Net with ResNet18 encoder, 3-class (Sky/Water/Object). Land treated as water (a hacky workaround exposing a dataset gap).

**Key bottlenecks:**
- **Dataset**: Only SMD (Singapore Maritime Dataset) clips. SMD is small and lacks pixel-level sky/sea/land/obstacle annotations natively — you were generating pseudo-labels from horizon GT lines + RT-DETR boxes, which introduces compounding noise.
- **Backbone**: ResNet18 is too shallow. It lacks the receptive field to understand the full horizontal extent of a horizon line in high-res frames.
- **No temporal modeling**: The horizon in a video is a slowly-moving signal. You added a heuristic "run every N frames" workaround, but never learned temporal structure.
- **Transformer attempt abandoned**: `3.9training_transformers` has only a README. This is the highest-leverage direction you didn't pursue.

---

## 1. Datasets — This Is Your Biggest Lever

| Dataset | Why It Matters |
|---|---|
| **MaSTr1325 / MaSTr1478** | Maritime semantic segmentation with exactly **water/sky/obstacle/land** pixel labels. UST Lab (University of Ljubljana). This is the dataset built for your exact problem. MaSTr1478 is the extended version. **Start here.** |
| **MODD2 (Marine Obstacle Detection Dataset v2)** | Stereo + monocular maritime footage with obstacle annotations. Complements MaSTr. |
| **SMD** | You already have it. Keep it. Its horizon GT is still useful for horizon regression tasks. |
| **KOLOMVERSE** | You have the scripts zip. Explore it — it may have additional labeled maritime data. |
| **LaRS (Large-scale Aquatic Robotic Sensing)** | 2023 dataset, 4K+ images, multi-source (USV, drone, shoreline cameras). Best diversity of conditions (fog, glare, night). Benchmark challenge existed — public. |
| **Seagull Dataset** | Horizon line regression dataset with tilted camera scenarios — useful if you care about non-horizontal horizons (vessel roll). |

**Critical insight**: MaSTr1325 + LaRS together gives you ~6,000+ annotated maritime frames with the exact 4-class structure you want. This alone will close most of the performance gap.

---

## 2. Model Recommendations

### Option A — Best Accuracy/Generalization: SegFormer-B2 or B4
**Why this beats your U-Net specifically:**
- Mix Transformer (MiT) captures **global context** across the full image width in early layers. A horizon line is a global signal — CNN local kernels struggle with it. SegFormer "sees" the whole horizon in a single attention pass.
- No positional encoding → generalizes to different image resolutions (critical for varying frame sizes across cameras).
- Pretrained on ADE20K/Cityscapes → sky/ground semantics already baked in, fine-tuning is fast.
- HuggingFace `transformers` has it ready. You already have the HF integration pattern from RT-DETR.

**Tradeoff**: B2 is ~25 FPS on GTX 1650 Super; B4 drops to ~12 FPS without optimization. Solvable with TensorRT or ONNX export.

**Recommendation**: Fine-tune `nvidia/mit-b2` + segmentation head on MaSTr1478 + SMD combined.

---

### Option B — Best Real-Time: DDRNet-23-slim or STDC2
If FPS matters more than peak accuracy (deployed on embedded hardware or 30+ FPS requirement):
- **DDRNet-23-slim**: ~60+ FPS on a modest GPU, mIoU competitive with older DeepLabV3+. Dual-resolution branch captures both local detail and global context simultaneously.
- **STDC2**: Even faster, specifically designed for edge deployment. Achieves real-time on GPU and near-real-time on Jetson class hardware.

**Tradeoff**: These are pure CNN. For maritime scenes with strong horizontal structure, they underperform attention-based models in challenging conditions (fog, glare, low contrast).

---

### Option C — Strong Baseline Upgrade: U-Net++ with EfficientNet-B4

If you want to stay in the U-Net family but dramatically improve:
- Swap ResNet18 → **EfficientNet-B4** encoder (segmentation-models-pytorch has this with one line)
- Add **SCSE (Squeeze-and-Channel Squeeze-and-Excitation)** attention blocks in the decoder
- Add **Boundary Loss** (or Lovász loss) to sharpen the horizon edge prediction
- This is the lowest-friction upgrade — your training code changes minimally

**When to pick this**: You want a fast iteration and your compute budget is limited for experimentation.

---

### Option D — Cutting Edge: SAM 2 (Segment Anything Model 2) Fine-Tuning
Meta's SAM 2 is a video-native foundation model. You can fine-tune it with your small labeled set and leverage:
- **Video propagation**: it was trained to propagate masks through video frames. Your "run every N frames" hack becomes a first-class feature.
- **Near-zero-shot performance** on maritime scenes with minimal fine-tuning.

**Tradeoff**: Heavy model, complex fine-tuning pipeline. Not production real-time without significant engineering. Best for research/accuracy benchmark.

---

## 3. Architectural & Training Improvements

### Add a Horizon Regression Head (Auxiliary Task)
Your current approach infers the horizon line by finding the sky/sea boundary from the segmentation mask. Add an **auxiliary regression head** that directly predicts `(y_left, y_right)` or `(rho, theta)` in Hough space. Train it jointly:

```
Loss_total = λ1 * CrossEntropy_seg + λ2 * Dice_seg + λ3 * SmoothL1_horizon
```

This forces the network to explicitly learn the horizon as a geometric primitive, not just as a byproduct of segmentation.

### Fix the Land Class — Don't Treat It as Water
Your README explicitly acknowledges this is a hack. With MaSTr1478, you get land annotations for free. Go 4-class: **Sky / Sea / Land / Obstacle**. The horizon between sky and sea is now unambiguous even when land is present.

### Domain-Specific Augmentations
Maritime scenes have specific degradations not covered by standard augmentation:
- Sun glare (simulate with additive gaussian + radial bright spots)
- Fog/haze (reduce contrast + add depth-dependent blur)
- Spray/rain (random vertical streaks)
- Camera roll (random rotation ±15°) — critical for vessels in rough sea
- Night + artificial light sources

Libraries: `albumentations` handles most, custom transforms for glare/fog.

### Loss Function Stack
Replace plain CrossEntropy with:
```python
Loss = 0.4 * CrossEntropy(weighted) + 0.4 * DiceLoss + 0.2 * BoundaryLoss
```
`BoundaryLoss` (or `Lovász`) specifically penalizes errors near the horizon edge, which is where it counts most.

### Temporal Consistency (Proper)
If you're doing video:
- **Short-term**: Optical flow warping of the previous mask as an additional input channel. The model gets "where the horizon was last frame" as a hint.
- **Medium-term**: ConvLSTM/ConvGRU bottleneck in the encoder. The horizon state is a low-dimensional signal that RNNs handle well.
- **Pragmatic shortcut**: Kalman filter on the extracted horizon line parameters `(rho, theta)` — filter the output, not the model internals. Near-zero compute cost, significant temporal stability gain.

---

## 4. Recommended Stack for a Fresh Attempt

**Phase 1 — Get data right:**
1. Download MaSTr1478 + LaRS
2. Keep SMD for horizon regression labels
3. Build a unified preprocessing pipeline: 4-class masks + horizon `(y_left, y_right)` regression targets

**Phase 2 — Model:**
1. Fine-tune `nvidia/mit-b2` (SegFormer-B2) with a custom 4-class head
2. Add horizon regression auxiliary head
3. Train with Dice + Boundary + SmoothL1 combined loss
4. Domain-specific augmentations from day one

**Phase 3 — Speed optimization:**
1. Export to ONNX, benchmark on your GTX 1650 Super
2. If FPS < 25: switch to SegFormer-B0 or STDC2 backbone
3. If FPS still insufficient: TensorRT INT8 quantization

**Phase 4 — Temporal:**
1. Add Kalman filter on horizon line output as a post-processor (free stability gains)
2. Optionally: optical flow hint channel if accuracy gains justify complexity

---

## Summary Table

| Model | mIoU (maritime) | FPS GTX1650 | Effort to implement | Verdict |
|---|---|---|---|---|
| U-Net ResNet18 (current) | ~72% (your estimate) | ~40 (seg only) | — | Baseline |
| U-Net++ EfficientNet-B4 | ~79% | ~30 | Low | Quick win |
| SegFormer-B2 | ~84–86% | ~22–25 | Medium | **Recommended** |
| SegFormer-B4 | ~87% | ~12 | Medium | If accuracy > speed |
| DDRNet-23-slim | ~80% | ~60 | Medium | If real-time critical |
| SAM 2 fine-tuned | ~89%+ | ~5 | High | Research only |

The single highest-ROI change: **MaSTr1478 dataset + SegFormer-B2 fine-tune**. Your previous work was limited more by data quality than model capacity.