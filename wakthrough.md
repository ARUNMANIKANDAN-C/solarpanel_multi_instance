# Analysis: FastViT vs MobileViTv2 for PV-Multi-Defect Detection

Both notebooks implement a **Faster R-CNN** object detection pipeline for **solar panel defect detection** on the PV-Multi-Defect dataset (5 defect classes + background). They share an identical data preprocessing pipeline but diverge significantly in model architecture.

---

## Shared Pipeline (Identical in Both)

| Stage | Details |
|---|---|
| **Dataset** | PV-Multi-Defect — 1,106 images, 3,981 objects |
| **Classes** | `scratch`, `black_border`, `no_electricity`, `hot_spot`, `broken` + `__background__` |
| **Augmentation** | `FixedAugmenter` — horizontal/vertical flip, brightness/contrast, noise, blur, HSV shift, rotation, scaling, translation, cutout |
| **Augmentation Target** | 9,000 samples per class |
| **Split Ratios** | 70% train / 15% val / 15% test |
| **DataLoader** | `FasterRCNNDataset` with VOC XML parsing, batch_size=16, 2 workers |
| **Training Config** | 200 epochs, lr=1e-4, weight_decay=1e-4 |
| **Metrics** | Precision, Recall, F1, mAP@0.5 (class-wise & overall) |
| **Optimizer** | AdamW with stage-wise backbone unfreezing |

---

## Architecture Differences

### Backbone

| Feature | [base-template(1).ipynb](file:///c:/Users/arunm/Documents/Project/analysis/base-template(1).ipynb) | [base-template-mobilevit(1).ipynb](file:///c:/Users/arunm/Documents/Project/analysis/base-template-mobilevit(1).ipynb) |
|---|---|---|
| **Backbone** | FastViT-T8 (`fastvit_t8`) | MobileViTv2-100 (`mobilevitv2_100`) |
| **Pretrained Source** | `timm/fastvit_t8.apple_in1k` (16.3 MB) | `timm/mobilevitv2_100.cvnets_in1k` (19.7 MB) |
| **Feature Channels** | `[48, 96, 192, 384]` (4 levels) | `[128, 256, 384]` (3 levels, `out_indices=(2,3,4)`) |
| **Architecture Type** | Hybrid CNN with structural re-parameterization | Hybrid CNN + Transformer (linear attention) |
| **Total Parameters** | **29,974,487** | **21,704,211** |
| **Trainable Parameters** | 26,802,551 (89.4%) | 17,315,370 (79.8%) |

### Neck / Feature Fusion

| Feature | FastViT Notebook | MobileViTv2 Notebook |
|---|---|---|
| **FPN Type** | Standard `FeaturePyramidNetwork` (torchvision) | Custom `BiFPNParallelAdapter` per level |
| **Post-FPN Modules** | `C2f` → `CoordAtt` → `SqueezeExcitation` → `ChannelAttention` + `DetailConvBlock` (levels 0–1) + `SPPF` (last level) | `GhostConv` stem → parallel branches (`LiteDetailBlock` + `LiteTextureBlock` + `LiteASPPv2` + Identity) → `BiFPNFusion` → `CBAM` |
| **Inter-level Fusion** | Bottom-up pathway with downsample convs + concat fusion | Progressive top-down → bottom-up `WeightedAdd` fusion → `cross_refine` Conv3×3 blocks |
| **Detail Extraction** | `DetailConvBlock` — 6-branch multi-kernel DW convs (7, 9, 11, 13, H-strip, V-strip) | `LiteDetailBlock` — 3-branch DW convs (7, 11, 13) with additive fusion |
| **Multi-Scale Pooling** | `SPPF` (cascaded MaxPool k=5) | `LiteASPPv2` (dilated DW convs d=1, d=2) |

### Attention Mechanisms

| FastViT Notebook | MobileViTv2 Notebook |
|---|---|
| `CoordAtt` — coordinate attention (H/W pooling) | `SpatialAttention` — 7×7 conv spatial gate |
| `SqueezeExcitation` — channel squeeze-excite (reduction=16) | `CBAM` — full channel + spatial attention |
| `ChannelAttention` — CBAM-style channel-only (avg+max pool) | `WeightedAdd` / `BiFPNFusion` — learnable weighted sum |

### Detection Head

| Feature | FastViT Notebook | MobileViTv2 Notebook |
|---|---|---|
| **RoI Pooling** | 4 feature maps (`"0","1","2","3"`) | 3 feature maps (`"0","1","2"`) |
| **Anchor Sizes** | `(33), (55), (80), (160), (330)` | `(16,32), (64,128), (256,512)` |
| **Aspect Ratios** | `(0.11, 0.24, 0.45, 0.99, 2.54)` | `(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)` |
| **Image Size** | `min=512, max=1024` | `min=800, max=1333` (config) / `min=640, max=640` (smoke test) |
| **Loss** | Standard Faster R-CNN loss | **Custom `WeightedRoIHeads`** — class-weighted cross-entropy for classification |
| **Class Weights** | None (uniform) | Computed dynamically: `[1.0, 1.56, 1.55, 0.42, 1.56, 1.40]` |

> [!IMPORTANT]
> The MobileViTv2 notebook uses **class-weighted loss** to handle the severe class imbalance in PV-Multi-Defect (hot_spot is heavily overrepresented with weight 0.42, while black_border and no_electricity get boosted to 1.56).

### Stage-wise Unfreezing Schedule

| Epoch | FastViT Notebook | MobileViTv2 Notebook |
|---|---|---|
| 0 | Freeze all backbone | Freeze all encoder |
| 5 | — | Unfreeze stages_4 (deepest) |
| 10 | Unfreeze stage 3 | Unfreeze stages_3 + stages_4 |
| 15 | Unfreeze stage 2 | — |
| 20 | — | Unfreeze full encoder |
| 25 | Unfreeze stage 0 | — |

---

## Key Design Decisions

### FastViT Notebook
- **Heavier architecture** (30M params) with a rich attention stack (CoordAtt + SE + CBAM channel)
- **4-level FPN** providing finer multi-scale coverage
- **Single anchor per level** with custom aspect ratios tuned for solar panel defects
- **Standard loss** — relies on data augmentation for class balance
- **Gradual unfreezing** over 25 epochs (one stage every 5 epochs)

### MobileViTv2 Notebook
- **Lighter architecture** (21.7M params) with transformer-based global context
- **3-level BiFPN-style fusion** with parallel detail/texture/context branches per level
- **Paired anchors per level** with extreme aspect ratios (0.1–10.0) for elongated scratches
- **Weighted loss** — explicitly addresses class imbalance via `WeightedRoIHeads`
- **Faster unfreezing** — full encoder unfrozen by epoch 20

---

## Augmentation Results

| Metric | FastViT Notebook | MobileViTv2 Notebook |
|---|---|---|
| Verified augmented train files | **5,466** | **5,096** |
| Val samples | 164 | 164 |
| Test samples | 170 | 169 |

> [!NOTE]
> The slight difference in augmented file counts (5,466 vs 5,096) is due to random seed differences in the stratified split, resulting in a slightly different train set composition.
