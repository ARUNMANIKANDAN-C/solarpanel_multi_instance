# FastViT vs MobileViTv2: Solar Panel Multi-Defect Detection

A comprehensive research comparison of two efficient architectures (**FastViT-T8** and **MobileViTv2-100**) for real-time solar panel defect detection using Faster R-CNN on the **PV-Multi-Defect dataset**.

## 🚀 Project Overview

This repository implements a modular Faster R-CNN pipeline comparing hybrid CNN architectures for detecting 5 defect classes in solar panels:
- **Scratch**
- **Black Border**
- **No Electricity**
- **Hot Spot**
- **Broken**

Both models share an identical data preprocessing and augmentation pipeline but diverge in architecture design, making this a valuable case study for efficient object detection.

## 📊 Dataset & Preprocessing

| Aspect | Value |
|--------|-------|
| **Dataset** | PV-Multi-Defect — 1,106 images, 3,981 objects |
| **Classes** | 5 defect types + background |
| **Augmentation** | FixedAugmenter with 11 techniques (flip, brightness/contrast, noise, blur, HSV shift, rotation, scaling, translation, cutout) |
| **Target Samples** | 9,000 per class (balanced via augmentation) |
| **Train/Val/Test Split** | 70% / 15% / 15% |
| **Batch Size** | 16 |
| **Epochs** | 200 |
| **Optimizer** | AdamW (lr=1e-4, weight_decay=1e-4) |

## 🏗️ Architecture Comparison

### FastViT-T8 Notebook

- **Backbone**: FastViT-T8 (16.3 MB, pretrained on ImageNet-1K)
- **Parameters**: 29,974,487 total (26.8M trainable)
- **Feature Levels**: 4 (channels: [48, 96, 192, 384])
- **Neck**: Standard FPN + Rich attention stack (CoordAtt + SqueezeExcitation + ChannelAttention)
- **Detail Extraction**: 6-branch DetailConvBlock (kernels: 7, 9, 11, 13, H-strip, V-strip)
- **Multi-Scale Pooling**: SPPF (cascaded MaxPool k=5)
- **Anchor Sizes**: (33), (55), (80), (160), (330)
- **Aspect Ratios**: (0.11, 0.24, 0.45, 0.99, 2.54)
- **Loss Strategy**: Standard Faster R-CNN (relies on augmentation for class balance)
- **Unfreezing**: Gradual stage-wise over 25 epochs

### MobileViTv2-100 Notebook

- **Backbone**: MobileViTv2-100 (19.7 MB, pretrained on ImageNet-1K)
- **Parameters**: 21,704,211 total (17.3M trainable)
- **Feature Levels**: 3 (channels: [128, 256, 384])
- **Neck**: Custom BiFPNParallelAdapter with parallel detail/texture/context branches
- **Detail Extraction**: 3-branch LiteDetailBlock (kernels: 7, 11, 13)
- **Multi-Scale Pooling**: LiteASPPv2 (dilated DW convs d=1, d=2)
- **Anchor Sizes**: (16,32), (64,128), (256,512)
- **Aspect Ratios**: (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
- **Loss Strategy**: WeightedRoIHeads with class-weighted cross-entropy [1.0, 1.56, 1.55, 0.42, 1.56, 1.40]
- **Unfreezing**: Full encoder unfrozen by epoch 20

## 📈 Key Design Insights

| Aspect | FastViT | MobileViTv2 |
|--------|---------|-------------|
| **Architecture Type** | Hybrid CNN + Structural Re-parameterization | Hybrid CNN + Transformer (Linear Attention) |
| **Model Size** | Heavier (30M) | Lighter (21.7M) |
| **FPN Strategy** | 4-level standard FPN | 3-level BiFPN with weighted fusion |
| **Class Imbalance Handling** | Data augmentation only | Explicit class weighting |
| **Attention Mechanisms** | CoordAtt + SE + CBAM | SpatialAttention + CBAM |

## 📊 Results & Metrics

### Dataset Splits & Augmentation

| Metric | FastViT Notebook | MobileViTv2 Notebook |
|--------|-----------------|---------------------|
| **Original Images** | 1,106 | 1,106 |
| **Original Objects** | 3,981 | 3,981 |
| **Augmented Train Files** | 5,466 | 5,096 |
| **Validation Samples** | 164 | 164 |
| **Test Samples** | 170 | 169 |

### Training Configuration

| Parameter | FastViT | MobileViTv2 |
|-----------|---------|-------------|
| **Epochs** | 200 | 200 |
| **Batch Size** | 16 | 16 |
| **Learning Rate** | 1e-4 | 1e-4 |
| **Weight Decay** | 1e-4 | 1e-4 |
| **Optimizer** | AdamW | AdamW |
| **Loss Function** | Standard Faster R-CNN | Weighted Faster R-CNN |

### Class Distribution & Weights

| Class | Weight | Notes |
|-------|--------|-------|
| Background | 1.0 | Base weight |
| Scratch | 1.0 / 1.56 | Rare class (MobileViTv2 boosted) |
| Black Border | 1.0 / 1.55 | Rare class (MobileViTv2 boosted) |
| No Electricity | 1.0 / 1.55 | Rare class (MobileViTv2 boosted) |
| Hot Spot | 1.0 / 0.42 | Heavily overrepresented (downweighted) |
| Broken | 1.0 / 1.40 | Rare class (MobileViTv2 boosted) |

> **Note**: FastViT relies purely on data augmentation for class balance, while MobileViTv2 uses explicit class weighting in the loss function to handle severe class imbalance.

### Expected Evaluation Metrics

Both models report on test set:
- **Precision** (per-class & overall)
- **Recall** (per-class & overall)
- **F1-Score** (per-class & overall)
- **mAP@0.5** (mean Average Precision at IoU=0.5)
- **Confusion Matrix** (class-wise predictions)

## 📂 Repository Structure

```
├── base-template(1).ipynb              # FastViT-T8 implementation
├── base-template-mobilevit.ipynb       # MobileViTv2-100 implementation
├── augmented-fasterrcnn.ipynb          # Augmentation analysis
├── kaggle_train.ipynb                  # Training and evaluation notebook
├── voc_dataset.py                      # VOC XML dataset parser
├── voc_aug.py                          # FixedAugmenter implementation
├── voc_utils.py                        # Dataset utilities
├── train.py                            # Training loop
├── main.py                             # Script entry point
├── requirements.txt                    # Dependencies
├── verify_aug.py                       # Augmentation verification
├── verify_modular_setup.py             # Module validation
└── wakthrough.md                       # Detailed architecture analysis
```

## 🔬 Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run FastViT notebook**:
   ```bash
   jupyter notebook base-template(1).ipynb
   ```

3. **Run MobileViTv2 notebook**:
   ```bash
   jupyter notebook base-template-mobilevit.ipynb
   ```

4. **Run training script**:
   ```bash
   python main.py --epochs 200 --batch_size 16
   ```

## 📊 Expected Metrics

Both models report:
- **Precision, Recall, F1** (class-wise & overall)
- **mAP@0.5** (mean Average Precision at IoU=0.5)
- **Per-class performance** accounting for severe class imbalance

## 🔎 Research Highlights

1. **Efficient architectures** for real-time solar panel inspection on edge devices
2. **Comparative study** of CNN vs Transformer-based hybrid designs
3. **Class imbalance handling** strategies (augmentation vs weighted loss)
4. **Stage-wise unfreezing** for effective transfer learning
5. **Modular pipeline** enabling easy architecture swaps and experimentation

## 📝 License

MIT License

