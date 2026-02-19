
# PVELAD: Pascal VOC Enhanced Learning & Augmentation Detection

A robust, modular Faster R-CNN training pipeline tailored for Kaggle and low-resource environments (3GB VRAM).

## 🚀 Features

-   **Modular Design**: Clean separation of configuration, dataset, augmentation, and training logic.
-   **Kaggle-Ready**: Organized for easy deployment as Utility Scripts.
-   **Adaptive Augmentation**: "Crop Cut" (`CoarseDropout`) and stronger augmentations for minority classes.
-   **Rich Logging**: Beautiful console output with progress bars and class distribution tables.
-   **Low VRAM Optimization**: Tuned for 3GB VRAM GPUs (`batch_size=2`, `num_workers=0`).
-   **Aggressive Class Balancing**: `WeightedRandomSampler` with inverse frequency sampling.

## 📂 Project Structure

```
├── kaggle_train.ipynb    # Main entry point (Jupyter Notebook)
├── voc_config.py         # Configuration settings (inside notebook)
├── voc_utils.py          # Logging, metrics, and helpers
├── voc_aug.py            # AdaptiveAugmentation class
├── voc_dataset.py        # VOCDataset and split management
├── main.py               # (Legacy) Single-file script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## 🛠️ Usage

### Local / Dedicated Server
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the legacy script (if desired):
    ```bash
    python main.py --epochs 15 --batch_size 2
    ```
    *OR* run the notebook `kaggle_train.ipynb`.

### Kaggle
1.  Upload `voc_utils.py`, `voc_aug.py`, and `voc_dataset.py` as a **Dataset** or **Utility Script**.
2.  Upload `kaggle_train.ipynb` as your main kernel.
3.  Add the **Pascal VOC 2012** dataset to your kernel.
4.  Run the notebook!

## 📊 Configuration
Adjust `VOCConfig` in `kaggle_train.ipynb`:
-   `data_root`: Path to your dataset.
-   `img_size`: Input image size (default 512).
-   `batch_size`: Adjust based on VRAM (2 for 3GB, 4+ for 8GB+).

## 📝 License
MIT License
