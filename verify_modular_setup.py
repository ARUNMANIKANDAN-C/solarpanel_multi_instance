
import os
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import logging
import albumentations as A

# Mocking the Notebook definitions
@dataclass
class VOCConfig:
    """Central configuration for VOC dataset."""
    data_root: str = "trainval" 
    img_size: int = 512
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    seed: int = 42
    num_workers: int = 0
    models_dir: Path = Path("models")
    
    def __post_init__(self):
        self.data_root = Path(self.data_root).resolve()
        self.images_dir = self.data_root / "JPEGImages"
        self.annotations_dir = self.data_root / "Annotations"
        self.splits_dir = self.data_root / "ImageSets" / "Main"
        self.models_dir.mkdir(exist_ok=True, parents=True)

# Imports from new modules
try:
    from voc_utils import setup_rich_logging, MetricLogger
    from voc_dataset import VOCSplitManager, VOCDataset
    from voc_aug import AdaptiveAugmentation
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

def test_modular_setup():
    logger = setup_rich_logging()
    
    config = VOCConfig()
    print("Config initialized.")
    
    # Initialize Dataset using the module
    print("Initializing VOCDataset from module...")
    # Triggering the dataset creation and stats calculation
    ds = VOCDataset(config, "trainval", class_list=["dummy"]) # fast init if mocks work, or it will scan
    
    print(f"Dataset init successful. Size: {len(ds)}")
    
    # Check Augmentation
    aug = AdaptiveAugmentation(config, {"dummy": 1.0})
    print("Augmentation initialized.")
    
    print("Modular setup verification PASSED.")

if __name__ == "__main__":
    test_modular_setup()
