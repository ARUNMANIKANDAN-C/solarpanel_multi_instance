
import os
import argparse
import torch
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from collections import Counter, defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import random
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Callable, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from tqdm import tqdm
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import pickle
import time
import math
import sys

import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Rich imports
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

# Suppress albumentations warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup Rich Console
console = Console()

# Setup Logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("KaggleTrainer")

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class VOCConfig:
    """Central configuration for VOC dataset."""
    data_root: str = "trainval" # Default relative path for Kaggle
    year: str = "2012"
    train_split: str = "trainval"
    val_split: str = "val"
    test_split: str = "test"
    img_size: int = 512
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    seed: int = 42
    num_workers: int = 2 # Reduced for stability in some envs
    pin_memory: bool = True
    debug: bool = False
    models_dir: Path = Path("models")
    
    def __post_init__(self):
        self.data_root = Path(self.data_root).resolve()
        self.images_dir = self.data_root / "JPEGImages"
        self.annotations_dir = self.data_root / "Annotations"
        self.splits_dir = self.data_root / "ImageSets" / "Main"
        self.models_dir.mkdir(exist_ok=True, parents=True)

# ==========================================
# TRAINING UTILITIES
# ==========================================

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values."""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = list()
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        if self.window_size > 0:
            if len(self.deque) > self.window_size:
                self.deque.pop(0)

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'MetricLogger' object has no attribute '{}'".format(attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

def reduce_dict(input_dict, average=True):
    return input_dict

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# AUGMENTATION
# ==========================================

class AdaptiveAugmentation:
    """
    Applies stronger augmentation to images containing minority classes.
    Includes "Crop Cut" (CoarseDropout) for minority classes.
    """
    
    def __init__(self, config: VOCConfig, class_weights: Dict[str, float]):
        self.config = config
        self.class_weights = class_weights
        self.logger = logging.getLogger(__name__)
        
        self.minority_classes = [
            cls for cls, weight in class_weights.items() 
            if weight > 1.2 and cls in class_weights
        ]
        
        self.majority_classes = [
            cls for cls, weight in class_weights.items() 
            if weight < 0.8 and cls in class_weights
        ]
        
    def get_transforms(self, image_id: str, class_distribution: List[str]) -> A.Compose:
        contains_minority = any(cls in self.minority_classes for cls in class_distribution)
        contains_majority = any(cls in self.majority_classes for cls in class_distribution)
        
        # Base transforms
        transforms = [
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(
                    height=self.config.img_size, 
                    width=self.config.img_size, 
                    p=0.5 if contains_minority else 0.3
                ),
                A.Resize(height=self.config.img_size, width=self.config.img_size)
            ], p=1.0),
            A.Resize(height=self.config.img_size, width=self.config.img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
        ]
        
        if contains_minority:
            # STRONG augmentation with "Crop Cut" (CoarseDropout)
            transforms.extend([
                A.VerticalFlip(p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0, 
                    p=0.5 # 50% chance for "Crop Cut" on minority images
                ),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.CLAHE(p=0.3),
            ])
        elif contains_majority:
            # LIGHT augmentation
            transforms.extend([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ])
        else:
            # MEDIUM augmentation
            transforms.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            ])
        
        # Always normalize
        transforms.extend([
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ])
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_visibility=0.3,
                label_fields=['labels']
            )
        )

# ==========================================
# DATASET
# ==========================================

class VOCDataset(Dataset):
    def __init__(
        self,
        config: VOCConfig,
        image_set: str,
        image_ids: Optional[List[str]] = None,
        transforms: Optional[Union[A.Compose, AdaptiveAugmentation]] = None,
        class_list: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.image_set = image_set
        self.transforms = transforms
        self.logger = logger or logging.getLogger(__name__)
        
        # Load image IDs
        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = self._load_image_ids()
        
        # Setup classes
        if class_list:
            self.class_list = class_list
        else:
            self.class_list = self._determine_classes()
        
        self.class_map = {name: idx for idx, name in enumerate(self.class_list, 1)}
        self.inv_class_map = {v: k for k, v in self.class_map.items()}
        
        # Filter and stats
        self.image_ids, self.class_distribution = self._filter_and_stats()
        self.class_weights = self._calculate_class_weights()
        self.minority_classes = [cls for cls, w in self.class_weights.items() if w > 1.2]
        self.majority_classes = [cls for cls, w in self.class_weights.items() if w < 0.8]
        
        if isinstance(transforms, AdaptiveAugmentation):
            self.adaptive_aug = transforms
            self.regular_transforms = None
        else:
            self.adaptive_aug = None
            self.regular_transforms = transforms

        self._log_init()

    def _log_init(self):
        self.logger.info(f"Initialized {self.image_set} set with {len(self)} images")
        
        if self.class_distribution:
            # Create a Rich Table for distribution
            table = Table(title=f"Class Distribution ({self.image_set})", show_header=True, header_style="bold magenta")
            table.add_column("Class", style="cyan")
            table.add_column("Count", justify="right", style="green")
            table.add_column("Percentage", justify="right", style="yellow")
            
            sorted_classes = sorted(self.class_distribution.items(), key=lambda x: x[1], reverse=True)
            total = sum(self.class_distribution.values())
            
            for cls, count in sorted_classes:
                percentage = (count / total) * 100 if total > 0 else 0
                table.add_row(cls, str(count), f"{percentage:.1f}%")
                
            # Print table to console directly since logger might mangle table structure
            console.print(table)
            # Also log a summary line for file logs
            self.logger.info(f"Top 3 classes: {', '.join([f'{c}: {n}' for c, n in sorted_classes[:3]])}")

    def _format_class_distribution(self) -> str:
        # Kept for compatibility if needed elsewhere, but mainly replaced by _log_init table
        if not self.class_distribution:
            return "  No classes found"
        sorted_classes = sorted(self.class_distribution.items(), key=lambda x: x[1], reverse=True)
        total = sum(self.class_distribution.values())
        return '\n'.join([f"  {cls}: {count} ({count/total*100:.1f}%)" for cls, count in sorted_classes])

    def _load_image_ids(self) -> List[str]:
        split_file = self.config.splits_dir / f"{self.image_set}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                return [line.strip().split()[0] for line in f if line.strip()]
        else:
            image_files = sorted(self.config.images_dir.glob("*.jpg"))
            return [f.stem for f in image_files]
    
    def _determine_classes(self) -> List[str]:
        classes = set()
        sample_size = min(500, len(self.image_ids))
        sample_ids = random.sample(self.image_ids, sample_size)
        for img_id in sample_ids:
            try:
                xml_path = self.config.annotations_dir / f"{img_id}.xml"
                if xml_path.exists():
                    tree = ET.parse(xml_path)
                    for obj in tree.findall("object"):
                        classes.add(obj.find("name").text)
            except: continue
        return sorted(classes)
    
    def _filter_and_stats(self) -> Tuple[List[str], Dict[str, int]]:
        valid_ids = []
        class_counter = Counter()
        for img_id in self.image_ids:
            try:
                boxes, labels, label_names, _ = self._parse_annotation(img_id)
                if len(boxes) > 0:
                    for name in label_names:
                        if name in self.class_map:
                            class_counter[name] += 1
                    valid_ids.append(img_id)
            except: continue
        return valid_ids, dict(class_counter)
    
    def _calculate_class_weights(self) -> Dict[str, float]:
        if not self.class_distribution: return {}
        total = sum(self.class_distribution.values())
        num_classes = len(self.class_distribution)
        return {cls: total / (count * num_classes) if count > 0 else 0 for cls, count in self.class_distribution.items()}
    
    def _parse_annotation(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, List[str], Tuple[float, float]]:
        xml_path = self.config.annotations_dir / f"{image_id}.xml"
        if not xml_path.exists(): raise FileNotFoundError(f"Annotation not found: {xml_path}")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            
            boxes = []
            label_names = []
            for obj in root.findall("object"):
                if obj.find("name").text not in self.class_map: continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                xmin = max(0, min(xmin, width))
                ymin = max(0, min(ymin, height))
                xmax = max(xmin, min(xmax, width))
                ymax = max(ymin, min(ymax, height))
                
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    label_names.append(obj.find("name").text)
            
            boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            labels = np.array([self.class_map[name] for name in label_names], dtype=np.int64) if label_names else np.zeros(0, dtype=np.int64)
            return boxes, labels, label_names, (width, height)
        except: raise ValueError(f"Error parsing {xml_path}")

    def __len__(self): return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.config.images_dir / f"{image_id}.jpg"
        
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            boxes, labels, label_names, _ = self._parse_annotation(image_id)
            
            if self.adaptive_aug:
                transforms = self.adaptive_aug.get_transforms(image_id, label_names)
            else:
                transforms = self.regular_transforms
            
            if transforms and len(boxes) > 0:
                transformed = transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed["image"]
                boxes = np.array(transformed["bboxes"]) if transformed["bboxes"] else np.zeros((0, 4))
                labels = np.array(transformed["labels"]) if transformed["labels"] else np.zeros(0)
            elif not isinstance(image, torch.Tensor): # Handling for no transforms case or empty boxes
                 image = ToTensorV2()(image=image)["image"]
                 image = A.Normalize(mean=self.config.mean, std=self.config.std)(image=image.permute(1,2,0).numpy())["image"]
                 image = torch.from_numpy(image).permute(2,0,1)

        except Exception as e:
            # Fallback
            return self.__getitem__((idx + 1) % len(self))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        return image, target

class VOCSplitManager:
    def __init__(self, config: VOCConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, use_adaptive_aug=True):
        full_dataset = VOCDataset(self.config, "trainval")
        all_ids = full_dataset.image_ids
        # Simple random split for demo purposes if files don't exist
        random.shuffle(all_ids)
        n = len(all_ids)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_ids = all_ids[:train_end]
        val_ids = all_ids[train_end:val_end]
        test_ids = all_ids[val_end:]
        
        # Create Datasets
        train_dataset = VOCDataset(self.config, "train", image_ids=train_ids, class_list=full_dataset.class_list)
        if use_adaptive_aug:
            train_dataset.transforms = AdaptiveAugmentation(self.config, train_dataset.class_weights)
            train_dataset.adaptive_aug = train_dataset.transforms
            
        val_dataset = VOCDataset(
            self.config, "val", image_ids=val_ids, class_list=full_dataset.class_list,
            transforms=A.Compose([
                A.Resize(self.config.img_size, self.config.img_size),
                A.Normalize(mean=self.config.mean, std=self.config.std),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        )
        
        return {'train': train_dataset, 'val': val_dataset}

# ==========================================
# TRAINING LOOPS
# ==========================================

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    all_losses = []
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[bold blue]{task.fields[loss]}"),
    )
    task_id = progress.add_task(f"[cyan]Epoch {epoch} Training", total=len(data_loader), loss="Loss: 0.0000")
    
    with progress:
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not math.isfinite(losses.item()):
                logger.error(f"Loss is {losses.item()}, skipping batch")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            all_losses.append(losses.item())
            progress.update(task_id, advance=1, loss=f"Loss: {losses.item():.4f}")

    return np.mean(all_losses)

def evaluate(model, data_loader, device):
    model.train() # Keep in train mode for loss
    losses = []
    progress = Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), MofNCompleteColumn()
    )
    task_id = progress.add_task("[green]Validating", total=len(data_loader))
    
    with progress:
        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]
                loss_dict = model(images, targets)
                losses.append(sum(l.item() for l in loss_dict.values()))
                progress.update(task_id, advance=1)
    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=15, type=int) 
    parser.add_argument('--batch_size', default=2, type=int, help="Batch size (2 recommended for 3GB VRAM)")
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold magenta]PVELAD Kaggle Pipeline[/bold magenta]", border_style="magenta"))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    console.print(f"[bold green]Using device:[/bold green] {device}")
    
    config = VOCConfig()
    split_manager = VOCSplitManager(config, logger)
    datasets = split_manager.create_splits()
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    
    # Balancing
    class_counts = train_dataset.class_distribution
    total = sum(class_counts.values())
    w = {k: total/(v+1e-6) for k,v in class_counts.items()}
    sample_weights = []
    for i in range(len(train_dataset)):
         # Approximate weight lookup for speed in this monolithic script
         # In real run, load target. Here we accept overhead or pre-calculate
         _, target = train_dataset._load_sample_no_mixup(i) if hasattr(train_dataset, '_load_sample_no_mixup') else train_dataset[i]
         labels = target['labels']
         if len(labels)>0:
             weight = max([w.get(train_dataset.inv_class_map.get(int(l),''), 0) for l in labels])
         else: weight = 0
         sample_weights.append(weight)
         
    num_classes = len(train_dataset.class_list) + 1
    train_sampler = WeightedRandomSampler(sample_weights, 1000, replacement=True) # Reduced samples for quick test
    
    train_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, args.batch_size, collate_fn=collate_fn, num_workers=0)
    
    model = get_model(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        console.print(f"[bold]Epoch {epoch}[/bold] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

if __name__ == "__main__":
    main()
