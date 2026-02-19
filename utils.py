import os
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
import time

# Suppress albumentations warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION AND LOGGING
# ==========================================

@dataclass
class VOCConfig:
    """Central configuration for VOC dataset."""
    data_root: str
    year: str = "2012"
    train_split: str = "trainval"
    val_split: str = "val"
    test_split: str = "test"
    img_size: int = 512
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    debug: bool = False
    
    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.images_dir = self.data_root / "JPEGImages"
        self.annotations_dir = self.data_root / "Annotations"
        self.splits_dir = self.data_root / "ImageSets" / "Main"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_dir.exists():
            warnings.warn(f"Annotations directory not found: {self.annotations_dir}")

def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup professional logging."""
    logger = logging.getLogger('VOCDataset')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        if debug:
            log_file = f"voc_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

# ==========================================
# ADAPTIVE AUGMENTATION (FIXED)
# ==========================================

class AdaptiveAugmentation:
    """
    Applies stronger augmentation to images containing minority classes.
    Fixed version with proper albumentations parameters.
    """
    
    def __init__(self, config: VOCConfig, class_weights: Dict[str, float]):
        self.config = config
        self.class_weights = class_weights
        self.logger = logging.getLogger(__name__)
        
        # Define minority classes (weight > 1.2) - these need strong augmentation
        self.minority_classes = [
            cls for cls, weight in class_weights.items() 
            if weight > 1.2 and cls in class_weights
        ]
        
        # Define majority classes (weight < 0.8) - these need light augmentation
        self.majority_classes = [
            cls for cls, weight in class_weights.items() 
            if weight < 0.8 and cls in class_weights
        ]
        
        self.logger.info(f"🔴 Minority classes (need strong aug): {self.minority_classes}")
        self.logger.info(f"🔵 Majority classes (need light aug): {self.majority_classes}")
    
    def get_transforms(self, image_id: str, class_distribution: List[str]) -> A.Compose:
        """
        Get adaptive transforms based on classes present in the image.
        """
        # Check if image contains minority classes
        contains_minority = any(cls in self.minority_classes for cls in class_distribution)
        contains_majority = any(cls in self.majority_classes for cls in class_distribution)
        
        # Calculate augmentation intensity
        if contains_minority:
            # Find the rarest class in this image
            rare_class_weight = max(
                [self.class_weights.get(cls, 1.0) for cls in class_distribution if cls in self.minority_classes]
            )
            intensity = min(rare_class_weight, 3.0)
            aug_level = "STRONG"
        elif contains_majority:
            intensity = 0.5
            aug_level = "LIGHT"
        else:
            intensity = 1.0
            aug_level = "MEDIUM"
        
        if self.config.debug:
            self.logger.debug(f"Image {image_id}: {aug_level} augmentation (intensity={intensity:.2f})")
        
        # Base transforms (always applied)
        transforms = [
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(
                    height=self.config.img_size, 
                    width=self.config.img_size, 
                    p=0.5 if contains_minority else 0.3
                ),
                A.Resize(height=self.config.img_size, width=self.config.img_size)
            ], p=1.0),
            # Mandatory resize to ensure output is always 512x512 even if SafeCrop fails
            A.Resize(height=self.config.img_size, width=self.config.img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
        ]
        
        if contains_minority:
            # STRONG augmentation for minority classes
            # STRONG augmentation for minority classes
            transforms.extend([
                A.VerticalFlip(p=0.3),
                # Rotation removed as per user request
            ])
        
        # Color/Noise augmentation (using fixed parameters to avoid warnings)
        if contains_minority:
            transforms.extend([
                A.ColorJitter(
                    brightness=0.3,  # Fixed instead of scaled
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=0.8
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Fixed tuple
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),  # Fixed odd numbers
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                A.CLAHE(p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
            ])
        elif contains_majority:
            # LIGHT augmentation for majority classes
            transforms.extend([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ])
        else:
            # MEDIUM augmentation for balanced classes
            transforms.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ])
        
        # Advanced augmentations for extremely rare classes (weight > 2.0)
        if contains_minority and any(self.class_weights.get(cls, 1.0) > 2.0 for cls in class_distribution):
            transforms.extend([
                # CoarseDropout removed due to "apply_to_bbox not implemented" warning
                A.ISONoise(color_shift=(0.01, 0.03), p=0.1),
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
# ENHANCED AUGMENTATION ANALYZER
# ==========================================

class AugmentationAnalyzer:
    """Analyzes the effect of augmentations on different classes."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_augmentations(self, dataset: VOCDataset, num_samples: int = 1000):
        """Analyze how augmentations affect different classes."""
        self.logger.info("Analyzing augmentation effects...")
        
        stats = {
            'pre_aug': defaultdict(int),
            'post_aug': defaultdict(int),
            'box_changes': [],
            'class_preservation': defaultdict(lambda: {'kept': 0, 'lost': 0, 'gained': 0}),
            'images_with_minority': 0,
            'images_with_majority': 0
        }
        
        samples = min(num_samples, len(dataset))
        
        for i in tqdm(range(samples), desc="Analyzing augmentations"):
            try:
                _, target = dataset[i]
                
                # Get classes
                pre_classes = [dataset.inv_class_map[int(l)] for l in target['orig_labels'] if int(l) in dataset.inv_class_map]
                post_classes = [dataset.inv_class_map[int(l)] for l in target['labels'] if int(l) in dataset.inv_class_map]
                
                # Track minority/majority images
                if any(cls in dataset.minority_classes for cls in pre_classes):
                    stats['images_with_minority'] += 1
                if any(cls in dataset.majority_classes for cls in pre_classes):
                    stats['images_with_majority'] += 1
                
                # Count pre-augmentation boxes
                for cls in pre_classes:
                    stats['pre_aug'][cls] += 1
                
                # Count post-augmentation boxes
                for cls in post_classes:
                    stats['post_aug'][cls] += 1
                
                # Track class preservation
                pre_set = set(pre_classes)
                post_set = set(post_classes)
                
                # Calculate preservation rate per class
                for cls in pre_set:
                    if cls in post_set:
                        stats['class_preservation'][cls]['kept'] += 1
                    else:
                        stats['class_preservation'][cls]['lost'] += 1
                
                for cls in post_set:
                    if cls not in pre_set:
                        stats['class_preservation'][cls]['gained'] += 1
                
                stats['box_changes'].append(len(post_classes) - len(pre_classes))
                
            except Exception as e:
                self.logger.warning(f"Error analyzing index {i}: {e}")
        
        return stats
    
    def print_augmentation_report(self, stats: Dict, dataset: VOCDataset):
        """Print augmentation analysis report."""
        print(f"\n{'='*70}")
        print("AUGMENTATION EFFECT ANALYSIS".center(70))
        print(f"{'='*70}")
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Samples Analyzed: {len(stats['box_changes']):,}")
        print(f"  Images with Minority Classes: {stats['images_with_minority']} ({stats['images_with_minority']/len(stats['box_changes'])*100:.1f}%)")
        print(f"  Images with Majority Classes: {stats['images_with_majority']} ({stats['images_with_majority']/len(stats['box_changes'])*100:.1f}%)")
        
        if stats['box_changes']:
            avg_change = np.mean(stats['box_changes'])
            std_change = np.std(stats['box_changes'])
            print(f"  Avg Box Change: {avg_change:.2f} (±{std_change:.2f})")
        
        print(f"\n📦 Class-wise Augmentation Effects:")
        print(f"{'Class':25} {'Pre-Aug':>10} {'Post-Aug':>10} {'Change':>15} {'Preserved':>12}")
        print("-" * 75)
        
        # Get minority/majority classes from dataset
        minority_classes = getattr(dataset, 'minority_classes', [])
        majority_classes = getattr(dataset, 'majority_classes', [])
        
        all_classes = sorted(set(stats['pre_aug'].keys()) | set(stats['post_aug'].keys()))
        
        for cls in all_classes:
            pre = stats['pre_aug'][cls]
            post = stats['post_aug'][cls]
            change = post - pre
            change_pct = (change / pre * 100) if pre > 0 else 0
            
            kept = stats['class_preservation'][cls]['kept']
            preserved_pct = (kept / pre * 100) if pre > 0 else 0
            
            # Color coding
            if cls in minority_classes:
                marker = "🔴"  # Red for minority
            elif cls in majority_classes:
                marker = "🔵"  # Blue for majority
            else:
                marker = "⚪"  # White for balanced
            
            # Format change with sign
            change_str = f"{change:+d} ({change_pct:+.1f}%)"
            
            print(f"{marker} {cls:23} {pre:10d} {post:10d} {change_str:>15} {preserved_pct:6.1f}%")
        
        print(f"\n{'='*70}\n")

# ==========================================
# VOC DATASET (with adaptive augmentation support)
# ==========================================

class VOCDataset(Dataset):
    """
    VOC Dataset with adaptive augmentation.
    """
    
    def __init__(
        self,
        config: VOCConfig,
        image_set: str,
        image_ids: Optional[List[str]] = None,
        transforms: Optional[Union[A.Compose, AdaptiveAugmentation]] = None,
        class_list: Optional[List[str]] = None,
        filter_empty: bool = True,
        use_difficult: bool = False,
        cache_annotations: bool = True,
        min_area: float = 0.0,
        max_samples: Optional[int] = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.image_set = image_set
        self.transforms = transforms
        self.filter_empty = filter_empty
        self.use_difficult = use_difficult
        self.min_area = min_area
        self.max_samples = max_samples
        self.cache_annotations = cache_annotations
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.config = config
        self.image_set = image_set
        self.transforms = transforms
        self.filter_empty = filter_empty
        self.use_difficult = use_difficult
        self.min_area = min_area
        self.max_samples = max_samples
        self.cache_annotations = cache_annotations
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize cache
        self._annotation_cache = {}
        
        # Load image IDs
        if image_ids is not None:
            self.image_ids = image_ids
            self.logger.info(f"Using provided {len(self.image_ids)} image IDs for {image_set}")
        else:
            self.image_ids = self._load_image_ids()
        
        # Setup classes
        if class_list:
            self.class_list = class_list
        else:
            self.class_list = self._determine_classes()
        
        self.class_map = {name: idx for idx, name in enumerate(self.class_list, 1)}
        self.inv_class_map = {v: k for k, v in self.class_map.items()}
        
        # Filter images and collect statistics
        self.image_ids, self.invalid_count, self.class_distribution = self._filter_and_stats()
        
        # Apply max_samples
        if max_samples and max_samples < len(self.image_ids):
            self.image_ids = self.image_ids[:max_samples]
            self.logger.info(f"Limited to {max_samples} samples")
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
        
        # Define minority/majority classes
        self.minority_classes = [cls for cls, w in self.class_weights.items() if w > 1.2]
        self.majority_classes = [cls for cls, w in self.class_weights.items() if w < 0.8]
        
        # Initialize adaptive augmentation
        if image_set == 'split_train' and isinstance(transforms, AdaptiveAugmentation):
            self.adaptive_aug = transforms
            self.regular_transforms = None
        else:
            self.adaptive_aug = None
            self.regular_transforms = transforms
        
        # Log initialization
        self._log_init()
    
    def _load_image_ids(self) -> List[str]:
        """Load image IDs from split file or directory."""
        split_file = self.config.splits_dir / f"{self.image_set}.txt"
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                image_ids = [line.strip().split()[0] for line in f if line.strip()]
            return image_ids
        else:
            image_files = sorted(self.config.images_dir.glob("*.jpg"))
            return [f.stem for f in image_files]
    
    def _determine_classes(self) -> List[str]:
        """Determine all classes present in the dataset."""
        classes = set()
        sample_size = min(1000, len(self.image_ids))
        sample_ids = random.sample(self.image_ids, sample_size) if len(self.image_ids) > sample_size else self.image_ids
        
        for img_id in tqdm(sample_ids, desc="Discovering classes", disable=not self.config.debug):
            try:
                xml_path = self.config.annotations_dir / f"{img_id}.xml"
                if xml_path.exists():
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        classes.add(obj.find("name").text)
            except:
                continue
        
        return sorted(classes)
    
    def _filter_and_stats(self) -> Tuple[List[str], int, Dict[str, int]]:
        """Filter images and collect class statistics."""
        valid_ids = []
        invalid_count = 0
        missing_annotation_count = 0
        class_counter = Counter()
        
        desc = f"Processing {self.image_set} images"
        for img_id in tqdm(self.image_ids, desc=desc, disable=not self.config.debug):
            try:
                boxes, labels, label_names, img_size = self._parse_annotation(img_id)
                
                for name in label_names:
                    if name in self.class_map:
                        class_counter[name] += 1
                
                if self.filter_empty and len(boxes) == 0:
                    invalid_count += 1
                    continue
                
                valid_ids.append(img_id)
                
                if self.config.debug and len(valid_ids) % 500 == 0:
                    self.logger.debug(f"Processed {len(valid_ids)} valid images...")
                    
            except FileNotFoundError:
                # Silent failure for missing annotations, just count them
                missing_annotation_count += 1
                invalid_count += 1
            except Exception as e:
                # Check for "Annotation not found" in string just in case
                if "Annotation not found" in str(e):
                    missing_annotation_count += 1
                else:
                    self.logger.warning(f"Error processing {img_id}: {e}")
                invalid_count += 1
        
        self.logger.info(f"Summary for {self.image_set}: Found {len(valid_ids)} valid images. {missing_annotation_count} annotations not found.")
        return valid_ids, invalid_count, dict(class_counter)
    
    def _calculate_class_weights(self) -> Dict[str, float]:
        """Calculate class weights for adaptive augmentation."""
        if not self.class_distribution:
            return {}
            
        total = sum(self.class_distribution.values())
        num_classes = len(self.class_distribution)
        
        weights = {}
        for cls, count in self.class_distribution.items():
            weight = total / (count * num_classes) if count > 0 else 0
            weights[cls] = weight
        
        # No clamping - use raw inverse frequency for aggressive balancing
        # This ensures rare classes (e.g. 0.1%) get 300x weight compared to common classes (30%)

        
        return weights
    
    def _parse_annotation(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, List[str], Tuple[float, float]]:
        """Parse VOC annotation with caching."""
        if self.cache_annotations and image_id in self._annotation_cache:
            return self._annotation_cache[image_id]
        
        xml_path = self.config.annotations_dir / f"{image_id}.xml"
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Annotation not found: {xml_path}")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find("size")
            if size is not None:
                width = float(size.find("width").text)
                height = float(size.find("height").text)
            else:
                img_path = self.config.images_dir / f"{image_id}.jpg"
                if img_path.exists():
                    with Image.open(img_path) as img:
                        width, height = img.size
                else:
                    width, height = 0, 0
            
            img_size = (width, height)
            
            boxes = []
            label_names = []
            
            for obj in root.findall("object"):
                difficult = obj.find("difficult")
                if not self.use_difficult and difficult is not None and difficult.text == "1":
                    continue
                
                label_name = obj.find("name").text
                
                if label_name not in self.class_map:
                    continue
                
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                if width > 0 and height > 0:
                    xmin = max(0, min(xmin, width))
                    ymin = max(0, min(ymin, height))
                    xmax = max(xmin, min(xmax, width))
                    ymax = max(ymin, min(ymax, height))
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                boxes.append([xmin, ymin, xmax, ymax])
                label_names.append(label_name)
            
            boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            labels = np.array([self.class_map[name] for name in label_names], dtype=np.int64) if label_names else np.zeros(0, dtype=np.int64)
            
            result = (boxes, labels, label_names, img_size)
            
            if self.cache_annotations:
                self._annotation_cache[image_id] = result
            
            return result
            
        except ET.ParseError as e:
            raise ValueError(f"XML parsing error for {xml_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_sample_no_mixup(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Helper to load a sample without mixup."""
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = self.config.images_dir / f"{image_id}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            self.logger.error(f"Error loading image {img_path}: {e}")
            # Return a random valid image instead of failing
            return self._load_sample_no_mixup((idx + 1) % len(self))
        
        # Parse annotation
        boxes, labels, label_names, (img_w, img_h) = self._parse_annotation(image_id)
        
        # Store original
        orig_boxes = boxes.copy() if len(boxes) > 0 else boxes
        orig_labels = labels.copy() if len(labels) > 0 else labels
        
        # Apply adaptive augmentation
        if self.adaptive_aug is not None and len(boxes) > 0:
            try:
                transforms = self.adaptive_aug.get_transforms(image_id, label_names)
                
                transformed = transforms(
                    image=image,
                    bboxes=boxes.tolist() if len(boxes) > 0 else [],
                    labels=labels.tolist() if len(labels) > 0 else []
                )
                
                image = transformed["image"]
                boxes = np.array(transformed["bboxes"]) if transformed["bboxes"] else np.zeros((0, 4))
                labels = np.array(transformed["labels"]) if transformed["labels"] else np.zeros(0)
                
            except Exception as e:
                self.logger.warning(f"Transform failed for {image_id}: {e}")
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                boxes = orig_boxes
                labels = orig_labels
        elif self.regular_transforms and len(boxes) > 0:
            try:
                transformed = self.regular_transforms(
                    image=image,
                    bboxes=boxes.tolist() if len(boxes) > 0 else [],
                    labels=labels.tolist() if len(labels) > 0 else []
                )
                
                image = transformed["image"]
                boxes = np.array(transformed["bboxes"]) if transformed["bboxes"] else np.zeros((0, 4))
                labels = np.array(transformed["labels"]) if transformed["labels"] else np.zeros(0)
                
            except Exception as e:
                self.logger.warning(f"Transform failed for {image_id}: {e}")
                # Manual resize to ensure correct shape for mixup
                img_pil = Image.fromarray(image).resize((self.config.img_size, self.config.img_size))
                image = np.array(img_pil)
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                boxes = orig_boxes
                labels = orig_labels
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Ensure correct shape
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        if image.dim() == 3 and image.shape[2] in [1, 3, 4]:
            image = image.permute(2, 0, 1)
            
        # Prepare target
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(
                [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], 
                dtype=torch.float32
            ) if len(boxes) > 0 else torch.zeros(0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "orig_boxes": torch.as_tensor(orig_boxes, dtype=torch.float32),
            "orig_labels": torch.as_tensor(orig_labels, dtype=torch.int64),
            "image_id_str": image_id,
            "img_size": torch.tensor([img_h, img_w], dtype=torch.int64),
            "class_distribution": label_names,
            "mixup": False
        }
        
        return image, target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get item with adaptive augmentation and Mixup."""
        image, target = self._load_sample_no_mixup(idx)
        
        # Apply Mixup if enabled and random chance
        if self.use_mixup and random.random() < 0.5:
            idx2 = random.randint(0, len(self) - 1)
            image2, target2 = self._load_sample_no_mixup(idx2)
            
            # Mixup parameters
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Blend images
            image = image * lam + image2 * (1 - lam)
            
            # Combine boxes and labels
            # For detection, we typically union the boxes
            target["boxes"] = torch.cat([target["boxes"], target2["boxes"]], dim=0)
            target["labels"] = torch.cat([target["labels"], target2["labels"]], dim=0)
            target["area"] = torch.cat([target["area"], target2["area"]], dim=0)
            target["iscrowd"] = torch.cat([target["iscrowd"], target2["iscrowd"]], dim=0)
            target["mixup_lam"] = lam
            target["mixup"] = True
            
        return image, target
    
    def _log_init(self):
        """Log dataset initialization details."""
        self.logger.info(f"""
        {'='*50}
        VOC Dataset Initialized
        {'='*50}
        Set: {self.image_set}
        Images: {len(self)}
        Classes: {len(self.class_list)}
        Invalid filtered: {self.invalid_count}
        
        Class Distribution:
        {self._format_class_distribution()}
        
        Class Weights:
        {self._format_class_weights()}
        
        Minority Classes (🔴): {self.minority_classes}
        Majority Classes (🔵): {self.majority_classes}
        {'='*50}
        """)
    
    def _format_class_distribution(self) -> str:
        """Format class distribution for logging."""
        if not self.class_distribution:
            return "  No classes found"
        
        sorted_classes = sorted(
            self.class_distribution.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return '\n'.join([f"  {cls}: {count}" for cls, count in sorted_classes])
    
    def _format_class_weights(self) -> str:
        """Format class weights for logging."""
        if not self.class_weights:
            return "  No weights calculated"
        
        sorted_weights = sorted(
            self.class_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return '\n'.join([f"  {cls}: {weight:.3f}" for cls, weight in sorted_weights])

# ==========================================
# SIMPLE ANALYZER
# ==========================================

class VOCAnalyzer:
    """Simple analysis tools."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_dataset(self, dataset: VOCDataset, num_samples: Optional[int] = None) -> Dict:
        """Comprehensive dataset analysis."""
        self.logger.info(f"Analyzing dataset: {dataset.image_set} with {len(dataset)} images")
        
        stats = {
            'total_images': len(dataset),
            'class_counts': defaultdict(int),
            'total_boxes': 0
        }
        
        samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        
        for i in tqdm(range(samples), desc=f"Analyzing {dataset.image_set}"):
            try:
                _, target = dataset[i]
                
                for label in target['labels']:
                    if int(label) in dataset.inv_class_map:
                        cls = dataset.inv_class_map[int(label)]
                        stats['class_counts'][cls] += 1
                        stats['total_boxes'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing index {i}: {e}")
        
        return stats
    
    def print_report(self, stats: Dict, dataset_name: str = "Dataset"):
        """Print comprehensive report."""
        print(f"\n{'='*60}")
        print(f"{dataset_name} ANALYSIS REPORT".center(60))
        print(f"{'='*60}")
        
        print(f"\n📊 Statistics:")
        print(f"  Total Images: {stats['total_images']:,}")
        print(f"  Total Boxes: {stats['total_boxes']:,}")
        print(f"  Avg Boxes/Image: {stats['total_boxes']/stats['total_images']:.2f}")
        
        print(f"\n📦 Class Distribution:")
        if stats['class_counts']:
            for cls, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_boxes']) * 100
                bar = '█' * int(percentage / 2)
                print(f"  {cls:25} {count:6,d} ({percentage:5.2f}%) {bar}")
        else:
            print("  No boxes found")
        
        print(f"\n{'='*60}\n")
        
    def analyze_loader_stats(self, loader: DataLoader, dataset: VOCDataset, num_batches: int = 50):
        """Analyze class distribution in the data loader (after sampling/balancing)."""
        self.logger.info(f"Analyzing effective distribution in DataLoader ({num_batches} batches)...")
        
        counts = defaultdict(int)
        total_boxes = 0
        
        for i, (images, targets) in tqdm(enumerate(loader), total=num_batches, desc="Scanning Loader"):
            if i >= num_batches:
                break
                
            for target in targets:
                for label in target['labels']:
                    if int(label) in dataset.inv_class_map:
                        cls = dataset.inv_class_map[int(label)]
                        counts[cls] += 1
                        total_boxes += 1
                        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("EFFECTIVE CLASS DISTRIBUTION (AFTER BALANCING)".center(60))
        self.logger.info(f"{'='*60}")
        
        # Determine original distribution for comparison
        orig_counts = getattr(dataset, 'class_distribution', {})
        if not orig_counts:
             # Fallback if not stored
             orig_counts = defaultdict(int) 
             
        total_orig = sum(orig_counts.values()) if orig_counts else 1
        
        self.logger.info(f"{'Class':20} {'Original %':>12} {'Balanced %':>12} {'Change factor':>15}")
        self.logger.info("-" * 65)
        
        sorted_classes = sorted(counts.keys())
        for cls in sorted_classes:
            count = counts[cls]
            pct = (count / total_boxes * 100) if total_boxes > 0 else 0
            
            orig_count = orig_counts.get(cls, 0)
            orig_pct = (orig_count / total_orig * 100) if total_orig > 0 else 0
            
            ratio = (pct / orig_pct) if orig_pct > 0 else 0.0
            
            self.logger.info(f"{cls:20} {orig_pct:11.2f}% {pct:11.2f}% {ratio:14.2f}x")
            
        self.logger.info(f"{'='*60}\n")
    
    def save_augmented_visualizations(self, loader: DataLoader, dataset: VOCDataset, num_images: int = 30, save_path: str = "augmented_samples.png"):
        """Save a grid of augmented samples."""
        self.logger.info(f"Saving {num_images} augmented samples to {save_path}...")
        
        images_collected = []
        targets_collected = []
        titles = []
        
        # Collect images
        iterator = iter(loader)
        while len(images_collected) < num_images:
            try:
                batch_images, batch_targets = next(iterator)
                for i in range(len(batch_images)):
                    if len(images_collected) >= num_images:
                        break
                        
                    img = batch_images[i]
                    target = batch_targets[i]
                    
                    # Unnormalize
                    if isinstance(img, torch.Tensor):
                        img = img.permute(1, 2, 0).numpy()
                        mean = np.array(dataset.config.mean)
                        std = np.array(dataset.config.std)
                        img = (img * std + mean).clip(0, 1)
                        

                    images_collected.append(img)
                    targets_collected.append(target)
                    
                    # Title (Mixup or classes)
                    if target.get("mixup", False):
                        titles.append(f"Mixup (lam={target.get('mixup_lam', 0):.2f})")
                    else:
                        classes = [dataset.inv_class_map[int(l)] for l in target['labels'] if int(l) in dataset.inv_class_map]
                        # Truncate if too long
                        title = ", ".join(classes)
                        if len(title) > 20: title = title[:17] + "..."
                        titles.append(title)
                        
            except StopIteration:
                break
        
        # Create Grid
        cols = 5
        rows = (len(images_collected) + cols - 1) // cols
        
        plt.figure(figsize=(cols * 4, rows * 4))
        
        for i, img in enumerate(images_collected):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            
            # Draw bounding boxes
            if i < len(targets_collected):
                target = targets_collected[i]
                if "boxes" in target:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    
                    labels = target.get("labels", [])
                    if isinstance(labels, torch.Tensor):
                        labels = labels.cpu().numpy()
                        
                    for j, box in enumerate(boxes):
                        xmin, ymin, xmax, ymax = box
                        w, h = xmax - xmin, ymax - ymin
                        
                        # Get class color (simple hash based)
                        cls_id = int(labels[j]) if len(labels) > j else 0
                        color = plt.cm.tab10(cls_id % 10)
                        
                        rect = Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)
                        
                        # Add class name text
                        class_name = dataset.inv_class_map.get(cls_id, str(cls_id))
                        ax.text(xmin, ymin - 2, class_name, fontsize=6, color='white', 
                                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))
            
            ax.set_title(titles[i], fontsize=8)
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        self.logger.info(f"Saved visualization to {save_path}")

# ==========================================
# SPLIT MANAGER
# ==========================================

class VOCSplitManager:
    """Split manager with adaptive augmentation support."""
    
    def __init__(
        self,
        config: VOCConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.cache_dir = Path("./voc_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        force_recreate: bool = False,
        use_adaptive_aug: bool = True
    ) -> Dict[str, VOCDataset]:
        """
        Create random splits with optional adaptive augmentation.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        cache_file = self.cache_dir / f"splits_adaptive_{self.config.year}_{hashlib.md5(str(self.config.data_root).encode()).hexdigest()}.pkl"
        
        if cache_file.exists() and not force_recreate:
            self.logger.info(f"Loading cached splits from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Get all image IDs
        self.logger.info("Loading all image IDs...")
        all_images = sorted(self.config.images_dir.glob("*.jpg"))
        all_ids = [f.stem for f in all_images]
        self.logger.info(f"Found {len(all_ids)} images")
        
        # Shuffle
        random.shuffle(all_ids)
        
        # Calculate split sizes
        total = len(all_ids)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        
        split_ids = {
            'train': all_ids[:train_end],
            'val': all_ids[train_end:val_end],
            'test': all_ids[val_end:]
        }
        
        for name, ids in split_ids.items():
            self.logger.info(f"{name}: {len(ids)} images")
        
        # Determine class list
        self.logger.info("Determining class list...")
        class_set = set()
        sample_size = min(1000, len(all_ids))
        for img_id in tqdm(random.sample(all_ids, sample_size), desc="Sampling classes"):
            try:
                xml_path = self.config.annotations_dir / f"{img_id}.xml"
                if xml_path.exists():
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        class_set.add(obj.find("name").text)
            except:
                continue
        
        class_list = sorted(class_set)
        self.logger.info(f"Found {len(class_list)} classes: {class_list}")
        
        # First create train dataset to get class weights
        temp_train = VOCDataset(
            config=self.config,
            image_set="temp_train",
            image_ids=split_ids['train'],
            transforms=None,
            class_list=class_list,
            filter_empty=True,
            cache_annotations=True,
            logger=self.logger
        )
        
        # Create adaptive augmentation for training
        if use_adaptive_aug:
            adaptive_aug = AdaptiveAugmentation(self.config, temp_train.class_weights)
            train_transforms = adaptive_aug
        else:
            train_transforms = self._get_train_transforms()
        
        # Create all datasets
        datasets = {}
        datasets['train'] = VOCDataset(
            config=self.config,
            image_set="split_train",
            image_ids=split_ids['train'],
            transforms=train_transforms,
            class_list=class_list,
            filter_empty=True,
            cache_annotations=True,
            logger=self.logger
        )
        
        datasets['val'] = VOCDataset(
            config=self.config,
            image_set="split_val",
            image_ids=split_ids['val'],
            transforms=self._get_val_transforms(),
            class_list=class_list,
            filter_empty=True,
            cache_annotations=True,
            logger=self.logger
        )
        
        datasets['test'] = VOCDataset(
            config=self.config,
            image_set="split_test",
            image_ids=split_ids['test'],
            transforms=self._get_val_transforms(),
            class_list=class_list,
            filter_empty=True,
            cache_annotations=True,
            logger=self.logger
        )
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(datasets, f)
        
        return datasets
    
    def _get_train_transforms(self):
        """Get standard training transforms."""
        return A.Compose([
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(height=self.config.img_size, width=self.config.img_size, p=0.5),
                A.Resize(height=self.config.img_size, width=self.config.img_size)
            ], p=1.0),
            # Mandatory resize to ensure output is always 512x512 even if SafeCrop fails
            A.Resize(height=self.config.img_size, width=self.config.img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.3,
            label_fields=['labels']
        ))
    
    
    def _calculate_sampling_weights(self, dataset: VOCDataset) -> torch.Tensor:
        """Calculate sampling weights for each image based on class rarity."""
        self.logger.info("Calculating sampling weights for class balancing...")
        
        # We need to assign a weight to each image. 
        # Strategy: Image weight = max(class_weight for class in image)
        # This prioritizes images that contain at least one instance of a rare class.
        
        weights = []
        for i in range(len(dataset)):
            _, target = dataset._load_sample_no_mixup(i) # Use no mixup for stats
            
            # Get classes in this image
            classes = [dataset.inv_class_map[int(l)] for l in target["labels"] if int(l) in dataset.inv_class_map]
            
            if not classes:
                img_weight = 0.0 # No classes (background only?)
            else:
                # Get the weight of the rarest class in this image
                img_weight = max(dataset.class_weights.get(c, 0.0) for c in classes)
            
            # Ensure non-zero
            img_weight = max(img_weight, 0.1)
            weights.append(img_weight)
            
        return torch.DoubleTensor(weights)

    def _get_val_transforms(self):
        """Get validation transforms."""
        return A.Compose([
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main execution with adaptive augmentation."""
    
    config = VOCConfig(
        data_root=r"C:\Users\arunm\Documents\Project\pvelad\trainval",
        year="2012",
        debug=True,
        seed=42,
        img_size=512
    )
    
    logger = setup_logging(debug=config.debug)
    logger.info("Starting VOC Dataset Pipeline with Adaptive Augmentation")
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    try:
        split_manager = VOCSplitManager(config, logger)
        
        logger.info("Creating random splits with adaptive augmentation...")
        datasets = split_manager.create_splits(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            force_recreate=True,
            use_adaptive_aug=True
        )
        
        # Calculate weights for training set
        train_sampler = None
        if datasets['train']:
            sample_weights = split_manager._calculate_sampling_weights(datasets['train'])
            
            # User wants 3000 samples per class. 
            # With balanced sampling, we need Total = 3000 * NumClasses
            num_classes = len(datasets['train'].class_list)
            target_samples = 3000 * num_classes
            logger.info(f"Setting sampler length to {target_samples} to achieve ~3000 samples/class")
            
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=target_samples,
                replacement=True
            )
            logger.info("Created WeightedRandomSampler for class balancing")
            
            # Disable Mixup temporarily to resolve dimension mismatch stability issues
            # Adaptive Augmentation still provides variety
            datasets['train'].use_mixup = False
            logger.info("Mixup augmentation disabled for stability")
        
        # Analyze datasets
        analyzer = VOCAnalyzer(logger)
        aug_analyzer = AugmentationAnalyzer(logger)
        
        for split_name, dataset in datasets.items():
            logger.info(f"Analyzing {split_name} split (all {len(dataset)} images)...")
            stats = analyzer.analyze_dataset(dataset, num_samples=None)
            analyzer.print_report(stats, f"{split_name.upper()} SET")
        
        
        # Analyze augmentation effects on training set
        logger.info("Analyzing augmentation effects on training set...")
        # Temporarily disable mixup for analysis to see effect of geometric/color augs on single images
        was_mixup = datasets['train'].use_mixup
        datasets['train'].use_mixup = False
        
        aug_stats = aug_analyzer.analyze_augmentations(datasets['train'], num_samples=1000)
        aug_analyzer.print_augmentation_report(aug_stats, datasets['train'])
        
        datasets['train'].use_mixup = was_mixup
        
        # Create data loaders
        logger.info("Creating data loaders...")
        
        train_loader = DataLoader(
            datasets['train'],
            batch_size=4,
            sampler=train_sampler, # Use sampler
            shuffle=False if train_sampler else True, # mutually exclusive
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: tuple(zip(*batch))
        )
        
        val_loader = DataLoader(
            datasets['val'],
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: tuple(zip(*batch))
        )
        
        test_loader = DataLoader(
            datasets['test'],
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: tuple(zip(*batch))
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        logger.info("Testing one batch from train loader...")
        images, targets = next(iter(train_loader))
        logger.info(f"Batch images: {len(images)}")
        logger.info(f"Sample image shape: {images[0].shape}")
        
        for i, target in enumerate(targets[:2]):
            class_names = [datasets['train'].inv_class_map[int(l)] for l in target['labels'] if int(l) in datasets['train'].inv_class_map]
            logger.info(f"Sample {i}: {len(target['boxes'])} boxes, classes: {class_names}")
        
        # Analyze effective distribution after sampling
        if datasets['train']:
            analyzer.analyze_loader_stats(train_loader, datasets['train'], num_batches=100)
            
        # Save augmented samples visualization
        logger.info("Saving augmented samples visualization (30 images)...")
        analyzer.save_augmented_visualizations(train_loader, datasets['train'], num_images=30, save_path="augmented_batch_visualization.png")
        
        return datasets, train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


# ==========================================
# TRAINING UTILITIES
# ==========================================

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average."""
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

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        pass

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

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as input_dict,
    after reduction.
    """
    # For single process, just return
    return input_dict

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    datasets, train_loader, val_loader, test_loader = main()