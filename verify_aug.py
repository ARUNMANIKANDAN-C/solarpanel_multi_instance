
import sys
import logging
import time
import torch
import numpy as np
from main import VOCConfig, VOCDataset, AdaptiveAugmentation, VOCSplitManager
import albumentations as A

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def test_augmentation():
    config = VOCConfig(data_root="trainval") 
    
    # Create dataset to get class weights
    logger.info("Initializing dataset...")
    # This will now trigger _log_init and show the distribution with percentages
    ds = VOCDataset(config, "trainval")
    
    # Setup Augmentation
    aug = AdaptiveAugmentation(config, ds.class_weights)
    ds.adaptive_aug = aug
    
    logger.info("Testing augmentation on 50 samples...")
    success_count = 0
    
    for i in range(min(50, len(ds))):
        try:
            image, target = ds[i]
            
            if image.shape[0] != 3 or image.shape[1] != 512 or image.shape[2] != 512:
                logger.error(f"Image {i} has wrong shape: {image.shape}")
                continue
                
            if torch.isnan(image).any():
                logger.error(f"Image {i} has NaNs!")
                continue
                
            success_count += 1
            if i % 10 == 0:
                logger.info(f"Processed {i} samples successfully.")
                
        except Exception as e:
            logger.error(f"Failed at index {i}: {e}")
            sys.exit(1)
            
    logger.info(f"Successfully processed {success_count} samples with augmentation.")
    
if __name__ == "__main__":
    test_augmentation()
