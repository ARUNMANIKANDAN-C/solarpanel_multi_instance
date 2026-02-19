
from main import VOCConfig, VOCDataset, console
import logging

# Setup config to point to existing data
config = VOCConfig(data_root="trainval") 

# Instantiate dataset (triggers _log_init -> table)
print("Creating dataset to show table...")
ds = VOCDataset(config, "train", class_list=["dummy"]) 

# Mock distribution for speed if needed, but VOCDataset will scan.
# Let's hope it scans fast or we mock it.
# Actually, let's allow it to scan a few images to populate the table.
