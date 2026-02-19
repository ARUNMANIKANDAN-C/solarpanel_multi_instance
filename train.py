
import os
import torch
import logging
import argparse
from pathlib import Path
import math
import sys
import numpy as np
import time
from datetime import datetime

import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, WeightedRandomSampler

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

# Import local utils
import utils
from utils import VOCConfig, VOCSplitManager, VOCDataset

# Setup Rich Console
console = Console()

# Setup Logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("Train")

def get_model(num_classes):
    """
    Create Faster R-CNN model with ResNet-50 FPN backbone.
    """
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    all_losses = []
    
    # Use Rich Progress
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
        # We manually iterate to update progress bar
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                logger.error(f"Loss is {loss_value}, stopping training")
                logger.error(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            all_losses.append(loss_value)
            
            # Update progress bar
            progress.update(task_id, advance=1, loss=f"Loss: {loss_value:.4f}")

    return np.mean(all_losses)

def evaluate(model, data_loader, device):
    """
    Simple evaluation loop with Rich progress.
    """
    model.train() # Keep in train mode for loss calculation
    losses = []
    
    # Use Rich Progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    
    task_id = progress.add_task("[green]Validating", total=len(data_loader))
    
    with progress:
        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                losses.append(loss.item())
                
                progress.update(task_id, advance=1)
            
    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on VOC')
    parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size per GPU')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold magenta]PVELAD Training Pipeline[/bold magenta]", border_style="magenta"))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    console.print(f"[bold green]Using device:[/bold green] {device}")
    
    # 1. Setup Config and Dataset
    config = VOCConfig()
    
    # Ensure directories exist
    config.models_dir.mkdir(exist_ok=True, parents=True)
    
    with console.status("[bold green]Creating dataset splits...", spinner="dots"):
        split_manager = VOCSplitManager(config, logger)
        datasets = split_manager.create_splits(
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15, 
            use_adaptive_aug=True
        )
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    
    # Num classes = (classes in dataset) + background
    num_classes = len(train_dataset.class_list) + 1
    console.print(f"[bold yellow]Training with {num_classes} classes (including background)[/bold yellow]")
    
    # 2. Data Loaders
    
    # Setup WeightedRandomSampler
    with console.status("[bold green]Setting up Sampler...", spinner="dots"):
        class_counts = train_dataset.class_distribution
        total_instances = sum(class_counts.values())
        class_weights = {k: total_instances / (v + 1e-6) for k, v in class_counts.items()}
        
        sample_weights = []
        for i in range(len(train_dataset)):
            _, target = train_dataset._load_sample_no_mixup(i)
            labels = target['labels']
            if len(labels) > 0:
                weight = max([class_weights.get(train_dataset.inv_class_map.get(int(l), ''), 0) for l in labels])
            else:
                weight = 0
            sample_weights.append(weight)
            
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
        num_target_samples = 3000 * (num_classes - 1) 
        console.print(f"Sampler target samples: [cyan]{num_target_samples}[/cyan]")
        
        train_sampler = WeightedRandomSampler(sample_weights, num_target_samples, replacement=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=utils.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=utils.collate_fn
    )
    
    # 3. Model
    model = get_model(num_classes)
    model.to(device)
    
    # 4. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        console.print(f"[bold yellow]Resumed from epoch {start_epoch}[/bold yellow]")

    # 5. Training Loop
    console.print("[bold magenta]Starting training...[/bold magenta]")
    
    # Create results table
    table = Table(title="Training Results")
    table.add_column("Epoch", justify="center", style="cyan", no_wrap=True)
    table.add_column("Train Loss", justify="right", style="green")
    table.add_column("Val Loss", justify="right", style="blue")
    table.add_column("Time", justify="right", style="magenta")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        epoch_time_str = str(datetime.fromtimestamp(epoch_time).strftime('%M:%S')) # not quite right for duration but good enough for small times
        # Better formatting for duration
        m, s = divmod(epoch_time, 60)
        epoch_str = f"{int(m):02d}:{int(s):02d}"

        # Update table
        table.add_row(str(epoch), f"{train_loss:.4f}", f"{val_loss:.4f}", epoch_str)
        console.clear() # Optional: clear to redraw table or just print it iteratively
        # Instead of clearing, let's print the table row or just print the table at the end?
        # A live table is nice, but might conflict with progress bars. 
        # Simpler: print log message and update a global table? 
        # Let's just print a summary line using console.log
        
        console.print(f"[bold]Epoch {epoch}[/bold] | Train Loss: [green]{train_loss:.4f}[/green] | Val Loss: [blue]{val_loss:.4f}[/blue] | Time: {epoch_str}")
        
        # Scheduler
        lr_scheduler.step()
        
        # Save Checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        
        # Save latest
        torch.save(checkpoint, str(config.models_dir / "latest.pth"))
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, str(config.models_dir / "best_model.pth"))
            console.print(f"[bold yellow]Saved new best model with loss {best_loss:.4f}[/bold yellow]")

    console.print(Panel.fit("[bold green]Training Complete![/bold green]", border_style="green"))

if __name__ == "__main__":
    main()
