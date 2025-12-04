"""
Baseline Training Script
Trains EfficientNet-B0 with standard cross-entropy loss (no distillation)
"""

import os
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

import config
from data_loader import get_dataloaders
from models import get_student_model, count_parameters
from utils import (
    set_seed, 
    AverageMeter, 
    accuracy, 
    save_checkpoint, 
    EarlyStopping,
    create_dirs
)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """
    Train the model for one epoch using standard cross-entropy.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        
        # Update meters
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc@1': f'{top1.avg:.2f}%',
            'acc@5': f'{top5.avg:.2f}%'
        })
    
    return {
        'loss': losses.avg,
        'top1_acc': top1.avg,
        'top5_acc': top5.avg
    }


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> dict:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            
            # Update meters
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc@1': f'{top1.avg:.2f}%',
                'acc@5': f'{top5.avg:.2f}%'
            })
    
    return {
        'loss': losses.avg,
        'top1_acc': top1.avg,
        'top5_acc': top5.avg
    }


def train_baseline():
    """Main training function for baseline model (no distillation)."""
    
    # Setup
    set_seed(config.SEED)
    create_dirs()
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating baseline model...")
    print("=" * 60)
    model = get_student_model(num_classes).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    if config.LR_SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    elif config.LR_SCHEDULER == "step":
        scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "=" * 60)
    print("Starting Baseline Training (No Distillation)")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if config.LR_SCHEDULER == "plateau":
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['top1_acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['top1_acc'])
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['top1_acc']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['top1_acc']:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        is_best = val_metrics['top1_acc'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['top1_acc']
            print(f"  New best validation accuracy: {best_val_acc:.2f}%")
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics['top1_acc'],
            os.path.join(config.CHECKPOINT_DIR, 'baseline_b0'),
            is_best=is_best
        )
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, 'baseline_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    # Load best model
    best_checkpoint = torch.load(
        os.path.join(config.CHECKPOINT_DIR, 'baseline_b0', 'best_model.pth'),
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device, epoch=-1)
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {test_metrics['top5_acc']:.2f}%")
    
    return history, test_metrics


if __name__ == "__main__":
    train_baseline()
