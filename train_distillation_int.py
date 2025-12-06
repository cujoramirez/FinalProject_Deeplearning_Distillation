"""
Knowledge Distillation Training Script
Ensemble distillation: Teacher = EfficientNet-B2 + ResNet18 (avg logits) -> Student = EfficientNet-B0
Configured for TinyImageNet (200 classes, 64x64)
"""

import os
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torchvision import models as tv_models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

import config
from data_loader import get_dataloaders
from models import get_teacher_model, get_student_model, DistillationLoss, count_parameters
from utils import (
    set_seed, 
    AverageMeter, 
    accuracy, 
    save_checkpoint, 
    EarlyStopping,
    create_dirs
)


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    criterion: DistillationLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """
    Train the student model for one epoch using knowledge distillation.
    
    Args:
        student: Student model to train
        teacher: Teacher model (frozen)
        train_loader: Training data loader
        criterion: Distillation loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        Dictionary with training metrics
    """
    student.train()
    teacher.eval()  # Teacher is always in eval mode
    
    losses = AverageMeter()
    soft_losses = AverageMeter()
    hard_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass through teachers (no gradients needed)
        with torch.no_grad():
            logits_b2 = teacher_b2(images)
            logits_r18 = teacher_r18(images)
            teacher_logits = (logits_b2 + logits_r18) / 2.0
        
        # Forward pass through student
        student_logits = student(images)
        
        # Calculate distillation loss
        total_loss, soft_loss, hard_loss = criterion(student_logits, teacher_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc1, acc5 = accuracy(student_logits, labels, topk=(1, 5))
        
        # Update meters
        batch_size = images.size(0)
        losses.update(total_loss.item(), batch_size)
        soft_losses.update(soft_loss.item(), batch_size)
        hard_losses.update(hard_loss.item(), batch_size)
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
        'soft_loss': soft_losses.avg,
        'hard_loss': hard_losses.avg,
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
        criterion: Loss function (CrossEntropyLoss for validation)
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    ce_criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = ce_criterion(logits, labels)
            
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


def train_distillation():
    """Main training function for knowledge distillation."""
    
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
    
    # Create models
    print("\n" + "=" * 60)
    print("Creating models...")
    print("=" * 60)
    teacher_b2 = get_teacher_model(num_classes).to(device)
    teacher_r18 = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    teacher_r18.fc = nn.Linear(teacher_r18.fc.in_features, num_classes)
    teacher_r18 = teacher_r18.to(device)
    student = get_student_model(num_classes).to(device)
    
    # Freeze teacher models
    for param in teacher_b2.parameters():
        param.requires_grad = False
    teacher_b2.eval()
    for param in teacher_r18.parameters():
        param.requires_grad = False
    teacher_r18.eval()
    
    print(f"\nTeacher B2 parameters: {count_parameters(teacher_b2):,} (frozen)")
    print(f"Teacher R18 parameters: {count_parameters(teacher_r18):,} (frozen)")
    print(f"Student parameters: {count_parameters(student):,} (trainable)")
    
    # Loss and optimizer
    criterion = DistillationLoss(
        temperature=config.TEMPERATURE,
        alpha=config.ALPHA
    )
    
    optimizer = optim.AdamW(
        student.parameters(),
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
        'soft_loss': [], 'hard_loss': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "=" * 60)
    print("Starting Knowledge Distillation Training")
    print(f"Temperature: {config.TEMPERATURE}, Alpha: {config.ALPHA}")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            student, teacher, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(student, val_loader, criterion, device, epoch)
        
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
        history['soft_loss'].append(train_metrics['soft_loss'])
        history['hard_loss'].append(train_metrics['hard_loss'])
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
            student, optimizer, epoch, val_metrics['top1_acc'],
            os.path.join(config.CHECKPOINT_DIR, 'distilled_b0'),
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
    history_path = os.path.join(config.RESULTS_DIR, 'distillation_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    # Load best model
    best_checkpoint = torch.load(
        os.path.join(config.CHECKPOINT_DIR, 'distilled_b0', 'best_model.pth'),
        map_location=device
    )
    student.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = validate(student, test_loader, criterion, device, epoch=-1)
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {test_metrics['top5_acc']:.2f}%")
    
    return history, test_metrics


if __name__ == "__main__":
    train_distillation()
