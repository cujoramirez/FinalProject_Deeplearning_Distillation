"""
Model definitions and utilities for EfficientNet Knowledge Distillation
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional

import config


def get_efficientnet(
    model_name: str,
    num_classes: int,
    pretrained: bool = True
) -> nn.Module:
    """
    Create an EfficientNet model using timm library.
    
    Args:
        model_name: Model name (e.g., "efficientnet_b0", "efficientnet_b4")
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        EfficientNet model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def get_teacher_model(num_classes: int) -> nn.Module:
    """
    Get the teacher model (larger EfficientNet).
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        Teacher model (EfficientNet-B4 by default)
    """
    print(f"Loading teacher model: {config.TEACHER_MODEL}")
    model = get_efficientnet(
        config.TEACHER_MODEL,
        num_classes=num_classes,
        pretrained=config.USE_PRETRAINED_TEACHER
    )
    print(f"  Parameters: {count_parameters(model):,}")
    return model


def get_student_model(num_classes: int) -> nn.Module:
    """
    Get the student model (smaller EfficientNet).
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        Student model (EfficientNet-B0 by default)
    """
    print(f"Loading student model: {config.STUDENT_MODEL}")
    model = get_efficientnet(
        config.STUDENT_MODEL,
        num_classes=num_classes,
        pretrained=config.USE_PRETRAINED_STUDENT
    )
    print(f"  Parameters: {count_parameters(model):,}")
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining soft and hard targets.
    
    Loss = alpha * KL_divergence(soft_student, soft_teacher) + (1-alpha) * CE(student, labels)
    
    Args:
        temperature: Temperature for softening probability distributions
        alpha: Weight for soft target loss (1-alpha for hard target loss)
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
            labels: Ground truth labels
        
        Returns:
            total_loss, soft_loss, hard_loss
        """
        # Soft targets (with temperature)
        soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence loss (scaled by T^2 as per Hinton et al.)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard targets (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss


def compare_models():
    """Print a comparison of different EfficientNet variants."""
    print("=" * 70)
    print("EfficientNet Model Comparison")
    print("=" * 70)
    
    models_info = [
        ("efficientnet_b0", "Student/Baseline"),
        ("efficientnet_b2", "Teacher (6GB VRAM)"),
        ("efficientnet_b4", "Teacher (8GB+ VRAM)"),
    ]
    
    print(f"{'Model':<20} {'Role':<25} {'Params':>12} {'Size (MB)':>12}")
    print("-" * 70)
    
    for model_name, role in models_info:
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
        params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        print(f"{model_name:<20} {role:<25} {params:>12,} {size_mb:>12.2f}")
        del model
    
    print("=" * 70)


if __name__ == "__main__":
    compare_models()
    
    print("\n" + "=" * 70)
    print("Testing model creation...")
    print("=" * 70)
    
    # Test model creation
    teacher = get_teacher_model(num_classes=100)
    student = get_student_model(num_classes=100)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    
    teacher.eval()
    student.eval()
    
    with torch.no_grad():
        teacher_out = teacher(dummy_input)
        student_out = student(dummy_input)
    
    print(f"\nTeacher output shape: {teacher_out.shape}")
    print(f"Student output shape: {student_out.shape}")
    
    # Test distillation loss
    print("\n" + "=" * 70)
    print("Testing distillation loss...")
    print("=" * 70)
    
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    labels = torch.randint(0, 100, (2,))
    
    total_loss, soft_loss, hard_loss = criterion(student_out, teacher_out, labels)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Soft loss (KL): {soft_loss.item():.4f}")
    print(f"Hard loss (CE): {hard_loss.item():.4f}")
