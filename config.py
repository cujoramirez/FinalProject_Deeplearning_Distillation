"""
Configuration file for EfficientNet Knowledge Distillation Experiment
Baseline B0 vs Distilled B0 (from B4 teacher)
"""

import torch

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

# =============================================================================
# DATASET CONFIG
# =============================================================================
DATASET_NAME = "TinyImageNet"  # Options: "CIFAR100", "Flowers102", "Food101", "TinyImageNet"
NUM_CLASSES = 200
IMAGE_SIZE = 64  # Use native TinyImageNet resolution

# =============================================================================
# MODEL CONFIG
# =============================================================================
TEACHER_MODEL = "efficientnet_b2"  # Smaller teacher for 6GB VRAM (9.1M params)
STUDENT_MODEL = "efficientnet_b0"  # Smaller model to be trained

# Pretrained weights
USE_PRETRAINED_TEACHER = True  # Use ImageNet pretrained weights for teacher
USE_PRETRAINED_STUDENT = False  # Train student from scratch (or set True for fine-tuning)

# =============================================================================
# TRAINING CONFIG
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Adjusted for RTX 5070 headroom; lower if OOM
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Learning rate scheduler
LR_SCHEDULER = "cosine"  # Options: "cosine", "step", "plateau"
LR_STEP_SIZE = 10
LR_GAMMA = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# KNOWLEDGE DISTILLATION CONFIG
# =============================================================================
# Temperature for softening probability distributions
# Higher T = softer probabilities = more knowledge transfer
TEMPERATURE = 4.0

# Loss weights: total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
# soft_loss = KL divergence between teacher and student soft predictions
# hard_loss = Cross-entropy with ground truth labels
ALPHA = 0.7  # Weight for distillation loss (soft targets)

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
TRAIN_AUGMENTATION = {
    "random_crop": True,
    "horizontal_flip": True,
    "color_jitter": True,
    "random_rotation": 15,
    "normalize": True,
}

# =============================================================================
# EVALUATION
# =============================================================================
EVAL_METRICS = ["accuracy", "top5_accuracy", "f1_score", "inference_time"]

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# LOGGING
# =============================================================================
LOG_INTERVAL = 50  # Log every N batches
SAVE_BEST_ONLY = True
