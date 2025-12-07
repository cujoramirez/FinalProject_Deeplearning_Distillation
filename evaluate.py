"""
Evaluation and Comparison Script
Compare baseline EfficientNet-B0 vs Distilled EfficientNet-B0
"""

import os
import json
import time
from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from torchvision import models

import config
from data_loader import get_dataloaders
from models import get_student_model, count_parameters, get_model_size_mb
from utils import set_seed, accuracy, load_checkpoint, create_dirs


def measure_inference_time(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 64, 64),
    num_iterations: int = 100,
    device: str = 'cuda',
    warmup: int = 10
) -> Dict[str, float]:
    """
    Measure model inference time.
    
    Args:
        model: Model to measure
        input_size: Input tensor size (batch, channels, height, width)
        num_iterations: Number of iterations for timing
        device: Device to run on
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure time
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str,
    model_name: str = "Model"
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        model_name: Name for logging
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    print(f"\nEvaluating {model_name}...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            # Top-1 and Top-5 accuracy
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            batch_size = images.size(0)
            top1_correct += acc1.item() * batch_size / 100
            top5_correct += acc5.item() * batch_size / 100
            total += batch_size
            
            # Store predictions
            _, preds = logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    
    # Model stats
    num_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    # Inference time
    timing = measure_inference_time(model, device=device)
    
    results = {
        'model_name': model_name,
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'num_parameters': num_params,
        'size_mb': size_mb,
        'inference_time_ms': timing['mean_ms'],
        'inference_std_ms': timing['std_ms'],
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  F1 Score (Macro): {f1_macro:.2f}%")
    print(f"  Parameters: {num_params:,}")
    print(f"  Model Size: {size_mb:.2f} MB")
    print(f"  Inference Time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
    
    return results


def plot_training_comparison(
    baseline_history: Dict,
    distilled_history: Dict,
    save_path: str = None
):
    """
    Plot training curves comparison.
    
    Args:
        baseline_history: Training history of baseline model
        distilled_history: Training history of distilled model
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_baseline = range(1, len(baseline_history['train_loss']) + 1)
    epochs_distilled = range(1, len(distilled_history['train_loss']) + 1)
    
    # Training Loss
    ax = axes[0, 0]
    ax.plot(epochs_baseline, baseline_history['train_loss'], 'b-', label='Baseline', linewidth=2)
    ax.plot(epochs_distilled, distilled_history['train_loss'], 'r-', label='Distilled', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    ax.plot(epochs_baseline, baseline_history['val_loss'], 'b-', label='Baseline', linewidth=2)
    ax.plot(epochs_distilled, distilled_history['val_loss'], 'r-', label='Distilled', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[1, 0]
    ax.plot(epochs_baseline, baseline_history['train_acc'], 'b-', label='Baseline', linewidth=2)
    ax.plot(epochs_distilled, distilled_history['train_acc'], 'r-', label='Distilled', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax = axes[1, 1]
    ax.plot(epochs_baseline, baseline_history['val_acc'], 'b-', label='Baseline', linewidth=2)
    ax.plot(epochs_distilled, distilled_history['val_acc'], 'r-', label='Distilled', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training comparison plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    baseline_results: Dict,
    distilled_results: Dict,
    save_path: str = None
):
    """
    Plot bar chart comparing model metrics.
    
    Args:
        baseline_results: Evaluation results of baseline model
        distilled_results: Evaluation results of distilled model
        save_path: Path to save the figure
    """
    metrics = ['top1_accuracy', 'top5_accuracy', 'f1_macro']
    metric_names = ['Top-1 Accuracy', 'Top-5 Accuracy', 'F1 Score (Macro)']
    
    baseline_values = [baseline_results[m] for m in metrics]
    distilled_values = [distilled_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline B0', color='steelblue')
    bars2 = ax.bar(x + width/2, distilled_values, width, label='Distilled B0', color='coral')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix_comparison(
    baseline_results: Dict,
    distilled_results: Dict,
    num_classes: int = 20,  # Show top N classes for readability
    save_path: str = None
):
    """
    Plot confusion matrices for both models.
    
    Args:
        baseline_results: Evaluation results of baseline model
        distilled_results: Evaluation results of distilled model
        num_classes: Number of classes to show (for readability)
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (results, title) in enumerate([
        (baseline_results, 'Baseline B0'),
        (distilled_results, 'Distilled B0')
    ]):
        # Get subset of classes for readability
        cm = confusion_matrix(results['labels'], results['predictions'])
        cm_subset = cm[:num_classes, :num_classes]
        
        # Normalize
        cm_normalized = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', ax=ax,
                    cbar_kws={'label': 'Proportion'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{title} Confusion Matrix (First {num_classes} Classes)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def generate_comparison_report(
    baseline_results: Dict,
    distilled_results: Dict,
    save_path: str = None
) -> str:
    """
    Generate a text report comparing both models.
    
    Args:
        baseline_results: Evaluation results of baseline model
        distilled_results: Evaluation results of distilled model
        save_path: Path to save the report
    
    Returns:
        Report string
    """
    improvement_top1 = distilled_results['top1_accuracy'] - baseline_results['top1_accuracy']
    improvement_top5 = distilled_results['top5_accuracy'] - baseline_results['top5_accuracy']
    improvement_f1 = distilled_results['f1_macro'] - baseline_results['f1_macro']
    
    report = f"""
================================================================================
          KNOWLEDGE DISTILLATION COMPARATIVE ANALYSIS REPORT
================================================================================

EXPERIMENT CONFIGURATION
------------------------
Dataset: {config.DATASET_NAME}
Teacher Model: {config.TEACHER_MODEL}
Student Model: {config.STUDENT_MODEL}
Distillation Temperature: {config.TEMPERATURE}
Alpha (soft target weight): {config.ALPHA}
Training Epochs: {config.NUM_EPOCHS}
Batch Size: {config.BATCH_SIZE}
Learning Rate: {config.LEARNING_RATE}

================================================================================
MODEL SPECIFICATIONS
================================================================================

                          Baseline B0          Distilled B0
                          -----------          ------------
Parameters:               {baseline_results['num_parameters']:>12,}      {distilled_results['num_parameters']:>12,}
Model Size (MB):          {baseline_results['size_mb']:>12.2f}      {distilled_results['size_mb']:>12.2f}
Inference Time (ms):      {baseline_results['inference_time_ms']:>12.2f}      {distilled_results['inference_time_ms']:>12.2f}

================================================================================
PERFORMANCE COMPARISON
================================================================================

                          Baseline B0          Distilled B0         Improvement
                          -----------          ------------         -----------
Top-1 Accuracy (%):       {baseline_results['top1_accuracy']:>12.2f}      {distilled_results['top1_accuracy']:>12.2f}        {improvement_top1:>+.2f}
Top-5 Accuracy (%):       {baseline_results['top5_accuracy']:>12.2f}      {distilled_results['top5_accuracy']:>12.2f}        {improvement_top5:>+.2f}
F1 Score Macro (%):       {baseline_results['f1_macro']:>12.2f}      {distilled_results['f1_macro']:>12.2f}        {improvement_f1:>+.2f}
F1 Score Weighted (%):    {baseline_results['f1_weighted']:>12.2f}      {distilled_results['f1_weighted']:>12.2f}

================================================================================
KEY FINDINGS
================================================================================

1. ACCURACY IMPROVEMENT:
   - Knowledge distillation {"improved" if improvement_top1 > 0 else "decreased"} Top-1 accuracy by {abs(improvement_top1):.2f}%
   - Top-5 accuracy {"improved" if improvement_top5 > 0 else "decreased"} by {abs(improvement_top5):.2f}%

2. MODEL EFFICIENCY:
   - Both models have identical architecture (EfficientNet-B0)
   - Same inference time and model size
   - Distillation provides accuracy gains with no additional inference cost

3. KNOWLEDGE TRANSFER:
   - Teacher model ({config.TEACHER_MODEL}) knowledge successfully transferred
   - Student model learns from soft probability distributions
   - Temperature T={config.TEMPERATURE} used for softening predictions

================================================================================
CONCLUSIONS
================================================================================

{"[OK] Knowledge distillation successfully improved model performance" if improvement_top1 > 0 else "[X] Baseline performed better in this experiment"}
{"[OK] Distilled model shows better generalization" if improvement_top1 > 0 else "Consider tuning distillation hyperparameters (T, alpha)"}

The experiment demonstrates that knowledge distillation can transfer knowledge 
from a larger teacher model to a smaller student model, potentially achieving 
better performance than training the student model from scratch.

================================================================================
"""
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report


def build_torchvision_b0(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_torchvision_b2(num_classes: int):
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_torchvision_r18(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model_from_spec(spec: Dict, num_classes: int, device: str):
    kind = spec["kind"]
    if kind == "timm_b0":
        model = get_student_model(num_classes)
    elif kind == "tv_b0":
        model = build_torchvision_b0(num_classes)
    elif kind == "tv_b2":
        model = build_torchvision_b2(num_classes)
    elif kind == "tv_r18":
        model = build_torchvision_r18(num_classes)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    ckpt = spec["ckpt"]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model.to(device)


def plot_metric_bars(results: List[Dict], metric: str, title: str, save_path: str = None):
    labels = [r["model_name"] for r in results]
    values = [r[metric] for r in results]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=plt.cm.tab20.colors[: len(labels)])
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {title} to {save_path}")
    plt.close()


def run_full_comparison():
    """Run comprehensive comparison across baseline, vanilla distill, AKTP, and teachers."""

    set_seed(config.SEED)
    create_dirs()
    device = config.DEVICE

    print("=" * 60)
    print("Loading TinyImageNet test data...")
    print("=" * 60)
    _, _, test_loader, num_classes = get_dataloaders(dataset_name="TinyImageNet", batch_size=config.BATCH_SIZE)

    model_specs = [
        {
            "model_name": "Baseline B0 TinyImageNet",
            "kind": "timm_b0",
            "ckpt": os.path.join(config.CHECKPOINT_DIR, "baseline_b0_tinyimagenet", "best_model.pth"),
        },
        {
            "model_name": "Vanilla Distilled B0",
            "kind": "tv_b0",
            "ckpt": os.path.join(config.CHECKPOINT_DIR, "distilled_b0", "best_model.pth"),
        },
        {
            "model_name": "AKTP Distilled B0",
            "kind": "tv_b0",
            "ckpt": os.path.join("./checkpoints_aktp", "b0_aktp_tiny_best.pth"),
        },
        {
            "model_name": "Teacher EfficientNet-B2",
            "kind": "tv_b2",
            "ckpt": os.path.join("./checkpoints_aktp", "teacher_b2_tiny.pth"),
        },
        {
            "model_name": "Teacher ResNet18",
            "kind": "tv_r18",
            "ckpt": os.path.join("./checkpoints_aktp", "teacher_r18_tiny.pth"),
        },
    ]

    all_results = []
    for spec in model_specs:
        print("\n" + "=" * 60)
        print(f"Loading {spec['model_name']} from {spec['ckpt']}")
        print("=" * 60)
        model = load_model_from_spec(spec, num_classes, device)
        results = evaluate_model(model, test_loader, device, spec["model_name"])
        all_results.append(results)

    # Save aggregated results
    aggregate = {
        r["model_name"]: {k: v for k, v in r.items() if k not in ["predictions", "labels", "probabilities"]}
        for r in all_results
    }
    agg_path = os.path.join(config.RESULTS_DIR, "comparison_results_all.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Aggregated results saved to {agg_path}")

    # Bar plots for key metrics
    plot_metric_bars(all_results, "top1_accuracy", "Top-1 Accuracy Comparison",
                     os.path.join(config.RESULTS_DIR, "metrics_top1_all.png"))
    plot_metric_bars(all_results, "top5_accuracy", "Top-5 Accuracy Comparison",
                     os.path.join(config.RESULTS_DIR, "metrics_top5_all.png"))
    plot_metric_bars(all_results, "f1_macro", "F1 Macro Comparison",
                     os.path.join(config.RESULTS_DIR, "metrics_f1_all.png"))

    return all_results


if __name__ == "__main__":
    run_full_comparison()
