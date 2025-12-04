"""
Evaluation and Comparison Script
Compare baseline EfficientNet-B0 vs Distilled EfficientNet-B0
"""

import os
import json
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

import config
from data_loader import get_dataloaders
from models import get_student_model, count_parameters, get_model_size_mb
from utils import set_seed, accuracy, load_checkpoint, create_dirs


def measure_inference_time(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
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


def run_full_comparison():
    """Run complete comparison between baseline and distilled models."""
    
    set_seed(config.SEED)
    create_dirs()
    device = config.DEVICE
    
    print("=" * 60)
    print("Loading test data...")
    print("=" * 60)
    _, _, test_loader, num_classes = get_dataloaders()
    
    # Load models
    print("\n" + "=" * 60)
    print("Loading trained models...")
    print("=" * 60)
    
    baseline_model = get_student_model(num_classes)
    distilled_model = get_student_model(num_classes)
    
    # Load checkpoints
    baseline_path = os.path.join(config.CHECKPOINT_DIR, 'baseline_b0', 'best_model.pth')
    distilled_path = os.path.join(config.CHECKPOINT_DIR, 'distilled_b0', 'best_model.pth')
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
    if not os.path.exists(distilled_path):
        raise FileNotFoundError(f"Distilled checkpoint not found: {distilled_path}")
    
    baseline_model, _, _ = load_checkpoint(baseline_model, baseline_path, device)
    distilled_model, _, _ = load_checkpoint(distilled_model, distilled_path, device)
    
    # Evaluate models
    print("\n" + "=" * 60)
    print("Evaluating models...")
    print("=" * 60)
    
    baseline_results = evaluate_model(baseline_model, test_loader, device, "Baseline B0")
    distilled_results = evaluate_model(distilled_model, test_loader, device, "Distilled B0")
    
    # Load training histories
    baseline_history_path = os.path.join(config.RESULTS_DIR, 'baseline_history.json')
    distilled_history_path = os.path.join(config.RESULTS_DIR, 'distillation_history.json')
    
    if os.path.exists(baseline_history_path) and os.path.exists(distilled_history_path):
        with open(baseline_history_path, 'r') as f:
            baseline_history = json.load(f)
        with open(distilled_history_path, 'r') as f:
            distilled_history = json.load(f)
        
        # Plot training curves
        plot_training_comparison(
            baseline_history, distilled_history,
            save_path=os.path.join(config.RESULTS_DIR, 'training_comparison.png')
        )
    
    # Plot metrics comparison
    plot_metrics_comparison(
        baseline_results, distilled_results,
        save_path=os.path.join(config.RESULTS_DIR, 'metrics_comparison.png')
    )
    
    # Plot confusion matrices
    plot_confusion_matrix_comparison(
        baseline_results, distilled_results,
        save_path=os.path.join(config.RESULTS_DIR, 'confusion_matrices.png')
    )
    
    # Generate report
    report = generate_comparison_report(
        baseline_results, distilled_results,
        save_path=os.path.join(config.RESULTS_DIR, 'comparison_report.txt')
    )
    print(report)
    
    # Save results as JSON
    results_json = {
        'baseline': {
            k: v for k, v in baseline_results.items() 
            if k not in ['predictions', 'labels', 'probabilities']
        },
        'distilled': {
            k: v for k, v in distilled_results.items() 
            if k not in ['predictions', 'labels', 'probabilities']
        }
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    return baseline_results, distilled_results


if __name__ == "__main__":
    run_full_comparison()
