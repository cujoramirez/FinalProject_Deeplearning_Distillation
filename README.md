# Knowledge Distillation: EfficientNet-B0 Baseline vs Distilled

A comparative analysis of EfficientNet-B0 trained with standard cross-entropy loss versus knowledge distillation from EfficientNet-B2.

## Project Structure

```
final_DL/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Dataset loading and preprocessing
├── models.py              # Model definitions and distillation loss
├── train_baseline.py      # Baseline training script
├── train_distillation.py  # Knowledge distillation training script
├── evaluate.py            # Evaluation and comparison utilities
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
├── data/                  # Downloaded datasets (auto-created)
├── checkpoints/           # Saved model weights (auto-created)
└── results/               # Training logs and plots (auto-created)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify GPU (Optional)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

## Usage

### Step 1: Train Baseline Model

```bash
python train_baseline.py
```

This trains EfficientNet-B0 with standard cross-entropy loss.

### Step 2: Train Distilled Model

```bash
python train_distillation.py
```

This trains EfficientNet-B0 using knowledge distillation from EfficientNet-B2.

### Step 3: Compare Results

```bash
python evaluate.py
```

This generates:
- Performance comparison plots
- Confusion matrices
- Detailed comparison report

## Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_NAME` | "CIFAR100" | Dataset to use |
| `TEACHER_MODEL` | "efficientnet_b2" | Teacher model architecture |
| `STUDENT_MODEL` | "efficientnet_b0" | Student model architecture |
| `TEMPERATURE` | 4.0 | Distillation temperature |
| `ALPHA` | 0.7 | Weight for soft targets |
| `BATCH_SIZE` | 16 | Batch size (optimized for 6GB VRAM) |
| `NUM_EPOCHS` | 50 | Training epochs |

## Model Sizes

| Model | Parameters | Size (MB) | GPU Memory (batch=16) |
|-------|------------|-----------|----------------------|
| EfficientNet-B0 (Student) | 5.3M | ~20 MB | ~1.5 GB |
| EfficientNet-B2 (Teacher) | 9.1M | ~35 MB | ~2.5 GB |
| **Combined (Training)** | - | - | **~4-5 GB** ✅ |

## Knowledge Distillation

The distillation loss combines:
- **Soft loss**: KL divergence between teacher and student soft predictions
- **Hard loss**: Cross-entropy with ground truth labels

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}(\sigma(z_s/T) \| \sigma(z_t/T)) + (1-\alpha) \cdot \text{CE}(z_s, y)$$

Where:
- $z_s$, $z_t$ = student and teacher logits
- $T$ = temperature (higher = softer probabilities)
- $\alpha$ = weight for soft targets
- $\sigma$ = softmax function

## Expected Results

After training, you should see:
- Distilled model outperforms baseline by ~1-3% on CIFAR-100
- Both models have identical inference time (same architecture)
- Distilled model learns "dark knowledge" from teacher's soft predictions

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` in `config.py` (try 8)
- Current config is optimized for 6GB VRAM with B2 teacher

### Slow Training
- Reduce `NUM_EPOCHS`
- Use smaller dataset (Flowers102)
- Ensure CUDA is being used

## For Your Colleagues

This project is designed for team collaboration:
- **Explainability team**: Use saved models in `checkpoints/` for GradCAM, SHAP, etc.
- **Demo app team**: Load `best_model.pth` for inference in Streamlit/Flask
- **Report team**: Use plots in `results/` and `comparison_report.txt`

## References

- [Hinton et al., "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
