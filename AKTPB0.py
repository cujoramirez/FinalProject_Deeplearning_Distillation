import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet18_Weights,
)
from tqdm import tqdm
import numpy as np

# --- Configuration ---
class Config:
    def __init__(self):
        self.dataset_name = "CIFAR100"  # Options: "CIFAR100", "TinyImageNet"
        self.data_path = "./data"
        self.num_classes = 100 if self.dataset_name == "CIFAR100" else 200
        self.image_size = None  # Use native dataset resolution
        self.batch_size = 32  # Safe on RTX 5070; lower if OOM
        self.num_workers = 4
        self.epochs = 50
        self.lr = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping_patience = 10
        
        # Paths to PRETRAINED Teachers (You must provide these)
        # For this script to run immediately, I will use torchvision pretrained 
        # but in practice, you load your checkpoints here.
        self.teacher_b2_pretrained = True 
        self.teacher_r18_pretrained = True 
        self.student_pretrained = False  # Keep student random init by default

# --- 1. Interchangeable Dataset Wrapper ---
def get_dataloaders(config):
    print(f"Loading {config.dataset_name}...")
    
    if config.dataset_name == "CIFAR100":
        # Native 32x32 resolution, CIFAR normalization
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        train_set = datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=test_transform)
    elif config.dataset_name == "TinyImageNet":
        # Native 64x64 resolution, ImageNet normalization
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        train_dir = os.path.join(config.data_path, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(config.data_path, 'tiny-imagenet-200', 'val') 
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(val_dir, transform=test_transform)
    else:
        raise ValueError("Dataset not supported")

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader

# --- 2. The Combiner Module (Logit Fusion) ---
class CombinerNetwork(nn.Module):
    """
    Fuses logits from multiple teachers into a single soft target.
    Reference CALM Paper Stage 2.
    """
    def __init__(self, num_teachers, num_classes, hidden_dim=256):
        super().__init__()
        input_dim = num_teachers * num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, logits_list):
        # Concatenate logits: [Batch, Class] + [Batch, Class] -> [Batch, Class*2]
        combined = torch.cat(logits_list, dim=1)
        return self.net(combined)

# --- 3. AKTP Weighting Module ---
class AKTP(nn.Module):
    """
    Adaptive Knowledge Transfer Protocol.
    Calculates lambda based on Student Entropy and Teacher Disagreement.
    Reference CALM Paper[cite: 219, 237].
    """
    def __init__(self):
        super().__init__()
        # Input: 2 dims (Entropy, Disagreement) -> Output: 1 scalar (Lambda)
        self.fc = nn.Linear(2, 1)
        # Initialize bias to negative to prefer distillation (lambda close to 0) initially
        nn.init.constant_(self.fc.bias, -1.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, student_logits, teacher_logits_1, teacher_logits_2):
        # 1. Calculate Student Entropy H(S(x))
        probs_student = F.softmax(student_logits, dim=1)
        log_probs_student = F.log_softmax(student_logits, dim=1)
        entropy = -torch.sum(probs_student * log_probs_student, dim=1, keepdim=True) # [Batch, 1]

        # 2. Calculate Teacher Disagreement D(T1, T2) using symmetric KL
        # Note: Paper uses disagreement between students, we adapt to disagreement between Teachers
        log_prob_t1 = F.log_softmax(teacher_logits_1, dim=1)
        prob_t2 = F.softmax(teacher_logits_2, dim=1)
        
        log_prob_t2 = F.log_softmax(teacher_logits_2, dim=1)
        prob_t1 = F.softmax(teacher_logits_1, dim=1)
        
        kl1 = F.kl_div(log_prob_t1, prob_t2, reduction='none', log_target=False).sum(1, keepdim=True)
        kl2 = F.kl_div(log_prob_t2, prob_t1, reduction='none', log_target=False).sum(1, keepdim=True)
        disagreement = 0.5 * (kl1 + kl2) # [Batch, 1]

        # 3. Compute Lambda
        # Normalize inputs roughly for stability
        features = torch.cat([entropy, disagreement], dim=1)
        return self.sigmoid(self.fc(features)) # Returns lambda per sample [Batch, 1]


class EarlyStopping:
    """Stop training if validation metric does not improve after patience epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience

# --- 4. Main Training System ---
def train_aktp(config):
    train_loader, test_loader = get_dataloaders(config)

    # --- Initialize Models ---
    print("Initializing Models...")
    
    # 1. Student: EfficientNet-B0 (Trainable)
    student_weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if config.student_pretrained else None
    student = models.efficientnet_b0(weights=student_weights)
    student.classifier[1] = nn.Linear(student.classifier[1].in_features, config.num_classes)
    student = student.to(config.device)

    # 2. Teacher 1: EfficientNet-B2 (Frozen)
    t1_weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if config.teacher_b2_pretrained else None
    t1 = models.efficientnet_b2(weights=t1_weights)
    t1.classifier[1] = nn.Linear(t1.classifier[1].in_features, config.num_classes)
    t1 = t1.to(config.device)
    t1.eval()  # Freeze
    for p in t1.parameters():
        p.requires_grad = False

    # 3. Teacher 2: ResNet18 (Frozen)
    t2_weights = ResNet18_Weights.IMAGENET1K_V1 if config.teacher_r18_pretrained else None
    t2 = models.resnet18(weights=t2_weights)
    t2.fc = nn.Linear(t2.fc.in_features, config.num_classes)
    t2 = t2.to(config.device)
    t2.eval()  # Freeze
    for p in t2.parameters():
        p.requires_grad = False
    
    # 4. Modules
    combiner = CombinerNetwork(num_teachers=2, num_classes=config.num_classes).to(config.device)
    aktp_module = AKTP().to(config.device)

    # Optimizer (Train Student, Combiner, and AKTP weights)
    optimizer = optim.AdamW([
        {'params': student.parameters(), 'lr': config.lr},
        {'params': combiner.parameters(), 'lr': config.lr},
        {'params': aktp_module.parameters(), 'lr': config.lr}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # --- Training Loop ---
    print("Starting Training with AKTP...")
    
    early_stop = EarlyStopping(patience=config.early_stopping_patience)
    best_acc = 0.0

    for epoch in range(config.epochs):
        student.train()
        combiner.train()
        aktp_module.train()
        
        total_loss = 0
        avg_lambda = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            
            # 1. Get Teacher Logits (No Grad)
            with torch.no_grad():
                # Resize if necessary for specific architectures (EfficientNet usually handles varying sizes, 
                # but B2 might prefer 260. Here we stick to input size for speed, or upscale if needed)
                l_t1 = t1(inputs)
                l_t2 = t2(inputs)
            
            # 2. Logit Fusion (Combiner)
            # Create a "Soft Target" from the teachers [cite: 218]
            fused_logits = combiner([l_t1, l_t2])
            p_comb = F.softmax(fused_logits, dim=1)

            # 3. Student Forward
            l_student = student(inputs)

            # 4. AKTP Weight Calculation 
            # lambda weighs CE (Ground Truth), (1-lambda) weighs KD (Fused Teachers)
            lambda_val = aktp_module(l_student, l_t1, l_t2) # shape [Batch, 1]
            
            # 5. Loss Calculation
            # CE Loss (Student vs Ground Truth)
            ce_loss = F.cross_entropy(l_student, targets, reduction='none')
            
            # KD Loss (Student vs Fused Teacher Soft Targets)
            # Note: KLDiv expects log_probs as input
            log_prob_student = F.log_softmax(l_student, dim=1)
            kd_loss = F.kl_div(log_prob_student, p_comb, reduction='none').sum(dim=1)
            
            # Calibration Loss (Optional, from CALM )
            # Simplified Brier score-like penalty for calibration
            conf, pred = torch.max(F.softmax(l_student, dim=1), 1)
            acc = (pred == targets).float()
            cal_loss = (conf - acc) ** 2
            
            # Final Weighted Loss [cite: 243]
            # lambda * CE + (1-lambda) * KD + gamma * Cal
            final_loss = (lambda_val.squeeze() * ce_loss) + \
                         ((1 - lambda_val.squeeze()) * kd_loss) + \
                         (0.5 * cal_loss) # gamma fixed at 0.5 for example
            
            final_loss = final_loss.mean()

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            total_loss += final_loss.item()
            avg_lambda += lambda_val.mean().item()
            
            pbar.set_postfix({'Loss': final_loss.item(), 'Mean λ': lambda_val.mean().item()})

        scheduler.step()
        
        # --- Evaluation ---
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = student(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}% | Avg AKTP Lambda: {avg_lambda/len(train_loader):.4f}")

        # Early stopping on test accuracy (proxy for val)
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), f"b0_aktp_{config.dataset_name}_best.pth")
            print(f"Saved best model at epoch {epoch+1} (acc={acc:.2f}%)")
            stop_now = False
        else:
            stop_now = early_stop.step(acc)
        if stop_now:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Save latest
    torch.save(student.state_dict(), f"b0_aktp_{config.dataset_name}.pth")
    print("Training Complete.")

if __name__ == "__main__":
    conf = Config()
    train_aktp(conf)