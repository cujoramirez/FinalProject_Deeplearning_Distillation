"""
AKTP ensemble distillation on TinyImageNet (EffNet-B2 + ResNet18 -> EffNet-B0).

Workflow
1) Fine-tune both teachers (EffNet-B2, ResNet18) on TinyImageNet starting from
   ImageNet-1k weights. Checkpoints are saved; if they already exist they are
   loaded instead of retraining.
2) (Optional) Pretrain the student on TinyImageNet with cross-entropy.
3) Distill from the two TinyImageNet-finetuned teachers into the student using
   the AKTP weight scheduler and logit fusion combiner.

This script is self-contained for TinyImageNet. Place the dataset at
`./data/tiny-imagenet-200` (train/val structure as standard). Adjust paths in
Config if needed.
"""

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet18_Weights,
 )
from tqdm import tqdm
class TinyImageNetValDataset(Dataset):
    """TinyImageNet validation dataset using val_annotations.txt labels."""

    def __init__(self, root: str, transform=None, class_to_idx=None):
        super().__init__()
        self.root = root
        self.transform = transform
        annotations = os.path.join(root, "val_annotations.txt")
        images_dir = os.path.join(root, "images")

        self.samples = []
        # Ensure class indices align with train set if provided
        self.class_to_idx = class_to_idx if class_to_idx is not None else {}

        with open(annotations, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img, cls = parts[0], parts[1]
                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = len(self.class_to_idx)
                self.samples.append((os.path.join(images_dir, img), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

@dataclass
class Config:
    # Paths
    data_path: str = "./data"
    checkpoints_dir: str = "./checkpoints_aktp"
    logs_dir: str = "./results_aktp"

    # Dataset
    dataset_name: str = "TinyImageNet"
    num_classes: int = 200
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 4

    # Teacher fine-tuning
    teacher_epochs: int = 25
    teacher_lr: float = 5e-4
    teacher_weight_decay: float = 1e-4
    teacher_early_stop: int = 5
    teacher_b2_ckpt: str = "./checkpoints_aktp/teacher_b2_tiny.pth"
    teacher_r18_ckpt: str = "./checkpoints_aktp/teacher_r18_tiny.pth"
    train_teachers_if_missing: bool = True

    # Student pretrain
    student_pretrain_epochs: int = 0  # not used when distilling from scratch
    student_pretrain_lr: float = 5e-4
    student_pretrain_ckpt: str = "./checkpoints_aktp/student_b0_tiny_pretrain.pth"  # optional load
    pretrain_student_if_missing: bool = False  # keep False to distill from ImageNet init

    # Distillation
    distill_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    temperature: float = 4.0
    gamma_cal: float = 0.5

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def serialize_cfg(cfg):
    return {k: (str(v) if isinstance(v, torch.device) else v) for k, v in cfg.__dict__.items()}

def ensure_tiny_imagenet(cfg: Config):
    data_root = os.path.join(cfg.data_path, "tiny-imagenet-200")
    if os.path.isdir(data_root):
        print(f"TinyImageNet found at {data_root}")
        return data_root
    os.makedirs(cfg.data_path, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(cfg.data_path, "tiny-imagenet-200.zip")
    if not os.path.isfile(zip_path):
        print("Downloading TinyImageNet (~240MB)...")
        urllib.request.urlretrieve(url, zip_path)
    print("Extracting TinyImageNet...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cfg.data_path)
    print("Done extracting.")
    return data_root

# --- 1. Interchangeable Dataset Wrapper ---
def get_tinyimagenet_loaders(config: Config):
    """TinyImageNet train/val loaders using official train/val split."""
    data_root = os.path.join(config.data_path, "tiny-imagenet-200")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"TinyImageNet not found at {data_root}. Download and extract to this path."
        )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = TinyImageNetValDataset(val_dir, transform=val_tf, class_to_idx=train_set.class_to_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def split_train_val(train_loader, val_ratio=0.1):
    """Utility to create an explicit val split from train if desired."""
    dataset = train_loader.dataset
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset

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

def build_effnet_b2(num_classes: int):
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_resnet18(num_classes: int):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_effnet_b0(num_classes: int):
    # Student starts from scratch (no ImageNet weights) as requested
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_classifier(model: nn.Module, train_loader, val_loader, config: Config, epochs: int, lr: float, weight_decay: float, patience: int, device: torch.device, tag: str, save_path: str):
    """Standard CE training loop with early stopping; saves best checkpoint."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    stopper = EarlyStopping(patience=patience)

    best_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"{tag} Train E{epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total
        print(f"{tag} Epoch {epoch+1}: train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved new best {tag} to {save_path} (val_acc={val_acc:.2f}%)")
            stop_now = False
        else:
            stop_now = stopper.step(val_acc)
        if stop_now:
            print(f"Early stopping {tag} at epoch {epoch+1}")
            break

    return save_path


def load_or_train_teachers(train_loader, val_loader, cfg: Config):
    """Ensure TinyImageNet-finetuned teachers are available."""
    b2_path = cfg.teacher_b2_ckpt
    r18_path = cfg.teacher_r18_ckpt
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)

    need_b2 = not os.path.isfile(b2_path)
    need_r18 = not os.path.isfile(r18_path)

    if (need_b2 or need_r18) and not cfg.train_teachers_if_missing:
        missing = ["B2" if need_b2 else None, "R18" if need_r18 else None]
        missing = [m for m in missing if m]
        raise FileNotFoundError(f"Missing teacher checkpoints: {missing}. Enable training or provide paths.")

    if need_b2:
        print("Training teacher EfficientNet-B2 on TinyImageNet...")
        model = build_effnet_b2(cfg.num_classes)
        train_classifier(model, train_loader, val_loader, cfg, cfg.teacher_epochs, cfg.teacher_lr, cfg.teacher_weight_decay, cfg.teacher_early_stop, cfg.device, "Teacher-B2", b2_path)
    if need_r18:
        print("Training teacher ResNet18 on TinyImageNet...")
        model = build_resnet18(cfg.num_classes)
        train_classifier(model, train_loader, val_loader, cfg, cfg.teacher_epochs, cfg.teacher_lr, cfg.teacher_weight_decay, cfg.teacher_early_stop, cfg.device, "Teacher-R18", r18_path)

    # Load teachers
    b2 = build_effnet_b2(cfg.num_classes)
    b2.load_state_dict(torch.load(b2_path, map_location=cfg.device))
    b2.to(cfg.device)
    b2.eval()
    for p in b2.parameters():
        p.requires_grad = False

    r18 = build_resnet18(cfg.num_classes)
    r18.load_state_dict(torch.load(r18_path, map_location=cfg.device))
    r18.to(cfg.device)
    r18.eval()
    for p in r18.parameters():
        p.requires_grad = False

    return b2, r18


def load_or_pretrain_student(train_loader, val_loader, cfg: Config):
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    path = cfg.student_pretrain_ckpt

    # If a checkpoint exists, load it; otherwise start from scratch (no ImageNet-1k weights).
    if os.path.isfile(path):
        print(f"Loading existing student checkpoint from {path}")
        model = build_effnet_b0(cfg.num_classes)
        model.load_state_dict(torch.load(path, map_location=cfg.device))
    else:
        print("No student checkpoint found; starting student from scratch (no ImageNet-1k weights) and distilling with AKTP.")
        model = build_effnet_b0(cfg.num_classes)

    model.to(cfg.device)
    return model


def distill_with_aktp(train_loader, val_loader, teachers, student, cfg: Config):
    t1, t2 = teachers
    combiner = CombinerNetwork(num_teachers=2, num_classes=cfg.num_classes).to(cfg.device)
    aktp_module = AKTP().to(cfg.device)

    optimizer = optim.AdamW([
        {"params": student.parameters(), "lr": cfg.lr},
        {"params": combiner.parameters(), "lr": cfg.lr},
        {"params": aktp_module.parameters(), "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.distill_epochs)
    stopper = EarlyStopping(patience=cfg.early_stopping_patience)

    best_acc = 0.0
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoints_dir, "b0_aktp_tiny_best.pth")

    for epoch in range(cfg.distill_epochs):
        student.train(); combiner.train(); aktp_module.train()
        total_loss = 0.0
        avg_lambda = 0.0
        pbar = tqdm(train_loader, desc=f"AKTP Distill E{epoch+1}/{cfg.distill_epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            with torch.no_grad():
                l_t1 = t1(inputs)
                l_t2 = t2(inputs)
            fused_logits = combiner([l_t1, l_t2])
            p_comb = F.softmax(fused_logits / cfg.temperature, dim=1)

            l_student = student(inputs)
            lambda_val = aktp_module(l_student, l_t1, l_t2)

            ce_loss = F.cross_entropy(l_student, targets, reduction="none")
            log_prob_student = F.log_softmax(l_student / cfg.temperature, dim=1)
            # Standard KD scaling multiplies by T^2 to keep gradient magnitudes stable
            kd_loss = F.kl_div(log_prob_student, p_comb, reduction="none").sum(dim=1) * (cfg.temperature ** 2)
            conf, pred = torch.max(F.softmax(l_student, dim=1), 1)
            acc = (pred == targets).float()
            cal_loss = (conf - acc) ** 2

            final_loss = (lambda_val.squeeze() * ce_loss) + ((1 - lambda_val.squeeze()) * kd_loss) + (cfg.gamma_cal * cal_loss)
            final_loss = final_loss.mean()

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            total_loss += final_loss.item()
            avg_lambda += lambda_val.mean().item()
            pbar.set_postfix({"loss": final_loss.item(), "mean_lambda": lambda_val.mean().item()})

        scheduler.step()

        # Validation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                outputs = student(inputs)
                pred = outputs.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
        acc = 100.0 * correct / total
        mean_lambda = avg_lambda / max(len(train_loader), 1)
        print(f"Epoch {epoch+1}: val_acc={acc:.2f}% mean_lambda={mean_lambda:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), best_path)
            print(f"  Saved best distilled student at {best_path} (acc={acc:.2f}%)")
            stop_now = False
        else:
            stop_now = stopper.step(acc)
        if stop_now:
            print(f"Early stopping distillation at epoch {epoch+1}")
            break

    latest_path = os.path.join(cfg.checkpoints_dir, "b0_aktp_tiny_latest.pth")
    torch.save(student.state_dict(), latest_path)
    return best_path, latest_path, best_acc


def main():
    cfg = Config()
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(cfg.data_path, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(cfg.logs_dir, "aktp_tiny_config.json"), "w") as f:
        json.dump(serialize_cfg(cfg), f, indent=2)

    ensure_tiny_imagenet(cfg)

    print(f"Using device: {cfg.device}")
    train_loader, val_loader = get_tinyimagenet_loaders(cfg)

    # Stage 1: Teachers
    t1, t2 = load_or_train_teachers(train_loader, val_loader, cfg)
    print("Teachers ready (TinyImageNet-finetuned).")

    # Stage 2: Student pretrain (optional)
    student = load_or_pretrain_student(train_loader, val_loader, cfg)

    # Stage 3: AKTP distillation
    best_path, latest_path, best_acc = distill_with_aktp(train_loader, val_loader, (t1, t2), student, cfg)
    print(f"Distillation complete. Best val acc: {best_acc:.2f}%")
    print(f"Best student checkpoint: {best_path}")
    print(f"Latest student checkpoint: {latest_path}")


if __name__ == "__main__":
    main()