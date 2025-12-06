"""
Data Loading Pipeline for Knowledge Distillation Experiment
Supports: CIFAR-100, Flowers-102, Food-101
"""

import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from PIL import Image
from PIL import Image

import config


def get_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get data transforms for training and validation.
    
    Args:
        image_size: Target image size (EfficientNet uses 224)
        is_training: Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training:
        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    
    return transforms.Compose(transform_list)


def get_cifar100_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """
    Load CIFAR-100 dataset.
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    train_transform = get_transforms(config.IMAGE_SIZE, is_training=True)
    test_transform = get_transforms(config.IMAGE_SIZE, is_training=False)
    
    # Download and load training data
    full_train_dataset = datasets.CIFAR100(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split training into train and validation (90-10 split)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Validation uses test transforms (no augmentation)
    val_dataset.dataset.transform = test_transform
    
    # Test dataset
    test_dataset = datasets.CIFAR100(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"CIFAR-100 loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: 100")
    
    return train_loader, val_loader, test_loader, 100


def get_flowers102_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """
    Load Oxford Flowers-102 dataset.
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    train_transform = get_transforms(config.IMAGE_SIZE, is_training=True)
    test_transform = get_transforms(config.IMAGE_SIZE, is_training=False)
    
    train_dataset = datasets.Flowers102(
        root=config.DATA_DIR,
        split='train',
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.Flowers102(
        root=config.DATA_DIR,
        split='val',
        download=True,
        transform=test_transform
    )
    
    test_dataset = datasets.Flowers102(
        root=config.DATA_DIR,
        split='test',
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Flowers-102 loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: 102")
    
    return train_loader, val_loader, test_loader, 102


def get_food101_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """
    Load Food-101 dataset.
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    train_transform = get_transforms(config.IMAGE_SIZE, is_training=True)
    test_transform = get_transforms(config.IMAGE_SIZE, is_training=False)
    
    full_train_dataset = datasets.Food101(
        root=config.DATA_DIR,
        split='train',
        download=True,
        transform=train_transform
    )
    
    # Split training into train and validation
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    test_dataset = datasets.Food101(
        root=config.DATA_DIR,
        split='test',
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Food-101 loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: 101")
    
    return train_loader, val_loader, test_loader, 101


class TinyImageNetValDataset(Dataset):
    """TinyImageNet validation dataset using val_annotations.txt labels."""

    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        annotations = os.path.join(root, "val_annotations.txt")
        images_dir = os.path.join(root, "images")

        # Load label mapping
        self.samples = []
        self.class_to_idx = {}

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


def get_tinyimagenet_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """Load TinyImageNet dataset (expects tiny-imagenet-200 folder inside DATA_DIR)."""
    data_root = os.path.join(config.DATA_DIR, "tiny-imagenet-200")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"TinyImageNet not found at {data_root}. Download from https://www.kaggle.com/c/tiny-imagenet/data or "
            "http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract to data/tiny-imagenet-200/."
        )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = TinyImageNetValDataset(val_dir, transform=val_transform)

    # For simplicity, use val split as both val and test
    test_dataset = val_dataset

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"TinyImageNet loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: 200")

    return train_loader, val_loader, test_loader, 200


def get_dataloaders(dataset_name: str = None, batch_size: int = None, num_workers: int = 4):
    """
    Get data loaders for the specified dataset.
    
    Args:
        dataset_name: Name of dataset ("CIFAR100", "Flowers102", "Food101")
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    if dataset_name is None:
        dataset_name = config.DATASET_NAME
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    dataset_loaders = {
        "CIFAR100": get_cifar100_dataloaders,
        "Flowers102": get_flowers102_dataloaders,
        "Food101": get_food101_dataloaders,
        "TinyImageNet": get_tinyimagenet_dataloaders,
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_loaders.keys())}")
    
    return dataset_loaders[dataset_name](batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading pipeline...")
    print("=" * 50)
    
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
