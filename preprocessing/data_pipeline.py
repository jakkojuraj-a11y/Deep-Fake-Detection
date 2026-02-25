"""
Data Pipeline Module.

Provides PyTorch Dataset classes and data loaders for deepfake
detection training with proper augmentation and stratified splitting.
"""

import logging
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DeepFakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.

    Expected directory structure:
        data_dir/
        ├── real/       # Authentic images
        │   ├── img001.jpg
        │   └── ...
        └── fake/       # Deepfake images
            ├── img001.jpg
            └── ...
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir: Root directory containing 'real/' and 'fake/' subdirs.
            transform: Torchvision transforms to apply.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        # Load real images (label=0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img_path in sorted(real_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.image_paths.append(img_path)
                    self.labels.append(0)

        # Load fake images (label=1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in sorted(fake_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.image_paths.append(img_path)
                    self.labels.append(1)

        logger.info(
            f"Loaded {len(self.image_paths)} images "
            f"(Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TransformedSubset(Dataset):
    """Wrapper to apply different transforms to a subset of a dataset."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = self.dataset.image_paths[self.indices[idx]]
        label = self.dataset.labels[self.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Build torchvision transforms for training or evaluation.

    Training includes aggressive augmentations to combat overfitting.
    Validation/test uses only resize + normalize.

    Args:
        image_size: Target image dimension (square).
        is_training: Whether to include augmentations.

    Returns:
        A composed transform pipeline.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Transform for single-image inference (no augmentation)."""
    return get_transforms(image_size=image_size, is_training=False)


def create_data_loaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    train_split: float = 0.70,
    val_split: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0 if platform.system() == "Windows" else 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create stratified train/val/test data loaders.

    Args:
        data_dir: Path to data directory.
        image_size: Target image size.
        batch_size: Batch size.
        train_split: Proportion for training set.
        val_split: Proportion for validation set.
        random_seed: Seed for reproducibility.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for CUDA transfer.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    # Build full dataset with validation transforms first (for splitting)
    full_dataset = DeepFakeDataset(
        data_dir=data_dir,
        transform=None,  # Transforms applied per-subset
    )

    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}/real/ or {data_dir}/fake/")

    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_split),
        stratify=labels,
        random_state=random_seed,
    )

    # Second split: val vs test (from the remaining)
    temp_labels = [labels[i] for i in temp_idx]
    relative_val = val_split / (val_split + (1.0 - train_split - val_split))
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - relative_val),
        stratify=temp_labels,
        random_state=random_seed,
    )

    logger.info(
        f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    # Create subsets with appropriate transforms
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)

    train_dataset = TransformedSubset(full_dataset, train_idx, train_transform)
    val_dataset = TransformedSubset(full_dataset, val_idx, val_transform)
    test_dataset = TransformedSubset(full_dataset, test_idx, val_transform)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders
