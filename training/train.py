"""
Training Pipeline for the DeepFake Detector.

Features:
    - Mixed-precision training (AMP) for faster GPU training
    - Gradient clipping for stable optimization
    - Early stopping to prevent overfitting
    - ReduceLROnPlateau learning-rate scheduling
    - Best-model checkpointing on validation loss
    - Comprehensive epoch-level logging

Usage:
    python -m training.train --data_dir data/processed
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from models.detector import DeepFakeDetector
from preprocessing.data_pipeline import create_data_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement."
                )
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: GradScaler,
    gradient_clip: float = 1.0,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass
        with autocast(device_type=device, enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """
    Validate the model.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(config: Config, data_dir: str = None) -> str:
    """
    Full training pipeline.

    Args:
        config: Configuration object.
        data_dir: Override data directory.

    Returns:
        Path to best model checkpoint.
    """
    data_path = data_dir or str(config.processed_data_dir)
    device = config.device
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {data_path}")

    # ── Data ───────────────────────────────────────────────────────
    loaders = create_data_loaders(
        data_dir=data_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        train_split=config.train_split,
        val_split=config.val_split,
        random_seed=config.random_seed,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(
        f"Data — Train: {len(loaders['train'].dataset)}, "
        f"Val: {len(loaders['val'].dataset)}, "
        f"Test: {len(loaders['test'].dataset)}"
    )

    # ── Model ──────────────────────────────────────────────────────
    model = DeepFakeDetector(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        dropout_rate_fc=config.dropout_rate_fc,
        hidden_dim=config.hidden_dim,
    ).to(device)

    logger.info(
        f"Model: {config.model_name} | "
        f"Trainable: {model.get_trainable_params():,} / "
        f"Total: {model.get_total_params():,} params"
    )

    # ── Training Setup ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.lr_patience,
        factor=config.lr_factor,
    )
    scaler = GradScaler(enabled=(device == "cuda"))
    early_stopping = EarlyStopping(patience=config.patience)

    best_val_loss = float("inf")
    save_path = str(config.model_save_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ──────────────────────────────────────────────
    logger.info(f"Starting training for {config.num_epochs} epochs...")
    logger.info("=" * 80)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer,
            device, scaler, config.gradient_clip_value,
        )

        # Validate
        val_loss, val_acc = validate(model, loaders["val"], criterion, device)

        # LR scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch:02d}/{config.num_epochs}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_config": {
                    "model_name": config.model_name,
                    "num_classes": config.num_classes,
                    "dropout_rate": config.dropout_rate,
                    "dropout_rate_fc": config.dropout_rate_fc,
                    "hidden_dim": config.hidden_dim,
                },
            }
            torch.save(checkpoint, save_path)
            logger.info(f"  ✓ Best model saved → {save_path}")

        # Early stopping check
        if early_stopping(val_loss):
            break

    logger.info("=" * 80)
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint: {save_path}")

    return save_path


def main():
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser(description="Train DeepFake Detector")
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to data directory (default: config.processed_data_dir)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    config = Config()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    train(config, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
