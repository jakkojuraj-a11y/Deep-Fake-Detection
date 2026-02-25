"""
Model Evaluation Module.

Computes comprehensive classification metrics and generates
publication-quality visualization plots for the deepfake detector.

Metrics:
    - Accuracy, Precision, Recall, F1-Score
    - ROC-AUC with curve plot
    - Confusion Matrix with heatmap

Usage:
    python -m training.evaluate --checkpoint outputs/best_model.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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


def collect_predictions(
    model: DeepFakeDetector,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> tuple:
    """
    Run inference on a DataLoader and collect all predictions.

    Returns:
        Tuple of (all_labels, all_predictions, all_probabilities).
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])  # Probability of FAKE class

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "f1_score": float(f1_score(y_true, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple,
    save_path: str,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix — DeepFake Detector", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    save_path: str,
) -> None:
    """Generate and save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — DeepFake Detector", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved → {save_path}")


def evaluate(
    config: Config,
    checkpoint_path: str = None,
    data_dir: str = None,
) -> dict:
    """
    Full evaluation pipeline.

    Args:
        config: Configuration object.
        checkpoint_path: Path to model checkpoint.
        data_dir: Path to data directory.

    Returns:
        Dictionary of evaluation metrics.
    """
    ckpt_path = checkpoint_path or str(config.model_save_path)
    data_path = data_dir or str(config.processed_data_dir)
    device = config.device

    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Data: {data_path}")

    # Load model
    model = DeepFakeDetector.load_from_checkpoint(ckpt_path, device=device)
    model.to(device)

    # Load test data
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

    logger.info(f"Test set size: {len(loaders['test'].dataset)}")

    # Collect predictions
    y_true, y_pred, y_prob = collect_predictions(model, loaders["test"], device)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for name, value in metrics.items():
        logger.info(f"  {name:>12s}: {value:.4f}")
    logger.info("=" * 60)

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=list(config.class_names),
        digits=4,
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Save plots
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=config.class_names,
        save_path=str(config.metrics_dir / "confusion_matrix.png"),
    )

    plot_roc_curve(
        y_true, y_prob,
        roc_auc=metrics["roc_auc"],
        save_path=str(config.metrics_dir / "roc_curve.png"),
    )

    # Save metrics JSON
    metrics_path = config.metrics_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    return metrics


def main():
    """CLI entrypoint for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DeepFake Detector")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: outputs/best_model.pth)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to data directory",
    )
    args = parser.parse_args()

    config = Config()
    evaluate(config, checkpoint_path=args.checkpoint, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
