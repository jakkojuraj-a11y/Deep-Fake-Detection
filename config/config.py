"""
Centralized configuration for the Deep Fake Detection System.

All hyperparameters, paths, and settings are defined here
to ensure reproducibility and easy experimentation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Master configuration for the deepfake detection pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def model_save_path(self) -> Path:
        return self.output_dir / "best_model.pth"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    # ── Data ───────────────────────────────────────────────────────────
    image_size: int = 224
    num_classes: int = 2
    class_names: tuple = ("REAL", "FAKE")
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "efficientnet_b4"
    pretrained: bool = True
    dropout_rate: float = 0.5
    dropout_rate_fc: float = 0.3
    hidden_dim: int = 512

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5          # Early-stopping patience
    lr_patience: int = 3       # ReduceLROnPlateau patience
    lr_factor: float = 0.5     # ReduceLROnPlateau factor
    gradient_clip_value: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True

    # ── Video Processing ───────────────────────────────────────────────
    frame_interval: int = 10   # Extract every Nth frame
    max_frames: int = 50       # Max frames per video
    face_confidence_threshold: float = 0.9

    # ── Device ─────────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.output_dir,
            self.metrics_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        self.ensure_dirs()
