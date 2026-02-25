"""
EfficientNet-B4 based DeepFake Detector.

Uses transfer learning from ImageNet-pretrained EfficientNet-B4
with a custom classification head optimized for binary deepfake detection.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class DeepFakeDetector(nn.Module):
    """
    EfficientNet-B4 based binary classifier for deepfake detection.

    Architecture:
        - EfficientNet-B4 backbone (pretrained on ImageNet)
        - Custom head: Dropout → Linear(1792, 512) → ReLU → Dropout → Linear(512, 2)

    Why EfficientNet-B4?
        1. Compound scaling (depth + width + resolution) — best accuracy/compute ratio
        2. Achieves ~96% on FaceForensics++ benchmark
        3. Only 19M parameters (vs 25M ResNet-50, 23M Xception)
        4. Excellent transfer-learning performance on face-forensics tasks
    """

    CLASS_NAMES = ("REAL", "FAKE")

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        dropout_rate_fc: float = 0.3,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
        )

        # Get feature dimension from backbone
        in_features = self.backbone.num_features  # 1792 for efficientnet_b4

        # Custom classification head with regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate_fc),
            nn.Linear(hidden_dim, num_classes),
        )

        logger.info(
            f"Initialized {model_name} | "
            f"Features: {in_features} → {hidden_dim} → {num_classes} | "
            f"Pretrained: {pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone features → classifier logits."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning (train head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only classifier head will be trained.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — full model will be trained.")

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def predict_image(
        self,
        image: Image.Image,
        transform: Optional[transforms.Compose] = None,
        device: str = "cpu",
    ) -> Tuple[str, float, dict]:
        """
        Predict whether a single image is REAL or FAKE.

        Args:
            image: PIL Image (RGB).
            transform: Preprocessing transform pipeline.
            device: Device string ('cpu', 'cuda', 'mps').

        Returns:
            Tuple of (prediction_label, confidence, class_probabilities).
        """
        self.eval()

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        tensor = transform(image).unsqueeze(0).to(device)
        logits = self(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        label = self.CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        class_probs = {
            self.CLASS_NAMES[i]: float(probs[i]) for i in range(self.num_classes)
        }

        return label, confidence, class_probs

    @torch.no_grad()
    def predict_video(
        self,
        frames: List[Image.Image],
        transform: Optional[transforms.Compose] = None,
        device: str = "cpu",
    ) -> Tuple[str, float, List[dict]]:
        """
        Predict whether a video is REAL or FAKE using extracted frames.

        Classifies each frame independently and aggregates results
        using average probability pooling.

        Args:
            frames: List of PIL Image frames (RGB).
            transform: Preprocessing transform pipeline.
            device: Device string.

        Returns:
            Tuple of (overall_label, overall_confidence, per_frame_results).
        """
        self.eval()

        if not frames:
            return "UNKNOWN", 0.0, []

        per_frame_results = []
        all_probs = []

        for i, frame in enumerate(frames):
            label, conf, probs = self.predict_image(frame, transform, device)
            per_frame_results.append({
                "frame_idx": i,
                "label": label,
                "confidence": conf,
                "probabilities": probs,
            })
            all_probs.append([probs["REAL"], probs["FAKE"]])

        # Average probability pooling across all frames
        avg_probs = np.mean(all_probs, axis=0)
        overall_idx = int(np.argmax(avg_probs))
        overall_label = self.CLASS_NAMES[overall_idx]
        overall_confidence = float(avg_probs[overall_idx])

        return overall_label, overall_confidence, per_frame_results

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DeepFakeDetector":
        """
        Load a model from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            device: Device to load model on.
            **kwargs: Override model constructor arguments.

        Returns:
            Loaded DeepFakeDetector model.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract model config from checkpoint if available
        model_config = checkpoint.get("model_config", {})
        model_config.update(kwargs)
        model_config["pretrained"] = False  # Don't re-download weights

        model = cls(**model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return model
