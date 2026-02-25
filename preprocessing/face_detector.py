"""
Face Detection Module using MTCNN.

Provides face detection and cropping for deepfake analysis.
Falls back to center-crop when no face is detected.
"""

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceDetector:
    """MTCNN-based face detector with fallback center-crop."""

    def __init__(self, confidence_threshold: float = 0.9, margin: int = 40):
        """
        Args:
            confidence_threshold: Minimum detection confidence to accept a face.
            margin: Pixel margin around detected face bounding box.
        """
        from facenet_pytorch import MTCNN

        self.confidence_threshold = confidence_threshold
        self.margin = margin
        self._detector = MTCNN(
            keep_all=False,
            post_process=False,
            select_largest=True,
            device="cpu",
        )

    def detect_and_crop(
        self,
        image: Image.Image,
        target_size: int = 224,
    ) -> Optional[Image.Image]:
        """
        Detect face in an image and return the cropped face region.

        Args:
            image: PIL Image (RGB).
            target_size: Output image dimensions (square).

        Returns:
            Cropped face as PIL Image, or center-cropped fallback.
        """
        try:
            boxes, probs = self._detector.detect(image)

            if boxes is not None and len(boxes) > 0:
                # Take the detection with the highest confidence
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])

                if confidence >= self.confidence_threshold:
                    box = boxes[best_idx].astype(int)
                    x1, y1, x2, y2 = box

                    # Add margin
                    w, h = image.size
                    x1 = max(0, x1 - self.margin)
                    y1 = max(0, y1 - self.margin)
                    x2 = min(w, x2 + self.margin)
                    y2 = min(h, y2 + self.margin)

                    face = image.crop((x1, y1, x2, y2))
                    face = face.resize((target_size, target_size), Image.LANCZOS)
                    return face

            logger.debug("No confident face found — using center crop.")
        except Exception as e:
            logger.warning(f"Face detection failed: {e} — using center crop.")

        return self._center_crop(image, target_size)

    @staticmethod
    def _center_crop(image: Image.Image, target_size: int) -> Image.Image:
        """Fallback: center-crop and resize the image."""
        w, h = image.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped = image.crop((left, top, left + min_dim, top + min_dim))
        return cropped.resize((target_size, target_size), Image.LANCZOS)
