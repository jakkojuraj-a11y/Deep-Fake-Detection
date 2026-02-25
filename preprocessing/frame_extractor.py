"""
Video Frame Extraction Module.

Extracts frames from video files at configurable intervals
using OpenCV for downstream face detection and classification.
"""

import logging
from pathlib import Path
from typing import List

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video files using OpenCV."""

    def __init__(self, frame_interval: int = 10, max_frames: int = 50):
        """
        Args:
            frame_interval: Extract every Nth frame.
            max_frames: Maximum number of frames to extract per video.
        """
        self.frame_interval = frame_interval
        self.max_frames = max_frames

    def extract_frames(self, video_path: str) -> List[Image.Image]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file (mp4, avi, mov, etc.).

        Returns:
            List of PIL Image objects (RGB).

        Raises:
            ValueError: If the video cannot be opened.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames: List[Image.Image] = []
        frame_count = 0

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(
                f"Video: {Path(video_path).name} | "
                f"Total frames: {total_frames} | FPS: {fps:.1f}"
            )

            while cap.isOpened() and len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_interval == 0:
                    # Convert BGR (OpenCV) → RGB (PIL)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    frames.append(pil_image)

                frame_count += 1

        finally:
            cap.release()

        logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
        return frames

    def extract_frames_from_bytes(self, video_bytes: bytes) -> List[Image.Image]:
        """
        Extract frames from in-memory video bytes (for Streamlit uploads).

        Args:
            video_bytes: Raw video file bytes.

        Returns:
            List of PIL Image objects (RGB).
        """
        import tempfile
        import os

        # Write bytes to a temp file and process
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            return self.extract_frames(tmp_path)
        finally:
            os.unlink(tmp_path)
