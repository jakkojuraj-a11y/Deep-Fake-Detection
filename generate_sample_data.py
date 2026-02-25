"""
Generate synthetic sample data for testing the DeepFake Detection pipeline.

Creates random face-like images in data/processed/real/ and data/processed/fake/
so you can verify that training, evaluation, and inference work end-to-end.

NOTE: These are NOT real deepfake samples — they are synthetic images with
different visual patterns to simulate a real vs fake classification task.
Replace with actual data (e.g., FaceForensics++) for production use.

Usage:
    python generate_sample_data.py
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Configuration
NUM_REAL = 200
NUM_FAKE = 200
IMAGE_SIZE = 256
OUTPUT_DIR = Path("data/processed")


def generate_real_image(idx: int) -> Image.Image:
    """Generate a 'real' looking synthetic face-like image."""
    # Skin-tone base color with natural variation
    r = random.randint(180, 230)
    g = random.randint(140, 190)
    b = random.randint(110, 160)

    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (r, g, b))
    draw = ImageDraw.Draw(img)

    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2

    # Face oval
    face_w = random.randint(70, 90)
    face_h = random.randint(85, 110)
    draw.ellipse(
        [cx - face_w, cy - face_h, cx + face_w, cy + face_h],
        fill=(r - 10, g - 10, b - 10),
        outline=(r - 30, g - 30, b - 30),
    )

    # Eyes
    eye_y = cy - random.randint(15, 30)
    eye_spacing = random.randint(20, 35)
    eye_size = random.randint(6, 10)
    draw.ellipse(
        [cx - eye_spacing - eye_size, eye_y - eye_size,
         cx - eye_spacing + eye_size, eye_y + eye_size],
        fill=(255, 255, 255),
    )
    draw.ellipse(
        [cx + eye_spacing - eye_size, eye_y - eye_size,
         cx + eye_spacing + eye_size, eye_y + eye_size],
        fill=(255, 255, 255),
    )
    # Pupils
    pupil_size = eye_size // 2
    draw.ellipse(
        [cx - eye_spacing - pupil_size, eye_y - pupil_size,
         cx - eye_spacing + pupil_size, eye_y + pupil_size],
        fill=(40, 30, 20),
    )
    draw.ellipse(
        [cx + eye_spacing - pupil_size, eye_y - pupil_size,
         cx + eye_spacing + pupil_size, eye_y + pupil_size],
        fill=(40, 30, 20),
    )

    # Nose
    nose_y = cy + random.randint(5, 15)
    draw.polygon(
        [(cx, nose_y - 8), (cx - 6, nose_y + 6), (cx + 6, nose_y + 6)],
        fill=(r - 20, g - 20, b - 20),
    )

    # Mouth
    mouth_y = cy + random.randint(25, 40)
    mouth_w = random.randint(15, 25)
    draw.arc(
        [cx - mouth_w, mouth_y - 5, cx + mouth_w, mouth_y + 10],
        0, 180,
        fill=(180, 80, 80),
        width=2,
    )

    # Smooth the image slightly (natural look)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))

    return img


def generate_fake_image(idx: int) -> Image.Image:
    """Generate a 'fake' synthetic image with manipulation artifacts."""
    # Start with a similar base but add deepfake-like artifacts
    img = generate_real_image(idx)
    draw = ImageDraw.Draw(img)

    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2

    # Artifact 1: Color inconsistency / blending boundary
    if random.random() > 0.3:
        band_y = random.randint(cy - 40, cy + 40)
        band_h = random.randint(5, 20)
        band_color = (
            random.randint(150, 255),
            random.randint(100, 200),
            random.randint(80, 180),
        )
        draw.rectangle(
            [0, band_y, IMAGE_SIZE, band_y + band_h],
            fill=(*band_color, 60),
        )

    # Artifact 2: Asymmetric distortion
    if random.random() > 0.3:
        pixels = np.array(img)
        shift = random.randint(2, 8)
        half = IMAGE_SIZE // 2
        pixels[:, half:] = np.roll(pixels[:, half:], shift, axis=0)
        img = Image.fromarray(pixels)

    # Artifact 3: Noise patches
    if random.random() > 0.4:
        pixels = np.array(img)
        patch_x = random.randint(0, IMAGE_SIZE - 40)
        patch_y = random.randint(0, IMAGE_SIZE - 40)
        patch_w = random.randint(20, 40)
        patch_h = random.randint(20, 40)
        noise = np.random.randint(0, 50, (patch_h, patch_w, 3), dtype=np.uint8)
        pixels[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w] = (
            np.clip(
                pixels[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w].astype(int) + noise,
                0, 255,
            ).astype(np.uint8)
        )
        img = Image.fromarray(pixels)

    # Artifact 4: Slight blur inconsistency (one half sharper than other)
    if random.random() > 0.5:
        left = img.crop((0, 0, IMAGE_SIZE // 2, IMAGE_SIZE))
        right = img.crop((IMAGE_SIZE // 2, 0, IMAGE_SIZE, IMAGE_SIZE))
        right = right.filter(ImageFilter.GaussianBlur(radius=2.5))
        img.paste(right, (IMAGE_SIZE // 2, 0))

    # Artifact 5: Edge artifacts / grid pattern
    if random.random() > 0.5:
        draw = ImageDraw.Draw(img)
        for i in range(0, IMAGE_SIZE, random.randint(30, 60)):
            alpha = random.randint(20, 60)
            draw.line([(i, 0), (i, IMAGE_SIZE)], fill=(alpha, alpha, alpha), width=1)

    return img


def main():
    """Generate sample dataset."""
    random.seed(42)
    np.random.seed(42)

    real_dir = OUTPUT_DIR / "real"
    fake_dir = OUTPUT_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {NUM_REAL} real images...")
    for i in range(NUM_REAL):
        img = generate_real_image(i)
        img.save(real_dir / f"real_{i:04d}.jpg", quality=95)
        if (i + 1) % 50 == 0:
            print(f"  Real: {i + 1}/{NUM_REAL}")

    print(f"Generating {NUM_FAKE} fake images...")
    for i in range(NUM_FAKE):
        img = generate_fake_image(i)
        img.save(fake_dir / f"fake_{i:04d}.jpg", quality=95)
        if (i + 1) % 50 == 0:
            print(f"  Fake: {i + 1}/{NUM_FAKE}")

    print(f"\n✅ Done! Generated {NUM_REAL + NUM_FAKE} images:")
    print(f"   Real: {real_dir}")
    print(f"   Fake: {fake_dir}")
    print(f"\nNow run: python -m training.train --data_dir data/processed")


if __name__ == "__main__":
    main()
