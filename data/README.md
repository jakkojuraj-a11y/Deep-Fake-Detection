# ── Data Directory ─────────────────────────────────────────────────
# This directory should contain the dataset for training.
#
# Expected Structure:
#   data/
#   ├── raw/           ← Original collected data
#   └── processed/     ← Training-ready images
#       ├── real/      ← Authentic images
#       │   ├── img001.jpg
#       │   └── ...
#       └── fake/      ← Deepfake images
#           ├── img001.jpg
#           └── ...
#
# Supported Datasets:
#   - FaceForensics++ (https://github.com/ondyari/FaceForensics)
#   - Celeb-DF (https://github.com/yuezunli/celeb-deepfakeforensics)
#   - DFDC (https://ai.facebook.com/datasets/dfdc/)
#   - DeeperForensics-1.0
#
# Preparation Steps:
#   1. Download a deepfake dataset
#   2. Extract face crops (or use our face detector)
#   3. Place real images in data/processed/real/
#   4. Place fake images in data/processed/fake/
#   5. Run training: python -m training.train
