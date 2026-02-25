"""Setup script for DeepFake Detection System."""

from setuptools import setup, find_packages

setup(
    name="deepfake-detector",
    version="1.0.0",
    description="Production-ready Deep Fake Detection System using EfficientNet-B4",
    author="Deep Fake Detection Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "facenet-pytorch>=2.5.3",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
    ],
    entry_points={
        "console_scripts": [
            "deepfake-train=training.train:main",
            "deepfake-eval=training.evaluate:main",
        ],
    },
)
