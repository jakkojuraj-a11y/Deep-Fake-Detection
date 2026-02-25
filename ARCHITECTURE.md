# 🏗️ Architecture & Design Decisions

## Why EfficientNet-B4?

We evaluated three leading architectures for deepfake detection:

| Criteria | EfficientNet-B4 ✅ | ResNet-50 | Xception |
|---|---|---|---|
| FaceForensics++ Accuracy | **~96%** | ~92% | ~95% |
| Parameters | **19M** | 25M | 23M |
| Inference Speed | Fast | Fastest | Medium |
| Compound Scaling | ✅ Yes | ❌ No | ❌ No |
| ImageNet Top-1 | **83.4%** | 76.1% | 79.0% |

**EfficientNet-B4 was chosen because it:**

1. **Compound Scaling** — Simultaneously scales depth, width, and resolution using a principled coefficient, unlike ResNet (depth-only) or Xception (width-only). This leads to better feature extraction from facial manipulation artifacts.

2. **Best Accuracy-to-Compute Ratio** — Achieves higher accuracy than Xception and ResNet with fewer parameters and FLOPs.

3. **Strong Transfer Learning** — The ImageNet-pretrained features transfer exceptionally well to face-forensics tasks because facial manipulation artifacts (texture inconsistencies, blending boundaries) are captured by mid-level features that EfficientNet learns efficiently.

4. **Production Viable** — 19M parameters fits comfortably in memory for both GPU and CPU inference, making it suitable for deployment.

---

## How Overfitting Is Handled

Deepfake detection is prone to overfitting because datasets are often small and models can memorize compression artifacts. We combat this with **six complementary strategies**:

### 1. Aggressive Data Augmentation
```
RandomHorizontalFlip(p=0.5)
RandomRotation(±15°)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
RandomAffine(translate=5%)
GaussianBlur(kernel=3, sigma=0.1-2.0)
RandomErasing(p=0.1)
```
These force the model to learn semantic manipulation features rather than memorizing pixel patterns.

### 2. Dropout Regularization
- **0.5 dropout** before the first fully-connected layer
- **0.3 dropout** before the final classification layer
- Combined dropout probability prevents co-adaptation of neurons

### 3. Early Stopping (patience=5)
Training halts after 5 epochs without validation loss improvement, preventing the model from memorizing training data.

### 4. Learning Rate Scheduling
`ReduceLROnPlateau` with patience=3 and factor=0.5. When validation loss plateaus, the learning rate is halved to enable finer convergence.

### 5. L2 Weight Decay (1e-4)
AdamW optimizer applies L2 regularization to prevent weight magnitudes from growing too large.

### 6. Batch Normalization
Added after the hidden layer in the classification head to stabilize training and act as a mild regularizer.

---

## How to Scale This System

### Horizontal Scaling (More Users)

```
                    ┌──────────────┐
                    │   Load       │
                    │  Balancer    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴─────┐ ┌───┴─────┐
        │ Instance 1 │ │ Inst. 2 │ │ Inst. 3 │
        │ (GPU/CPU)  │ │ (GPU)   │ │ (GPU)   │
        └────────────┘ └─────────┘ └─────────┘
```

1. **Containerize** with Docker (already provided)
2. **Deploy** on Kubernetes/ECS with auto-scaling
3. **GPU inference** with NVIDIA Triton or TorchServe for batch processing
4. **Async processing** — Use Celery + Redis for video analysis queues

### Vertical Scaling (Better Accuracy)

1. **Larger Models** — Upgrade to EfficientNet-B7 or EfficientNet-V2
2. **Ensemble Methods** — Combine multiple architectures (EfficientNet + Xception + ResNet)
3. **Attention Mechanisms** — Add SE blocks or attention modules to focus on manipulation regions
4. **Multi-task Learning** — Train to detect specific manipulation types alongside binary classification
5. **Temporal Analysis** — For video, add LSTM/Transformer layers to capture temporal inconsistencies

### Data Scaling

1. **More Datasets** — Combine FaceForensics++, Celeb-DF, and DFDC
2. **Cross-dataset Training** — Improves generalization to unseen manipulation techniques
3. **Hard Negative Mining** — Focus training on difficult-to-detect manipulations
4. **Synthetic Augmentation** — Use GANs to generate additional training samples

### Production Checklist

- [ ] Model versioning (MLflow / DVC)
- [ ] A/B testing for model updates
- [ ] Monitoring & alerting (prediction drift)
- [ ] API rate limiting
- [ ] Input validation & sanitization
- [ ] Logging & audit trail
- [ ] ONNX export for cross-platform deployment

---

## Pipeline Overview

```
Input (Image/Video)
       │
       ▼
┌──────────────┐
│ Frame Extract │ ← (Video only: extract every Nth frame)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ MTCNN Face   │ ← Detect & crop face region
│ Detection    │   (fallback: center-crop)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocessing│ ← Resize 224×224, normalize (ImageNet stats)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ EfficientNet │ ← Feature extraction (1792-dim)
│ B4 Backbone  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Classifier   │ ← Dropout → 512 → ReLU → BN → Dropout → 2
│ Head         │
└──────┬───────┘
       │
       ▼
   REAL / FAKE
  (+ confidence)
```
