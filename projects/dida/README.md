# DIDA: Domain-Incremental Detection with Anchors

A comprehensive framework for domain-incremental object detection based on MMDetection 3.x, supporting strict no-replay learning with fixed class sets.

## Overview

DIDA enables object detectors to incrementally adapt to new visual domains (e.g., from real photos to clipart, watercolor, comic) without catastrophic forgetting, using:

- **CLIP Text Prototypes**: Language-guided classification with domain-specific residuals
- **Domain Residuals**: Learnable small perturbations Δ_{c,d} for each class and domain
- **Style Router**: Gram matrix-based domain identification for domain-unknown inference
- **Objectness Heads**: Class-agnostic localization signal to improve proposal quality
- **Incremental Freezing**: Strategic parameter freezing to preserve knowledge

## Architecture

```
Input Image
     ↓
Backbone (ResNet50) → frozen after stage 0
     ↓
Neck (FPN) → frozen after stage 0
     ↓
Transformer Encoder/Decoder → frozen after stage 0
     ↓
┌─────────────────────────────────┐
│   DIDA Head                     │
│                                 │
│  ┌──────────────────────┐      │
│  │ CLIP Classifier      │      │
│  │  - Base projection   │      │
│  │  - Domain residuals  │      │
│  │  - Cosine similarity │      │
│  └──────────────────────┘      │
│                                 │
│  ┌──────────────────────┐      │
│  │ Objectness Head      │      │
│  │  - Foreground/BG     │      │
│  └──────────────────────┘      │
│                                 │
│  ┌──────────────────────┐      │
│  │ Bbox Regression      │      │
│  │  (frozen after stage 0) │  │
│  └──────────────────────┘      │
└─────────────────────────────────┘
     ↓
Style Router (for domain weights)
     ↓
Predictions
```

## Installation

### Requirements

- Python >= 3.7
- PyTorch >= 1.8
- MMDetection 3.x
- MMEngine
- MMCV

### Setup

1. Install MMDetection following the [official installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html)

2. Install DIDA-specific dependencies:

```bash
cd projects/dida
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare CLIP Prototypes

Extract CLIP text embeddings for your classes:

```bash
python projects/dida/tools/extract_clip_prototypes.py \
    --classes aeroplane bicycle bird boat bottle bus car cat chair cow \
             diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor \
    --output data/clip_prototypes/voc_clip_prototypes.pt \
    --model ViT-B-32 \
    --pretrained openai \
    --templates "a photo of a {}" "a rendering of a {}" "a cropped photo of the {}"
```

### 2. Prepare Datasets

Organize your datasets in COCO format:

```
data/
├── VOCdevkit/
│   └── annotations/
│       ├── voc0712_trainval.json
│       └── voc07_test.json
├── clipart/
│   └── annotations/
│       ├── clipart_train.json
│       └── clipart_test.json
├── watercolor/
└── comic/
```

### 3. Train Domain-Incrementally

#### Option A: Using the Shell Script

```bash
bash projects/dida/tools/train_incremental.sh voc_clipart_watercolor_comic 0
```

#### Option B: Using Python Script

```bash
python projects/dida/tools/train_incremental.py \
    voc_clipart_watercolor_comic \
    --work-dir work_dirs \
    --gpu-ids 0
```

#### Option C: Manual Stage-by-Stage Training

```bash
# Stage 0: Source domain (VOC)
python tools/train.py projects/dida/configs/voc_clipart_watercolor_comic/stage0_voc.py

# Stage 1: Adapt to Clipart
python tools/train.py projects/dida/configs/voc_clipart_watercolor_comic/stage1_clipart.py

# Stage 2: Adapt to Watercolor
python tools/train.py projects/dida/configs/voc_clipart_watercolor_comic/stage2_watercolor.py

# Stage 3: Adapt to Comic
python tools/train.py projects/dida/configs/voc_clipart_watercolor_comic/stage3_comic.py
```

### 4. Test with Domain-Unknown Inference

```bash
python projects/dida/tools/test_domain_aware.py \
    projects/dida/configs/voc_clipart_watercolor_comic/stage3_comic.py \
    work_dirs/dida_voc_series/stage3/epoch_12.pth \
    --domain-unknown
```

### 5. Demo Inference

```bash
python projects/dida/tools/demo_inference.py \
    projects/dida/configs/voc_clipart_watercolor_comic/stage3_comic.py \
    work_dirs/dida_voc_series/stage3/epoch_12.pth \
    path/to/test/image.jpg \
    --out result.jpg
```

## Configuration Guide

### Key Configuration Options

#### Model Configuration

```python
model = dict(
    type='DIDADINO',
    current_domain=0,  # Training domain index
    bbox_head=dict(
        type='DIDAHead',
        clip_classifier_cfg=dict(
            projection_mode='shared',  # or 'low_rank' for domain-specific projection
            cosine_scale=20.0,  # Temperature for cosine similarity
        ),
        domain_residual_cfg=dict(
            magnitude_weight=0.1,  # Weight for residual magnitude loss
            orthogonal_weight=0.1,  # Weight for orthogonality loss
        ),
    ),
    style_router_cfg=dict(
        style_dim=64,  # Dimension of style vectors
        sigma=0.5,  # Temperature for domain weight computation
    ),
)
```

#### Freezing Strategy

```python
custom_hooks = [
    dict(
        type='FreezeHook',
        stage=1,  # 0 for source, >0 for incremental
        freeze_backbone=True,
        freeze_neck=True,
        freeze_transformer=True,
        freeze_bbox_head=True,
        freeze_base_classifier=True,
        verbose=True
    )
]
```

## Domain Shift Validation Experiments

### 1. Oracle Classification Analysis

Measure upper bound of performance by replacing predicted labels with GT labels:

```python
val_evaluator = dict(
    type='OracleCocoMetric',
    iou_threshold=0.5,
    matching_strategy='greedy',  # or 'hungarian'
    metric='bbox'
)
```

This reveals how much performance degradation is due to classification vs. localization.

### 2. Class-Agnostic Evaluation

Evaluate pure localization by treating all objects as a single class:

```python
val_evaluator = dict(
    type='ClassAgnosticCocoMetric',
    metric='bbox'
)
```

### 3. Geometric Error Analysis

Analyze localization errors (center shift, width/height errors):

```bash
python projects/dida/tools/geometric_error_analysis.py \
    results.pkl \
    --ann-file data/VOCdevkit/annotations/voc07_test.json
```

## Module Documentation

### DomainResidualPrototype

Learns domain-specific residuals for CLIP prototypes:

```python
from projects.dida.models import DomainResidualPrototype

residual = DomainResidualPrototype(
    num_classes=20,
    embed_dim=512,
    magnitude_weight=0.1
)

# Add new domain
domain_idx = residual.add_domain()

# Get adapted prototypes
prototypes = residual.get_domain_prototypes(base_prototypes, domain_idx)

# Freeze old domains
residual.freeze_domain(0)
```

### CLIPPrototypeClassifier

CLIP-based classifier with optional domain-specific projection:

```python
from projects.dida.models import CLIPPrototypeClassifier

classifier = CLIPPrototypeClassifier(
    num_classes=20,
    clip_dim=512,
    query_dim=256,
    projection_mode='low_rank',
    low_rank_dim=4
)

# Load CLIP prototypes
classifier.load_clip_prototypes('path/to/prototypes.pt')

# Forward pass
scores = classifier(queries, domain_idx=1)
```

### StyleRouter

Domain identification via Gram matrix style features:

```python
from projects.dida.models import StyleRouter

router = StyleRouter(
    feature_channels=256,
    style_dim=64,
    sigma=0.5
)

# Add domain
router.add_domain()

# Compute domain weights
domain_weights, info = router(features, update_prototype=True, domain_idx=0)
```

## Hyperparameter Tuning Guide

### Key Hyperparameters

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `cosine_scale` | 20.0 | Temperature for classification | Higher = more confident predictions |
| `magnitude_weight` | 0.1 | Residual magnitude regularization | Increase to keep residuals small |
| `orthogonal_weight` | 0.1 | Residual orthogonality loss | Increase to enforce domain diversity |
| `low_rank_dim` | 4 | Rank of domain projection | Higher = more capacity, slower |
| `style_dim` | 64 | Dimension of style vectors | Higher = better domain discrimination |
| `sigma` | 0.5 | Domain weight temperature | Lower = sharper domain selection |
| `ema_momentum` | 0.1 | EMA for domain prototypes | Higher = faster adaptation |
| `lr` (incremental) | 1e-5 | Learning rate for stages >0 | ~10x smaller than source |

### Training Tips

1. **Source Domain (Stage 0)**:
   - Train normally with standard learning rate
   - Ensure good performance before proceeding
   - Target: >70% mAP for VOC

2. **Incremental Stages (Stage >0)**:
   - Use 10x smaller learning rate
   - Train for fewer epochs (6-12)
   - Monitor residual norms (should be small)
   - Check domain weight entropy (should be low for clear assignment)

3. **Domain-Unknown Inference**:
   - Enable style router
   - Domain weights should be well-separated
   - Entropy <0.5 indicates good domain discrimination

## Expected Performance

### VOC → Clipart → Watercolor → Comic

| Domain | Baseline | DIDA | Oracle (Upper Bound) |
|--------|----------|------|----------------------|
| VOC | 75.2 | 75.2 | 82.1 |
| Clipart | 32.5 | 58.3 | 67.8 |
| Watercolor | 28.9 | 53.6 | 64.2 |
| Comic | 25.1 | 50.4 | 62.5 |

*Note: Numbers are illustrative. Actual performance depends on dataset and hyperparameters.*

## FAQ

### Q: How many domains can DIDA handle?

A: Tested with up to 4 domains. More domains may require larger style vectors and more training data per domain.

### Q: Can I use a different backbone?

A: Yes, modify the `backbone` config. Swin-Transformer is recommended for better performance.

### Q: What if my domains have different class sets?

A: DIDA assumes fixed classes. For class-incremental learning, see related work on class-incremental detection.

### Q: How to handle domain-unknown inference?

A: Set `model.domain_unknown_mode=True` and ensure style router is trained. Domain weights will be computed automatically.

### Q: Can I skip intermediate domains?

A: Not recommended. The residuals are learned incrementally and skipping may hurt performance.

## Citation

```bibtex
@inproceedings{dida2024,
  title={DIDA: Domain-Incremental Detection with Anchors},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

## License

This project is released under the Apache 2.0 license (same as MMDetection).

## Acknowledgments

- Built on [MMDetection](https://github.com/open-mmlab/mmdetection)
- CLIP prototypes from [OpenCLIP](https://github.com/mlfoundations/open_clip)
- Inspired by DINO detector architecture

## Contact

For questions and issues, please open an issue on GitHub.
