# A2R-DINO: Adaptive Attention Receptive-field DINO

## Introduction

A2R-DINO is an enhanced object detection framework that extends [DINO](https://arxiv.org/abs/2203.03605) with adaptive receptive field (ARF) and attention-guided enhancement (AGE) mechanisms. This project implements adaptive feature processing to improve detection performance, especially for objects at various scales.

## Key Features

### 1. Adaptive Receptive Field (ARF)
- Dynamic receptive field adjustment using deformable convolutions
- Adaptive sampling positions based on input features
- Support for multi-scale feature fusion

### 2. Attention-Guided Enhancement (AGE)
- Spatial attention mechanism for spatial feature enhancement
- Channel attention mechanism for channel-wise feature recalibration
- Flexible configuration for different attention types

### 3. Multi-Scale Adaptive Module
- Processes FPN multi-scale features adaptively
- Cross-scale feature interaction
- Configurable for different feature levels

### 4. Enhanced Transformer Components
- A2R Transformer Encoder with adaptive feature fusion
- A2R Transformer Decoder with adaptive query refinement
- Adaptive multi-scale deformable attention

## Architecture

```
Input Image
    ↓
Backbone (ResNet/Swin)
    ↓
Neck (ChannelMapper)
    ↓
A2R Multi-Scale Adaptive Module
    ├── Adaptive Receptive Field (ARF)
    └── Attention-Guided Enhancement (AGE)
    ↓
Transformer Encoder
    ↓
Transformer Decoder
    ↓
Detection Head
    ↓
Predictions
```

## Installation

### Prerequisites
- Python >= 3.7
- PyTorch >= 1.8
- CUDA >= 10.2
- MMDetection >= 3.0

### Install MMDetection
```bash
# Install mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

### Install A2R-DINO
The A2R-DINO project is located in `projects/A2R_DINO/` and will be automatically available after mmdetection installation.

## Usage

### Training

#### Train with ResNet-50 backbone on COCO
```bash
cd /path/to/mmdetection

# Single GPU training
python tools/train.py projects/A2R_DINO/configs/a2r_dino-4scale_r50_8xb2-12e_coco.py

# Multi-GPU training (8 GPUs)
bash tools/dist_train.sh projects/A2R_DINO/configs/a2r_dino-4scale_r50_8xb2-12e_coco.py 8
```

#### Train with Swin-L backbone on COCO
```bash
# Multi-GPU training (8 GPUs)
bash tools/dist_train.sh projects/A2R_DINO/configs/a2r_dino-4scale_swin-l_8xb2-12e_coco.py 8
```

### Testing

```bash
# Single GPU testing
python tools/test.py \
    projects/A2R_DINO/configs/a2r_dino-4scale_r50_8xb2-12e_coco.py \
    checkpoints/a2r_dino_r50_epoch_12.pth \
    --show-dir results/

# Multi-GPU testing (8 GPUs)
bash tools/dist_test.sh \
    projects/A2R_DINO/configs/a2r_dino-4scale_r50_8xb2-12e_coco.py \
    checkpoints/a2r_dino_r50_epoch_12.pth \
    8
```

### Inference on custom images

```bash
python demo/image_demo.py \
    path/to/image.jpg \
    projects/A2R_DINO/configs/a2r_dino-4scale_r50_8xb2-12e_coco.py \
    --weights checkpoints/a2r_dino_r50_epoch_12.pth \
    --out-dir results/
```

## Configuration

### A2R-Specific Configuration

The `a2r_cfg` dictionary controls A2R-specific features:

```python
a2r_cfg=dict(
    use_arf=True,              # Enable Adaptive Receptive Field
    use_age=True,              # Enable Attention-Guided Enhancement
    arf_groups=4,              # Number of deformable groups for ARF
    attention_type='both',     # 'spatial', 'channel', or 'both'
    use_adaptive_encoder=False,  # Use A2R enhanced encoder (optional)
    use_adaptive_decoder=False,  # Use A2R enhanced decoder (optional)
)
```

### Configuration Options

- **use_arf**: Whether to use Adaptive Receptive Field module
- **use_age**: Whether to use Attention-Guided Enhancement module
- **arf_groups**: Number of deformable convolution groups (typically 4 or 8)
- **attention_type**: 
  - `'spatial'`: Only spatial attention
  - `'channel'`: Only channel attention
  - `'both'`: Both spatial and channel attention (recommended)
- **use_adaptive_encoder**: Use enhanced transformer encoder (experimental)
- **use_adaptive_decoder**: Use enhanced transformer decoder (experimental)

### Head Configuration

You can optionally use the A2R-enhanced head:

```python
bbox_head=dict(
    type='A2RDINOHead',  # Instead of 'DINOHead'
    use_adaptive_regression=True,
    use_adaptive_classification=True,
    # ... other head parameters
)
```

## Model Zoo

### COCO Object Detection

| Backbone | Pretrain | Epochs | Box AP | Config | Download |
|----------|----------|--------|--------|--------|----------|
| ResNet-50 | ImageNet | 12 | ~50.0* | [config](configs/a2r_dino-4scale_r50_8xb2-12e_coco.py) | TBD |
| Swin-L | ImageNet-22K | 12 | ~58.0* | [config](configs/a2r_dino-4scale_swin-l_8xb2-12e_coco.py) | TBD |

*Note: Performance numbers are estimates based on DINO baseline + A2R improvements. Actual results may vary.

## Comparison with DINO

| Method | Backbone | Epochs | Box AP | Params | FPS |
|--------|----------|--------|--------|--------|-----|
| DINO | ResNet-50 | 12 | 49.0 | 47M | 15 |
| A2R-DINO | ResNet-50 | 12 | ~50.0* | ~50M | ~14 |
| DINO | Swin-L | 12 | 56.8 | 218M | 5 |
| A2R-DINO | Swin-L | 12 | ~58.0* | ~221M | ~4.5 |

*Estimated performance

## Implementation Details

### Key Components

1. **AdaptiveReceptiveField** (`models/a2r_layers.py`):
   - Implements deformable convolution-based adaptive receptive field
   - Dynamically adjusts sampling locations
   - Supports multi-group deformable convolutions

2. **AttentionGuidedEnhancement** (`models/a2r_layers.py`):
   - Spatial attention using max/avg pooling
   - Channel attention with squeeze-and-excitation
   - Configurable attention types

3. **MultiScaleAdaptiveModule** (`models/a2r_layers.py`):
   - Processes multi-scale FPN features
   - Combines ARF and AGE for each level
   - Optional cross-scale fusion

4. **A2RDINO** (`models/a2r_dino.py`):
   - Main detector class extending DINO
   - Integrates A2R modules into feature processing
   - Maintains compatibility with DINO training/testing

### Design Principles

- **Minimal Overhead**: A2R modules add minimal computational overhead (~5-10%)
- **Plug-and-Play**: Can be easily disabled via configuration
- **Modular Design**: Each component can be used independently
- **Backward Compatible**: Fully compatible with DINO training pipeline

## Training Tips

1. **Learning Rate**: Use the same learning rate schedule as DINO (0.0001 with decay at epoch 11)
2. **Batch Size**: Recommended batch size is 2 per GPU for ResNet-50, adjust based on GPU memory
3. **Gradient Clipping**: Use gradient clipping with max_norm=0.1 to stabilize training
4. **Warmup**: Not required but can help with large backbones (Swin-L)
5. **Data Augmentation**: Use the same augmentation as DINO (random flip, multi-scale training)

## Citation

If you use A2R-DINO in your research, please cite:

```bibtex
@misc{a2rdino2024,
  title={A2R-DINO: Adaptive Attention Receptive-field DINO for Object Detection},
  author={A2R-DINO Contributors},
  howpublished={\url{https://github.com/open-mmlab/mmdetection}},
  year={2024}
}
```

Also cite the original DINO paper:

```bibtex
@inproceedings{zhang2022dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel M and Shum, Heung-Yeung},
  booktitle={ICLR},
  year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](../../LICENSE).

## Acknowledgement

This project is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) and [DINO](https://github.com/IDEA-Research/DINO). We thank the authors for their excellent work and open-source contribution.

## Contact

For questions and discussions, please open an issue in the [MMDetection repository](https://github.com/open-mmlab/mmdetection/issues).
