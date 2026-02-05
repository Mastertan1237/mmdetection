# RetroCalib ETF Head with Orthogonal Constraints

## Overview

This implementation provides components for domain incremental object detection using orthogonal constraints to minimize interference between old and new tasks.

## Components

### 1. Losses (`mmdet/models/losses/retro_losses.py`)

- **OrthogonalConstraintLoss**: Keeps parameter updates orthogonal to semantic basis
- **SemanticResponseLoss**: Regularizes network to minimize semantic noise
- **SemanticOutputPreservationLoss**: Ensures classifier output consistency
- **CombinedOrthogonalLoss**: Aggregates all three losses

### 2. Head (`mmdet/models/roi_heads/bbox_heads/retro_etf_head.py`)

- **RetroCalibETFHeadWithOrthLoss**: Extends ConvFCBBoxHead with:
  - Task-specific adapter banks
  - Semantic basis buffer for orthogonal projections
  - Background feature decoupling support

## Usage

### In a config file:

```python
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='RetroCalibETFHeadWithOrthLoss',
            num_adapters=4,
            adapter_dim=256,
            enable_orthogonal=True,
            in_channels=256,
            fc_out_channels=1024,
            num_classes=80
        )
    )
)
```

### Using the losses:

```python
from mmdet.models.losses import CombinedOrthogonalLoss

loss_fn = CombinedOrthogonalLoss(
    orth_weight=1.0,
    resp_weight=1.0,
    pres_weight=1.0
)

# In training loop
loss_dict = loss_fn(
    # Pass required tensors as kwargs
)
```

## Implementation Details

This is a minimal placeholder implementation that provides the framework for:

1. **Multi-stage projection** - Enforces updates in orthogonal subspaces
2. **Adapter banks** - Task-specific parameter handling
3. **Background decoupling** - Separates foreground and background features

The actual loss computation logic is stubbed out and returns zero values. This allows the components to be registered and used in the MMDetection framework while providing a foundation for full implementation.

## Testing

Run the provided unit tests:

```bash
pytest tests/test_models/test_losses/test_retro_losses.py
pytest tests/test_models/test_roi_heads/test_bbox_heads/test_retro_etf_head.py
```

## Security

All components have been validated with CodeQL security scanning with no vulnerabilities detected.
