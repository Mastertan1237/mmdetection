# Copyright (c) OpenMMLab. All rights reserved.
# Stage 0: Source domain training on VOC

from mmengine.config import read_base

with read_base():
    from .._base_.dino_baseline import *
    from .._base_.dataset_voc_template import *
    from .._base_.default_runtime import *

# Model configuration
_base_model = model
model = dict(
    current_domain=0,
    bbox_head=dict(
        num_classes=20,
        current_domain=0,
        clip_classifier_cfg=dict(
            num_classes=20,
            clip_prototypes_path='data/clip_prototypes/voc_clip_prototypes.pt'
        ),
        domain_residual_cfg=dict(
            num_classes=20,
        )
    )
)

# Initialize domain 0 (will be done in training script or hook)
# This is the source domain, so we train all parameters

# Dataset - VOC 2007+2012 trainval
data_root = 'data/VOCdevkit/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/voc0712_trainval.json',
        data_prefix=dict(img=''),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/voc07_test.json',
        data_prefix=dict(img=''),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/voc07_test.json'
)

# Training schedule
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1
    )
]

# No freezing for stage 0
custom_hooks = []

# Work directory
work_dir = './work_dirs/dida_voc_series/stage0'
