# Copyright (c) OpenMMLab. All rights reserved.
# Stage 1: Incremental adaptation to Clipart domain

from mmengine.config import read_base

with read_base():
    from .stage0_voc import *

# Model configuration - Domain 1
model = dict(
    current_domain=1,
    bbox_head=dict(
        current_domain=1,
    )
)

# Dataset - Clipart1k
data_root = 'data/clipart/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/clipart_train.json',
        data_prefix=dict(img=''),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/clipart_test.json',
        data_prefix=dict(img=''),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/clipart_test.json'
)

# Load checkpoint from stage 0
load_from = './work_dirs/dida_voc_series/stage0/epoch_12.pth'

# Incremental learning: freeze most parameters
custom_hooks = [
    dict(
        type='FreezeHook',
        stage=1,
        freeze_backbone=True,
        freeze_neck=True,
        freeze_transformer=True,
        freeze_bbox_head=True,
        freeze_base_classifier=True,
        verbose=True
    )
]

# Reduced learning rate for incremental stage
optim_wrapper = dict(
    optimizer=dict(lr=0.00001),  # 10x smaller
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0),  # Frozen
            'neck': dict(lr_mult=0.0),  # Frozen
        }
    )
)

# Training schedule
max_epochs = 12
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

# Work directory
work_dir = './work_dirs/dida_voc_series/stage1'
