# Copyright (c) OpenMMLab. All rights reserved.
# Stage 2: Incremental adaptation to Watercolor domain

from mmengine.config import read_base

with read_base():
    from .stage1_clipart import *

# Model configuration - Domain 2
model = dict(
    current_domain=2,
    bbox_head=dict(
        current_domain=2,
    )
)

# Dataset - Watercolor2k
data_root = 'data/watercolor/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/watercolor_train.json',
        data_prefix=dict(img=''),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/watercolor_test.json',
        data_prefix=dict(img=''),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/watercolor_test.json'
)

# Load checkpoint from stage 1
load_from = './work_dirs/dida_voc_series/stage1/epoch_12.pth'

# Freeze hook for stage 2
custom_hooks = [
    dict(
        type='FreezeHook',
        stage=2,
        freeze_backbone=True,
        freeze_neck=True,
        freeze_transformer=True,
        freeze_bbox_head=True,
        freeze_base_classifier=True,
        verbose=True
    )
]

# Work directory
work_dir = './work_dirs/dida_voc_series/stage2'
