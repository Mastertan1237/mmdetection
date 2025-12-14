# Copyright (c) OpenMMLab. All rights reserved.
# Stage 3: Incremental adaptation to Comic domain

from mmengine.config import read_base

with read_base():
    from .stage2_watercolor import *

# Model configuration - Domain 3
model = dict(
    current_domain=3,
    bbox_head=dict(
        current_domain=3,
    )
)

# Dataset - Comic2k
data_root = 'data/comic/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/comic_train.json',
        data_prefix=dict(img=''),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/comic_test.json',
        data_prefix=dict(img=''),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/comic_test.json'
)

# Load checkpoint from stage 2
load_from = './work_dirs/dida_voc_series/stage2/epoch_12.pth'

# Freeze hook for stage 3
custom_hooks = [
    dict(
        type='FreezeHook',
        stage=3,
        freeze_backbone=True,
        freeze_neck=True,
        freeze_transformer=True,
        freeze_bbox_head=True,
        freeze_base_classifier=True,
        verbose=True
    )
]

# Work directory
work_dir = './work_dirs/dida_voc_series/stage3'
