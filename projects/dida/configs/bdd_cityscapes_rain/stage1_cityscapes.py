# Copyright (c) OpenMMLab. All rights reserved.
# Stage 1: Incremental adaptation to Cityscapes

from mmengine.config import read_base

with read_base():
    from .stage0_bdd import *

# Model - Domain 1
model = dict(
    current_domain=1,
    bbox_head=dict(current_domain=1)
)

# Dataset - Cityscapes
data_root = 'data/cityscapes/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/cityscapes_train.json',
        data_prefix=dict(img='leftImg8bit/train/'),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/cityscapes_val.json',
        data_prefix=dict(img='leftImg8bit/val/'),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/cityscapes_val.json'
)

load_from = './work_dirs/dida_bdd_series/stage0/epoch_12.pth'

custom_hooks = [
    dict(type='FreezeHook', stage=1, verbose=True)
]

optim_wrapper = dict(optimizer=dict(lr=0.00001))

work_dir = './work_dirs/dida_bdd_series/stage1'
