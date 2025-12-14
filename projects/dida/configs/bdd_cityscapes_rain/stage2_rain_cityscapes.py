# Copyright (c) OpenMMLab. All rights reserved.
# Stage 2: Incremental adaptation to RainCityscapes

from mmengine.config import read_base

with read_base():
    from .stage1_cityscapes import *

# Model - Domain 2
model = dict(
    current_domain=2,
    bbox_head=dict(current_domain=2)
)

# Dataset - RainCityscapes
data_root = 'data/cityscapes_rain/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/rain_train.json',
        data_prefix=dict(img='leftImg8bit_rain/train/'),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/rain_val.json',
        data_prefix=dict(img='leftImg8bit_rain/val/'),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/rain_val.json'
)

load_from = './work_dirs/dida_bdd_series/stage1/epoch_12.pth'

custom_hooks = [
    dict(type='FreezeHook', stage=2, verbose=True)
]

work_dir = './work_dirs/dida_bdd_series/stage2'
