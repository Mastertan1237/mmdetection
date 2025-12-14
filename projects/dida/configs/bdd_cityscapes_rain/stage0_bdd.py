# Copyright (c) OpenMMLab. All rights reserved.
# Stage 0: Source domain training on BDD100K

from mmengine.config import read_base

with read_base():
    from .._base_.dino_baseline import *
    from .._base_.dataset_bdd_template import *
    from .._base_.default_runtime import *

# Model configuration
model = dict(
    current_domain=0,
    bbox_head=dict(
        num_classes=10,
        current_domain=0,
        clip_classifier_cfg=dict(
            num_classes=10,
            clip_prototypes_path='data/clip_prototypes/bdd_clip_prototypes.pt'
        ),
        domain_residual_cfg=dict(
            num_classes=10,
        )
    )
)

# Dataset - BDD100K
data_root = 'data/bdd100k/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/bdd_det_train.json',
        data_prefix=dict(img='images/100k/train/'),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/bdd_det_val.json',
        data_prefix=dict(img='images/100k/val/'),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/bdd_det_val.json'
)

# Training settings
max_epochs = 12
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[11], gamma=0.1)
]

work_dir = './work_dirs/dida_bdd_series/stage0'
