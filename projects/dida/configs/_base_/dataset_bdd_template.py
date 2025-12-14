# Copyright (c) OpenMMLab. All rights reserved.
# BDD100K dataset template configuration

from mmengine.dataset import DefaultSampler

# BDD100K uses COCO classes (subset of 10 classes for domain adaptation experiments)
bdd_classes = (
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'traffic light', 'traffic sign', 'rider'
)

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/bdd100k/'  # Placeholder
metainfo = dict(classes=bdd_classes)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='PackDetInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')
    )
]

# Dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',  # Placeholder
        data_prefix=dict(img='images/train/'),  # Placeholder
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',  # Placeholder
        data_prefix=dict(img='images/val/'),  # Placeholder
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',  # Placeholder
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator
