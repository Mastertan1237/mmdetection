# Copyright (c) OpenMMLab. All rights reserved.
# VOC dataset template configuration

from mmengine.dataset import DefaultSampler

# VOC class names (20 classes)
voc_classes = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'  # Placeholder
metainfo = dict(classes=voc_classes)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='Resize', scale=(480, 800), keep_ratio=True)],
            [dict(type='Resize', scale=(512, 800), keep_ratio=True)],
            [dict(type='Resize', scale=(544, 800), keep_ratio=True)],
            [dict(type='Resize', scale=(576, 800), keep_ratio=True)],
            [dict(type='Resize', scale=(608, 800), keep_ratio=True)],
        ]
    ),
    dict(type='PackDetInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
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
        ann_file='annotations/voc_train.json',  # Placeholder
        data_prefix=dict(img='VOC2007/JPEGImages/'),  # Placeholder
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
        ann_file='annotations/voc_val.json',  # Placeholder
        data_prefix=dict(img='VOC2007/JPEGImages/'),  # Placeholder
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/voc_val.json',  # Placeholder
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator
