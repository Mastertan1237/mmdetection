# Copyright (c) OpenMMLab. All rights reserved.
# Base DIDA-DINO configuration

from mmengine.config import read_base

# Import base DINO config
with read_base():
    from mmdet.configs.dino.dino_4scale_r50_8xb2_12e_coco import *

# Override model to use DIDA components
model = dict(
    type='DIDADINO',
    # Style router configuration
    style_router_cfg=dict(
        type='StyleRouter',
        feature_channels=256,  # Match neck output channels
        style_dim=64,
        sigma=0.5,
        ema_momentum=0.1,
        update_mode='ema',
        extract_layer='neck'
    ),
    # Override bbox_head with DIDA head
    bbox_head=dict(
        type='DIDAHead',
        num_classes=20,  # Will be overridden in specific configs
        # CLIP classifier configuration
        clip_classifier_cfg=dict(
            type='CLIPPrototypeClassifier',
            num_classes=20,
            clip_dim=512,
            query_dim=256,  # Match decoder hidden dim
            projection_mode='shared',  # or 'low_rank'
            low_rank_dim=4,
            cosine_scale=20.0,
            clip_prototypes_path=None,  # Set in specific configs
            normalize_prototypes=True
        ),
        # Domain residual configuration
        domain_residual_cfg=dict(
            type='DomainResidualPrototype',
            num_classes=20,
            embed_dim=512,  # Match CLIP dim
            init_std=0.01,
            magnitude_weight=0.1,
            orthogonal_weight=0.1
        ),
        # Objectness head configuration
        objectness_cfg=dict(
            type='MultiLevelObjectnessHead',
            embed_dim=256,
            num_decoder_layers=6,
            num_mlp_layers=2,
            share_head=True
        ),
        use_objectness=True,
        current_domain=0,  # Will be set in specific configs
        # Standard DINO head parameters
        num_query=900,
        num_classes=20,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        embed_dims=256,
        num_reg_fcs=2,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    # Domain settings
    current_domain=0,  # Will be overridden
    domain_unknown_mode=False
)

# Reduced classification cost for better localization focus
train_cfg = dict(
    assigner=dict(
        type='ReducedClassificationHungarianAssigner',
        cls_cost_weight_factor=0.5,  # Reduce cls cost to half
        match_costs=[
            dict(type='FocalLossCost', weight=1.0),  # Reduced from 2.0
            dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ]
    )
)
