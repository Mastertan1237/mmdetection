# Copyright (c) OpenMMLab. All rights reserved.
"""A2R-DINO project for mmdetection.

This project implements A2R-DINO (Adaptive Attention Receptive-field DINO),
an enhanced version of DINO with adaptive receptive fields and attention
mechanisms for improved object detection.
"""
from .models import (A2RDINO, A2RDINOHead, A2RDeformableAttention,
                     A2RTransformerDecoder, A2RTransformerEncoder,
                     AdaptiveReceptiveField, AttentionGuidedEnhancement,
                     MultiScaleAdaptiveModule)

__all__ = [
    'A2RDINO', 'A2RDINOHead', 'AdaptiveReceptiveField',
    'AttentionGuidedEnhancement', 'MultiScaleAdaptiveModule',
    'A2RTransformerEncoder', 'A2RTransformerDecoder', 'A2RDeformableAttention'
]
