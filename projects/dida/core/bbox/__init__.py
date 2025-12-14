# Copyright (c) OpenMMLab. All rights reserved.
from .objectness_assigner import (ObjectnessAwareHungarianAssigner,
                                  ReducedClassificationHungarianAssigner)

__all__ = [
    'ObjectnessAwareHungarianAssigner',
    'ReducedClassificationHungarianAssigner'
]
