# Copyright (c) OpenMMLab. All rights reserved.
"""DIDA: Domain-Incremental Detection with Anchors.

A comprehensive framework for domain-incremental object detection based on
MMDetection 3.x with support for strict no-replay learning.
"""

# Import all modules to register them
from . import core, datasets, engine, evaluation, models

# Register all components
from .core import *  # noqa: F401, F403
from .datasets import *  # noqa: F401, F403
from .engine import *  # noqa: F401, F403
from .evaluation import *  # noqa: F401, F403
from .models import *  # noqa: F401, F403

__version__ = '1.0.0'

__all__ = [
    # Models
    'CLIPPrototypeClassifier', 'DIDADINO', 'DIDAHead',
    'DomainResidualPrototype', 'ObjectnessHead', 'EncoderObjectnessHead',
    'MultiLevelObjectnessHead', 'StyleRouter',
    # Core
    'ObjectnessAwareHungarianAssigner', 'ReducedClassificationHungarianAssigner',
    # Engine
    'FreezeHook',
    # Evaluation
    'OracleCocoMetric', 'ClassAgnosticCocoMetric', 'DomainAwareMetric',
]
