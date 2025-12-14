# Copyright (c) OpenMMLab. All rights reserved.
# Default runtime configuration for DIDA

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.runner import LogProcessor

default_scope = 'mmdet'

# Hooks
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=1, save_best='auto'),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Logger
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
