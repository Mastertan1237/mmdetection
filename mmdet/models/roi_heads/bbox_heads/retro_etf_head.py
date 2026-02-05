# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from .convfc_bbox_head import ConvFCBBoxHead


@MODELS.register_module()
class RetroCalibETFHeadWithOrthLoss(ConvFCBBoxHead):
    def __init__(self, num_adapters=4, adapter_dim=256, enable_orthogonal=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_adapters = num_adapters
        self.adapter_dim = adapter_dim
        self.enable_orthogonal = enable_orthogonal
        self.adapter_banks = nn.ModuleList([nn.Linear(self.fc_out_channels, adapter_dim) for _ in range(num_adapters)])
        self.register_buffer('semantic_basis', torch.randn(num_adapters, self.fc_out_channels))
    
    def forward(self, x):
        return super().forward(x)
