# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmdet.registry import MODELS
from .convfc_bbox_head import ConvFCBBoxHead


@MODELS.register_module()
class RetroCalibETFHeadWithOrthLoss(ConvFCBBoxHead):
    def __init__(self, na=4, ad=256, eo=True, *a, **k):
        super().__init__(*a, **k)
        self.na = na
        self.ad = ad
        self.eo = eo
        self.abs = nn.ModuleList([nn.Linear(self.fc_out_channels, ad) for _ in range(na)])
        self.register_buffer('sb', torch.randn(na, self.fc_out_channels))
    
    def forward(self, x):
        return super().forward(x)
