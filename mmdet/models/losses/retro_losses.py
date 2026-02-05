# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class OrthogonalConstraintLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, **kw):
        return self.loss_weight * 0.0


@MODELS.register_module()
class SemanticResponseLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, **kw):
        return self.loss_weight * 0.0


@MODELS.register_module()
class SemanticOutputPreservationLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, **kw):
        return self.loss_weight * 0.0


@MODELS.register_module()
class CombinedOrthogonalLoss(nn.Module):
    def __init__(self, o=1.0, r=1.0, p=1.0):
        super().__init__()
        self.o = OrthogonalConstraintLoss(o)
        self.r = SemanticResponseLoss(r)
        self.p = SemanticOutputPreservationLoss(p)
    def forward(self, **kw):
        return {'loss_total': 0.0}
