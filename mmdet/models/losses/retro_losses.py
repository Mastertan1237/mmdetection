# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class OrthogonalConstraintLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, **kwargs):
        return self.loss_weight * 0.0


@MODELS.register_module()
class SemanticResponseLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, **kwargs):
        return self.loss_weight * 0.0


@MODELS.register_module()
class SemanticOutputPreservationLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, **kwargs):
        return self.loss_weight * 0.0


@MODELS.register_module()
class CombinedOrthogonalLoss(nn.Module):
    def __init__(self, orth_weight=1.0, resp_weight=1.0, pres_weight=1.0):
        super().__init__()
        self.orth = OrthogonalConstraintLoss(orth_weight)
        self.resp = SemanticResponseLoss(resp_weight)
        self.pres = SemanticOutputPreservationLoss(pres_weight)
    def forward(self, **kwargs):
        return {'loss_total': 0.0}
