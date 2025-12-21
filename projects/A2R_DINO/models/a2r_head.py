# Copyright (c) OpenMMLab. All rights reserved.
"""A2R DINO Head with adaptive regression and classification."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.dense_heads import DINOHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class A2RDINOHead(DINOHead):
    """A2R DINO Head with enhanced adaptive prediction branches.
    
    This head extends DINOHead with adaptive regression and classification
    branches for better detection performance.
    
    Args:
        use_adaptive_regression (bool): Whether to use adaptive regression
            branch. Defaults to True.
        use_adaptive_classification (bool): Whether to use adaptive 
            classification branch. Defaults to True.
        adaptive_hidden_dim (int, optional): Hidden dimension for adaptive
            branches. If None, uses embed_dims. Defaults to None.
        **kwargs: Other arguments for DINOHead.
    """
    
    def __init__(
        self,
        *args,
        use_adaptive_regression: bool = True,
        use_adaptive_classification: bool = True,
        adaptive_hidden_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_adaptive_regression = use_adaptive_regression
        self.use_adaptive_classification = use_adaptive_classification
        
        if adaptive_hidden_dim is None:
            adaptive_hidden_dim = self.embed_dims
        
        # Adaptive regression enhancement
        if use_adaptive_regression:
            self.adaptive_reg_modules = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embed_dims, adaptive_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(adaptive_hidden_dim, self.embed_dims),
                    nn.LayerNorm(self.embed_dims)
                ) for _ in range(self.num_pred_layer)
            ])
        
        # Adaptive classification enhancement
        if use_adaptive_classification:
            self.adaptive_cls_modules = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embed_dims, adaptive_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(adaptive_hidden_dim, self.embed_dims),
                    nn.LayerNorm(self.embed_dims)
                ) for _ in range(self.num_pred_layer)
            ])
    
    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward function.
        
        Args:
            hidden_states (Tensor): Hidden states from decoder, with shape
                (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of reference points from decoder,
                each with shape (bs, num_queries, 2 or 4).
        
        Returns:
            tuple[Tensor, Tensor]: Outputs of classification and regression
                branches.
                
                - all_layers_cls_scores (Tensor): Classification scores of all
                  decoder layers, with shape (num_decoder_layers, bs,
                  num_queries, cls_out_channels).
                - all_layers_bbox_preds (Tensor): Regression outputs of all
                  decoder layers, with shape (num_decoder_layers, bs,
                  num_queries, 4).
        """
        all_layers_cls_scores = []
        all_layers_bbox_preds = []
        
        for layer_id in range(hidden_states.shape[0]):
            hidden_state = hidden_states[layer_id]
            reference = references[layer_id]
            
            # Apply adaptive enhancements
            if self.use_adaptive_regression:
                reg_feat = hidden_state + self.adaptive_reg_modules[layer_id](
                    hidden_state)
            else:
                reg_feat = hidden_state
            
            if self.use_adaptive_classification:
                cls_feat = hidden_state + self.adaptive_cls_modules[layer_id](
                    hidden_state)
            else:
                cls_feat = hidden_state
            
            # Classification
            cls_score = self.cls_branches[layer_id](cls_feat)
            
            # Regression
            if reference.shape[-1] == 4:
                # When reference has 4 dimensions (x, y, w, h)
                bbox_pred = self.reg_branches[layer_id](reg_feat)
                bbox_pred = bbox_pred + reference
            else:
                # When reference has 2 dimensions (x, y)
                bbox_pred = self.reg_branches[layer_id](reg_feat)
                bbox_pred[..., :2] = bbox_pred[..., :2] + reference
            
            all_layers_cls_scores.append(cls_score)
            all_layers_bbox_preds.append(bbox_pred)
        
        all_layers_cls_scores = torch.stack(all_layers_cls_scores)
        all_layers_bbox_preds = torch.stack(all_layers_bbox_preds)
        
        return all_layers_cls_scores, all_layers_bbox_preds
