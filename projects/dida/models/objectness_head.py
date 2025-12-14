# Copyright (c) OpenMMLab. All rights reserved.
"""Objectness heads for class-agnostic object detection signal.

Provides objectness scoring at both encoder and decoder levels to improve
localization and proposal selection.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class ObjectnessHead(nn.Module):
    """Objectness head for decoder-level objectness scoring.
    
    Provides a class-agnostic foreground/background signal to complement
    class-specific classification.
    
    Args:
        embed_dim (int): Embedding dimension of queries. Defaults to 256.
        num_layers (int): Number of MLP layers. Defaults to 2.
        output_dim (int): Output dimension (1 for binary objectness). Defaults to 1.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Build MLP
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(embed_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, queries: Tensor) -> Tensor:
        """Forward pass to compute objectness scores.
        
        Args:
            queries (Tensor): Query features, shape (batch_size, num_queries, embed_dim).
        
        Returns:
            Tensor: Objectness scores, shape (batch_size, num_queries, 1).
        """
        return self.mlp(queries)


@MODELS.register_module()
class EncoderObjectnessHead(nn.Module):
    """Encoder-level objectness head for top-k proposal selection.
    
    Similar to ObjectnessHead but specifically designed for encoder features
    to select high-quality proposals.
    
    Args:
        embed_dim (int): Embedding dimension of encoder features. Defaults to 256.
        num_layers (int): Number of MLP layers. Defaults to 2.
        output_dim (int): Output dimension (1 for binary objectness). Defaults to 1.
        use_feature_pyramid (bool): Whether to use multi-scale features. Defaults to False.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 1,
        use_feature_pyramid: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_feature_pyramid = use_feature_pyramid
        
        # Build MLP
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(embed_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features: Tensor) -> Tensor:
        """Forward pass to compute objectness scores.
        
        Args:
            features (Tensor): Encoder features, shape:
                - If single-scale: (batch_size, num_points, embed_dim)
                - If multi-scale: (batch_size, num_levels, num_points_per_level, embed_dim)
        
        Returns:
            Tensor: Objectness scores, shape matching input but with last dim = 1.
        """
        original_shape = features.shape
        
        if self.use_feature_pyramid and features.dim() == 4:
            # Multi-scale features: (bs, num_levels, num_points, embed_dim)
            bs, num_levels, num_points, embed_dim = features.shape
            # Reshape to (bs * num_levels * num_points, embed_dim)
            features_flat = features.reshape(-1, embed_dim)
            # Apply MLP
            scores = self.mlp(features_flat)
            # Reshape back to (bs, num_levels, num_points, 1)
            scores = scores.reshape(bs, num_levels, num_points, self.output_dim)
        else:
            # Single-scale features: (bs, num_points, embed_dim)
            scores = self.mlp(features)
        
        return scores


@MODELS.register_module()
class MultiLevelObjectnessHead(nn.Module):
    """Multi-level objectness head for hierarchical object detection.
    
    Combines objectness scores from multiple decoder layers with different weights.
    
    Args:
        embed_dim (int): Embedding dimension. Defaults to 256.
        num_decoder_layers (int): Number of decoder layers. Defaults to 6.
        num_mlp_layers (int): Number of MLP layers per head. Defaults to 2.
        share_head (bool): Whether to share head across layers. Defaults to True.
        layer_weights (List[float], optional): Weights for each layer's objectness loss.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_decoder_layers: int = 6,
        num_mlp_layers: int = 2,
        share_head: bool = True,
        layer_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_mlp_layers = num_mlp_layers
        self.share_head = share_head
        
        # Layer weights for loss computation
        if layer_weights is None:
            layer_weights = [1.0] * num_decoder_layers
        assert len(layer_weights) == num_decoder_layers
        self.register_buffer('layer_weights', torch.tensor(layer_weights))
        
        # Build heads
        if share_head:
            # Single shared head
            self.objectness_head = ObjectnessHead(
                embed_dim=embed_dim,
                num_layers=num_mlp_layers,
                output_dim=1
            )
        else:
            # Separate head per layer
            self.objectness_heads = nn.ModuleList([
                ObjectnessHead(
                    embed_dim=embed_dim,
                    num_layers=num_mlp_layers,
                    output_dim=1
                )
                for _ in range(num_decoder_layers)
            ])
    
    def forward(self, all_layer_queries: Tensor) -> Tensor:
        """Forward pass on all decoder layer queries.
        
        Args:
            all_layer_queries (Tensor): Queries from all layers,
                shape (num_decoder_layers, batch_size, num_queries, embed_dim).
        
        Returns:
            Tensor: Objectness scores for all layers,
                shape (num_decoder_layers, batch_size, num_queries, 1).
        """
        num_layers, bs, num_queries, embed_dim = all_layer_queries.shape
        
        if self.share_head:
            # Reshape to (num_layers * bs, num_queries, embed_dim)
            queries_flat = all_layer_queries.reshape(num_layers * bs, num_queries, embed_dim)
            # Apply shared head
            scores = self.objectness_head(queries_flat)
            # Reshape back
            scores = scores.reshape(num_layers, bs, num_queries, 1)
        else:
            # Apply separate head per layer
            scores_list = []
            for i in range(num_layers):
                layer_scores = self.objectness_heads[i](all_layer_queries[i])
                scores_list.append(layer_scores)
            scores = torch.stack(scores_list, dim=0)
        
        return scores
    
    def compute_loss(
        self,
        objectness_scores: Tensor,
        objectness_targets: Tensor,
        weights: Optional[Tensor] = None
    ) -> Tensor:
        """Compute objectness loss.
        
        Args:
            objectness_scores (Tensor): Predicted scores,
                shape (num_layers, batch_size, num_queries, 1).
            objectness_targets (Tensor): Target labels (0 or 1),
                shape (batch_size, num_queries) or (num_layers, batch_size, num_queries).
            weights (Tensor, optional): Loss weights per sample,
                shape (batch_size, num_queries) or (num_layers, batch_size, num_queries).
        
        Returns:
            Tensor: Objectness loss (scalar).
        """
        num_layers = objectness_scores.size(0)
        
        # Ensure target has correct shape
        if objectness_targets.dim() == 2:
            # Broadcast to all layers
            objectness_targets = objectness_targets.unsqueeze(0).expand(num_layers, -1, -1)
        
        # Compute BCE loss
        objectness_scores = objectness_scores.squeeze(-1)  # (num_layers, bs, nq)
        loss = F.binary_cross_entropy_with_logits(
            objectness_scores,
            objectness_targets.float(),
            reduction='none'
        )
        
        # Apply sample weights if provided
        if weights is not None:
            if weights.dim() == 2:
                weights = weights.unsqueeze(0).expand(num_layers, -1, -1)
            loss = loss * weights
        
        # Weight by layer
        layer_weights = self.layer_weights.view(-1, 1, 1)  # (num_layers, 1, 1)
        loss = loss * layer_weights
        
        # Average
        return loss.mean()
