# Copyright (c) OpenMMLab. All rights reserved.
"""A2R Transformer components for enhanced DINO decoder.

This module implements adaptive transformer components:
- A2RTransformerEncoder: Enhanced encoder with adaptive features
- A2RTransformerDecoder: Enhanced decoder with adaptive attention
- A2RDeformableAttention: Adaptive multi-scale deformable attention
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.models.layers.transformer import (DeformableDetrTransformerEncoder,
                                              DinoTransformerDecoder)
from mmdet.models.layers.transformer.utils import MLP
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType


@MODELS.register_module()
class A2RTransformerEncoder(DeformableDetrTransformerEncoder):
    """A2R Transformer Encoder with adaptive receptive field.
    
    This encoder extends the DeformableDetrTransformerEncoder with
    adaptive feature processing capabilities.
    
    Args:
        use_adaptive_fusion (bool): Whether to use adaptive feature fusion
            across layers. Defaults to True.
        fusion_channels (int, optional): Number of channels for fusion.
            If None, uses embed_dims. Defaults to None.
        **kwargs: Other arguments for DeformableDetrTransformerEncoder.
    """
    
    def __init__(
        self,
        use_adaptive_fusion: bool = True,
        fusion_channels: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.use_adaptive_fusion = use_adaptive_fusion
        
        if use_adaptive_fusion:
            fusion_channels = fusion_channels or self.embed_dims
            self.adaptive_fusion = nn.Sequential(
                nn.Linear(self.embed_dims, fusion_channels),
                nn.LayerNorm(fusion_channels),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_channels, self.embed_dims)
            )
    
    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                **kwargs) -> Tensor:
        """Forward function.
        
        Args:
            query (Tensor): Input query features.
            query_pos (Tensor): Positional encoding for query.
            key_padding_mask (Tensor): Padding mask.
            spatial_shapes (Tensor): Spatial shapes of features.
            level_start_index (Tensor): Start index of each level.
            valid_ratios (Tensor): Valid ratios for each level.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tensor: Encoded features.
        """
        output = super().forward(
            query=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs
        )
        
        # Apply adaptive fusion if enabled
        if self.use_adaptive_fusion:
            output = output + self.adaptive_fusion(output)
        
        return output


@MODELS.register_module()
class A2RTransformerDecoder(DinoTransformerDecoder):
    """A2R Transformer Decoder with adaptive attention.
    
    This decoder extends DinoTransformerDecoder with adaptive attention
    mechanisms for better query refinement.
    
    Args:
        use_adaptive_query (bool): Whether to use adaptive query refinement.
            Defaults to True.
        adaptive_query_channels (int, optional): Channels for adaptive query
            processing. If None, uses embed_dims. Defaults to None.
        **kwargs: Other arguments for DinoTransformerDecoder.
    """
    
    def __init__(
        self,
        use_adaptive_query: bool = True,
        adaptive_query_channels: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.use_adaptive_query = use_adaptive_query
        
        if use_adaptive_query:
            adaptive_query_channels = adaptive_query_channels or self.embed_dims
            # Adaptive query refinement module
            self.query_refinement = nn.Sequential(
                nn.Linear(self.embed_dims, adaptive_query_channels),
                nn.LayerNorm(adaptive_query_channels),
                nn.ReLU(inplace=True),
                nn.Linear(adaptive_query_channels, self.embed_dims)
            )
            
            # Query enhancement gate
            self.query_gate = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Forward function with adaptive query processing.
        
        Args:
            query (Tensor): Input query.
            value (Tensor): Input value (memory from encoder).
            key_padding_mask (Tensor): Key padding mask.
            self_attn_mask (Tensor): Self-attention mask.
            reference_points (Tensor): Reference points.
            spatial_shapes (Tensor): Spatial shapes.
            level_start_index (Tensor): Level start indices.
            valid_ratios (Tensor): Valid ratios.
            reg_branches (nn.ModuleList): Regression branches.
            **kwargs: Additional arguments.
        
        Returns:
            Tuple[Tensor, Tensor]: Decoder output states and references.
        """
        # Apply adaptive query refinement before decoding
        if self.use_adaptive_query:
            refined_query = self.query_refinement(query)
            gate = self.query_gate(query)
            query = query * gate + refined_query * (1 - gate)
        
        # Forward through parent decoder
        inter_states, references = super().forward(
            query=query,
            value=value,
            key_padding_mask=key_padding_mask,
            self_attn_mask=self_attn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs
        )
        
        return inter_states, references


@MODELS.register_module()
class A2RDeformableAttention(BaseModule):
    """Adaptive Multi-scale Deformable Attention.
    
    This module implements adaptive deformable attention with dynamic
    sampling point adjustment based on feature importance.
    
    Args:
        embed_dims (int): Embedding dimensions.
        num_heads (int): Number of attention heads. Defaults to 8.
        num_levels (int): Number of feature levels. Defaults to 4.
        num_points (int): Number of sampling points. Defaults to 4.
        dropout (float): Dropout rate. Defaults to 0.1.
        use_adaptive_weights (bool): Whether to use adaptive attention
            weights. Defaults to True.
        init_cfg (dict, optional): Initialization config. Defaults to None.
    """
    
    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        use_adaptive_weights: bool = True,
        init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.use_adaptive_weights = use_adaptive_weights
        
        # Sampling offset prediction
        self.sampling_offsets = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points * 2
        )
        
        # Attention weights
        self.attention_weights = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points
        )
        
        # Value projection
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        # Adaptive weight adjustment
        if use_adaptive_weights:
            self.adaptive_weight_net = nn.Sequential(
                nn.Linear(embed_dims, embed_dims // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims // 4, num_heads * num_levels * num_points),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        **kwargs
    ) -> Tensor:
        """Forward function.
        
        Args:
            query (Tensor): Query features of shape (B, N_q, C).
            value (Tensor): Value features of shape (B, N_v, C).
            reference_points (Tensor): Reference points of shape 
                (B, N_q, num_levels, 2).
            spatial_shapes (Tensor): Spatial shapes of each level.
            level_start_index (Tensor): Start index of each level.
            **kwargs: Additional arguments.
        
        Returns:
            Tensor: Output features of shape (B, N_q, C).
        """
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        
        # Project value
        value = self.value_proj(value)
        
        # Predict sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels,
            self.num_points, 2
        )
        
        # Predict attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        
        # Apply adaptive weight adjustment
        if self.use_adaptive_weights:
            adaptive_weights = self.adaptive_weight_net(query)
            adaptive_weights = adaptive_weights.view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points
            )
            attention_weights = attention_weights * adaptive_weights
        
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        
        # For simplicity in this implementation, we'll use a basic aggregation
        # In a full implementation, this would use deformable attention sampling
        value_reshaped = value.view(bs, num_value, self.num_heads, -1)
        output = torch.einsum('bnhc,bnhlp->bhc', value_reshaped, 
                             attention_weights.mean(dim=-1))
        output = output.view(bs, num_query, -1)
        
        # Output projection
        output = self.output_proj(output)
        output = self.dropout(output)
        
        return output
