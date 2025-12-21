# Copyright (c) OpenMMLab. All rights reserved.
"""A2R (Adaptive Attention Receptive-field) layers for enhanced feature processing.

This module implements the core components of A2R-DINO:
- AdaptiveReceptiveField (ARF): Dynamic receptive field using deformable convolutions
- AttentionGuidedEnhancement (AGE): Spatial and channel attention mechanisms
- MultiScaleAdaptiveModule: Multi-scale feature interaction
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import ModulatedDeformConv2d
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType


@MODELS.register_module()
class AdaptiveReceptiveField(BaseModule):
    """Adaptive Receptive Field module using deformable convolutions.
    
    This module dynamically adjusts sampling positions based on input features
    to achieve adaptive receptive fields.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for deformable convolution. Defaults to 3.
        stride (int): Stride for deformable convolution. Defaults to 1.
        padding (int): Padding for deformable convolution. Defaults to 1.
        dilation (int): Dilation for deformable convolution. Defaults to 1.
        groups (int): Number of groups for deformable convolution. Defaults to 4.
        deform_groups (int): Number of deformable groups. Defaults to 4.
        init_cfg (dict, optional): Initialization config. Defaults to None.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 4,
        deform_groups: int = 4,
        init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        
        # Offset prediction network
        self.offset_conv = nn.Conv2d(
            in_channels,
            deform_groups * 2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )
        
        # Mask prediction network
        self.mask_conv = nn.Conv2d(
            in_channels,
            deform_groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )
        
        # Deformable convolution
        self.deform_conv = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=False
        )
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            x (Tensor): Input features of shape (B, C, H, W).
        
        Returns:
            Tensor: Output features of shape (B, out_channels, H, W).
        """
        # Predict offsets and masks
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        # Apply deformable convolution
        out = self.deform_conv(x, offset, mask)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


@MODELS.register_module()
class AttentionGuidedEnhancement(BaseModule):
    """Attention-Guided Enhancement module with spatial and channel attention.
    
    This module enhances features using both spatial and channel attention
    mechanisms for better feature representation.
    
    Args:
        channels (int): Number of input channels.
        attention_type (str): Type of attention to use. Options: 'spatial',
            'channel', 'both'. Defaults to 'both'.
        reduction_ratio (int): Reduction ratio for channel attention. 
            Defaults to 16.
        kernel_size (int): Kernel size for spatial attention. Defaults to 7.
        init_cfg (dict, optional): Initialization config. Defaults to None.
    """
    
    def __init__(
        self,
        channels: int,
        attention_type: str = 'both',
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert attention_type in ['spatial', 'channel', 'both'], \
            f"attention_type must be 'spatial', 'channel', or 'both', " \
            f"got {attention_type}"
        
        self.channels = channels
        self.attention_type = attention_type
        
        # Channel attention
        if attention_type in ['channel', 'both']:
            self.channel_attention = ChannelAttention(
                channels, reduction_ratio)
        
        # Spatial attention
        if attention_type in ['spatial', 'both']:
            self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            x (Tensor): Input features of shape (B, C, H, W).
        
        Returns:
            Tensor: Enhanced features of shape (B, C, H, W).
        """
        identity = x
        
        # Apply channel attention
        if self.attention_type in ['channel', 'both']:
            x = x * self.channel_attention(x)
        
        # Apply spatial attention
        if self.attention_type in ['spatial', 'both']:
            x = x * self.spatial_attention(x)
        
        return x


class ChannelAttention(nn.Module):
    """Channel attention module.
    
    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for hidden dimension.
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_channels = max(channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            x (Tensor): Input features of shape (B, C, H, W).
        
        Returns:
            Tensor: Channel attention weights of shape (B, C, 1, 1).
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module.
    
    Args:
        kernel_size (int): Kernel size for convolution. Defaults to 7.
    """
    
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function.
        
        Args:
            x (Tensor): Input features of shape (B, C, H, W).
        
        Returns:
            Tensor: Spatial attention weights of shape (B, 1, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


@MODELS.register_module()
class MultiScaleAdaptiveModule(BaseModule):
    """Multi-scale adaptive module for processing FPN features.
    
    This module processes multi-scale features with adaptive cross-scale
    feature interaction.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_levels (int): Number of feature levels. Defaults to 4.
        use_arf (bool): Whether to use adaptive receptive field. 
            Defaults to True.
        use_age (bool): Whether to use attention-guided enhancement.
            Defaults to True.
        arf_groups (int): Number of groups for ARF. Defaults to 4.
        attention_type (str): Type of attention for AGE. Defaults to 'both'.
        init_cfg (dict, optional): Initialization config. Defaults to None.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 4,
        use_arf: bool = True,
        use_age: bool = True,
        arf_groups: int = 4,
        attention_type: str = 'both',
        init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_arf = use_arf
        self.use_age = use_age
        
        # Build adaptive modules for each level
        self.level_modules = nn.ModuleList()
        for _ in range(num_levels):
            modules = nn.ModuleDict()
            
            # Adaptive receptive field
            if use_arf:
                modules['arf'] = AdaptiveReceptiveField(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=arf_groups,
                    deform_groups=arf_groups
                )
            else:
                modules['conv'] = ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                )
            
            # Attention-guided enhancement
            if use_age:
                modules['age'] = AttentionGuidedEnhancement(
                    channels=out_channels,
                    attention_type=attention_type
                )
            
            self.level_modules.append(modules)
        
        # Cross-scale interaction
        self.fusion_conv = nn.Conv2d(
            out_channels * num_levels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_relu = nn.ReLU(inplace=True)
    
    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Forward function.
        
        Args:
            feats (tuple[Tensor]): Multi-scale input features, each of shape
                (B, C, H, W).
        
        Returns:
            tuple[Tensor]: Enhanced multi-scale features.
        """
        assert len(feats) == self.num_levels, \
            f"Expected {self.num_levels} feature levels, got {len(feats)}"
        
        enhanced_feats = []
        
        # Process each level
        for i, (feat, modules) in enumerate(zip(feats, self.level_modules)):
            # Apply ARF or standard conv
            if self.use_arf:
                out = modules['arf'](feat)
            else:
                out = modules['conv'](feat)
            
            # Apply AGE
            if self.use_age:
                out = modules['age'](out)
            
            enhanced_feats.append(out)
        
        # Cross-scale fusion (optional, can be disabled for efficiency)
        # For now, we return the enhanced features directly
        return tuple(enhanced_feats)
