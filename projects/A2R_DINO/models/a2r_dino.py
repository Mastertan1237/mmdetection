# Copyright (c) OpenMMLab. All rights reserved.
"""A2R-DINO: Adaptive Attention Receptive-field DINO detector."""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.models.detectors import DINO
from mmdet.models.detectors.deformable_detr import \
    MultiScaleDeformableAttention
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from .a2r_layers import MultiScaleAdaptiveModule


@MODELS.register_module()
class A2RDINO(DINO):
    """A2R-DINO: Adaptive Attention Receptive-field DINO detector.
    
    This detector extends DINO with adaptive receptive field (ARF) and
    attention-guided enhancement (AGE) mechanisms for improved detection.
    
    Args:
        a2r_cfg (dict, optional): Configuration for A2R modules. Should contain:
            - use_arf (bool): Whether to use adaptive receptive field. 
              Defaults to True.
            - use_age (bool): Whether to use attention-guided enhancement.
              Defaults to True.
            - arf_groups (int): Number of groups for ARF. Defaults to 4.
            - attention_type (str): Type of attention ('spatial', 'channel', 
              'both'). Defaults to 'both'.
            - use_adaptive_encoder (bool): Use adaptive encoder. Defaults to False.
            - use_adaptive_decoder (bool): Use adaptive decoder. Defaults to False.
        **kwargs: Other arguments for DINO.
    """
    
    def __init__(
        self,
        *args,
        a2r_cfg: OptConfigType = None,
        **kwargs
    ) -> None:
        # Set default A2R config
        self.a2r_cfg = dict(
            use_arf=True,
            use_age=True,
            arf_groups=4,
            attention_type='both',
            use_adaptive_encoder=False,
            use_adaptive_decoder=False
        )
        if a2r_cfg is not None:
            self.a2r_cfg.update(a2r_cfg)
        
        super().__init__(*args, **kwargs)
    
    def _init_layers(self) -> None:
        """Initialize layers with A2R components."""
        # Initialize parent layers
        super()._init_layers()
        
        # Add A2R multi-scale adaptive module for feature enhancement
        if self.a2r_cfg['use_arf'] or self.a2r_cfg['use_age']:
            self.a2r_module = MultiScaleAdaptiveModule(
                in_channels=self.embed_dims,
                out_channels=self.embed_dims,
                num_levels=self.num_feature_levels,
                use_arf=self.a2r_cfg['use_arf'],
                use_age=self.a2r_cfg['use_age'],
                arf_groups=self.a2r_cfg['arf_groups'],
                attention_type=self.a2r_cfg['attention_type']
            )
        
        # Optional: Replace encoder/decoder with A2R versions
        # This is controlled by config and kept optional for flexibility
        if self.a2r_cfg.get('use_adaptive_encoder', False):
            from .a2r_transformer import A2RTransformerEncoder
            encoder_cfg = self.encoder.to_dict() if hasattr(
                self.encoder, 'to_dict') else {}
            encoder_cfg['type'] = 'A2RTransformerEncoder'
            self.encoder = MODELS.build(encoder_cfg)
        
        if self.a2r_cfg.get('use_adaptive_decoder', False):
            from .a2r_transformer import A2RTransformerDecoder
            decoder_cfg = self.decoder.to_dict() if hasattr(
                self.decoder, 'to_dict') else {}
            decoder_cfg['type'] = 'A2RTransformerDecoder'
            self.decoder = MODELS.build(decoder_cfg)
    
    def init_weights(self) -> None:
        """Initialize weights for Transformer and A2R components."""
        super().init_weights()
        
        # Initialize A2R module weights if present
        if hasattr(self, 'a2r_module'):
            for m in self.a2r_module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer with A2R enhancement.
        
        This method enhances the features with A2R modules before feeding
        them to the transformer encoder.
        
        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. Defaults to None.
        
        Returns:
            dict: The dictionary of bbox_head function inputs.
        """
        # Apply A2R enhancement to features if enabled
        if hasattr(self, 'a2r_module') and (
            self.a2r_cfg['use_arf'] or self.a2r_cfg['use_age']
        ):
            # Enhance features with A2R module
            img_feats = self.a2r_module(img_feats)
        
        # Continue with standard DINO forward
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features with optional A2R processing.
        
        Args:
            batch_inputs (Tensor): Input images with shape (bs, C, H, W).
        
        Returns:
            tuple[Tensor]: Multi-level features from the neck.
        """
        # Extract features using parent method
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        
        return x
