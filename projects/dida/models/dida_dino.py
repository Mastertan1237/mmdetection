# Copyright (c) OpenMMLab. All rights reserved.
"""DIDA DINO Detector for domain-incremental object detection.

Integrates DIDA head and style router for domain-incremental learning.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.detectors import DINO
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType


@MODELS.register_module()
class DIDADINO(DINO):
    """DIDA DINO detector for domain-incremental detection.
    
    Args:
        *args: Arguments for DINO.
        style_router_cfg (dict, optional): Config for style router.
        current_domain (int, optional): Current training domain index.
        domain_unknown_mode (bool): Enable domain-unknown inference. Defaults to False.
        **kwargs: Keyword arguments for DINO.
    """
    
    def __init__(
        self,
        *args,
        style_router_cfg: OptConfigType = None,
        current_domain: Optional[int] = None,
        domain_unknown_mode: bool = False,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        # Build style router if provided
        if style_router_cfg is not None:
            self.style_router = MODELS.build(style_router_cfg)
        else:
            self.style_router = None
        
        self.current_domain = current_domain
        self.domain_unknown_mode = domain_unknown_mode
        
        # Set domain in head if it's a DIDA head
        if hasattr(self.bbox_head, 'set_current_domain') and current_domain is not None:
            self.bbox_head.set_current_domain(current_domain)
    
    def set_current_domain(self, domain_idx: int) -> None:
        """Set current training domain.
        
        Args:
            domain_idx (int): Domain index.
        """
        self.current_domain = domain_idx
        if hasattr(self.bbox_head, 'set_current_domain'):
            self.bbox_head.set_current_domain(domain_idx)
        if self.style_router is not None:
            self.style_router.set_training_domain(domain_idx)
    
    def set_domain_unknown_mode(self, enable: bool = True) -> None:
        """Enable/disable domain-unknown inference mode.
        
        Args:
            enable (bool): Whether to enable.
        """
        self.domain_unknown_mode = enable
        if hasattr(self.bbox_head, 'set_domain_unknown_mode'):
            self.bbox_head.set_domain_unknown_mode(enable)
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images.
        
        This is overridden to store intermediate features for style computation.
        
        Args:
            batch_inputs (Tensor): Input images, shape (bs, C, H, W).
        
        Returns:
            tuple[Tensor]: Multi-scale features from neck.
        """
        # Extract features via backbone
        x = self.backbone(batch_inputs)
        
        # Store for style computation if needed
        if hasattr(self, '_store_backbone_features'):
            self._backbone_features = x
        
        # Extract features via neck
        if self.with_neck:
            x = self.neck(x)
        
        return x
    
    def compute_style_features(
        self,
        img_feats: Tuple[Tensor],
        update_prototype: bool = False
    ) -> Tuple[Tensor, Dict]:
        """Compute style features and domain weights.
        
        Args:
            img_feats (tuple[Tensor]): Multi-scale features from neck.
            update_prototype (bool): Whether to update domain prototypes.
        
        Returns:
            Tuple[Tensor, Dict]:
                - domain_weights: Shape (batch_size, num_domains)
                - style_info: Dictionary with style information
        """
        if self.style_router is None:
            return None, {}
        
        # Use first scale features for style computation
        # Shape: (bs, C, H, W)
        style_features = img_feats[0]
        
        # Compute domain weights
        domain_weights, style_info = self.style_router(
            style_features,
            update_prototype=update_prototype and self.training,
            domain_idx=self.current_domain if self.training else None
        )
        
        return domain_weights, style_info
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: OptSampleList) -> Dict:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            batch_inputs (Tensor): Input images, shape (bs, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Batch data samples.
        
        Returns:
            dict: A dictionary of loss components.
        """
        # Extract features
        img_feats = self.extract_feat(batch_inputs)
        
        # Compute style features and update prototypes
        if self.style_router is not None:
            domain_weights, style_info = self.compute_style_features(
                img_feats,
                update_prototype=True
            )
        
        # Forward transformer
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        
        # Add domain index to head inputs
        if hasattr(self.bbox_head, 'loss'):
            losses = self.bbox_head.loss(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples,
                domain_idx=self.current_domain
            )
        else:
            losses = self.bbox_head.loss(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples
            )
        
        return losses
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: OptSampleList,
                rescale: bool = True) -> OptSampleList:
        """Predict results from a batch of inputs and data samples.
        
        Args:
            batch_inputs (Tensor): Input images, shape (bs, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Batch data samples.
            rescale (bool): Whether to rescale predictions. Defaults to True.
        
        Returns:
            list[:obj:`DetDataSample`]: Detection results.
        """
        # Extract features
        img_feats = self.extract_feat(batch_inputs)
        
        # Compute style features for domain-unknown mode
        domain_prototypes = None
        if self.domain_unknown_mode and self.style_router is not None:
            domain_weights, style_info = self.compute_style_features(
                img_feats,
                update_prototype=False
            )
            
            # Get weighted domain prototypes if using DIDA head
            if hasattr(self.bbox_head, 'domain_residual') and \
               hasattr(self.bbox_head, 'clip_classifier'):
                clip_prototypes = self.bbox_head.clip_classifier.clip_prototypes
                domain_prototypes = self.bbox_head.domain_residual.get_multi_domain_prototypes(
                    clip_prototypes,
                    domain_weights
                )
        
        # Forward transformer
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        
        # Predict with domain information
        if hasattr(self.bbox_head, 'predict'):
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples,
                domain_idx=self.current_domain if not self.domain_unknown_mode else None,
                domain_prototypes=domain_prototypes,
                rescale=rescale
            )
        else:
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples,
                rescale=rescale
            )
        
        # Add predictions to data samples
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        
        return batch_data_samples
