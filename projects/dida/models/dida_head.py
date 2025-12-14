# Copyright (c) OpenMMLab. All rights reserved.
"""DIDA DINO Head for domain-incremental object detection.

Extends DINOHead with domain-incremental capabilities including:
- CLIP-based classification with domain residuals
- Objectness scoring
- Style-based domain routing
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.dense_heads import DINOHead
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList

from .clip_classifier import CLIPPrototypeClassifier
from .domain_residual import DomainResidualPrototype
from .objectness_head import MultiLevelObjectnessHead


@MODELS.register_module()
class DIDAHead(DINOHead):
    """DIDA DINO Head for domain-incremental detection.
    
    Args:
        *args: Arguments for DINOHead.
        clip_classifier_cfg (dict): Config for CLIP classifier.
        domain_residual_cfg (dict): Config for domain residual prototype.
        objectness_cfg (dict, optional): Config for objectness head.
        use_objectness (bool): Whether to use objectness branch. Defaults to True.
        current_domain (int, optional): Current training domain index.
        **kwargs: Keyword arguments for DINOHead.
    """
    
    def __init__(
        self,
        *args,
        clip_classifier_cfg: dict,
        domain_residual_cfg: dict,
        objectness_cfg: Optional[dict] = None,
        use_objectness: bool = True,
        current_domain: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        # Build CLIP classifier
        self.clip_classifier = MODELS.build(clip_classifier_cfg)
        
        # Build domain residual prototype
        self.domain_residual = MODELS.build(domain_residual_cfg)
        
        # Build objectness head if enabled
        self.use_objectness = use_objectness
        if use_objectness:
            if objectness_cfg is None:
                objectness_cfg = dict(
                    type='MultiLevelObjectnessHead',
                    embed_dim=self.embed_dims,
                    num_decoder_layers=self.num_pred_layer,
                )
            self.objectness_head = MODELS.build(objectness_cfg)
        
        # Current domain for training
        self.current_domain = current_domain
        self._domain_unknown_mode = False
    
    def set_current_domain(self, domain_idx: int) -> None:
        """Set current training domain.
        
        Args:
            domain_idx (int): Domain index.
        """
        self.current_domain = domain_idx
    
    def set_domain_unknown_mode(self, enable: bool = True) -> None:
        """Enable/disable domain-unknown inference mode.
        
        Args:
            enable (bool): Whether to enable domain-unknown mode.
        """
        self._domain_unknown_mode = enable
    
    def forward_cls_branch(
        self,
        hidden_states: Tensor,
        domain_idx: Optional[int] = None,
        domain_prototypes: Optional[Tensor] = None
    ) -> Tensor:
        """Forward classification branch using CLIP classifier.
        
        Args:
            hidden_states (Tensor): Query features from decoder,
                shape (num_layers, batch_size, num_queries, embed_dim).
            domain_idx (int, optional): Domain index for domain-known inference.
            domain_prototypes (Tensor, optional): Batch-wise prototypes for
                domain-unknown inference, shape (batch_size, num_classes, clip_dim).
        
        Returns:
            Tensor: Classification scores,
                shape (num_layers, batch_size, num_queries, num_classes).
        """
        num_layers, bs, num_queries, embed_dim = hidden_states.shape
        
        # Get base CLIP prototypes
        clip_prototypes = self.clip_classifier.clip_prototypes
        
        # Get domain-adapted prototypes
        if domain_prototypes is not None:
            # Domain-unknown mode: use provided batch-wise prototypes
            prototypes = domain_prototypes
        elif domain_idx is not None:
            # Domain-known mode: get single-domain prototypes
            prototypes = self.domain_residual.get_domain_prototypes(
                clip_prototypes, domain_idx
            )
        else:
            # No domain adaptation, use base prototypes
            prototypes = clip_prototypes
        
        # Process each layer
        all_cls_scores = []
        for layer_idx in range(num_layers):
            layer_queries = hidden_states[layer_idx]  # (bs, nq, embed_dim)
            
            # Get classification scores via CLIP classifier
            cls_scores = self.clip_classifier(
                layer_queries,
                domain_idx=domain_idx if domain_prototypes is None else None,
                prototypes=prototypes
            )
            all_cls_scores.append(cls_scores)
        
        # Stack: (num_layers, bs, nq, num_classes)
        all_cls_scores = torch.stack(all_cls_scores, dim=0)
        
        return all_cls_scores
    
    def forward_objectness_branch(self, hidden_states: Tensor) -> Tensor:
        """Forward objectness branch.
        
        Args:
            hidden_states (Tensor): Query features,
                shape (num_layers, batch_size, num_queries, embed_dim).
        
        Returns:
            Tensor: Objectness scores,
                shape (num_layers, batch_size, num_queries, 1).
        """
        if not self.use_objectness:
            return None
        
        return self.objectness_head(hidden_states)
    
    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        domain_idx: Optional[int] = None,
        domain_prototypes: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward function.
        
        Args:
            hidden_states (Tensor): Hidden states from decoder,
                shape (num_layers, bs, num_queries, embed_dim).
            references (list[Tensor]): Reference boxes from decoder.
            domain_idx (int, optional): Domain index.
            domain_prototypes (Tensor, optional): Domain prototypes for
                domain-unknown mode.
        
        Returns:
            Tuple[Tensor, Tensor]:
                - all_layers_cls_scores: (num_layers, bs, nq, num_classes)
                - all_layers_bbox_preds: (num_layers, bs, nq, 4)
        """
        # Classification using CLIP classifier
        all_layers_cls_scores = self.forward_cls_branch(
            hidden_states,
            domain_idx=domain_idx,
            domain_prototypes=domain_prototypes
        )
        
        # Bbox regression (unchanged from DINOHead)
        all_layers_bbox_preds = []
        for layer_idx in range(hidden_states.shape[0]):
            layer_hidden = hidden_states[layer_idx]
            layer_ref = references[layer_idx + 1]
            
            # Apply bbox regression branch
            bbox_pred = self.reg_branches[layer_idx](layer_hidden)
            
            # Update reference
            if layer_ref.shape[-1] == 4:
                bbox_pred = bbox_pred + layer_ref.inverse_sigmoid()
            else:
                assert layer_ref.shape[-1] == 2
                bbox_pred[..., :2] = bbox_pred[..., :2] + layer_ref.inverse_sigmoid()
            
            all_layers_bbox_preds.append(bbox_pred)
        
        all_layers_bbox_preds = torch.stack(all_layers_bbox_preds, dim=0)
        
        return all_layers_cls_scores, all_layers_bbox_preds
    
    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
        domain_idx: Optional[int] = None
    ) -> dict:
        """Compute losses.
        
        Args:
            hidden_states (Tensor): Hidden states from decoder.
            references (list[Tensor]): Reference boxes.
            enc_outputs_class (Tensor): Encoder classification outputs.
            enc_outputs_coord (Tensor): Encoder bbox outputs.
            batch_data_samples (list[:obj:`DetDataSample`]): Batch data samples.
            dn_meta (Dict[str, int]): Denoising meta information.
            domain_idx (int, optional): Current training domain index.
        
        Returns:
            dict: Loss dictionary.
        """
        # Use current domain if not specified
        if domain_idx is None:
            domain_idx = self.current_domain
        
        # Forward pass
        outs = self(hidden_states, references, domain_idx=domain_idx)
        
        # Prepare loss inputs
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
        
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        
        # Add regularization losses from domain residuals
        if self.training and domain_idx is not None:
            mag_loss, orth_loss = self.domain_residual.compute_regularization_loss()
            losses['loss_residual_mag'] = mag_loss
            losses['loss_residual_orth'] = orth_loss
        
        # Objectness loss
        if self.use_objectness and self.training:
            objectness_scores = self.forward_objectness_branch(hidden_states)
            # TODO: Implement objectness loss computation
            # This would require objectness targets from the assigner
            # For now, we skip it and it can be added later
        
        return losses
    
    def predict(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        batch_data_samples: SampleList,
        domain_idx: Optional[int] = None,
        domain_prototypes: Optional[Tensor] = None,
        rescale: bool = True
    ) -> InstanceList:
        """Predict boxes and labels.
        
        Args:
            hidden_states (Tensor): Hidden states from decoder.
            references (list[Tensor]): Reference boxes.
            batch_data_samples (list[:obj:`DetDataSample`]): Batch data samples.
            domain_idx (int, optional): Domain index for domain-known inference.
            domain_prototypes (Tensor, optional): Prototypes for domain-unknown mode.
            rescale (bool): Whether to rescale predictions. Defaults to True.
        
        Returns:
            list[:obj:`InstanceData`]: Detection results.
        """
        # Forward pass
        all_layers_cls_scores, all_layers_bbox_preds = self(
            hidden_states,
            references,
            domain_idx=domain_idx,
            domain_prototypes=domain_prototypes
        )
        
        # Use last layer predictions
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        
        # Post-process predictions
        result_list = self.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_data_samples,
            rescale=rescale
        )
        
        return result_list
