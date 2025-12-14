# Copyright (c) OpenMMLab. All rights reserved.
"""Freeze Hook for domain-incremental training.

Controls which parameters are trainable at each training stage.
"""
from typing import List, Optional, Sequence

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class FreezeHook(Hook):
    """Hook to freeze specific parameters during training.
    
    For domain-incremental learning, this hook freezes:
    - Stage 0: Train all parameters (source domain)
    - Stage d > 0: Freeze backbone, neck, transformer, bbox regression,
                   and previous domain parameters. Only train current domain
                   parameters (Δ_{c,d}, U_d, V_d, etc.)
    
    Args:
        stage (int): Training stage (0 for source, >0 for incremental).
        freeze_backbone (bool): Freeze backbone. Defaults to True for stage > 0.
        freeze_neck (bool): Freeze neck. Defaults to True for stage > 0.
        freeze_transformer (bool): Freeze transformer. Defaults to True for stage > 0.
        freeze_bbox_head (bool): Freeze bbox regression. Defaults to True for stage > 0.
        freeze_base_classifier (bool): Freeze base classifier projection. Defaults to True.
        trainable_modules (List[str], optional): Additional modules to keep trainable.
        verbose (bool): Print freezing information. Defaults to True.
    """
    
    priority = 'VERY_HIGH'  # Run before optimizer initialization
    
    def __init__(
        self,
        stage: int = 0,
        freeze_backbone: Optional[bool] = None,
        freeze_neck: Optional[bool] = None,
        freeze_transformer: Optional[bool] = None,
        freeze_bbox_head: Optional[bool] = None,
        freeze_base_classifier: Optional[bool] = None,
        trainable_modules: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.verbose = verbose
        
        # Set defaults based on stage
        if stage == 0:
            # Source domain: train everything
            self.freeze_backbone = False if freeze_backbone is None else freeze_backbone
            self.freeze_neck = False if freeze_neck is None else freeze_neck
            self.freeze_transformer = False if freeze_transformer is None else freeze_transformer
            self.freeze_bbox_head = False if freeze_bbox_head is None else freeze_bbox_head
            self.freeze_base_classifier = False if freeze_base_classifier is None else freeze_base_classifier
        else:
            # Incremental domains: freeze most things
            self.freeze_backbone = True if freeze_backbone is None else freeze_backbone
            self.freeze_neck = True if freeze_neck is None else freeze_neck
            self.freeze_transformer = True if freeze_transformer is None else freeze_transformer
            self.freeze_bbox_head = True if freeze_bbox_head is None else freeze_bbox_head
            self.freeze_base_classifier = True if freeze_base_classifier is None else freeze_base_classifier
        
        self.trainable_modules = trainable_modules or []
    
    def _freeze_module(self, module: nn.Module, name: str = '') -> None:
        """Freeze all parameters in a module.
        
        Args:
            module (nn.Module): Module to freeze.
            name (str): Name of the module for logging.
        """
        for param in module.parameters():
            param.requires_grad = False
        
        if self.verbose and name:
            print(f'[FreezeHook] Frozen module: {name}')
    
    def _unfreeze_module(self, module: nn.Module, name: str = '') -> None:
        """Unfreeze all parameters in a module.
        
        Args:
            module (nn.Module): Module to unfreeze.
            name (str): Name of the module for logging.
        """
        for param in module.parameters():
            param.requires_grad = True
        
        if self.verbose and name:
            print(f'[FreezeHook] Unfrozen module: {name}')
    
    def _check_module_exists(self, model: nn.Module, attr_path: str) -> bool:
        """Check if a nested module exists.
        
        Args:
            model (nn.Module): Root model.
            attr_path (str): Dot-separated attribute path (e.g., 'bbox_head.reg_branches').
        
        Returns:
            bool: Whether the module exists.
        """
        parts = attr_path.split('.')
        current = model
        for part in parts:
            if not hasattr(current, part):
                return False
            current = getattr(current, part)
        return True
    
    def _get_module(self, model: nn.Module, attr_path: str) -> Optional[nn.Module]:
        """Get a nested module.
        
        Args:
            model (nn.Module): Root model.
            attr_path (str): Dot-separated attribute path.
        
        Returns:
            Optional[nn.Module]: The module, or None if it doesn't exist.
        """
        if not self._check_module_exists(model, attr_path):
            return None
        
        parts = attr_path.split('.')
        current = model
        for part in parts:
            current = getattr(current, part)
        return current
    
    def before_train(self, runner: Runner) -> None:
        """Freeze parameters before training starts.
        
        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        
        if self.verbose:
            print(f'\n[FreezeHook] Stage {self.stage} - Freezing parameters...')
        
        # Freeze backbone
        if self.freeze_backbone and hasattr(model, 'backbone'):
            self._freeze_module(model.backbone, 'backbone')
        
        # Freeze neck
        if self.freeze_neck and hasattr(model, 'neck'):
            self._freeze_module(model.neck, 'neck')
        
        # Freeze transformer (encoder + decoder)
        if self.freeze_transformer:
            if hasattr(model, 'encoder'):
                self._freeze_module(model.encoder, 'encoder')
            if hasattr(model, 'decoder'):
                self._freeze_module(model.decoder, 'decoder')
            # Also freeze positional encoding and related
            if hasattr(model, 'positional_encoding'):
                self._freeze_module(model.positional_encoding, 'positional_encoding')
            if hasattr(model, 'level_embed'):
                model.level_embed.requires_grad = False
            if hasattr(model, 'query_embedding'):
                self._freeze_module(model.query_embedding, 'query_embedding')
        
        # Freeze bbox head regression branches
        if self.freeze_bbox_head:
            reg_branches = self._get_module(model, 'bbox_head.reg_branches')
            if reg_branches is not None:
                self._freeze_module(reg_branches, 'bbox_head.reg_branches')
        
        # Freeze base classifier projection
        if self.freeze_base_classifier:
            shared_proj = self._get_module(model, 'bbox_head.clip_classifier.shared_projection')
            if shared_proj is not None:
                self._freeze_module(shared_proj, 'bbox_head.clip_classifier.shared_projection')
        
        # For incremental stages, freeze previous domain parameters
        if self.stage > 0:
            # Freeze old domain residuals
            domain_residual = self._get_module(model, 'bbox_head.domain_residual')
            if domain_residual is not None:
                for i in range(self.stage):
                    if i < len(domain_residual.domain_residuals):
                        domain_residual.freeze_domain(i)
                        if self.verbose:
                            print(f'[FreezeHook] Frozen domain {i} residuals')
            
            # Freeze old domain projections (low-rank)
            clip_classifier = self._get_module(model, 'bbox_head.clip_classifier')
            if clip_classifier is not None and hasattr(clip_classifier, 'domain_U'):
                for i in range(self.stage):
                    if i < len(clip_classifier.domain_U):
                        clip_classifier.freeze_domain(i)
                        if self.verbose:
                            print(f'[FreezeHook] Frozen domain {i} projections')
        
        # Unfreeze explicitly trainable modules
        for module_name in self.trainable_modules:
            module = self._get_module(model, module_name)
            if module is not None:
                self._unfreeze_module(module, module_name)
        
        # Print summary
        if self.verbose:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            print(f'\n[FreezeHook] Parameter Summary:')
            print(f'  Total parameters: {total_params:,}')
            print(f'  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)')
            print(f'  Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)')
            print()
