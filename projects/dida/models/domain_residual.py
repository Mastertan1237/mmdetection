# Copyright (c) OpenMMLab. All rights reserved.
"""Domain Residual Prototype module for domain-incremental learning.

This module learns small residuals Δ_{c,d} for each domain and class,
enabling domain-specific adaptation while preserving shared knowledge.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class DomainResidualPrototype(nn.Module):
    """Domain Residual Prototype for domain-incremental learning.
    
    Learns domain-specific residuals Δ_{c,d} that are added to base prototypes
    to create domain-adapted prototypes: P_{c,d} = P_c + Δ_{c,d}
    
    Args:
        num_classes (int): Number of object classes.
        embed_dim (int): Embedding dimension. Defaults to 512.
        init_std (float): Standard deviation for initialization. Defaults to 0.01.
        magnitude_weight (float): Weight for magnitude regularization. Defaults to 0.1.
        orthogonal_weight (float): Weight for orthogonal regularization. Defaults to 0.1.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        init_std: float = 0.01,
        magnitude_weight: float = 0.1,
        orthogonal_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.magnitude_weight = magnitude_weight
        self.orthogonal_weight = orthogonal_weight
        
        # List of domain residuals, each shape (num_classes, embed_dim)
        self.domain_residuals = nn.ParameterList()
        
        # Track which domains are frozen
        self.register_buffer('frozen_domains', torch.tensor([], dtype=torch.long))
        self.num_domains = 0
    
    def add_domain(self) -> int:
        """Add a new domain with initialized residuals.
        
        Returns:
            int: Index of the newly added domain.
        """
        # Initialize new domain residuals with small values
        new_residual = nn.Parameter(
            torch.randn(self.num_classes, self.embed_dim) * self.init_std
        )
        self.domain_residuals.append(new_residual)
        self.num_domains += 1
        return self.num_domains - 1
    
    def freeze_domain(self, domain_idx: int) -> None:
        """Freeze residuals for a specific domain.
        
        Args:
            domain_idx (int): Index of the domain to freeze.
        """
        if domain_idx >= self.num_domains:
            raise ValueError(f"Domain {domain_idx} does not exist")
        
        # Mark domain as frozen
        if domain_idx not in self.frozen_domains:
            self.frozen_domains = torch.cat([
                self.frozen_domains,
                torch.tensor([domain_idx], device=self.frozen_domains.device)
            ])
        
        # Set requires_grad to False
        self.domain_residuals[domain_idx].requires_grad = False
    
    def unfreeze_domain(self, domain_idx: int) -> None:
        """Unfreeze residuals for a specific domain.
        
        Args:
            domain_idx (int): Index of the domain to unfreeze.
        """
        if domain_idx >= self.num_domains:
            raise ValueError(f"Domain {domain_idx} does not exist")
        
        # Remove from frozen list
        mask = self.frozen_domains != domain_idx
        self.frozen_domains = self.frozen_domains[mask]
        
        # Set requires_grad to True
        self.domain_residuals[domain_idx].requires_grad = True
    
    def get_domain_prototypes(
        self,
        base_prototypes: Tensor,
        domain_idx: int
    ) -> Tensor:
        """Get domain-adapted prototypes.
        
        Args:
            base_prototypes (Tensor): Base prototypes, shape (num_classes, embed_dim).
            domain_idx (int): Index of the domain.
        
        Returns:
            Tensor: Domain-adapted prototypes, shape (num_classes, embed_dim).
        """
        if domain_idx >= self.num_domains:
            raise ValueError(f"Domain {domain_idx} does not exist")
        
        # P_{c,d} = P_c + Δ_{c,d}
        return base_prototypes + self.domain_residuals[domain_idx]
    
    def get_multi_domain_prototypes(
        self,
        base_prototypes: Tensor,
        domain_weights: Tensor
    ) -> Tensor:
        """Get weighted combination of domain prototypes.
        
        Args:
            base_prototypes (Tensor): Base prototypes, shape (num_classes, embed_dim).
            domain_weights (Tensor): Domain weights, shape (batch_size, num_domains).
        
        Returns:
            Tensor: Weighted prototypes, shape (batch_size, num_classes, embed_dim).
        """
        # Stack all domain residuals: (num_domains, num_classes, embed_dim)
        all_residuals = torch.stack([r for r in self.domain_residuals], dim=0)
        
        # Weighted sum: (bs, num_domains, 1, 1) * (1, num_domains, num_classes, embed_dim)
        # -> (bs, num_domains, num_classes, embed_dim) -> (bs, num_classes, embed_dim)
        weighted_residuals = torch.einsum(
            'bd,dce->bce',
            domain_weights,
            all_residuals
        )
        
        # Add to base prototypes
        return base_prototypes.unsqueeze(0) + weighted_residuals
    
    def compute_regularization_loss(self) -> Tuple[Tensor, Tensor]:
        """Compute regularization losses on residuals.
        
        Returns:
            Tuple[Tensor, Tensor]: (magnitude_loss, orthogonal_loss)
                - magnitude_loss: L2 norm of residuals to keep them small
                - orthogonal_loss: Encourage orthogonality between domain residuals
        """
        if self.num_domains == 0:
            device = next(self.parameters()).device
            return torch.tensor(0., device=device), torch.tensor(0., device=device)
        
        # Stack all residuals: (num_domains, num_classes, embed_dim)
        all_residuals = torch.stack([r for r in self.domain_residuals], dim=0)
        
        # Magnitude loss: encourage small residuals
        # ||Δ_{c,d}||^2 for all c, d
        magnitude_loss = (all_residuals ** 2).mean()
        
        # Orthogonal loss: encourage different domains to learn orthogonal residuals
        if self.num_domains > 1:
            # Flatten class and embedding dimensions
            # (num_domains, num_classes * embed_dim)
            flat_residuals = all_residuals.reshape(self.num_domains, -1)
            
            # Compute Gram matrix: (num_domains, num_domains)
            gram = torch.mm(flat_residuals, flat_residuals.t())
            
            # Normalize by magnitude
            norms = torch.sqrt(torch.diag(gram))
            normalized_gram = gram / (norms.unsqueeze(1) * norms.unsqueeze(0) + 1e-8)
            
            # Off-diagonal elements should be close to 0
            mask = 1 - torch.eye(self.num_domains, device=gram.device)
            orthogonal_loss = (normalized_gram * mask).abs().mean()
        else:
            orthogonal_loss = torch.tensor(0., device=all_residuals.device)
        
        return magnitude_loss * self.magnitude_weight, orthogonal_loss * self.orthogonal_weight
    
    def forward(
        self,
        base_prototypes: Tensor,
        domain_idx: Optional[int] = None,
        domain_weights: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass to get domain-adapted prototypes.
        
        Args:
            base_prototypes (Tensor): Base prototypes, shape (num_classes, embed_dim).
            domain_idx (int, optional): Single domain index for domain-known inference.
            domain_weights (Tensor, optional): Domain weights for domain-unknown inference,
                shape (batch_size, num_domains).
        
        Returns:
            Tensor: Domain-adapted prototypes.
                If domain_idx provided: (num_classes, embed_dim)
                If domain_weights provided: (batch_size, num_classes, embed_dim)
        """
        if domain_idx is not None:
            return self.get_domain_prototypes(base_prototypes, domain_idx)
        elif domain_weights is not None:
            return self.get_multi_domain_prototypes(base_prototypes, domain_weights)
        else:
            raise ValueError("Either domain_idx or domain_weights must be provided")
