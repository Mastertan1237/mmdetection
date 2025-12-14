# Copyright (c) OpenMMLab. All rights reserved.
"""CLIP-based prototype classifier for domain-incremental detection.

Uses cosine similarity between query features and CLIP text prototypes
for classification, with optional domain-specific projection.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class CLIPPrototypeClassifier(nn.Module):
    """CLIP Prototype Classifier with domain-specific projections.
    
    Supports two projection modes:
    - 'shared': W_d = W_0 for all domains (no domain-specific projection)
    - 'low_rank': W_d = W_0 + U_d @ V_d^T (low-rank adaptation)
    
    Args:
        num_classes (int): Number of object classes.
        clip_dim (int): Dimension of CLIP embeddings. Defaults to 512.
        query_dim (int): Dimension of query features. Defaults to 256.
        projection_mode (str): Projection mode, 'shared' or 'low_rank'. Defaults to 'shared'.
        low_rank_dim (int): Rank for low-rank projection. Defaults to 4.
        cosine_scale (float): Scale factor for cosine similarity. Defaults to 20.0.
        clip_prototypes_path (str, optional): Path to pre-extracted CLIP prototypes.
        normalize_prototypes (bool): Whether to L2-normalize prototypes. Defaults to True.
    """
    
    def __init__(
        self,
        num_classes: int,
        clip_dim: int = 512,
        query_dim: int = 256,
        projection_mode: str = 'shared',
        low_rank_dim: int = 4,
        cosine_scale: float = 20.0,
        clip_prototypes_path: Optional[str] = None,
        normalize_prototypes: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.clip_dim = clip_dim
        self.query_dim = query_dim
        self.projection_mode = projection_mode
        self.low_rank_dim = low_rank_dim
        self.cosine_scale = cosine_scale
        self.normalize_prototypes = normalize_prototypes
        
        # Shared projection W_0: query_dim -> clip_dim
        self.shared_projection = nn.Linear(query_dim, clip_dim)
        
        # Domain-specific low-rank projections
        if projection_mode == 'low_rank':
            # U_d: (query_dim, low_rank_dim), V_d: (clip_dim, low_rank_dim)
            self.domain_U = nn.ParameterList()
            self.domain_V = nn.ParameterList()
        elif projection_mode != 'shared':
            raise ValueError(f"Unknown projection_mode: {projection_mode}")
        
        # CLIP text prototypes (num_classes, clip_dim)
        # Use register_buffer so they're saved in checkpoint but not trained
        self.register_buffer('clip_prototypes', torch.zeros(num_classes, clip_dim))
        
        # Load pre-extracted prototypes if provided
        if clip_prototypes_path is not None:
            self.load_clip_prototypes(clip_prototypes_path)
        
        self.num_domains = 0
    
    def load_clip_prototypes(self, path: str) -> None:
        """Load pre-extracted CLIP text prototypes.
        
        Args:
            path (str): Path to .pt file containing prototypes.
        """
        data = torch.load(path, map_location='cpu')
        
        if isinstance(data, dict) and 'prototypes' in data:
            prototypes = data['prototypes']
        else:
            prototypes = data
        
        # Ensure correct shape
        assert prototypes.shape == (self.num_classes, self.clip_dim), \
            f"Expected shape ({self.num_classes}, {self.clip_dim}), got {prototypes.shape}"
        
        # Normalize if required
        if self.normalize_prototypes:
            prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        self.clip_prototypes.copy_(prototypes)
    
    def add_domain(self) -> int:
        """Add a new domain with low-rank projection parameters.
        
        Returns:
            int: Index of the newly added domain.
        """
        if self.projection_mode == 'low_rank':
            # Initialize U_d and V_d with small random values
            U = nn.Parameter(torch.randn(self.query_dim, self.low_rank_dim) * 0.01)
            V = nn.Parameter(torch.randn(self.clip_dim, self.low_rank_dim) * 0.01)
            self.domain_U.append(U)
            self.domain_V.append(V)
        
        self.num_domains += 1
        return self.num_domains - 1
    
    def freeze_domain(self, domain_idx: int) -> None:
        """Freeze projection parameters for a specific domain.
        
        Args:
            domain_idx (int): Index of the domain to freeze.
        """
        if self.projection_mode == 'low_rank':
            if domain_idx >= self.num_domains:
                raise ValueError(f"Domain {domain_idx} does not exist")
            self.domain_U[domain_idx].requires_grad = False
            self.domain_V[domain_idx].requires_grad = False
    
    def project_queries(
        self,
        queries: Tensor,
        domain_idx: Optional[int] = None
    ) -> Tensor:
        """Project query features to CLIP space.
        
        Args:
            queries (Tensor): Query features, shape (batch_size, num_queries, query_dim).
            domain_idx (int, optional): Domain index for low-rank projection.
        
        Returns:
            Tensor: Projected features, shape (batch_size, num_queries, clip_dim).
        """
        # Apply shared projection: W_0 @ q
        projected = self.shared_projection(queries)
        
        # Add low-rank domain-specific projection if applicable
        if self.projection_mode == 'low_rank' and domain_idx is not None:
            if domain_idx >= self.num_domains:
                raise ValueError(f"Domain {domain_idx} does not exist")
            
            # Compute U_d @ V_d^T @ (W_0 @ q)
            # U_d: (query_dim, low_rank_dim), V_d: (clip_dim, low_rank_dim)
            U = self.domain_U[domain_idx]
            V = self.domain_V[domain_idx]
            
            # Alternative: add residual to queries before projection
            # delta = queries @ U @ V^T  # (bs, nq, query_dim)
            # projected = self.shared_projection(queries + delta)
            
            # Current: add residual in CLIP space
            # delta_clip = (W_0 @ q) @ V @ V^T = projected @ V @ V^T
            # But we want: queries @ U @ V^T
            # So compute: queries @ U @ V^T directly
            delta = torch.matmul(queries, U)  # (bs, nq, low_rank_dim)
            delta = torch.matmul(delta, V.t())  # (bs, nq, clip_dim)
            projected = projected + delta
        
        # L2 normalize
        projected = F.normalize(projected, p=2, dim=-1)
        
        return projected
    
    def forward(
        self,
        queries: Tensor,
        domain_idx: Optional[int] = None,
        prototypes: Optional[Tensor] = None
    ) -> Tensor:
        """Compute classification scores via cosine similarity.
        
        Args:
            queries (Tensor): Query features, shape (batch_size, num_queries, query_dim).
            domain_idx (int, optional): Domain index for projection.
            prototypes (Tensor, optional): Override prototypes, shape (num_classes, clip_dim)
                or (batch_size, num_classes, clip_dim) for domain-unknown mode.
        
        Returns:
            Tensor: Classification scores, shape (batch_size, num_queries, num_classes).
        """
        # Project queries to CLIP space
        projected_queries = self.project_queries(queries, domain_idx)
        
        # Use provided prototypes or default CLIP prototypes
        if prototypes is None:
            prototypes = self.clip_prototypes
        
        # Normalize prototypes
        if prototypes.dim() == 2:
            # Single set of prototypes for all batch items
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            # Compute cosine similarity: (bs, nq, clip_dim) x (num_classes, clip_dim)^T
            # -> (bs, nq, num_classes)
            scores = torch.matmul(projected_queries, prototypes.t())
        else:
            # Batch-wise prototypes (bs, num_classes, clip_dim)
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            # Compute: (bs, nq, clip_dim) x (bs, num_classes, clip_dim)^T
            # -> (bs, nq, num_classes)
            scores = torch.einsum('bqd,bcd->bqc', projected_queries, prototypes)
        
        # Scale scores
        scores = scores * self.cosine_scale
        
        return scores
    
    def forward_multi_domain(
        self,
        queries: Tensor,
        domain_prototypes: Tensor,
        domain_weights: Tensor
    ) -> Tensor:
        """Forward with domain-unknown inference using weighted prototypes.
        
        Args:
            queries (Tensor): Query features, shape (batch_size, num_queries, query_dim).
            domain_prototypes (Tensor): Per-domain prototypes from domain residuals,
                shape (batch_size, num_classes, clip_dim).
            domain_weights (Tensor): Domain weights, shape (batch_size, num_domains).
        
        Returns:
            Tensor: Classification scores, shape (batch_size, num_queries, num_classes).
        """
        # For multi-domain, use shared projection (no specific domain adaptation)
        projected_queries = self.project_queries(queries, domain_idx=None)
        
        # Use batch-wise prototypes
        prototypes = F.normalize(domain_prototypes, p=2, dim=-1)
        
        # Compute cosine similarity
        scores = torch.einsum('bqd,bcd->bqc', projected_queries, prototypes)
        scores = scores * self.cosine_scale
        
        return scores
