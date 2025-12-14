# Copyright (c) OpenMMLab. All rights reserved.
"""Style Router for domain identification using Gram matrices.

Computes domain weights based on style similarity measured by Gram matrices
of feature maps.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class StyleRouter(nn.Module):
    """Style-based domain router using Gram matrices.
    
    Computes style vectors from feature maps using Gram matrices,
    maintains domain prototypes via EMA, and computes soft domain weights
    based on style similarity.
    
    Args:
        feature_channels (int): Number of channels in input features.
        style_dim (int): Dimension of style vectors. Defaults to 64.
        sigma (float): Temperature for softmax weights. Defaults to 0.5.
        ema_momentum (float): Momentum for EMA updates. Defaults to 0.1.
        update_mode (str): Update mode, 'ema' or 'mean'. Defaults to 'ema'.
        extract_layer (str): Which layer to extract features from.
            Options: 'encoder', 'decoder', 'neck'. Defaults to 'neck'.
    """
    
    def __init__(
        self,
        feature_channels: int,
        style_dim: int = 64,
        sigma: float = 0.5,
        ema_momentum: float = 0.1,
        update_mode: str = 'ema',
        extract_layer: str = 'neck',
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.style_dim = style_dim
        self.sigma = sigma
        self.ema_momentum = ema_momentum
        self.update_mode = update_mode
        self.extract_layer = extract_layer
        
        # Compute Gram matrix size
        # Gram matrix is symmetric, so we take upper triangular part
        # For C channels: C*(C+1)/2 unique elements
        gram_size = feature_channels * (feature_channels + 1) // 2
        
        # Project Gram features to style vector
        self.style_projector = nn.Linear(gram_size, style_dim)
        
        # Domain style prototypes (maintained via EMA)
        # Shape: (num_domains, style_dim)
        self.register_buffer('domain_prototypes', torch.zeros(0, style_dim))
        
        # Counters for mean updates
        self.register_buffer('domain_counts', torch.zeros(0, dtype=torch.long))
        
        self.num_domains = 0
        self.training_domain = None
    
    def add_domain(self) -> int:
        """Add a new domain.
        
        Returns:
            int: Index of the newly added domain.
        """
        # Initialize new domain prototype as zeros (will be updated during training)
        new_prototype = torch.zeros(1, self.style_dim, device=self.domain_prototypes.device)
        self.domain_prototypes = torch.cat([self.domain_prototypes, new_prototype], dim=0)
        
        # Initialize count
        new_count = torch.zeros(1, dtype=torch.long, device=self.domain_counts.device)
        self.domain_counts = torch.cat([self.domain_counts, new_count], dim=0)
        
        self.num_domains += 1
        return self.num_domains - 1
    
    def set_training_domain(self, domain_idx: int) -> None:
        """Set the current training domain for prototype updates.
        
        Args:
            domain_idx (int): Index of the training domain.
        """
        if domain_idx >= self.num_domains:
            raise ValueError(f"Domain {domain_idx} does not exist")
        self.training_domain = domain_idx
    
    def compute_gram_matrix(self, features: Tensor) -> Tensor:
        """Compute Gram matrix from features.
        
        Args:
            features (Tensor): Input features, shape (batch_size, C, H, W).
        
        Returns:
            Tensor: Gram matrix, shape (batch_size, C, C).
        """
        batch_size, C, H, W = features.shape
        
        # Reshape to (batch_size, C, H*W)
        features_flat = features.reshape(batch_size, C, H * W)
        
        # Compute Gram matrix: features @ features^T
        # (batch_size, C, H*W) @ (batch_size, H*W, C) -> (batch_size, C, C)
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
        
        # Normalize by spatial size
        gram = gram / (H * W)
        
        return gram
    
    def gram_to_vector(self, gram: Tensor) -> Tensor:
        """Convert Gram matrix to vector by taking upper triangular part.
        
        Args:
            gram (Tensor): Gram matrix, shape (batch_size, C, C).
        
        Returns:
            Tensor: Vectorized Gram features, shape (batch_size, C*(C+1)/2).
        """
        batch_size, C, _ = gram.shape
        
        # Get upper triangular indices (including diagonal)
        indices = torch.triu_indices(C, C, device=gram.device)
        
        # Extract upper triangular values
        gram_vec = gram[:, indices[0], indices[1]]
        
        return gram_vec
    
    def compute_style_vector(self, features: Tensor) -> Tensor:
        """Compute style vector from features.
        
        Args:
            features (Tensor): Input features, shape (batch_size, C, H, W).
        
        Returns:
            Tensor: Style vectors, shape (batch_size, style_dim).
        """
        # Compute Gram matrix
        gram = self.compute_gram_matrix(features)
        
        # Convert to vector
        gram_vec = self.gram_to_vector(gram)
        
        # Project to style space
        style_vec = self.style_projector(gram_vec)
        
        # L2 normalize
        style_vec = F.normalize(style_vec, p=2, dim=-1)
        
        return style_vec
    
    @torch.no_grad()
    def update_domain_prototype(self, style_vectors: Tensor, domain_idx: int) -> None:
        """Update domain prototype with new style vectors.
        
        Args:
            style_vectors (Tensor): Style vectors, shape (batch_size, style_dim).
            domain_idx (int): Index of the domain to update.
        """
        if not self.training:
            return
        
        if domain_idx >= self.num_domains:
            raise ValueError(f"Domain {domain_idx} does not exist")
        
        # Average style vectors in batch
        batch_mean = style_vectors.mean(dim=0)
        
        if self.update_mode == 'ema':
            # Exponential moving average
            if self.domain_counts[domain_idx] == 0:
                # First update, directly set
                self.domain_prototypes[domain_idx] = batch_mean
            else:
                # EMA update: μ_d = (1-α) * μ_d + α * batch_mean
                self.domain_prototypes[domain_idx] = (
                    (1 - self.ema_momentum) * self.domain_prototypes[domain_idx] +
                    self.ema_momentum * batch_mean
                )
        elif self.update_mode == 'mean':
            # Running mean
            count = self.domain_counts[domain_idx].item()
            self.domain_prototypes[domain_idx] = (
                (self.domain_prototypes[domain_idx] * count + batch_mean) / (count + 1)
            )
        
        self.domain_counts[domain_idx] += 1
    
    def compute_domain_weights(self, style_vectors: Tensor) -> Tensor:
        """Compute soft domain weights based on style similarity.
        
        Args:
            style_vectors (Tensor): Style vectors, shape (batch_size, style_dim).
        
        Returns:
            Tensor: Domain weights, shape (batch_size, num_domains).
        """
        if self.num_domains == 0:
            raise ValueError("No domains added yet")
        
        # Compute L2 distances to all domain prototypes
        # (batch_size, style_dim) vs (num_domains, style_dim)
        # -> (batch_size, num_domains)
        distances = torch.cdist(style_vectors, self.domain_prototypes, p=2)
        
        # Compute weights: w_d = softmax(-||s - μ_d||^2 / σ^2)
        weights = F.softmax(-distances.pow(2) / (self.sigma ** 2), dim=-1)
        
        return weights
    
    def forward(
        self,
        features: Tensor,
        update_prototype: bool = False,
        domain_idx: Optional[int] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass to compute domain weights.
        
        Args:
            features (Tensor): Input features, shape (batch_size, C, H, W).
            update_prototype (bool): Whether to update domain prototype. Defaults to False.
            domain_idx (int, optional): Domain index for prototype update.
        
        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                - domain_weights: Shape (batch_size, num_domains)
                - info_dict: Additional information (style_vectors, distances)
        """
        # Compute style vectors
        style_vectors = self.compute_style_vector(features)
        
        # Update domain prototype if in training and requested
        if update_prototype and domain_idx is not None:
            self.update_domain_prototype(style_vectors, domain_idx)
        
        # Compute domain weights
        if self.num_domains > 0:
            domain_weights = self.compute_domain_weights(style_vectors)
            
            # Compute distances for logging
            distances = torch.cdist(style_vectors, self.domain_prototypes, p=2)
        else:
            # No domains yet, return empty weights
            domain_weights = torch.zeros(
                features.size(0), 0,
                device=features.device,
                dtype=features.dtype
            )
            distances = torch.zeros_like(domain_weights)
        
        info_dict = {
            'style_vectors': style_vectors,
            'distances': distances,
        }
        
        return domain_weights, info_dict
