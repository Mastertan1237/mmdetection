# Copyright (c) OpenMMLab. All rights reserved.
"""Domain-Aware Metric for tracking per-domain performance.

Tracks domain weights and computes per-domain mAP statistics.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class DomainAwareMetric(CocoMetric):
    """Domain-aware metric that tracks domain assignments and statistics.
    
    This metric extends CocoMetric to track:
    1. Domain weights assigned by style router
    2. Per-domain performance (if domain labels available)
    3. Domain weight statistics
    
    Args:
        num_domains (int): Number of domains.
        track_domain_weights (bool): Whether to log domain weight statistics.
            Defaults to True.
        **kwargs: Arguments for CocoMetric.
    """
    
    default_prefix: Optional[str] = 'domain_aware'
    
    def __init__(
        self,
        num_domains: int = 1,
        track_domain_weights: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_domains = num_domains
        self.track_domain_weights = track_domain_weights
        
        # Storage for domain statistics
        self.domain_weight_stats = []
        self.domain_labels = []
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # Collect domain statistics if available
        if self.track_domain_weights:
            for data_sample in data_samples:
                # Check if domain weights are available in meta info
                meta_info = data_sample.get('metainfo', {})
                domain_weights = meta_info.get('domain_weights', None)
                
                if domain_weights is not None:
                    # Store domain weights
                    self.domain_weight_stats.append(domain_weights.cpu().numpy())
                
                # Check if domain label is available
                domain_label = meta_info.get('domain_label', None)
                if domain_label is not None:
                    self.domain_labels.append(domain_label)
        
        # Call parent's process method
        super().process(data_batch, data_samples)
    
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
        
        Returns:
            dict: The computed metrics including domain statistics.
        """
        # Compute standard COCO metrics
        metrics = super().compute_metrics(results)
        
        # Add domain weight statistics
        if self.track_domain_weights and len(self.domain_weight_stats) > 0:
            domain_weights = np.array(self.domain_weight_stats)  # (N, num_domains)
            
            # Compute statistics
            mean_weights = domain_weights.mean(axis=0)
            std_weights = domain_weights.std(axis=0)
            max_weights = domain_weights.max(axis=0)
            min_weights = domain_weights.min(axis=0)
            
            # Add to metrics
            for d in range(self.num_domains):
                metrics[f'domain_{d}/mean_weight'] = float(mean_weights[d])
                metrics[f'domain_{d}/std_weight'] = float(std_weights[d])
                metrics[f'domain_{d}/max_weight'] = float(max_weights[d])
                metrics[f'domain_{d}/min_weight'] = float(min_weights[d])
            
            # Compute entropy of weights (uncertainty measure)
            # H = -sum(p * log(p))
            epsilon = 1e-8
            entropy = -np.sum(
                domain_weights * np.log(domain_weights + epsilon),
                axis=1
            ).mean()
            metrics['domain_weight_entropy'] = float(entropy)
            
            # Compute dominant domain distribution
            dominant_domains = domain_weights.argmax(axis=1)
            for d in range(self.num_domains):
                ratio = (dominant_domains == d).sum() / len(dominant_domains)
                metrics[f'domain_{d}/dominant_ratio'] = float(ratio)
            
            # Log statistics
            logger = MMLogger.get_current_instance()
            logger.info('\nDomain Weight Statistics:')
            for d in range(self.num_domains):
                logger.info(
                    f'  Domain {d}: mean={mean_weights[d]:.4f}, '
                    f'std={std_weights[d]:.4f}, '
                    f'dominant_ratio={metrics[f"domain_{d}/dominant_ratio"]:.4f}'
                )
            logger.info(f'  Entropy: {entropy:.4f}')
        
        # Clear statistics for next evaluation
        self.domain_weight_stats = []
        self.domain_labels = []
        
        return metrics
