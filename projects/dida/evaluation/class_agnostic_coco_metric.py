# Copyright (c) OpenMMLab. All rights reserved.
"""Class-Agnostic COCO Metric for pure localization evaluation.

Maps all classes to a single foreground class to evaluate localization
performance independent of classification.
"""
from typing import Optional, Sequence

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class ClassAgnosticCocoMetric(CocoMetric):
    """Class-agnostic COCO metric for localization evaluation.
    
    Maps all object classes to a single foreground class (label 0) to
    measure pure localization quality without classification impact.
    
    This is useful for diagnosing whether poor performance is due to
    classification errors or localization errors.
    
    Args:
        **kwargs: Arguments for CocoMetric.
    """
    
    default_prefix: Optional[str] = 'class_agnostic'
    
    def __init__(self, **kwargs) -> None:
        # Force single class
        super().__init__(**kwargs)
        self._original_classwise = self.classwise
        self.classwise = False  # No per-class metrics for class-agnostic
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        This method modifies both predictions and ground truth to use
        a single foreground class.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # Process each sample
        for data_sample in data_samples:
            # Modify predictions: map all labels to 0
            pred_instances = data_sample.get('pred_instances', None)
            if pred_instances is not None and len(pred_instances) > 0:
                # Set all predicted labels to 0 (foreground)
                pred_instances['labels'] = pred_instances['labels'].new_zeros(
                    len(pred_instances['labels'])
                )
            
            # Modify ground truth: map all labels to 0
            gt_instances = data_sample.get('gt_instances', None)
            if gt_instances is not None and len(gt_instances) > 0:
                gt_instances['labels'] = gt_instances['labels'].new_zeros(
                    len(gt_instances['labels'])
                )
        
        # Call parent's process method with modified data
        super().process(data_batch, data_samples)
    
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
        
        Returns:
            dict: The computed metrics. The keys are metric names and the
            values are corresponding results.
        """
        # Call parent's compute_metrics
        metrics = super().compute_metrics(results)
        
        # Add prefix to distinguish from standard COCO metrics
        prefixed_metrics = {}
        for key, value in metrics.items():
            if not key.startswith(self.default_prefix):
                prefixed_key = f'{self.default_prefix}/{key}'
            else:
                prefixed_key = key
            prefixed_metrics[prefixed_key] = value
        
        return prefixed_metrics
