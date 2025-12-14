# Copyright (c) OpenMMLab. All rights reserved.
"""Oracle Classification Metric for upper-bound evaluation.

Matches predictions to ground truth via IoU and replaces predicted labels
with ground truth labels to measure localization quality.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from mmdet.structures.bbox import bbox_overlaps


@METRICS.register_module()
class OracleCocoMetric(CocoMetric):
    """Oracle COCO metric that replaces predicted classes with GT classes.
    
    This metric measures the upper bound of detection performance if
    classification were perfect, revealing the impact of localization quality.
    
    Workflow:
    1. Match predictions to ground truth via IoU (Hungarian or greedy)
    2. For matched predictions, replace class label with GT label
    3. Evaluate using standard COCO metrics
    
    Args:
        iou_threshold (float): IoU threshold for matching. Defaults to 0.5.
        matching_strategy (str): 'hungarian' or 'greedy'. Defaults to 'greedy'.
        **kwargs: Arguments for CocoMetric.
    """
    
    default_prefix: Optional[str] = 'oracle_coco'
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        matching_strategy: str = 'greedy',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.matching_strategy = matching_strategy
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        This method modifies predictions by replacing class labels with
        ground truth labels for matched boxes.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # Process each sample
        for data_sample in data_samples:
            # Get predictions
            pred_instances = data_sample.get('pred_instances', None)
            if pred_instances is None:
                continue
            
            # Get ground truth
            gt_instances = data_sample.get('gt_instances', None)
            if gt_instances is None:
                continue
            
            # Convert to numpy for easier manipulation
            pred_bboxes = pred_instances['bboxes'].cpu().numpy()
            pred_scores = pred_instances['scores'].cpu().numpy()
            pred_labels = pred_instances['labels'].cpu().numpy()
            
            gt_bboxes = gt_instances['bboxes'].cpu().numpy()
            gt_labels = gt_instances['labels'].cpu().numpy()
            
            # Match predictions to ground truth
            if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
                # No matching needed
                continue
            
            # Compute IoU matrix
            ious = bbox_overlaps(
                pred_instances['bboxes'],
                gt_instances['bboxes']
            ).cpu().numpy()
            
            # Perform matching
            if self.matching_strategy == 'greedy':
                matched_gt_indices = self._greedy_matching(ious, self.iou_threshold)
            elif self.matching_strategy == 'hungarian':
                matched_gt_indices = self._hungarian_matching(ious, self.iou_threshold)
            else:
                raise ValueError(f"Unknown matching strategy: {self.matching_strategy}")
            
            # Replace labels for matched predictions
            new_labels = pred_labels.copy()
            for pred_idx, gt_idx in enumerate(matched_gt_indices):
                if gt_idx >= 0:  # Matched
                    new_labels[pred_idx] = gt_labels[gt_idx]
            
            # Update predictions with oracle labels
            pred_instances['labels'] = pred_instances['labels'].new_tensor(new_labels)
        
        # Call parent's process method with modified predictions
        super().process(data_batch, data_samples)
    
    def _greedy_matching(
        self,
        ious: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Greedy matching based on IoU.
        
        Args:
            ious (np.ndarray): IoU matrix, shape (num_preds, num_gts).
            threshold (float): IoU threshold.
        
        Returns:
            np.ndarray: Matched GT indices for each prediction, -1 if unmatched.
        """
        num_preds = ious.shape[0]
        matched_gt_indices = np.full(num_preds, -1, dtype=np.int32)
        matched_gts = set()
        
        # For each prediction, find best GT match
        for pred_idx in range(num_preds):
            best_iou = threshold
            best_gt_idx = -1
            
            for gt_idx in range(ious.shape[1]):
                if gt_idx in matched_gts:
                    continue
                
                if ious[pred_idx, gt_idx] > best_iou:
                    best_iou = ious[pred_idx, gt_idx]
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matched_gt_indices[pred_idx] = best_gt_idx
                matched_gts.add(best_gt_idx)
        
        return matched_gt_indices
    
    def _hungarian_matching(
        self,
        ious: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Hungarian matching based on IoU.
        
        Args:
            ious (np.ndarray): IoU matrix, shape (num_preds, num_gts).
            threshold (float): IoU threshold.
        
        Returns:
            np.ndarray: Matched GT indices for each prediction, -1 if unmatched.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError("scipy is required for Hungarian matching")
        
        num_preds, num_gts = ious.shape
        
        # Convert IoU to cost (maximize IoU = minimize negative IoU)
        cost_matrix = 1 - ious
        
        # Solve assignment problem
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Filter by threshold
        matched_gt_indices = np.full(num_preds, -1, dtype=np.int32)
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if ious[pred_idx, gt_idx] >= threshold:
                matched_gt_indices[pred_idx] = gt_idx
        
        return matched_gt_indices
