# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses import (CombinedOrthogonalLoss,
                                 OrthogonalConstraintLoss,
                                 SemanticOutputPreservationLoss,
                                 SemanticResponseLoss)


def test_orthogonal_constraint_loss():
    """Test OrthogonalConstraintLoss."""
    loss = OrthogonalConstraintLoss(loss_weight=1.0)
    result = loss()
    assert result == 0.0


def test_semantic_response_loss():
    """Test SemanticResponseLoss."""
    loss = SemanticResponseLoss(loss_weight=1.0)
    result = loss()
    assert result == 0.0


def test_semantic_output_preservation_loss():
    """Test SemanticOutputPreservationLoss."""
    loss = SemanticOutputPreservationLoss(loss_weight=1.0)
    result = loss()
    assert result == 0.0


def test_combined_orthogonal_loss():
    """Test CombinedOrthogonalLoss."""
    loss = CombinedOrthogonalLoss(o=1.0, r=1.0, p=1.0)
    result = loss()
    assert isinstance(result, dict)
    assert 'loss_total' in result
    assert result['loss_total'] == 0.0
