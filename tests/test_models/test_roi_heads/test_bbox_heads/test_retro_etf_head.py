# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
import torch
from mmdet.models.roi_heads.bbox_heads import RetroCalibETFHeadWithOrthLoss


class TestRetroCalibETFHead(TestCase):

    def test_init(self):
        """Test initialization of RetroCalibETFHeadWithOrthLoss."""
        bbox_head = RetroCalibETFHeadWithOrthLoss(
            in_channels=256,
            fc_out_channels=1024,
            num_classes=80,
            na=4,
            ad=256,
            eo=True)
        
        self.assertTrue(bbox_head.fc_cls)
        self.assertTrue(bbox_head.fc_reg)
        self.assertEqual(bbox_head.na, 4)
        self.assertEqual(bbox_head.ad, 256)
        self.assertTrue(bbox_head.eo)
        self.assertEqual(len(bbox_head.abs), 4)

    def test_forward(self):
        """Test forward pass."""
        bbox_head = RetroCalibETFHeadWithOrthLoss(
            in_channels=256,
            fc_out_channels=1024,
            num_classes=80,
            na=4,
            ad=256)
        
        # Create dummy input
        x = torch.randn(2, 256, 7, 7)
        cls_score, bbox_pred = bbox_head(x)
        
        self.assertEqual(cls_score.shape[0], 2)
        self.assertEqual(cls_score.shape[1], 81)  # num_classes + 1
        self.assertEqual(bbox_pred.shape[0], 2)
