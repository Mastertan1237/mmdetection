#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Validation script for DIDA framework structure."""

import os
import sys
from pathlib import Path

def check_file_exists(path: str, description: str):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def main():
    """Run validation checks."""
    print("="*60)
    print("DIDA Framework Structure Validation")
    print("="*60)
    
    base_path = "projects/dida"
    all_passed = True
    
    # Check core structure
    print("\n1. Core Structure:")
    core_files = [
        (f"{base_path}/__init__.py", "Main init"),
        (f"{base_path}/README.md", "Documentation"),
        (f"{base_path}/requirements.txt", "Requirements"),
    ]
    for path, desc in core_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check models
    print("\n2. Models:")
    model_files = [
        (f"{base_path}/models/__init__.py", "Models init"),
        (f"{base_path}/models/domain_residual.py", "Domain Residual"),
        (f"{base_path}/models/clip_classifier.py", "CLIP Classifier"),
        (f"{base_path}/models/style_router.py", "Style Router"),
        (f"{base_path}/models/objectness_head.py", "Objectness Heads"),
        (f"{base_path}/models/dida_head.py", "DIDA Head"),
        (f"{base_path}/models/dida_dino.py", "DIDA Detector"),
    ]
    for path, desc in model_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check engine
    print("\n3. Training Infrastructure:")
    engine_files = [
        (f"{base_path}/engine/hooks/freeze_hook.py", "Freeze Hook"),
    ]
    for path, desc in engine_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check evaluation
    print("\n4. Evaluation Metrics:")
    eval_files = [
        (f"{base_path}/evaluation/oracle_coco_metric.py", "Oracle Metric"),
        (f"{base_path}/evaluation/class_agnostic_coco_metric.py", "Class-Agnostic Metric"),
        (f"{base_path}/evaluation/domain_aware_metric.py", "Domain-Aware Metric"),
    ]
    for path, desc in eval_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check core
    print("\n5. Core Components:")
    core_comp_files = [
        (f"{base_path}/core/bbox/objectness_assigner.py", "Custom Assigner"),
    ]
    for path, desc in core_comp_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check tools
    print("\n6. Tools:")
    tool_files = [
        (f"{base_path}/tools/extract_clip_prototypes.py", "CLIP Extractor"),
        (f"{base_path}/tools/train_incremental.py", "Training Script (Python)"),
        (f"{base_path}/tools/train_incremental.sh", "Training Script (Bash)"),
        (f"{base_path}/tools/test_domain_aware.py", "Testing Script"),
        (f"{base_path}/tools/demo_inference.py", "Demo Script"),
        (f"{base_path}/tools/geometric_error_analysis.py", "Error Analysis"),
    ]
    for path, desc in tool_files:
        all_passed &= check_file_exists(path, desc)
    
    # Check configs
    print("\n7. Configuration Files:")
    config_files = [
        # Base configs
        (f"{base_path}/configs/_base_/dino_baseline.py", "Base DINO Config"),
        (f"{base_path}/configs/_base_/dataset_voc_template.py", "VOC Dataset"),
        (f"{base_path}/configs/_base_/dataset_bdd_template.py", "BDD Dataset"),
        (f"{base_path}/configs/_base_/default_runtime.py", "Runtime"),
        # VOC series
        (f"{base_path}/configs/voc_clipart_watercolor_comic/stage0_voc.py", "VOC Stage 0"),
        (f"{base_path}/configs/voc_clipart_watercolor_comic/stage1_clipart.py", "Clipart Stage 1"),
        (f"{base_path}/configs/voc_clipart_watercolor_comic/stage2_watercolor.py", "Watercolor Stage 2"),
        (f"{base_path}/configs/voc_clipart_watercolor_comic/stage3_comic.py", "Comic Stage 3"),
        # BDD series
        (f"{base_path}/configs/bdd_cityscapes_rain/stage0_bdd.py", "BDD Stage 0"),
        (f"{base_path}/configs/bdd_cityscapes_rain/stage1_cityscapes.py", "Cityscapes Stage 1"),
        (f"{base_path}/configs/bdd_cityscapes_rain/stage2_rain_cityscapes.py", "Rain Stage 2"),
    ]
    for path, desc in config_files:
        all_passed &= check_file_exists(path, desc)
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        print("\nDIDA framework is ready to use!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r projects/dida/requirements.txt")
        print("2. Prepare datasets and CLIP prototypes")
        print("3. Start training: bash projects/dida/tools/train_incremental.sh")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease verify missing files.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
