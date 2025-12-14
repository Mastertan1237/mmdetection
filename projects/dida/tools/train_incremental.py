#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Python script for incremental training across multiple domains."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Incremental training for DIDA')
    parser.add_argument('config_dir', help='Config directory (e.g., voc_clipart_watercolor_comic)')
    parser.add_argument('--work-dir', default='work_dirs', help='Working directory')
    parser.add_argument('--gpu-ids', default='0', help='GPU IDs')
    parser.add_argument('--stages', nargs='+', type=int, help='Stages to run')
    return parser.parse_args()


def main():
    args = parse_args()
    
    config_base = Path('projects/dida/configs') / args.config_dir
    stages = args.stages or [0, 1, 2, 3]
    
    for stage in stages:
        print(f"\n{'='*60}")
        print(f"Stage {stage}")
        print('='*60)
        
        # Find config file
        config_files = list(config_base.glob(f'stage{stage}_*.py'))
        if not config_files:
            print(f"No config found for stage {stage}, skipping...")
            continue
        
        config_file = config_files[0]
        work_dir = Path(args.work_dir) / args.config_dir / f'stage{stage}'
        
        # Build command
        cmd = [
            'python', 'tools/train.py',
            str(config_file),
            '--work-dir', str(work_dir),
        ]
        
        # Add checkpoint loading for incremental stages
        if stage > 0:
            prev_work_dir = Path(args.work_dir) / args.config_dir / f'stage{stage-1}'
            checkpoint = prev_work_dir / 'epoch_12.pth'
            if checkpoint.exists():
                cmd.extend(['--cfg-options', f'load_from={checkpoint}'])
        
        # Run training
        env = {'CUDA_VISIBLE_DEVICES': args.gpu_ids}
        subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
