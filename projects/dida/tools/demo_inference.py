#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Demo inference with domain weight visualization."""

import argparse
from mmdet.apis import inference_detector, init_detector


def main():
    parser = argparse.ArgumentParser(description='DIDA demo inference')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image file')
    parser.add_argument('--out', default='output.jpg', help='output file')
    args = parser.parse_args()
    
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    result = inference_detector(model, args.image)
    
    print(f"Detection completed. Results: {len(result.pred_instances)} objects")
    # TODO: Add visualization with domain weights


if __name__ == '__main__':
    main()
