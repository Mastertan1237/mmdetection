#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Test script with domain-aware inference."""

import argparse
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Test with domain-aware mode')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--domain-unknown', action='store_true',
                        help='Enable domain-unknown inference mode')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Init model
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    
    # Enable domain-unknown mode if requested
    if args.domain_unknown and hasattr(model, 'set_domain_unknown_mode'):
        model.set_domain_unknown_mode(True)
        print("Domain-unknown mode enabled")
    
    print("Model loaded successfully!")
    # TODO: Add actual testing loop


if __name__ == '__main__':
    main()
