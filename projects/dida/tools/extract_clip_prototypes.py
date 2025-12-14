#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Extract CLIP text prototypes for object classes.

This script extracts CLIP text embeddings for a list of class names using
various prompt templates and saves them for use in the DIDA framework.

Usage:
    python tools/extract_clip_prototypes.py \\
        --classes aeroplane bicycle bird boat ... \\
        --output clip_prototypes_voc.pt \\
        --model ViT-B-32 \\
        --pretrained openai \\
        --templates "a photo of a {}" "a rendering of a {}"
"""

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract CLIP text prototypes for object classes')
    parser.add_argument(
        '--classes',
        nargs='+',
        required=True,
        help='List of class names')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (.pt)')
    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B-32',
        help='CLIP model name (default: ViT-B-32)')
    parser.add_argument(
        '--pretrained',
        type=str,
        default='openai',
        help='Pretrained weights (default: openai)')
    parser.add_argument(
        '--templates',
        nargs='+',
        default=['a photo of a {}'],
        help='Prompt templates with {} placeholder for class name')
    parser.add_argument(
        '--ensemble-method',
        type=str,
        default='mean',
        choices=['mean', 'max'],
        help='Method to combine multiple prompts (default: mean)')
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='L2-normalize the prototypes (default: True)')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for extraction')
    
    args = parser.parse_args()
    return args


def extract_clip_prototypes(
    class_names: List[str],
    templates: List[str],
    model_name: str = 'ViT-B-32',
    pretrained: str = 'openai',
    ensemble_method: str = 'mean',
    normalize: bool = True,
    device: str = 'cuda'
) -> torch.Tensor:
    """Extract CLIP text embeddings for class names.
    
    Args:
        class_names (List[str]): List of object class names.
        templates (List[str]): List of prompt templates with {} placeholder.
        model_name (str): CLIP model architecture.
        pretrained (str): Pretrained weights identifier.
        ensemble_method (str): How to combine multiple prompts ('mean' or 'max').
        normalize (bool): Whether to L2-normalize embeddings.
        device (str): Device for computation.
    
    Returns:
        torch.Tensor: Text prototypes, shape (num_classes, embed_dim).
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip is required. Install with: pip install open_clip_torch"
        )
    
    print(f"Loading CLIP model: {model_name} with {pretrained} weights...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    model.eval()
    
    tokenizer = open_clip.get_tokenizer(model_name)
    
    print(f"Extracting prototypes for {len(class_names)} classes...")
    print(f"Using {len(templates)} prompt templates")
    
    all_prototypes = []
    
    with torch.no_grad():
        for class_name in class_names:
            # Generate prompts for this class
            prompts = [template.format(class_name) for template in templates]
            
            # Tokenize
            text_tokens = tokenizer(prompts).to(device)
            
            # Extract features
            text_features = model.encode_text(text_tokens)
            
            # Normalize if required
            if normalize:
                text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Ensemble across prompts
            if ensemble_method == 'mean':
                class_prototype = text_features.mean(dim=0)
            elif ensemble_method == 'max':
                class_prototype = text_features.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")
            
            all_prototypes.append(class_prototype)
    
    # Stack into tensor: (num_classes, embed_dim)
    prototypes = torch.stack(all_prototypes, dim=0)
    
    # Final normalization
    if normalize:
        prototypes = F.normalize(prototypes, p=2, dim=-1)
    
    print(f"Extracted prototypes shape: {prototypes.shape}")
    
    return prototypes


def main():
    args = parse_args()
    
    # Extract prototypes
    prototypes = extract_clip_prototypes(
        class_names=args.classes,
        templates=args.templates,
        model_name=args.model,
        pretrained=args.pretrained,
        ensemble_method=args.ensemble_method,
        normalize=args.normalize,
        device=args.device
    )
    
    # Prepare output data
    output_data = {
        'prototypes': prototypes.cpu(),
        'class_names': args.classes,
        'model_name': args.model,
        'pretrained': args.pretrained,
        'templates': args.templates,
        'ensemble_method': args.ensemble_method,
        'embed_dim': prototypes.shape[1],
        'num_classes': len(args.classes)
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(output_data, output_path)
    print(f"\nSaved prototypes to: {output_path}")
    print(f"  - Classes: {len(args.classes)}")
    print(f"  - Embedding dimension: {prototypes.shape[1]}")
    print(f"  - Templates used: {len(args.templates)}")
    
    # Print class names for verification
    print("\nClass names:")
    for i, name in enumerate(args.classes):
        print(f"  {i}: {name}")


if __name__ == '__main__':
    main()
