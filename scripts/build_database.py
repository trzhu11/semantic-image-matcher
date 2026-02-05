#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建图像数据库脚本
使用 DINOv3-ViT-H/16+ 提取全局特征并构建 FAISS 索引

Usage:
    python scripts/build_database.py --image-dir /path/to/images --output-dir /path/to/output
"""
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.m1_retrieval import build_database


def main():
    parser = argparse.ArgumentParser(description='Build image database with DINOv3-ViT-H/16+')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for database')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to DINOv3 model (default: facebook/dinov2-giant)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Building Image Database (DINOv3-ViT-H/16+)")
    print("=" * 60)
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    if args.model_path:
        print(f"Model path: {args.model_path}")
    print()
    
    # 构建数据库
    num_images = build_database(
        image_dir=image_dir,
        output_dir=output_dir,
        model_path=args.model_path,
        device=args.device
    )
    
    print()
    print("=" * 60)
    print(f"✅ Database built successfully!")
    print(f"   Total images indexed: {num_images}")
    print(f"   Index saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
