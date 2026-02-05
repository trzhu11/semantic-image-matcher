#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图像匹配脚本
对查询图像进行 M1 检索 + M3 验证

Usage:
    python scripts/test_matching.py \
        --query-dir /path/to/query/images \
        --db-dir /path/to/database \
        --output-dir /path/to/results
"""
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.m1_retrieval import M1GlobalRetrieval
from core.m3_verifier import LocalM3Verifier, get_available_prompts


def main():
    parser = argparse.ArgumentParser(description='Test image matching with M1+M3')
    parser.add_argument('--query-dir', type=str, required=True,
                        help='Directory containing query images')
    parser.add_argument('--db-dir', type=str, required=True,
                        help='Database directory (from build_database.py)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--top-k', type=int, default=1,
                        help='Number of candidates to retrieve from M1')
    parser.add_argument('--prompt', type=str, default='simple',
                        choices=['simple', 'scenic_guide', 'strict', 'chinese', 'detailed'],
                        help='Prompt style for M3 verification')
    parser.add_argument('--m1-model', type=str, default=None,
                        help='Path to DINOv3 model')
    parser.add_argument('--m3-model', type=str, default=None,
                        help='Path to Qwen-VL model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--list-prompts', action='store_true',
                        help='List available prompt styles and exit')
    
    args = parser.parse_args()
    
    # 列出可用的 prompt
    if args.list_prompts:
        print("\nAvailable prompt styles:")
        print("-" * 50)
        for name, desc in get_available_prompts().items():
            print(f"  {name}: {desc}")
        print()
        return
    
    query_dir = Path(args.query_dir)
    db_dir = Path(args.db_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Image Matching Test (M1 + M3)")
    print("=" * 60)
    print(f"Query directory: {query_dir}")
    print(f"Database directory: {db_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Top-K: {args.top_k}")
    print(f"Prompt style: {args.prompt}")
    print()
    
    # 加载 M1
    print("Loading M1 (DINOv3-ViT-H/16+)...")
    m1 = M1GlobalRetrieval(model_path=args.m1_model, device=args.device)
    m1.load_index(db_dir)
    
    # 加载 M3
    print("\nLoading M3 (Qwen-VL)...")
    m3 = LocalM3Verifier(
        model_path=args.m3_model,
        device=args.device,
        prompt_style=args.prompt
    )
    
    # 收集查询图像
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    query_images = sorted([
        p for p in query_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])
    
    print(f"\nFound {len(query_images)} query images")
    print("=" * 60)
    
    # 测试
    results = []
    m1_times = []
    m3_times = []
    
    for img_path in tqdm(query_images, desc="Processing"):
        # M1 检索
        m1_start = time.time()
        candidates = m1.query(img_path, top_k=args.top_k)
        m1_time = time.time() - m1_start
        m1_times.append(m1_time)
        
        # M3 验证 Top-1
        m3_start = time.time()
        if candidates:
            is_match, response, confidence = m3.verify_pair(
                str(img_path),
                str(candidates[0]['path'])
            )
            top1_name = candidates[0]['name']
            top1_distance = candidates[0]['distance']
        else:
            is_match = False
            response = ""
            confidence = 0.0
            top1_name = None
            top1_distance = None
        m3_time = time.time() - m3_start
        m3_times.append(m3_time)
        
        result = {
            'query': img_path.name,
            'm1_top1': top1_name,
            'm1_distance': top1_distance,
            'm3_verified': is_match,
            'm3_response': response,
            'm3_confidence': confidence,
            'm1_time_ms': m1_time * 1000,
            'm3_time_ms': m3_time * 1000,
            'total_time_ms': (m1_time + m3_time) * 1000
        }
        results.append(result)
    
    # 统计
    n = len(results)
    matched_count = sum(r['m3_verified'] for r in results)
    avg_m1_time = sum(m1_times) / n * 1000
    avg_m3_time = sum(m3_times) / n * 1000
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total queries: {n}")
    print(f"M3 verified matches: {matched_count} ({matched_count/n*100:.1f}%)")
    print(f"Average M1 time: {avg_m1_time:.1f}ms")
    print(f"Average M3 time: {avg_m3_time:.1f}ms")
    print(f"Average total time: {avg_m1_time + avg_m3_time:.1f}ms")
    
    # 保存结果
    summary = {
        'total_queries': n,
        'matched_count': matched_count,
        'match_rate': matched_count / n * 100,
        'avg_m1_time_ms': avg_m1_time,
        'avg_m3_time_ms': avg_m3_time,
        'avg_total_time_ms': avg_m1_time + avg_m3_time,
        'prompt_style': args.prompt,
        'top_k': args.top_k
    }
    
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
