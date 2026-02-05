# -*- coding: utf-8 -*-
"""
M1: Global Image Retrieval Module
基于 DINOv3-ViT-H/16+ 构建全局特征索引，用于快速召回 Top-K 候选位置。

Model: facebook/dinov2-giant (DINOv3-ViT-H/16+)
Feature: CLS token from last hidden state
Dimension: 1536
"""
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import faiss
import h5py
from typing import List, Tuple, Optional, Dict, Any


class M1GlobalRetrieval:
    """DINOv3-ViT-H/16+ 全局图像检索模块"""
    
    # 默认模型路径 - 可以是本地路径或 HuggingFace 模型名
    DEFAULT_MODEL_PATH = 'facebook/dinov2-giant'
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'cuda'):
        """初始化 M1 检索模块
        
        Args:
            model_path: DINOv3 模型路径，可以是:
                - HuggingFace 模型名: 'facebook/dinov2-giant'
                - 本地路径: '/path/to/model'
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = None
        self.processor = None
        self._load_model()
        
        # 索引相关
        self.index = None
        self.image_names = None
        self.image_paths = None

    def _load_model(self):
        """加载 DINOv3-ViT-H/16+ 模型"""
        print(f"Loading DINOv3-ViT-H/16+ model from: {self.model_path}...")
        try:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            print("✅ Loaded DINOv3-ViT-H/16+ successfully")
            
            # 获取特征维度
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224).to(self.device)
                out = self.model(dummy)
                self.feature_dim = out.last_hidden_state[:, 0, :].shape[-1]
                print(f"   Feature dimension: {self.feature_dim}")
        except Exception as e:
            print(f"Error loading DINOv3-ViT-H/16+: {e}")
            raise RuntimeError("Failed to load DINOv3-ViT-H/16+ model")

    def extract_features(self, image_paths: List[Path], batch_size: int = 16) -> np.ndarray:
        """提取图像的全局特征
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小 (大模型建议较小值)
            
        Returns:
            np.ndarray: L2 归一化后的特征矩阵 [N, D]
        """
        features = []
        print(f"Extracting DINOv3-ViT-H/16+ features for {len(image_paths)} images...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size)):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert('RGB')
                        batch_images.append(img)
                    except Exception as e:
                        print(f"Warning: Could not read image {p}: {e}")
                        batch_images.append(Image.new('RGB', (224, 224)))

                if not batch_images:
                    continue

                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                # 使用 CLS token 作为全局特征
                batch_features = outputs.last_hidden_state[:, 0, :]
                features.append(batch_features.cpu().numpy())
        
        if not features:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
            
        all_features = np.concatenate(features, axis=0)
        faiss.normalize_L2(all_features)
        return all_features

    def build_index(self, features: np.ndarray, image_names: List[str], 
                    image_paths: List[Path], output_dir: Path):
        """构建 FAISS 索引并保存
        
        Args:
            features: 特征矩阵 [N, D]
            image_names: 图像文件名列表
            image_paths: 图像完整路径列表
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存描述符到 HDF5
        h5_path = output_dir / 'db_descriptors.h5'
        print(f"Saving descriptors to {h5_path}...")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('descriptors', data=features)
            f.create_dataset('names', data=np.array([n.encode('utf-8') for n in image_names]))
            f.create_dataset('paths', data=np.array([str(p).encode('utf-8') for p in image_paths]))
            
        # 构建 FAISS 索引
        print("Building FAISS index...")
        d = features.shape[1]
        n = features.shape[0]
        
        if n < 1000:
            # 小数据集使用精确搜索
            print("Using IndexFlatL2 for small dataset")
            index = faiss.IndexFlatL2(d)
            index.add(features)
        else:
            # 大数据集使用 IVF 索引
            nlist = min(100, int(4 * np.sqrt(n)))
            print(f"Using IndexIVFFlat with nlist={nlist}")
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.train(features)
            index.add(features)
            
        index_path = output_dir / 'global_index.faiss'
        faiss.write_index(index, str(index_path))
        print(f"✅ Index saved to {index_path}")

    def load_index(self, index_dir: Path):
        """加载已构建的索引
        
        Args:
            index_dir: 索引目录路径
        """
        index_dir = Path(index_dir)
        index_path = index_dir / 'global_index.faiss'
        h5_path = index_dir / 'db_descriptors.h5'
        
        print(f"Loading DINOv3-ViT-H/16+ index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        
        with h5py.File(h5_path, 'r') as f:
            self.image_names = [n.decode('utf-8') for n in f['names'][:]]
            self.image_paths = [Path(p.decode('utf-8')) for p in f['paths'][:]]
        
        print(f"✅ Loaded index with {len(self.image_names)} images")

    def query(self, query_image_path: Path, top_k: int = 1) -> List[Dict[str, Any]]:
        """查询 Top-K 候选
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回的候选数量
            
        Returns:
            List[Dict]: 候选列表，每个包含 rank, name, path, distance, score
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        # 提取查询特征
        query_features = self.extract_features([Path(query_image_path)])
        
        # 搜索
        distances, indices = self.index.search(query_features, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:
                results.append({
                    'rank': i + 1,
                    'name': self.image_names[idx],
                    'path': self.image_paths[idx],
                    'distance': float(dist),
                    'score': 1.0 / (1.0 + float(dist))  # 距离转相似度
                })
        
        return results


def build_database(image_dir: Path, output_dir: Path, 
                   model_path: str = None, device: str = 'cuda') -> int:
    """构建图像数据库的便捷函数
    
    Args:
        image_dir: 图像目录
        output_dir: 输出目录
        model_path: 模型路径 (可选)
        device: 计算设备
        
    Returns:
        int: 索引的图像数量
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    
    # 收集图像
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = sorted([
        p for p in image_dir.iterdir() 
        if p.suffix.lower() in image_extensions
    ])
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # 初始化 M1
    m1 = M1GlobalRetrieval(model_path=model_path, device=device)
    
    # 提取特征
    features = m1.extract_features(image_paths)
    
    # 构建索引
    image_names = [p.name for p in image_paths]
    m1.build_index(features, image_names, image_paths, output_dir)
    
    return len(image_paths)
