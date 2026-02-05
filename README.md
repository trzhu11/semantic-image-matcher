# Semantic Image Matcher

基于 DINOv3 + Qwen3-VL 的语义图像匹配系统，用于景区/博物馆场景的图像定位与匹配。

## 系统架构

```
Query Image → M1 (DINOv3 全局检索) → Top-K 候选 → M3 (VLM 语义验证) → 最终匹配结果
```

- **M1 模块**: 使用 DINOv3-ViT-H/16+ 提取全局特征，通过 FAISS 索引进行快速检索
- **M3 模块**: 使用 Qwen3-VL-8B 进行图像对的语义验证，判断是否为同一地点

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n semantic-matcher python=3.10
conda activate semantic-matcher
```

### 2. 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.45.0
pip install faiss-gpu  # 或 faiss-cpu
pip install h5py
pip install opencv-python
pip install pillow
pip install tqdm
pip install qwen-vl-utils
```

### 3. 模型下载

#### DINOv3-ViT-H/16+ (M1 模块)

```bash
# 从 HuggingFace 下载
# 模型: facebook/dinov2-giant 或自定义路径
# 默认路径: /share/shared_weights/dinov3/facebook/dinov3-vith16plus-pretrain-lvd1689m
```

或使用 transformers 自动下载:
```python
from transformers import AutoModel, AutoImageProcessor
model = AutoModel.from_pretrained("facebook/dinov2-giant")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
```

#### Qwen3-VL-8B (M3 模块)

```bash
# 从 HuggingFace 下载
# 模型: Qwen/Qwen2-VL-7B-Instruct 或 Qwen3-VL-8B-Instruct
# 默认路径: /share/shared_weights/Qwen3-VL-8B-Instruct
```

## 模型版本说明

| 模块 | 模型 | 特征维度 | 说明 |
|------|------|----------|------|
| M1 | DINOv3-ViT-H/16+ | 1536 | 全局图像检索，提取 CLS token 作为全局特征 |
| M3 | Qwen3-VL-8B-Instruct | - | 多模态语义验证，判断图像对是否为同一地点 |

## 使用方法

### 1. 构建图像数据库

```bash
python scripts/build_database.py \
    --image-dir /path/to/your/images \
    --output-dir /path/to/output/database
```

### 2. 运行匹配测试

```bash
python scripts/test_matching.py \
    --query-dir /path/to/query/images \
    --db-dir /path/to/database \
    --output-dir /path/to/results
```

### 3. Python API 使用

```python
from core.m1_retrieval import M1GlobalRetrieval
from core.m3_verifier import LocalM3Verifier
from pathlib import Path

# 初始化模块
m1 = M1GlobalRetrieval(device='cuda')
m3 = LocalM3Verifier(prompt_style='simple')

# 加载数据库
m1.load_index(Path('/path/to/database'))

# 查询
query_image = Path('/path/to/query.jpg')
candidates = m1.query(query_image, top_k=5)

# 验证 Top-1
if candidates:
    is_match, response, confidence = m3.verify_pair(
        str(query_image),
        str(candidates[0]['path'])
    )
    print(f"Match: {is_match}, Response: {response}")
```

## 项目结构

```
semantic-image-matcher/
├── README.md
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── m1_retrieval.py      # M1: DINOv3 全局检索
│   └── m3_verifier.py       # M3: VLM 语义验证
└── scripts/
    ├── build_database.py    # 构建图像数据库
    └── test_matching.py     # 测试匹配效果
```

## M3 Prompt 风格

支持多种 prompt 风格，可根据场景选择：

| 风格 | 说明 |
|------|------|
| `simple` | 简洁直接版 (推荐首选) |
| `scenic_guide` | 景区讲解专用版 |
| `strict` | 严格匹配版 (减少误报) |
| `chinese` | 中文版 |
| `detailed` | 详细分析版 |

```python
m3 = LocalM3Verifier(prompt_style='strict')
```

## 性能参考

在南大苏州数据集上的测试结果：

| 模型 | 准确率 | M1 时间 | M3 时间 | 总时间 |
|------|--------|---------|---------|--------|
| DINOv3-ViT-H/16+ | ~85%+ | ~50ms | ~800ms | ~850ms |

## License

MIT License

## 致谢

- [DINOv2](https://github.com/facebookresearch/dinov2) - Meta AI
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - Alibaba Cloud
