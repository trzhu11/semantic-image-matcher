# Semantic Image Matcher: Int8 Single GPU Deploy

这个仓库的 `int8-single-gpu-deploy` 分支用于承载一套可直接替换线上 `18080/10003` 的部署：

- `18080`: `llama.cpp` 原生 `Qwen3-VL-8B-Instruct-GGUF` 后端
- `10003`: 包装服务，兼容图搜接口、问答接口和数据库接口
- 默认单卡部署
- 可选分卡部署
- 数据库存储固定为 Elasticsearch，不再使用 Faiss 文件索引

当前已经包含这些能力：

- `POST /v1/chat/completions`
- `POST /vector/encode`
- `POST /vector/rerank`
- `POST /add_image`
- `POST /search_vector`
- `GET /health`
- `GET /stats`

## 目录结构

```text
semantic-image-matcher/
├── .env.example
├── .gitignore
├── README.md
├── docs/
│   ├── database_operations_es.md
│   └── github_publish.md
├── requirements.txt
├── benchmark/
│   └── roamii/
│       ├── README.md
│       ├── manifests/
│       ├── results/
│       └── run_api_benchmark.py
├── scripts/
│   ├── bootstrap_llama_cpp.sh
│   ├── load_env.sh
│   ├── smoke_test.sh
│   ├── start_llama_backend.sh
│   ├── start_replace_stack.sh
│   ├── start_service.sh
│   ├── status_replace_stack.sh
│   └── stop_replace_stack.sh
├── src/
│   └── qwen3_vl_single_gpu/
│       ├── app.py
│       ├── config.py
│       ├── prompts.py
│       ├── api/
│       │   └── schemas.py
│       ├── clients/
│       │   └── llama_backend.py
│       ├── models/
│       │   └── dino_encoder.py
│       ├── services/
│       │   ├── chat_proxy.py
│       │   ├── qwen_verifier.py
│       │   ├── rerank.py
│       │   └── vector_store.py
│       └── utils/
│           └── image_io.py
└── tests/
    ├── test_chat_proxy.py
    ├── test_config.py
    ├── test_rerank.py
    └── test_vector_store.py
```

不进入 Git 的本地产物：

- `.env.local`
- `logs/`
- `vendor/`
- `benchmark/roamii/results/`
- `__pycache__/`

## 代码职责

`scripts/`

- `bootstrap_llama_cpp.sh`: 拉取并构建 `llama.cpp`
- `load_env.sh`: 加载仓库根目录 `.env.local` 或 `.env`
- `start_llama_backend.sh`: 启动 `18080` 后端
- `start_service.sh`: 启动 `10003` 包装服务
- `start_replace_stack.sh`: 启动或复用整套服务
- `status_replace_stack.sh`: 查看 tmux、端口和健康状态
- `stop_replace_stack.sh`: 停止整套服务
- `smoke_test.sh`: 基础联调

`src/qwen3_vl_single_gpu/`

- `app.py`: FastAPI 入口
- `config.py`: 环境变量与默认配置
- `models/dino_encoder.py`: `DINOv3 large` 编码器
- `services/vector_store.py`: ES 向量库存储与检索
- `services/rerank.py`: query/candidate 的 rerank 流程
- `services/chat_proxy.py`: Qwen 问答代理

`docs/`

- `database_operations_es.md`: 数据库接口、GPU 和运维说明
- `github_publish.md`: 如何把本分支更新推到 GitHub

## 模型与依赖

- Qwen 权重目录：`/share/shared_weights/Qwen3-VL-8B-Instruct-GGUF`
- DINO 权重目录：`/share/shared_weights/dinov3/facebook/dinov3-vith16plus-pretrain-lvd1689m`
- Benchmark 数据目录：`/share/shared_datasets/roamii-benchmark-frozen/merged_all`

首次部署前：

```bash
cd /data1/students/zzh/semantic-image-matcher
cp .env.example .env.local
```

至少补全这些 `.env.local` 字段：

- `ES_HOST`
- `ES_USER`
- `ES_PASS`
- `ES_CERT`
- `IMAGE_URL_REWRITE_FROM`
- `IMAGE_URL_REWRITE_TO`

## 启动方式

推荐环境：

```bash
conda activate /home/test/.conda/envs/semantic_matcher
cd /data1/students/zzh/semantic-image-matcher
```

### 单卡默认

这条命令会让后端和包装服务都落在同一张卡上：

```bash
CUDA_DEVICE=0 ./scripts/start_replace_stack.sh
```

对应关系：

- 后端使用 `BACKEND_CUDA_DEVICE=${CUDA_DEVICE}`
- 包装服务使用 `SERVICE_CUDA_DEVICE=${CUDA_DEVICE}`
- 包装服务内部 `DINO_DEVICE=cuda:0`
- `/vector/encode`、`/add_image`、`/search_vector` 共用同一份 `DINOv3 large`

### 可选分卡

如果单卡显存紧张，可以显式拆成两张卡：

```bash
BACKEND_CUDA_DEVICE=4 SERVICE_CUDA_DEVICE=5 ./scripts/start_replace_stack.sh
```

### 常用运维

查看状态：

```bash
./scripts/status_replace_stack.sh
```

停止服务：

```bash
./scripts/stop_replace_stack.sh
```

### 首次构建 llama.cpp

```bash
./scripts/bootstrap_llama_cpp.sh
```

`vendor/` 不建议提交到 GitHub，首次拉起时按脚本构建即可。

## 接口

服务基地址：

```text
http://127.0.0.1:10003
```

### `POST /v1/chat/completions`

OpenAI 风格问答接口，支持文本和图片 URL。

```bash
curl --noproxy '*' http://127.0.0.1:10003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请描述这张图片"},
          {"type": "image_url", "image_url": {"url": "http://<image-host>:9000/demo.jpg"}}
        ]
      }
    ],
    "max_tokens": 128
  }'
```

### `POST /vector/encode`

输入图片 URL，返回 `DINOv3 large` 的 `1280` 维向量。

```bash
curl --noproxy '*' -X POST http://127.0.0.1:10003/vector/encode \
  -H 'Content-Type: application/json' \
  -d '{
    "imageUrl": "http://<image-host>:9000/demo/query.jpg"
  }'
```

### `POST /vector/rerank`

输入 query 图和按 `DINO` 排序后的候选图，核验 Top-1。

```bash
curl --noproxy '*' -X POST http://127.0.0.1:10003/vector/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "queryImageUrl": "http://<image-host>:9000/demo/query.jpg",
    "candidateImageUrls": [
      "http://<image-host>:9000/demo/c1.jpg",
      "http://<image-host>:9000/demo/c2.jpg"
    ]
  }'
```

### `POST /add_image`

兼容历史入库接口，底层直接写 Elasticsearch。

```bash
curl --noproxy '*' -X POST http://127.0.0.1:10003/add_image \
  -H 'Content-Type: application/json' \
  -d '{
    "image_base64": "<base64_image>"
  }'
```

### `POST /search_vector`

兼容历史检索接口，底层直接查 Elasticsearch。

```bash
curl --noproxy '*' -X POST http://127.0.0.1:10003/search_vector \
  -H 'Content-Type: application/json' \
  -d '{
    "image_base64": "<base64_image>",
    "k": 5,
    "index_list": [123]
  }'
```

### `GET /stats`

```bash
curl --noproxy '*' http://127.0.0.1:10003/stats
```

## 网络与代理

- 如果配置了 `IMAGE_URL_REWRITE_FROM` 和 `IMAGE_URL_REWRITE_TO`，服务会先做图片 URL 前缀改写
- 默认不继承系统 `HTTP_PROXY/HTTPS_PROXY`
- 本地回环和内网图片地址默认直连

## Benchmark

```bash
cd /data1/students/zzh/semantic-image-matcher
python benchmark/roamii/run_api_benchmark.py \
  --base-url http://127.0.0.1:10003 \
  --warmup 1 \
  --manifest benchmark/roamii/manifests/merged_all_dinov3_top1_v4_smoke16.json
```

## 验证

已经补齐数据库兼容相关测试：

- `tests/test_config.py`
- `tests/test_vector_store.py`

包装服务相关测试：

- `tests/test_chat_proxy.py`
- `tests/test_rerank.py`

## 日志

默认日志目录：

```text
/data1/students/zzh/semantic-image-matcher/logs
```

主要文件：

- `logs/llama_backend.log`
- `logs/wrapper_service.log`
