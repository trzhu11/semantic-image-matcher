# Qwen3-VL 数据库兼容接口运维文档（ES 版）

更新时间：2026-03-16  
适用目录：`/data1/students/zzh/semantic-image-matcher`

## 1. 当前形态

这个分支的 `10003` 统一承载：

- `POST /v1/chat/completions`
- `POST /vector/encode`
- `POST /vector/rerank`
- `POST /add_image`
- `POST /search_vector`
- `GET /health`
- `GET /stats`

数据库兼容接口底层固定使用 Elasticsearch。

## 2. 向量库配置

默认索引：

- `intour_vector_store`

字段：

- `faiss_index`: `integer`
- `image_vector`: `dense_vector`, `1280` 维, `cosine`
- `created_at`: `date`

默认图片编码模型：

- `dinov3-vith16plus-pretrain-lvd1689m`

## 3. GPU 约定

### 单卡默认

```bash
CUDA_DEVICE=0 ./scripts/start_replace_stack.sh
```

这时：

- Qwen 后端和包装服务都落在物理 GPU `0`
- 包装服务内部的 `DINO_DEVICE=cuda:0`
- `/vector/encode`、`/add_image`、`/search_vector` 共用同一个 `DINOv3 large`

### 可选分卡

```bash
BACKEND_CUDA_DEVICE=4 SERVICE_CUDA_DEVICE=5 ./scripts/start_replace_stack.sh
```

这时：

- Qwen 后端在物理 GPU `4`
- 包装服务和数据库兼容接口在物理 GPU `5`
- 对包装服务进程来说，`DINO_DEVICE=cuda:0` 实际映射到物理 GPU `5`

## 4. 启停命令

单卡重启：

```bash
cd /data1/students/zzh/semantic-image-matcher
./scripts/stop_replace_stack.sh
CUDA_DEVICE=0 ./scripts/start_replace_stack.sh
```

分卡重启：

```bash
cd /data1/students/zzh/semantic-image-matcher
./scripts/stop_replace_stack.sh
BACKEND_CUDA_DEVICE=4 SERVICE_CUDA_DEVICE=5 ./scripts/start_replace_stack.sh
```

查看状态：

```bash
./scripts/status_replace_stack.sh
```

## 5. 敏感配置

把这些内容写在仓库根目录 `.env.local`：

- `ES_HOST`
- `ES_USER`
- `ES_PASS`
- `ES_CERT`
- `ES_VERIFY_CERTS`
- `IMAGE_URL_REWRITE_FROM`
- `IMAGE_URL_REWRITE_TO`

`.env.local` 不提交 Git。

## 6. 核验命令

健康检查：

```bash
curl --noproxy '*' -s http://127.0.0.1:10003/health
curl --noproxy '*' -s http://127.0.0.1:10003/stats
```

查看 ES mapping：

```bash
cd /data1/students/zzh/semantic-image-matcher
source .env.local
${CONDA_PYTHON:-python} - <<'PY'
import os
from elasticsearch import Elasticsearch

kwargs = {
    "hosts": [os.environ["ES_HOST"]],
    "verify_certs": os.environ.get("ES_VERIFY_CERTS", "1") == "1",
}
if os.environ.get("ES_USER") or os.environ.get("ES_PASS"):
    kwargs["basic_auth"] = (os.environ.get("ES_USER", ""), os.environ.get("ES_PASS", ""))
if os.environ.get("ES_CERT"):
    kwargs["ca_certs"] = os.environ["ES_CERT"]

es = Elasticsearch(**kwargs)
mapping = es.indices.get_mapping(index="intour_vector_store")
print(mapping["intour_vector_store"]["mappings"]["properties"]["image_vector"])
PY
```

## 7. 工程师最小联调示例

局域网访问时，把 `<lan-host>` 替换成当前机器内网地址。

添加图片向量：

```bash
curl --noproxy '*' -X POST http://<lan-host>:10003/add_image \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64_image>"
  }'
```

按指定索引检索：

```bash
curl --noproxy '*' -X POST http://<lan-host>:10003/search_vector \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64_image>",
    "k": 5,
    "index_list": [123]
  }'
```
