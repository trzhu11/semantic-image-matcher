# Semantic Image Matcher: Int8 Single GPU Deploy

这个仓库的 `int8-single-gpu-deploy` 分支是独立于 `/data1/students/zzh/ztr` 的单卡替换部署，目标是：

- 用 `Qwen3-VL-8B-Instruct-GGUF` 在一张 `16GB` 卡上提供稳定的 VLM 能力
- 保持 `ztr` 两个图搜接口的调用方式和返回结构不变
- 额外提供一个可直接用于日常问答的 `Qwen3` 接口
- 默认替换线上旧端口：`18080` 和 `10003`

当前默认部署模式：

- `18080`: `llama.cpp` 原生 OpenAI 风格后端
- `10003`: 包装服务，提供 `Qwen3` 问答和两个图搜接口
- 单卡优先：`Qwen + DINOv3 large` 放同一张卡
- 如果单卡不稳，再把 DINO 单独挪到另一张卡

## 目录结构

```text
semantic-image-matcher/
├── README.md
├── requirements.txt
├── benchmark/
│   └── roamii/
│       ├── README.md
│       ├── manifests/
│       ├── results/
│       └── run_api_benchmark.py
├── scripts/
│   ├── bootstrap_llama_cpp.sh
│   ├── smoke_test.sh
│   ├── start_llama_backend.sh
│   ├── start_replace_stack.sh
│   ├── status_replace_stack.sh
│   ├── stop_replace_stack.sh
│   └── start_service.sh
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
│       │   └── rerank.py
│       └── utils/
│           └── image_io.py
└── tests/
    ├── test_chat_proxy.py
    └── test_rerank.py
```

## 代码职责

`scripts/`

- `bootstrap_llama_cpp.sh`: 构建或更新 `llama.cpp`
- `start_llama_backend.sh`: 启动 `18080` 上的原生 Qwen 后端
- `start_service.sh`: 启动 `10003` 上的替换包装服务
- `start_replace_stack.sh`: 一键启动整套服务，默认用 `tmux` 常驻托管
- `status_replace_stack.sh`: 查看 `tmux` 会话、端口和健康状态
- `stop_replace_stack.sh`: 停掉整套替换服务
- `smoke_test.sh`: 走真实接口做基础联调

`src/qwen3_vl_single_gpu/`

- `app.py`: FastAPI 入口，暴露三个接口
- `config.py`: 统一环境变量和默认配置
- `prompts.py`: benchmark 对齐的 VLM prompt
- `clients/llama_backend.py`: 对 `llama-server` 的本地 HTTP 调用
- `models/dino_encoder.py`: `DINOv3 large` 编码器
- `services/chat_proxy.py`: `Qwen3` 问答代理，支持远程图片转 `data:` URL
- `services/qwen_verifier.py`: 双图核验逻辑
- `services/rerank.py`: Top-1 候选选择与核验流程
- `utils/image_io.py`: 图片下载、URL 改写、缩放和编码

`benchmark/roamii/`

- `run_api_benchmark.py`: 用真实接口跑完整 benchmark
- `manifests/`: 冻结候选集
- `results/`: 输出结果和摘要

## 模型与权重

- Qwen 权重目录：`/share/shared_weights/Qwen3-VL-8B-Instruct-GGUF`
- DINO 权重目录：`/share/shared_weights/dinov3/facebook/dinov3-vith16plus-pretrain-lvd1689m`
- Benchmark 数据目录：`/share/shared_datasets/roamii-benchmark-frozen/merged_all`

如果后续需要下载新权重，优先用 `ModelScope`，并统一落到 `/share/shared_weights`。

## 默认端口与替换关系

- 旧 Qwen 服务端口：`18080`
- 新 Qwen 原生后端端口：`18080`
- `ztr` 图搜服务端口：`10003`
- 新替换包装服务端口：`10003`

因此，替换完成后：

- 原来调 `http://127.0.0.1:18080` 的调用方无需改地址
- 原来调 `http://127.0.0.1:10003/vector/encode` 和 `http://127.0.0.1:10003/vector/rerank` 的调用方无需改地址

## 启动

推荐环境：

```bash
conda activate /home/test/.conda/envs/semantic_matcher
cd /data1/students/zzh/semantic-image-matcher
```

### 一键启动

工程师直接用这一条即可：

```bash
CUDA_DEVICE=2 ./scripts/start_replace_stack.sh
```

这条命令会：

- 启动或复用 `18080` 上的 Qwen 后端
- 等待后端就绪
- 启动或复用 `10003` 上的包装服务
- 等待 `/health` 返回成功
- 默认使用 `tmux` 常驻托管，不会因为当前终端退出而自动停掉
- 默认使用独立的 `tmux` socket: `semantic_image_matcher`

说明：

- 这台机器已经准备好可用的 `llama.cpp`
- 如果换到一台新机器，第一次启动前先执行一次 `./scripts/bootstrap_llama_cpp.sh`

### 常用运维命令

查看状态：

```bash
./scripts/status_replace_stack.sh
```

停止服务：

```bash
./scripts/stop_replace_stack.sh
```

### 1. 构建 `llama.cpp`

```bash
./scripts/bootstrap_llama_cpp.sh
```

### 2. 单卡启动 Qwen 后端

下面命令会把 `Qwen3-VL-8B-Instruct-GGUF` 启到 `18080`：

```bash
CUDA_DEVICE=2 ./scripts/start_llama_backend.sh
```

默认行为：

- `CUDA_VISIBLE_DEVICES=<CUDA_DEVICE>`
- `--split-mode none`
- `--main-gpu 0`
- `flash-attn` 开启
- KV cache 使用 `q8_0`
- `reasoning-budget=0`

### 3. 单卡启动替换包装服务

下面命令会把三个接口启到 `10003`：

```bash
CUDA_DEVICE=2 ./scripts/start_service.sh
```

默认行为：

- 访问本地 `Qwen` 后端 `http://127.0.0.1:18080`
- `DINO_DEVICE=cuda:0`
- `QWEN_MAX_NEW_TOKENS=4`
- `MAX_IMAGE_DIM=768`
- 服务启动时预加载 `DINO`

### 4. 显存不够时的降级方案

如果某张 `16GB` 卡无法同时承载 `Qwen + DINO large`，保持 Qwen 单卡不变，只把 DINO 挪到另一张卡：

```bash
CUDA_DEVICE=2 DINO_DEVICE=cuda:1 CUDA_VISIBLE_DEVICES=2,3 ./scripts/start_service.sh
```

这会让：

- `Qwen` 继续走 `CUDA_VISIBLE_DEVICES` 中的第 `0` 张卡
- `DINO` 走第 `1` 张卡

## 三个接口

包装服务统一地址：

```text
http://127.0.0.1:10003
```

当前机器在 `2026-03-13` 可用的局域网地址包含：

```text
http://10.168.100.13:10003
```

如果局域网用户访问，优先让他们用这个地址；如果后续网卡地址有变化，用 `hostname -I` 重新确认。

### 1. `POST /v1/chat/completions`

用途：日常问答，兼容 OpenAI 风格请求；支持纯文本，也支持消息里带图片 URL。

示例：

```bash
curl --noproxy '*' http://10.168.100.13:10003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-vl-8b-instruct-q8_0",
    "messages": [
      {
        "role": "user",
        "content": "南京有哪些适合一日游的景点？"
      }
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

带图示例：

```bash
curl --noproxy '*' http://127.0.0.1:10003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请描述这张图片"},
          {"type": "image_url", "image_url": {"url": "http://10.168.100.13:9000/demo.jpg"}}
        ]
      }
    ],
    "max_tokens": 128
  }'
```

说明：

- 若消息中包含远程图片 URL，包装层会先下载、缩放，再转成 `data:` URL 发给后端
- 如果调用方只需要原生后端，也可以直接访问 `http://127.0.0.1:18080/v1/chat/completions`

### 2. `POST /vector/encode`

用途：输入一张图片，返回 `DINOv3 large` 的 `1280` 维向量。调用方式与 `/data1/students/zzh/ztr` 完全一致。

请求：

```json
{
  "imageUrl": "https://example.com/query.jpg"
}
```

响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "vector": [0.123, -0.045, "..."],
    "dim": 1280,
    "modelVersion": "dinov3-vith16plus-pretrain-lvd1689m"
  }
}
```

示例：

```bash
curl --noproxy '*' -X POST http://10.168.100.13:10003/vector/encode \
  -H 'Content-Type: application/json' \
  -d '{
    "imageUrl": "http://10.168.100.13:9000/demo/query.jpg"
  }'
```

### 3. `POST /vector/rerank`

用途：输入 query 图和已按 `DINO` 相似度降序排好的候选列表，当前只核验 Top-1，返回是否命中。调用方式与 `/data1/students/zzh/ztr` 完全一致。

请求：

```json
{
  "queryImageUrl": "https://example.com/query.jpg",
  "candidateImageUrls": [
    "https://example.com/c1.jpg",
    "https://example.com/c2.jpg",
    "https://example.com/c3.jpg"
  ]
}
```

命中响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "matchedImageUrl": "https://example.com/c1.jpg",
    "matchedIndex": 0,
    "checkedCount": 1,
    "isMatch": true,
    "strategy": "top1_ordered",
    "modelVersion": "Qwen3-VL-8B-Instruct"
  }
}
```

未命中响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "matchedImageUrl": null,
    "matchedIndex": null,
    "checkedCount": 1,
    "isMatch": false,
    "strategy": "top1_ordered",
    "modelVersion": "Qwen3-VL-8B-Instruct"
  }
}
```

示例：

```bash
curl --noproxy '*' -X POST http://10.168.100.13:10003/vector/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "queryImageUrl": "http://10.168.100.13:9000/demo/query.jpg",
    "candidateImageUrls": [
      "http://10.168.100.13:9000/demo/c1.jpg",
      "http://10.168.100.13:9000/demo/c2.jpg"
    ]
  }'
```

## 健康检查

- 包装服务：`GET /health`
- 原生后端：`GET /health`、`GET /v1/models`

示例：

```bash
curl --noproxy '*' http://10.168.100.13:10003/health
curl --noproxy '*' http://10.168.100.13:18080/v1/models
```

## 与 `ztr` 的兼容性

当前兼容范围：

- `/vector/encode` 请求字段一致
- `/vector/encode` 默认返回字段一致
- `/vector/rerank` 请求字段一致
- `/vector/rerank` 默认返回字段一致
- `400 / 422 / 500` 错误格式一致
- `candidateImageUrls` 为空时返回 `400`

额外能力：

- `POST /v1/chat/completions`
- benchmark 可显式带 `?includeTimings=1` 获取 timing 字段

注意：

- `?includeTimings=1` 只用于 benchmark，不会影响默认兼容响应

## 网络与代理

图片下载默认行为：

- `https://1001pqej17305.vicp.fun/` 会改写到 `http://10.168.100.13:9000/`
- 默认不继承系统 `HTTP_PROXY/HTTPS_PROXY`
- 本地回环和内网图片地址默认直连

相关环境变量：

- `IMAGE_URL_REWRITE_FROM`
- `IMAGE_URL_REWRITE_TO`
- `IMAGE_HTTP_TRUST_ENV`
- `IMAGE_FORCE_DIRECT_HOSTS`

如果当前 shell 配了代理，手工调本地服务时建议带：

```bash
curl --noproxy '*'
```

如果出现你截图里的 `Failed to connect ... port 10003`，这不是代理问题，而是服务没有在监听。先执行：

```bash
./scripts/status_replace_stack.sh
```

如果状态异常，重新执行：

```bash
CUDA_DEVICE=2 ./scripts/start_replace_stack.sh
```

## Benchmark

评测方式是走真实服务接口，而不是直接调内部 Python 函数。

烟测：

```bash
python benchmark/roamii/run_api_benchmark.py \
  --base-url http://127.0.0.1:10003 \
  --warmup 1 \
  --manifest benchmark/roamii/manifests/merged_all_dinov3_top1_v4_smoke16.json
```

完整集：

```bash
python benchmark/roamii/run_api_benchmark.py \
  --base-url http://127.0.0.1:10003 \
  --warmup 1 \
  --manifest benchmark/roamii/manifests/merged_all_dinov3_top1_v4.json
```

当前完整 benchmark 结果：

- 结果目录：`benchmark/roamii/results/20260312_175712`
- `DINO` 平均用时：`185.74 ms`
- `VLM` 平均用时：`1040.34 ms`
- 单次完整链路平均用时：`1226.57 ms`
- `miss_rate`: `16.49%`
- `fp_rate`: `2.06%`

## 日志

默认日志目录：

```text
/data1/students/zzh/semantic-image-matcher/logs
```

主要日志文件：

- `llama_backend.log`
- `wrapper_service.log`
