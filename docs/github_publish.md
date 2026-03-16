# GitHub 分支更新指南

适用目录：`/data1/students/zzh/semantic-image-matcher`  
目标仓库：`https://github.com/trzhu11/semantic-image-matcher`  
目标分支：`int8-single-gpu-deploy`

## 1. 更新前确认

先确认当前就在目标分支：

```bash
cd /data1/students/zzh/semantic-image-matcher
git branch --show-current
```

期望输出：

```text
int8-single-gpu-deploy
```

## 2. 查看本次会提交什么

```bash
git status --short --ignored
```

应当提交的内容包括：

- `README.md`
- `docs/`
- `src/`
- `scripts/`
- `tests/`
- `.env.example`

应当忽略的内容包括：

- `.env.local`
- `logs/`
- `vendor/`
- `benchmark/roamii/results/`
- `__pycache__/`

## 3. 提交本地改动

```bash
cd /data1/students/zzh/semantic-image-matcher
git add .
git commit -m "Add ES-backed vector DB compatibility to single-gpu deploy"
```

## 4. 推到远程分支

```bash
git push origin int8-single-gpu-deploy
```

如果你想先同步远程再推：

```bash
git fetch origin
git pull --rebase origin int8-single-gpu-deploy
git push origin int8-single-gpu-deploy
```

## 5. 推送后检查

重点检查：

- `README.md` 是否已经包含 `/add_image`、`/search_vector`、`/stats`
- `docs/database_operations_es.md` 是否已上传
- `src/qwen3_vl_single_gpu/services/vector_store.py` 是否已上传
- `tests/test_vector_store.py` 是否已上传
- `.env.local` 是否没有进入提交

## 6. 新机器拉起

```bash
git clone -b int8-single-gpu-deploy https://github.com/trzhu11/semantic-image-matcher
cd semantic-image-matcher
cp .env.example .env.local
```

补全 `.env.local` 后：

单卡：

```bash
CUDA_DEVICE=0 ./scripts/start_replace_stack.sh
```

分卡：

```bash
BACKEND_CUDA_DEVICE=4 SERVICE_CUDA_DEVICE=5 ./scripts/start_replace_stack.sh
```
