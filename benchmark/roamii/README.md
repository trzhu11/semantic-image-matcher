# Roamii API Benchmark

这个目录复用 `roamii` 冻结后的 benchmark 口径，但评测方式改成真实调用当前服务的两个接口：

- `POST /vector/encode`
- `POST /vector/rerank`

说明：

- 准确率指标仍然是 `漏判率 = 1 - TP / N`、`误判率 = FP / N`
- 候选图片不再重新跑 FAISS，而是直接使用冻结 manifest 里的 `DINOv3 top-1`
- DINO 侧记录真实 `/vector/encode` 耗时
- VLM 侧记录真实 `/vector/rerank` 耗时
- benchmark 会先抓 `/health`，把部署时的 `dinoDevice`、模型版本等信息一起写进结果
- 默认先做 1 轮 warmup，再进入正式计时，避免把首轮模型加载记进 steady-state 延迟
- 为了保持与 `ztr` 的接口完全兼容，benchmark 会通过 `?includeTimings=1` 显式请求 timing 字段

## 数据

默认使用：

```bash
/share/shared_datasets/roamii-benchmark-frozen/merged_all
```

## 运行

先启动后端和包装服务，然后执行：

```bash
cd /data1/students/zzh/semantic-image-matcher
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

结果会写到 `benchmark/roamii/results/<timestamp>/`：

- `benchmark_summary.json`
- `benchmark_result.json`
