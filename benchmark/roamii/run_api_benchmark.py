#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATASET_DIR = Path("/share/shared_datasets/roamii-benchmark-frozen/merged_all")
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "merged_all_dinov3_top1_v4.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Roamii benchmark through the API wrapper")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:10003", help="Wrapper API base URL")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Benchmark manifest")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset directory")
    parser.add_argument("--area", type=str, default="merged_all", help="Area name or merged_all")
    parser.add_argument("--limit", type=int, default=0, help="Optional item limit")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds before timed benchmark")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional result directory")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_json(url: str) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, parsed


def post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read().decode("utf-8")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return response.status, json.loads(body), elapsed_ms
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, parsed, elapsed_ms


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 2)
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return round(ordered[index], 2)


def summarize_latencies(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "avg": round(statistics.fmean(values), 2),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    tp = 0
    fp = 0
    yes_count = 0
    for item in results:
        if item["vlm_match"]:
            yes_count += 1
            if item["candidate_in_gt"]:
                tp += 1
            else:
                fp += 1
    miss_rate = round((1 - tp / n) * 100, 2) if n else 100.0
    fp_rate = round(fp / n * 100, 2) if n else 0.0
    return {
        "total_test": n,
        "tp": tp,
        "fp": fp,
        "yes_count": yes_count,
        "miss_rate": miss_rate,
        "fp_rate": fp_rate,
    }


def select_items(manifest: dict[str, Any], area: str, limit: int) -> list[dict[str, Any]]:
    items = manifest["items"]
    if area != "merged_all":
        items = [item for item in items if item["area_name"] == area]
    if limit > 0:
        items = items[:limit]
    return items


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RESULTS_DIR / stamp


def load_reference_metrics(manifest: dict[str, Any], area: str, total_test: int) -> dict[str, Any] | None:
    reference = manifest.get("reference_metrics_by_area", {}).get(area)
    if not reference:
        return None
    if reference.get("total_test") != total_test:
        return None
    return reference


def format_delta(current: float, reference: float) -> float:
    return round(current - reference, 2)


def verify_item_paths(dataset_dir: Path, item: dict[str, Any]) -> tuple[Path, Path]:
    query_path = dataset_dir / item["query_image"]
    candidate_path = dataset_dir / item["candidate_image"]
    if not query_path.is_file():
        raise FileNotFoundError(f"Missing query image: {query_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Missing candidate image: {candidate_path}")
    return query_path, candidate_path


def run_pair(base_url: str, query_path: Path, candidate_path: Path) -> dict[str, Any]:
    encode_status, encode_body, encode_http_ms = post_json(
        f"{base_url}/vector/encode?includeTimings=1",
        {"imageUrl": str(query_path)},
    )
    if encode_status != 200:
        raise RuntimeError(f"/vector/encode failed for {query_path}: {encode_body}")

    rerank_status, rerank_body, rerank_http_ms = post_json(
        f"{base_url}/vector/rerank?includeTimings=1",
        {
            "queryImageUrl": str(query_path),
            "candidateImageUrls": [str(candidate_path)],
        },
    )
    if rerank_status != 200:
        raise RuntimeError(f"/vector/rerank failed for {query_path}: {rerank_body}")

    encode_data = encode_body["data"]
    rerank_data = rerank_body["data"]
    return {
        "encode_data": encode_data,
        "rerank_data": rerank_data,
        "dino_http_ms": encode_http_ms,
        "dino_server_ms": float(encode_data.get("timings", {}).get("dinoEncodeMs", encode_http_ms)),
        "vlm_http_ms": rerank_http_ms,
        "vlm_server_ms": float(rerank_data.get("timings", {}).get("vlmMs", rerank_http_ms)),
        "rerank_total_server_ms": float(rerank_data.get("timings", {}).get("totalMs", rerank_http_ms)),
    }


def warmup(base_url: str, dataset_dir: Path, item: dict[str, Any], warmup_rounds: int) -> None:
    if warmup_rounds <= 0:
        return
    query_path, candidate_path = verify_item_paths(dataset_dir, item)
    for round_index in range(1, warmup_rounds + 1):
        pair = run_pair(base_url, query_path, candidate_path)
        print(
            f"[warmup {round_index}/{warmup_rounds}] "
            f"dino={pair['dino_server_ms']:.2f}ms "
            f"vlm={pair['vlm_server_ms']:.2f}ms"
        )


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    items = select_items(manifest, args.area, args.limit)
    if not items:
        raise ValueError("No benchmark items selected")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url.rstrip("/")
    health_status, health_body = get_json(f"{base_url}/health")
    if health_status != 200:
        raise RuntimeError(f"Wrapper health check failed: {health_body}")

    warmup(base_url, args.dataset_dir, items[0], args.warmup)

    results: list[dict[str, Any]] = []
    dino_http_times: list[float] = []
    dino_server_times: list[float] = []
    vlm_http_times: list[float] = []
    vlm_server_times: list[float] = []
    rerank_total_times: list[float] = []

    benchmark_started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        query_path, candidate_path = verify_item_paths(args.dataset_dir, item)
        pair = run_pair(base_url, query_path, candidate_path)

        dino_http_times.append(pair["dino_http_ms"])
        dino_server_times.append(pair["dino_server_ms"])
        vlm_http_times.append(pair["vlm_http_ms"])
        vlm_server_times.append(pair["vlm_server_ms"])
        rerank_total_times.append(pair["rerank_total_server_ms"])

        rerank_data = pair["rerank_data"]
        vlm_match = bool(rerank_data["isMatch"])
        result = {
            **item,
            "query_path": str(query_path),
            "candidate_path": str(candidate_path),
            "vlm_match": vlm_match,
            "candidate_in_gt": bool(item["candidate_in_gt"]),
            "matched_image_url": rerank_data["matchedImageUrl"],
            "matched_index": rerank_data["matchedIndex"],
            "checked_count": rerank_data["checkedCount"],
            "dino_encode_http_ms": pair["dino_http_ms"],
            "dino_encode_server_ms": pair["dino_server_ms"],
            "vlm_http_ms": pair["vlm_http_ms"],
            "vlm_server_ms": pair["vlm_server_ms"],
            "rerank_total_server_ms": pair["rerank_total_server_ms"],
        }
        results.append(result)
        print(
            f"[{index:03d}/{len(items):03d}] "
            f"yes={vlm_match} gt={item['candidate_in_gt']} "
            f"dino={pair['dino_server_ms']:.2f}ms "
            f"vlm={pair['vlm_server_ms']:.2f}ms"
        )

    metrics = compute_metrics(results)
    reference = load_reference_metrics(manifest, args.area, metrics["total_test"])
    summary = {
        "benchmark_name": manifest["benchmark_name"],
        "area": args.area,
        "manifest": str(args.manifest),
        "dataset_dir": str(args.dataset_dir),
        "base_url": base_url,
        "service_health": health_body,
        "warmup_rounds": args.warmup,
        "metrics": metrics,
        "timing": {
            "wall_time_s": round(time.perf_counter() - benchmark_started, 2),
            "dino_encode_http_ms": summarize_latencies(dino_http_times),
            "dino_encode_server_ms": summarize_latencies(dino_server_times),
            "vlm_http_ms": summarize_latencies(vlm_http_times),
            "vlm_server_ms": summarize_latencies(vlm_server_times),
            "rerank_total_server_ms": summarize_latencies(rerank_total_times),
        },
    }
    if reference:
        summary["reference_metrics"] = reference
        summary["delta_vs_reference"] = {
            "miss_rate": format_delta(metrics["miss_rate"], reference["miss_rate"]),
            "fp_rate": format_delta(metrics["fp_rate"], reference["fp_rate"]),
            "tp": metrics["tp"] - reference["tp"],
            "fp": metrics["fp"] - reference["fp"],
        }

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark_result.json").write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Output: {output_dir}")
    print(
        f"TP={metrics['tp']} FP={metrics['fp']} "
        f"miss={metrics['miss_rate']}% fp={metrics['fp_rate']}%"
    )
    if reference:
        print(
            "Reference: "
            f"miss={reference['miss_rate']}% fp={reference['fp_rate']}% "
            f"(delta miss={summary['delta_vs_reference']['miss_rate']} pp, "
            f"delta fp={summary['delta_vs_reference']['fp_rate']} pp)"
        )
    print(
        "Timing: "
        f"DINO(avg={summary['timing']['dino_encode_server_ms']['avg']}ms, "
        f"p95={summary['timing']['dino_encode_server_ms']['p95']}ms), "
        f"VLM(avg={summary['timing']['vlm_server_ms']['avg']}ms, "
        f"p95={summary['timing']['vlm_server_ms']['p95']}ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
