# -*- coding: utf-8 -*-
"""
代理模型推理耗时 benchmark

测试场景:
  1. CPU 单样本 (batch=1)
  2. CPU 批量 (batch=1024)
  3. GPU 单样本 (batch=1)
  4. GPU 批量 (batch=1024)
  5. GPU 批量 sweep (5,000,000 Dirichlet 候选，分块执行)

每场景: 预热 100 次 + 正式测 1000 次, 取中位数与 P95.
"""
import os
import sys
import time
import platform
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from surrogate_model.model import SurrogateModel

MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'surrogate_model', 'best_model.pth')
LOG_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'benchmark_inference.log')

N_WARMUP = 100
N_TRIALS = 1000
N_CANDIDATES = 5_000_000  # Dirichlet 采样候选数（每工况），对齐成稿


def time_inference(model, x, device, n_trials=N_TRIALS, n_warmup=N_WARMUP):
    model.eval()
    is_cuda = device.type == 'cuda'

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()

        times = np.empty(n_trials, dtype=np.float64)
        for i in range(n_trials):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times[i] = time.perf_counter() - t0
    return times


def fmt_stats(times, n_samples):
    med = np.median(times)
    p95 = np.percentile(times, 95)
    per_sample_med = med / n_samples
    return med, p95, per_sample_med


def run_full_sweep(model, device, n_total=N_CANDIDATES, chunk=65536):
    """分块推理全部候选点，返回总耗时"""
    model.eval()
    is_cuda = device.type == 'cuda'
    n_chunks = (n_total + chunk - 1) // chunk
    # warm up chunk
    x_warm = torch.randn(chunk, 7, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x_warm)
    if is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_chunks):
            bs = chunk if (i + 1) * chunk <= n_total else n_total - i * chunk
            x = torch.randn(bs, 7, device=device)
            _ = model(x)
    if is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    lines = []
    def log(s):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("Surrogate Model Inference Benchmark")
    log("=" * 70)
    log(f"Platform : {platform.platform()}")
    log(f"CPU      : {platform.processor()}")
    log(f"Python   : {platform.python_version()}")
    log(f"PyTorch  : {torch.__version__}")
    log(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU      : {torch.cuda.get_device_name(0)}")
    log("")

    model = SurrogateModel.load(MODEL_PATH)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model    : {total_params:,} params, hidden={model.hidden_dim}, layers={model.n_layers}")
    log("")

    results = {}
    for dev_name, dev in [('CPU', torch.device('cpu')),
                          ('GPU', torch.device('cuda') if torch.cuda.is_available() else None)]:
        if dev is None:
            continue
        model_dev = SurrogateModel.load(MODEL_PATH).to(dev)
        for bs in [1, 1024]:
            x = torch.randn(bs, 7, device=dev)
            times = time_inference(model_dev, x, dev)
            med, p95, per = fmt_stats(times, bs)
            key = f"{dev_name} batch={bs}"
            results[key] = (med, p95, per)
            log(f"{key:15s}: median {med*1e3:8.4f} ms | P95 {p95*1e3:8.4f} ms"
                f" | per-sample {per*1e6:8.3f} us")

        if dev_name == 'GPU':
            t_full = run_full_sweep(model_dev, dev)
            log(f"GPU full sweep ({N_CANDIDATES:,} pts) : {t_full:.3f} s"
                f" ({t_full*1e6/N_CANDIDATES:.3f} us/pt amortized)")
            results['GPU_full_sweep_s'] = t_full

    log("")
    log("=" * 70)
    log("Speedup vs FEM (assume single FEM run = 5~10 min)")
    log("=" * 70)
    fem_low, fem_high = 5 * 60, 10 * 60
    gpu_single = results['GPU batch=1'][0]
    log(f"GPU single inference median: {gpu_single*1e3:.4f} ms")
    log(f"Speedup per case : {fem_low/gpu_single:,.0f}x  (vs 5 min FEM)")
    log(f"Speedup per case : {fem_high/gpu_single:,.0f}x  (vs 10 min FEM)")
    if 'GPU_full_sweep_s' in results:
        t_full = results['GPU_full_sweep_s']
        fem_total_low = N_CANDIDATES * fem_low / 3600 / 24 / 365
        fem_total_high = N_CANDIDATES * fem_high / 3600 / 24 / 365
        log(f"Full sweep {N_CANDIDATES:,} pts: surrogate {t_full:.2f} s"
            f" vs FEM {fem_total_low:.1f}-{fem_total_high:.1f} years")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"\nLog saved to: {LOG_PATH}")


if __name__ == '__main__':
    main()
