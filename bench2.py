import time
import numpy as np
from MetalPy import MetalPy

mp = MetalPy()

N = 10_000_000
X_np = np.random.randn(N).astype(np.float32)
X = mp.array(X_np.tolist())

def bench(name, gpu_fn, np_fn, warmup=5, runs=100):
    # Warmup GPU
    for _ in range(warmup):
        gpu_fn()
    
    # Benchmark GPU
    gpu_times = []
    for _ in range(runs):
        start = time.perf_counter()
        gpu_result = gpu_fn()
        gpu_times.append((time.perf_counter() - start) * 1000)
    
    # Benchmark NumPy
    np_times = []
    for _ in range(runs):
        start = time.perf_counter()
        np_result = np_fn()
        np_times.append((time.perf_counter() - start) * 1000)
    
    gpu_avg = sum(gpu_times) / len(gpu_times)
    gpu_min = min(gpu_times)
    np_avg = sum(np_times) / len(np_times)
    np_min = min(np_times)
    
    print(f"=== {name} ===")
    print(f"GPU:   {gpu_result} ({gpu_min:.3f}ms min, {gpu_avg:.3f}ms avg)")
    print(f"NumPy: {np_result} ({np_min:.3f}ms min, {np_avg:.3f}ms avg)")
    print()

print(f"\nBenchmarking on {N:,} elements")
print("=" * 50)

bench("SUM", lambda: mp.sum(X), lambda: np.sum(X_np))
bench("PRODUCT", lambda: mp.product(X), lambda: np.prod(X_np))
bench("MAX", lambda: mp.max(X), lambda: np.max(X_np))
bench("MIN", lambda: mp.min(X), lambda: np.min(X_np))
bench("ARGMAX", lambda: mp.argmax(X), lambda: np.argmax(X_np))
bench("ARGMIN", lambda: mp.argmin(X), lambda: np.argmin(X_np))
