import numpy as np
import time
from MetalPy import MetalPy

mp = MetalPy()

# ============================
# Benchmark Configuration
# ============================
N = 10_000_000
print(f"\nBenchmarking REDUCTIONS on {N:,} elements")
print("==========================================\n")

X_np = np.random.rand(N).astype(np.float32)
X = mp.array(X_np.tolist())

# Warm up GPU
mp.sum(X)


# ==========================================
# Benchmark helper
# ==========================================
def bench(name, gpu_fn, np_fn, X, X_np):
    print(f"=== {name} ===")

    # GPU timed
    t0 = time.time()
    gpu_val = gpu_fn(X)
    t1 = time.time()
    gpu_time = (t1 - t0) * 1000

    # NumPy timed
    t2 = time.time()
    np_val = np_fn(X_np)
    t3 = time.time()
    np_time = (t3 - t2) * 1000

    print(f"GPU:   {gpu_val}      ({gpu_time:.3f} ms)")
    print(f"NumPy: {np_val}      ({np_time:.3f} ms)\n")


# ==========================================
# Run all reduction benchmarks
# ==========================================
bench("SUM",      mp.sum,      np.sum,      X, X_np)
bench("PRODUCT",  mp.product,  np.prod,     X, X_np)
bench("MEAN",     mp.mean,     np.mean,     X, X_np)
bench("MAX",      mp.max,      np.max,      X, X_np)
bench("MIN",      mp.min,      np.min,      X, X_np)
bench("ARGMAX",   mp.argmax,   np.argmax,   X, X_np)
bench("ARGMIN",   mp.argmin,   np.argmin,   X, X_np)

