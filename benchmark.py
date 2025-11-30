import time
import numpy as np
from MetalPy import MetalPy

def benchmark():
    print("\n" + "="*80)
    print("🚀 MetalPy vs NumPy COMPLETE Benchmark - ALL KERNELS")
    print("="*80)
    
    gpu = MetalPy()
    SIZE = 5_000_000
    
    print(f"\n📊 Array Size: {SIZE:,} elements (5 million)")
    print(f"🖥️  GPU: {gpu.device.name()}")
    print("\n" + "-"*80)
    
    # Create test data ONCE
    print("Creating test data...")
    np_a = np.random.rand(SIZE).astype(np.float32)
    np_b = np.random.rand(SIZE).astype(np.float32)
    np_c = np.random.randn(SIZE).astype(np.float32)  # Can have negative values
    
    # Upload to GPU ONCE
    print("Uploading to GPU...")
    gpu_a = gpu.array(np_a.tolist())
    gpu_b = gpu.array(np_b.tolist())
    gpu_c = gpu.array(np_c.tolist())
    
    print("\nStarting benchmarks...\n")
    
    def run_benchmark(name, metal_func, numpy_func, runs=10):
        # Warmup
        for _ in range(3):
            metal_func()
            numpy_func()
        
        # Benchmark Metal
        metal_times = []
        for _ in range(runs):
            start = time.perf_counter()
            metal_func()
            metal_times.append(time.perf_counter() - start)
        metal_time = min(metal_times) * 1000
        
        # Benchmark NumPy
        numpy_times = []
        for _ in range(runs):
            start = time.perf_counter()
            numpy_func()
            numpy_times.append(time.perf_counter() - start)
        numpy_time = min(numpy_times) * 1000
        
        speedup = numpy_time / metal_time
        speedup_str = f"{speedup:.2f}x"
        if speedup > 1.5:
            speedup_str = "🚀 " + speedup_str
        elif speedup < 0.7:
            speedup_str = "⚠️  " + speedup_str
        
        print(f"{name:<25} {metal_time:<15.4f} {numpy_time:<15.4f} {speedup_str:<10}")
        return speedup
    
    speedups = []
    
    # ======================================================================
    # CATEGORY 1: ELEMENTWISE MATH (23 kernels)
    # ======================================================================
    print("="*80)
    print("CATEGORY 1: ELEMENTWISE MATH (23 kernels)")
    print("="*80)
    print(f"{'Operation':<25} {'Metal (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    speedups.append(run_benchmark("add", 
        lambda: gpu.add(gpu_a, gpu_b),
        lambda: np_a + np_b))
    
    speedups.append(run_benchmark("sub",
        lambda: gpu.sub(gpu_a, gpu_b),
        lambda: np_a - np_b))
    
    speedups.append(run_benchmark("multiply",
        lambda: gpu.multiply(gpu_a, gpu_b),
        lambda: np_a * np_b))
    
    speedups.append(run_benchmark("divide",
        lambda: gpu.divide(gpu_a, gpu_b),
        lambda: np_a / np_b))
    
    speedups.append(run_benchmark("negate",
        lambda: gpu.negate(gpu_a),
        lambda: -np_a))
    
    speedups.append(run_benchmark("abs",
        lambda: gpu.abs(gpu_c),
        lambda: np.abs(np_c)))
    
    speedups.append(run_benchmark("pow",
        lambda: gpu.pow(gpu_a, 2.5),
        lambda: np.power(np_a, 2.5)))
    
    speedups.append(run_benchmark("square",
        lambda: gpu.square(gpu_a),
        lambda: np_a * np_a))
    
    speedups.append(run_benchmark("sqrt",
        lambda: gpu.sqrt(gpu_a),
        lambda: np.sqrt(np_a)))
    
    speedups.append(run_benchmark("exp",
        lambda: gpu.exp(gpu_a),
        lambda: np.exp(np_a)))
    
    speedups.append(run_benchmark("log",
        lambda: gpu.log(gpu_a),
        lambda: np.log(np_a)))
    
    speedups.append(run_benchmark("sin",
        lambda: gpu.sin(gpu_a),
        lambda: np.sin(np_a)))
    
    speedups.append(run_benchmark("cos",
        lambda: gpu.cos(gpu_a),
        lambda: np.cos(np_a)))
    
    speedups.append(run_benchmark("tan",
        lambda: gpu.tan(gpu_a),
        lambda: np.tan(np_a)))
    
    # Create data in valid range for asin
    np_asin = np.random.rand(SIZE).astype(np.float32) * 0.9  # Range [0, 0.9]
    gpu_asin = gpu.array(np_asin.tolist())
    speedups.append(run_benchmark("asin",
        lambda: gpu.asin(gpu_asin),
        lambda: np.arcsin(np_asin)))
    
    speedups.append(run_benchmark("atan2",
        lambda: gpu.atan2(gpu_a, gpu_b),
        lambda: np.arctan2(np_a, np_b)))
    
    speedups.append(run_benchmark("floor",
        lambda: gpu.floor(gpu_c),
        lambda: np.floor(np_c)))
    
    speedups.append(run_benchmark("ceil",
        lambda: gpu.ceil(gpu_c),
        lambda: np.ceil(np_c)))
    
    speedups.append(run_benchmark("sign",
        lambda: gpu.sign(gpu_c),
        lambda: np.sign(np_c)))
    
    speedups.append(run_benchmark("clip",
        lambda: gpu.clip(gpu_c, -1.0, 1.0),
        lambda: np.clip(np_c, -1.0, 1.0)))
    
    speedups.append(run_benchmark("round",
        lambda: gpu.round(gpu_c),
        lambda: np.round(np_c)))
    
    # Broadcast operations - FIXED
    SMALL_SIZE = 1000
    np_small = np.random.rand(SMALL_SIZE).astype(np.float32)
    # Create large array that's exactly divisible by small array size
    LARGE_SIZE = SMALL_SIZE * 5000  # 5,000,000
    np_large = np.random.rand(LARGE_SIZE).astype(np.float32)
    gpu_small = gpu.array(np_small.tolist())
    gpu_large = gpu.array(np_large.tolist())
    
    # NumPy broadcast: tile the small array to match large array size
    np_small_tiled = np.tile(np_small, LARGE_SIZE // SMALL_SIZE)
    
    speedups.append(run_benchmark("broadcast_add",
        lambda: gpu.broadcast_add(gpu_large, gpu_small),
        lambda: np_large + np_small_tiled))
    
    speedups.append(run_benchmark("broadcast_multiply",
        lambda: gpu.broadcast_multiply(gpu_large, gpu_small),
        lambda: np_large * np_small_tiled))
    
    # ======================================================================
    # CATEGORY 2: ACTIVATIONS (7 kernels)
    # ======================================================================
    print("\n" + "="*80)
    print("CATEGORY 2: ACTIVATIONS (7 kernels)")
    print("="*80)
    print(f"{'Operation':<25} {'Metal (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    speedups.append(run_benchmark("relu",
        lambda: gpu.relu(gpu_c),
        lambda: np.maximum(np_c, 0)))
    
    speedups.append(run_benchmark("leaky_relu",
        lambda: gpu.leaky_relu(gpu_c, 0.1),
        lambda: np.where(np_c > 0, np_c, 0.1 * np_c)))
    
    speedups.append(run_benchmark("sigmoid",
        lambda: gpu.sigmoid(gpu_c),
        lambda: 1 / (1 + np.exp(-np_c))))
    
    speedups.append(run_benchmark("tanh",
        lambda: gpu.tanh(gpu_c),
        lambda: np.tanh(np_c)))
    
    speedups.append(run_benchmark("softplus",
        lambda: gpu.softplus(gpu_c),
        lambda: np.log(1 + np.exp(np_c))))
    
    speedups.append(run_benchmark("swish",
        lambda: gpu.swish(gpu_c),
        lambda: np_c / (1 + np.exp(-np_c))))
    
    # GELU - no simple numpy equivalent, just test Metal
    print(f"{'gelu':<25} ", end='')
    metal_times = []
    for _ in range(10):
        start = time.perf_counter()
        gpu.gelu(gpu_c)
        metal_times.append(time.perf_counter() - start)
    metal_time = min(metal_times) * 1000
    print(f"{metal_time:<15.4f} {'N/A':<15} {'N/A':<10}")
    
    # ======================================================================
    # CATEGORY 3: REDUCTIONS (7 kernels)
    # ======================================================================
    print("\n" + "="*80)
    print("CATEGORY 3: REDUCTIONS (7 kernels)")
    print("="*80)
    print(f"{'Operation':<25} {'Metal (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    speedups.append(run_benchmark("sum",
        lambda: gpu.sum(gpu_a),
        lambda: np.sum(np_a)))
    
    speedups.append(run_benchmark("product",
        lambda: gpu.product(gpu_a),
        lambda: np.prod(np_a)))
    
    speedups.append(run_benchmark("max",
        lambda: gpu.max(gpu_a),
        lambda: np.max(np_a)))
    
    speedups.append(run_benchmark("min",
        lambda: gpu.min(gpu_a),
        lambda: np.min(np_a)))
    
    speedups.append(run_benchmark("argmax",
        lambda: gpu.argmax(gpu_a),
        lambda: np.argmax(np_a)))
    
    speedups.append(run_benchmark("argmin",
        lambda: gpu.argmin(gpu_a),
        lambda: np.argmin(np_a)))
    
    speedups.append(run_benchmark("mean",
        lambda: gpu.mean(gpu_a),
        lambda: np.mean(np_a)))
    
    # ======================================================================
    # CATEGORY 4: LINEAR ALGEBRA (7 kernels)
    # ======================================================================
    print("\n" + "="*80)
    print("CATEGORY 4: LINEAR ALGEBRA (7 kernels)")
    print("="*80)
    print(f"{'Operation':<25} {'Metal (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    # Matrix operations - use 2000x2000 for better GPU utilization
    MAT_SIZE = 2000
    np_mat_a = np.random.rand(MAT_SIZE, MAT_SIZE).astype(np.float32)
    np_mat_b = np.random.rand(MAT_SIZE, MAT_SIZE).astype(np.float32)
    gpu_mat_a = gpu.array(np_mat_a.flatten().tolist())
    gpu_mat_b = gpu.array(np_mat_b.flatten().tolist())
    
    speedups.append(run_benchmark("hadamard",
        lambda: gpu.hadamard(gpu_mat_a, gpu_mat_b, MAT_SIZE, MAT_SIZE),
        lambda: np_mat_a * np_mat_b))
    
    speedups.append(run_benchmark("matmul",
        lambda: gpu.matmul(gpu_mat_a, gpu_mat_b, MAT_SIZE, MAT_SIZE, MAT_SIZE),
        lambda: np.matmul(np_mat_a, np_mat_b), runs=5))
    
    speedups.append(run_benchmark("transpose",
        lambda: gpu.transpose(gpu_mat_a, MAT_SIZE, MAT_SIZE),
        lambda: np_mat_a.T))
    
    speedups.append(run_benchmark("row_sum",
        lambda: gpu.row_sum(gpu_mat_a, MAT_SIZE, MAT_SIZE),
        lambda: np.sum(np_mat_a, axis=1)))
    
    speedups.append(run_benchmark("col_sum",
        lambda: gpu.col_sum(gpu_mat_a, MAT_SIZE, MAT_SIZE),
        lambda: np.sum(np_mat_a, axis=0)))
    
    np_scale = np.random.rand(MAT_SIZE).astype(np.float32)
    gpu_scale = gpu.array(np_scale.tolist())
    
    speedups.append(run_benchmark("row_scale",
        lambda: gpu.row_scale(gpu_mat_a, gpu_scale, MAT_SIZE, MAT_SIZE),
        lambda: np_mat_a * np_scale[:, np.newaxis]))
    
    speedups.append(run_benchmark("col_scale",
        lambda: gpu.col_scale(gpu_mat_a, gpu_scale, MAT_SIZE, MAT_SIZE),
        lambda: np_mat_a * np_scale))
    
    # ======================================================================
    # CATEGORY 5: MISCELLANEOUS (2 kernels)
    # ======================================================================
    print("\n" + "="*80)
    print("CATEGORY 5: MISCELLANEOUS (2 kernels)")
    print("="*80)
    print(f"{'Operation':<25} {'Metal (ms)':<15} {'NumPy (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    speedups.append(run_benchmark("slice",
        lambda: gpu.slice(gpu_a, 1000000, 4000000),
        lambda: np_a[1000000:4000000]))
    
    # Bitonic sort - use smaller array (must be power of 2)
    SORT_SIZE = 65536  # 2^16
    np_sort = np.random.rand(SORT_SIZE).astype(np.float32)
    gpu_sort = gpu.array(np_sort.tolist())
    
    speedups.append(run_benchmark("bitonic_sort",
        lambda: gpu.bitonic_sort(gpu_sort),
        lambda: np.sort(np_sort), runs=3))
    
    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    
    avg_speedup = sum(speedups) / len(speedups)
    gpu_wins = sum(1 for s in speedups if s > 1.0)
    gpu_major_wins = sum(1 for s in speedups if s > 2.0)
    
    print(f"\n   Total Operations Tested: {len(speedups)}")
    print(f"   Average Speedup: {avg_speedup:.2f}x")
    print(f"   Best Speedup: {max(speedups):.2f}x")
    print(f"   Worst Speedup: {min(speedups):.2f}x")
    print(f"   GPU Wins (>1x): {gpu_wins}/{len(speedups)} ({100*gpu_wins/len(speedups):.1f}%)")
    print(f"   GPU Major Wins (>2x): {gpu_major_wins}/{len(speedups)} ({100*gpu_major_wins/len(speedups):.1f}%)")
    
    print(f"\n   🏆 Top 10 GPU Speedups:")
    speedup_with_idx = [(s, i) for i, s in enumerate(speedups)]
    speedup_with_idx.sort(reverse=True)
    for i, (speedup, _) in enumerate(speedup_with_idx[:10], 1):
        print(f"      {i}. {speedup:.2f}x")
    
    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    benchmark()