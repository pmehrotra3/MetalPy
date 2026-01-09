import ctypes
import numpy as np
import time

# ========================
# Load MetalPy library
# ========================
lib = ctypes.CDLL("./metalpy.so")

# ------------------------
# Function signatures
# ------------------------

# void metal_init(const char*)
lib.metal_init.argtypes = [ctypes.c_char_p]
lib.metal_init.restype  = None

# void* metal_create_array(float* data, uint32_t length)
lib.metal_create_array.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32
]
lib.metal_create_array.restype = ctypes.c_void_p

# void metal_to_cpu(void* gpu, float* out)
lib.metal_to_cpu.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float)
]
lib.metal_to_cpu.restype = None

# void metal_add(void* out, void* a, void* b)
lib.metal_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.metal_add.restype  = None

# float metal_sum(void* a)
lib.metal_sum.argtypes = [ctypes.c_void_p]
lib.metal_sum.restype  = ctypes.c_float


# ========================
# INIT METAL
# ========================
lib.metal_init(b"./metal_kernels.metallib")


# ========================
# Benchmark function
# ========================
def benchmark(name, func, *args, warmup=5, iterations=100):
    """Run warmup iterations, then time the function."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    
    total_ms = (end - start) * 1000
    avg_us = (total_ms / iterations) * 1000
    
    print(f"{name}: {avg_us:.2f} µs avg ({iterations} iterations)")
    return avg_us


# ========================
# Test different sizes
# ========================
sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

for N in sizes:
    print(f"\n{'='*50}")
    print(f"Array size: {N:,} elements ({N * 4 / 1024 / 1024:.2f} MB)")
    print('='*50)
    
    # Create test data
    X = np.random.randn(N).astype(np.float32)
    ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # ========================
    # Time: CPU → GPU transfer
    # ========================
    start = time.perf_counter()
    gpu_arr = lib.metal_create_array(ptr, N)
    transfer_time = (time.perf_counter() - start) * 1000
    print(f"CPU → GPU transfer: {transfer_time:.3f} ms")
    
    # Create output buffer
    out = np.zeros_like(X)
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_arr = lib.metal_create_array(out_ptr, N)
    
    # ========================
    # Benchmark: metal_add
    # ========================
    benchmark("Metal add", lib.metal_add, out_arr, gpu_arr, gpu_arr)
    
    # ========================
    # Benchmark: metal_sum
    # ========================
    benchmark("Metal sum", lib.metal_sum, gpu_arr)
    
    # ========================
    # Benchmark: NumPy (for comparison)
    # ========================
    benchmark("NumPy add", np.add, X, X)
    benchmark("NumPy sum", np.sum, X)
    
    # ========================
    # Time: GPU → CPU transfer
    # ========================
    result = np.zeros_like(X)
    start = time.perf_counter()
    lib.metal_to_cpu(out_arr, result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    transfer_back = (time.perf_counter() - start) * 1000
    print(f"GPU → CPU transfer: {transfer_back:.3f} ms")
    
    # Verify correctness
    lib.metal_add(out_arr, gpu_arr, gpu_arr)
    lib.metal_to_cpu(out_arr, result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    expected = X + X
    if np.allclose(result, expected):
        print("✅ Results verified correct")
    else:
        print("❌ Results mismatch!")


# ========================
# Throughput calculation
# ========================
print(f"\n{'='*50}")
print("Throughput Analysis (largest size)")
print('='*50)

N = 10_000_000
X = np.random.randn(N).astype(np.float32)
ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
gpu_arr = lib.metal_create_array(ptr, N)

out = np.zeros_like(X)
out_arr = lib.metal_create_array(out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N)

# Warmup
for _ in range(10):
    lib.metal_add(out_arr, gpu_arr, gpu_arr)

# Time many iterations
iterations = 1000
start = time.perf_counter()
for _ in range(iterations):
    lib.metal_add(out_arr, gpu_arr, gpu_arr)
end = time.perf_counter()

total_time = end - start
ops_per_sec = (N * iterations) / total_time
gb_per_sec = (N * 4 * 3 * iterations) / total_time / 1e9  # 2 reads + 1 write

print(f"Elements processed: {N * iterations:,}")
print(f"Total time: {total_time:.3f} s")
print(f"Throughput: {ops_per_sec/1e9:.2f} billion elements/sec")
print(f"Memory bandwidth: {gb_per_sec:.2f} GB/s")
