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
    
    
      # ======================================================
    # GENERIC REDUCTION LAUNCHER (float -> scalar)
    # ======================================================
    def _reduce_scalar(self, init_kernel_name, final_kernel_name, a):
        """Run a 2-pass reduction using (init, final) kernels and return a Python float."""

        if not isinstance(a, MetalArray):
            raise TypeError("Reduction expects a MetalArray input.")

        length = a.length
        TPT = self._TPT          # threads per threadgroup (must match kernels)
        num_groups = self._num_groups  # number of threadgroups (must match partial buffer size)

        # ------------------------------
        # 1st pass: parallel block reduction
        # ------------------------------
        # Load / cache initial reduction kernel PSO
        if init_kernel_name not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(init_kernel_name)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[init_kernel_name] = pso
        init_pso = self.kernel_cache[init_kernel_name]

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(init_pso)

        # buffer(0): input array
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)

        # buffer(1): partial output buffer (size = num_groups floats)
        enc.setBuffer_offset_atIndex_(self._partial_buffer, 0, 1)

        # buffer(2): N
        N_val = (ctypes.c_uint * 1)(length)
        enc.setBytes_length_atIndex_(N_val, ctypes.sizeof(ctypes.c_uint), 2)

        # buffer(3): TPT
        TPT_val = (ctypes.c_uint * 1)(TPT)
        enc.setBytes_length_atIndex_(TPT_val, ctypes.sizeof(ctypes.c_uint), 3)

        # Dispatch: num_groups threadgroups, each with TPT threads
        tg_size = Metal.MTLSizeMake(TPT, 1, 1)
        num_tg = Metal.MTLSizeMake(num_groups, 1, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(num_tg, tg_size)

        enc.endEncoding()

        # ------------------------------
        # 2nd pass: final reduction on partials
        # ------------------------------
        if final_kernel_name not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(final_kernel_name)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[final_kernel_name] = pso
        final_pso = self.kernel_cache[final_kernel_name]

        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(final_pso)

        # buffer(0): partials
        enc.setBuffer_offset_atIndex_(self._partial_buffer, 0, 0)

        # buffer(1): final result (single float)
        enc.setBuffer_offset_atIndex_(self._result_buffer, 0, 1)

        # buffer(2): N for final kernel (number of partials)
        N_final_val = (ctypes.c_uint * 1)(num_groups)
        enc.setBytes_length_atIndex_(N_final_val, ctypes.sizeof(ctypes.c_uint), 2)

        # We only need tid == 0 to run, but kernel expects a grid of threads.
        # Launch one threadgroup with TPT threads; only tid=0 does work.
        threads = Metal.MTLSizeMake(TPT, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg_size)

        enc.endEncoding()

        # Submit and wait
        cmd.commit()
        cmd.waitUntilCompleted()
        
        # Read scalar result back from _result_buffer
        ptr = self._result_buffer.contents()
        raw = ptr.as_buffer(4)  # 4 bytes = 1 float32
        out = array.array('f')
        out.frombytes(raw)

        return float(out[0])
    
    
    def test_elementwise_math():
    print("\n=== Testing Category 1: Elementwise Math ===")
    gpu = MetalPy()
    
    # Basic operations
    a = gpu.array([1.0, 2.0, 3.0, 4.0])
    b = gpu.array([5.0, 6.0, 7.0, 8.0])
    
    print("a:", a.to_list())
    print("b:", b.to_list())
    
    # Test add
    result = gpu.add(a, b)
    print("add(a, b):", result.to_list())
    assert result.to_list() == [6.0, 8.0, 10.0, 12.0], "Add failed"
    
    # Test sub
    result = gpu.sub(a, b)
    print("sub(a, b):", result.to_list())
    assert result.to_list() == [-4.0, -4.0, -4.0, -4.0], "Sub failed"
    
    # Test multiply
    result = gpu.multiply(a, b)
    print("multiply(a, b):", result.to_list())
    assert result.to_list() == [5.0, 12.0, 21.0, 32.0], "Multiply failed"
    
    # Test divide
    result = gpu.divide(a, b)
    print("divide(a, b):", result.to_list())
    expected = [1.0/5.0, 2.0/6.0, 3.0/7.0, 4.0/8.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Divide failed"
    
    # Test negate
    result = gpu.negate(a)
    print("negate(a):", result.to_list())
    assert result.to_list() == [-1.0, -2.0, -3.0, -4.0], "Negate failed"
    
    # Test abs
    c = gpu.array([-1.0, -2.0, 3.0, -4.0])
    result = gpu.abs(c)
    print("abs([-1, -2, 3, -4]):", result.to_list())
    assert result.to_list() == [1.0, 2.0, 3.0, 4.0], "Abs failed"
    
    # Test pow
    result = gpu.pow(a, 2.0)
    print("pow(a, 2):", result.to_list())
    assert result.to_list() == [1.0, 4.0, 9.0, 16.0], "Pow failed"
    
    # Test square
    result = gpu.square(a)
    print("square(a):", result.to_list())
    assert result.to_list() == [1.0, 4.0, 9.0, 16.0], "Square failed"
    
    # Test sqrt
    result = gpu.sqrt(a)
    print("sqrt(a):", result.to_list())
    expected = [1.0, math.sqrt(2), math.sqrt(3), 2.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Sqrt failed"
    
    # Test exp
    result = gpu.exp(gpu.array([0.0, 1.0, 2.0]))
    print("exp([0, 1, 2]):", result.to_list())
    expected = [1.0, math.e, math.e**2]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result.to_list(), expected)), "Exp failed"
    
    # Test log
    result = gpu.log(gpu.array([1.0, math.e, math.e**2]))
    print("log([1, e, e^2]):", result.to_list())
    expected = [0.0, 1.0, 2.0]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result.to_list(), expected)), "Log failed"
    
    # Test sin
    result = gpu.sin(gpu.array([0.0, math.pi/2, math.pi]))
    print("sin([0, π/2, π]):", result.to_list())
    expected = [0.0, 1.0, 0.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Sin failed"
    
    # Test cos
    result = gpu.cos(gpu.array([0.0, math.pi/2, math.pi]))
    print("cos([0, π/2, π]):", result.to_list())
    expected = [1.0, 0.0, -1.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Cos failed"
    
    # Test tan
    result = gpu.tan(gpu.array([0.0, math.pi/4]))
    print("tan([0, π/4]):", result.to_list())
    expected = [0.0, 1.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Tan failed"
    
    # Test asin
    result = gpu.asin(gpu.array([0.0, 0.5, 1.0]))
    print("asin([0, 0.5, 1]):", result.to_list())
    expected = [0.0, math.asin(0.5), math.pi/2]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Asin failed"
    
    # Test atan2
    y = gpu.array([1.0, 1.0, -1.0])
    x = gpu.array([1.0, 0.0, 0.0])
    result = gpu.atan2(y, x)
    print("atan2(y, x):", result.to_list())
    expected = [math.pi/4, math.pi/2, -math.pi/2]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Atan2 failed"
    
    # Test floor
    result = gpu.floor(gpu.array([1.2, 2.7, -1.5]))
    print("floor([1.2, 2.7, -1.5]):", result.to_list())
    assert result.to_list() == [1.0, 2.0, -2.0], "Floor failed"
    
    # Test ceil
    result = gpu.ceil(gpu.array([1.2, 2.7, -1.5]))
    print("ceil([1.2, 2.7, -1.5]):", result.to_list())
    assert result.to_list() == [2.0, 3.0, -1.0], "Ceil failed"
    
    # Test sign
    result = gpu.sign(gpu.array([-5.0, 0.0, 3.0]))
    print("sign([-5, 0, 3]):", result.to_list())
    assert result.to_list() == [-1.0, 0.0, 1.0], "Sign failed"
    
    # Test clip
    result = gpu.clip(gpu.array([1.0, 5.0, 10.0, 15.0]), 3.0, 12.0)
    print("clip([1, 5, 10, 15], 3, 12):", result.to_list())
    assert result.to_list() == [3.0, 5.0, 10.0, 12.0], "Clip failed"
    
    # Test round
    result = gpu.round(gpu.array([1.4, 1.5, 2.5, 3.7]))
    print("round([1.4, 1.5, 2.5, 3.7]):", result.to_list())
    # Note: round uses banker's rounding (round half to even)
    
    # Test broadcast_add
    a_broad = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # length 6
    b_broad = gpu.array([10.0, 20.0])  # length 2
    result = gpu.broadcast_add(a_broad, b_broad)
    print("broadcast_add([1,2,3,4,5,6], [10,20]):", result.to_list())
    assert result.to_list() == [11.0, 22.0, 13.0, 24.0, 15.0, 26.0], "Broadcast add failed"
    
    # Test broadcast_multiply
    result = gpu.broadcast_multiply(a_broad, b_broad)
    print("broadcast_multiply([1,2,3,4,5,6], [10,20]):", result.to_list())
    assert result.to_list() == [10.0, 40.0, 30.0, 80.0, 50.0, 120.0], "Broadcast multiply failed"
    
    print("✅ All elementwise math tests passed!")


def test_activations():
    print("\n=== Testing Category 2: Activations ===")
    gpu = MetalPy()
    
    # Test ReLU
    a = gpu.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = gpu.relu(a)
    print("relu([-2, -1, 0, 1, 2]):", result.to_list())
    assert result.to_list() == [0.0, 0.0, 0.0, 1.0, 2.0], "ReLU failed"
    
    # Test Leaky ReLU - USE TOLERANCE
    result = gpu.leaky_relu(a, alpha=0.1)
    print("leaky_relu([-2, -1, 0, 1, 2], α=0.1):", result.to_list())
    expected = [-0.2, -0.1, 0.0, 1.0, 2.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Leaky ReLU failed"
    
    # Test Sigmoid
    result = gpu.sigmoid(gpu.array([0.0, 1.0, -1.0]))
    print("sigmoid([0, 1, -1]):", result.to_list())
    expected = [0.5, 1/(1+math.e**-1), 1/(1+math.e**1)]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Sigmoid failed"
    
    # Test Tanh
    result = gpu.tanh(gpu.array([0.0, 1.0, -1.0]))
    print("tanh([0, 1, -1]):", result.to_list())
    expected = [0.0, math.tanh(1.0), math.tanh(-1.0)]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Tanh failed"
    
    # Test Softplus
    result = gpu.softplus(gpu.array([0.0, 1.0, 2.0]))
    print("softplus([0, 1, 2]):", result.to_list())
    expected = [math.log(2), math.log(1 + math.e), math.log(1 + math.e**2)]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result.to_list(), expected)), "Softplus failed"
    
    # Test Swish
    result = gpu.swish(gpu.array([0.0, 1.0, 2.0]))
    print("swish([0, 1, 2]):", result.to_list())
    # swish(x) = x * sigmoid(x)
    
    # Test GELU
    result = gpu.gelu(gpu.array([0.0, 1.0, -1.0]))
    print("gelu([0, 1, -1]):", result.to_list())
    # GELU is complex, just verify it runs
    
    print("✅ All activation tests passed!")

def test_reductions():
    print("\n=== Testing Category 3: Reductions ===")
    gpu = MetalPy()
    
    # Test sum
    a = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gpu.sum(a)
    print(f"sum([1,2,3,4,5]): {result}")
    assert abs(result - 15.0) < 1e-5, "Sum failed"
    
    # Test sum on large array
    large = gpu.array([1.0] * 10000)
    result = gpu.sum(large)
    print(f"sum([1.0] * 10000): {result}")
    assert abs(result - 10000.0) < 1e-3, "Large sum failed"
    
    # Test product
    a = gpu.array([2.0, 3.0, 4.0])
    result = gpu.product(a)
    print(f"product([2,3,4]): {result}")
    assert abs(result - 24.0) < 1e-5, "Product failed"
    
    # Test max
    a = gpu.array([3.0, 7.0, 2.0, 9.0, 1.0])
    result = gpu.max(a)
    print(f"max([3,7,2,9,1]): {result}")
    assert abs(result - 9.0) < 1e-5, "Max failed"
    
    # Test min
    result = gpu.min(a)
    print(f"min([3,7,2,9,1]): {result}")
    assert abs(result - 1.0) < 1e-5, "Min failed"
    
    # Test argmax
    result = gpu.argmax(a)
    print(f"argmax([3,7,2,9,1]): {result}")
    assert result == 3, "Argmax failed"
    
    # Test argmin
    result = gpu.argmin(a)
    print(f"argmin([3,7,2,9,1]): {result}")
    assert result == 4, "Argmin failed"
    
    # Test mean
    a = gpu.array([2.0, 4.0, 6.0, 8.0])
    result = gpu.mean(a)
    print(f"mean([2,4,6,8]): {result}")
    assert abs(result - 5.0) < 1e-5, "Mean failed"
    
    print("✅ All reduction tests passed!")


def test_linear_algebra():
    print("\n=== Testing Category 4: Linear Algebra ===")
    gpu = MetalPy()
    
    # Test Hadamard (element-wise matrix multiply)
    A = gpu.array([1.0, 2.0, 3.0, 4.0])  # 2x2 matrix
    B = gpu.array([5.0, 6.0, 7.0, 8.0])  # 2x2 matrix
    result = gpu.hadamard(A, B, rows=2, cols=2)
    print(f"hadamard([[1,2],[3,4]], [[5,6],[7,8]]): {result.to_list()}")
    assert result.to_list() == [5.0, 12.0, 21.0, 32.0], "Hadamard failed"
    
    # Test Matrix multiplication (2x3 @ 3x2 = 2x2)
    # A = [[1, 2, 3],
    #      [4, 5, 6]]  (2x3)
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # B = [[7,  8],
    #      [9, 10],
    #      [11, 12]]  (3x2)
    B = gpu.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    
    result = gpu.matmul(A, B, M=2, N=2, K=3)
    print(f"matmul(2x3, 3x2): {result.to_list()}")
    # Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12],
    #            [4*7+5*9+6*11, 4*8+5*10+6*12]]
    #         = [[58, 64], [139, 154]]
    expected = [58.0, 64.0, 139.0, 154.0]
    assert all(abs(r - e) < 1e-4 for r, e in zip(result.to_list(), expected)), "Matmul failed"
    
    # Test Transpose
    # A = [[1, 2, 3],
    #      [4, 5, 6]]  (2x3)
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = gpu.transpose(A, rows=2, cols=3)
    print(f"transpose(2x3): {result.to_list()}")
    # Expected: [[1, 4],
    #            [2, 5],
    #            [3, 6]]  (3x2, row-major)
    expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    assert result.to_list() == expected, "Transpose failed"
    
    # Test Row sum
    # A = [[1, 2, 3],
    #      [4, 5, 6]]
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = gpu.row_sum(A, rows=2, cols=3)
    print(f"row_sum([[1,2,3],[4,5,6]]): {result.to_list()}")
    assert result.to_list() == [6.0, 15.0], "Row sum failed"
    
    # Test Column sum
    result = gpu.col_sum(A, rows=2, cols=3)
    print(f"col_sum([[1,2,3],[4,5,6]]): {result.to_list()}")
    assert result.to_list() == [5.0, 7.0, 9.0], "Col sum failed"
    
    # Test Row scale
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2x3
    s = gpu.array([2.0, 3.0])  # scale factors for 2 rows
    result = gpu.row_scale(A, s, rows=2, cols=3)
    print(f"row_scale([[1,2,3],[4,5,6]], [2,3]): {result.to_list()}")
    expected = [2.0, 4.0, 6.0, 12.0, 15.0, 18.0]
    assert result.to_list() == expected, "Row scale failed"
    
    # Test Column scale
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2x3
    s = gpu.array([10.0, 100.0, 1000.0])  # scale factors for 3 cols
    result = gpu.col_scale(A, s, rows=2, cols=3)
    print(f"col_scale([[1,2,3],[4,5,6]], [10,100,1000]): {result.to_list()}")
    expected = [10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]
    assert result.to_list() == expected, "Col scale failed"
    
    print("✅ All linear algebra tests passed!")


def test_miscellaneous():
    print("\n=== Testing Category 5: Miscellaneous ===")
    gpu = MetalPy()
    
    # Test slice
    a = gpu.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result = gpu.slice(a, start=1, end=4)
    print(f"slice([10,20,30,40,50,60], 1:4): {result.to_list()}")
    assert result.to_list() == [20.0, 30.0, 40.0], "Slice failed"
    
    result = gpu.slice(a, start=0, end=2)
    print(f"slice([10,20,30,40,50,60], 0:2): {result.to_list()}")
    assert result.to_list() == [10.0, 20.0], "Slice failed"
    
    # Test bitonic sort
    # Must be power of 2
    a = gpu.array([64.0, 2.0, 25.0, 12.0, 22.0, 11.0, 90.0, 8.0])
    print(f"Before sort: {a.to_list()}")
    result = gpu.bitonic_sort(a)
    print(f"After bitonic_sort: {result.to_list()}")
    sorted_vals = result.to_list()
    assert sorted_vals == sorted(sorted_vals), "Bitonic sort failed"
    
    # Test with powers of 2 array
    a = gpu.array([5.0, 1.0, 9.0, 3.0])
    result = gpu.bitonic_sort(a)
    print(f"bitonic_sort([5,1,9,3]): {result.to_list()}")
    assert result.to_list() == [1.0, 3.0, 5.0, 9.0], "Bitonic sort failed"
    
    print("✅ All miscellaneous tests passed!")


def test_edge_cases():
    print("\n=== Testing Edge Cases ===")
    gpu = MetalPy()
    
    # Single element
    a = gpu.array([42.0])
    result = gpu.sum(a)
    print(f"sum([42]): {result}")
    assert abs(result - 42.0) < 1e-5, "Single element sum failed"
    
    # Large array reduction
    n = 1000000
    large = gpu.array([1.0] * n)
    result = gpu.sum(large)
    print(f"sum([1.0] * {n}): {result}")
    assert abs(result - float(n)) < 100, "Large array sum failed"
    
    # Negative numbers
    a = gpu.array([-5.0, -10.0, -15.0])
    result = gpu.sum(a)
    print(f"sum([-5,-10,-15]): {result}")
    assert abs(result - (-30.0)) < 1e-5, "Negative sum failed"
    
    print("✅ All edge case tests passed!")


def main():
    print("🚀 Starting MetalPy Test Suite\n")
    
    try:
        test_elementwise_math()
        test_activations()
        test_reductions()
        test_linear_algebra()
        test_miscellaneous()
        test_edge_cases()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()