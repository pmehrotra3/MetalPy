import Metal
import ctypes
import array
from pathlib import Path


# ==========================================================
# GPU Array Wrapper
# ==========================================================
class MetalArray:
    """GPU array wrapper around a Metal buffer."""

    def __init__(self, gpu, data, length=None, buffer=None):
        self.gpu = gpu

        # If kernel already created buffer
        if buffer is not None:
            self.buffer = buffer
            self.length = length
            return

        if isinstance(data, list):
            data = array.array('f', data)
        elif isinstance(data, array.array):
            pass
        else:
            raise TypeError("MetalArray expects list or array('f').")

        self.length = len(data)

        self.buffer = gpu.device.newBufferWithBytes_length_options_(
            data.tobytes(), self.length * 4,
            Metal.MTLResourceStorageModeShared
        )

    def to_list(self):
        ptr = self.buffer.contents()
        arr = (ctypes.c_float * self.length).from_buffer(ptr.as_buffer(self.length * 4))
        return list(arr)


# ==========================================================
# Main Metal Compute Class
# ==========================================================
class MetalPy:

    def __init__(self, kernel_path="metal_kernels.metallib"):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("❌ No Metal GPU detected.")

        self.queue = self.device.newCommandQueue()
        print(f"✅ Using GPU: {self.device.name()}")

        # Load compiled library instead of source
        from pathlib import Path
        lib_path = str(Path(kernel_path).absolute())
        library, err = self.device.newLibraryWithFile_error_(lib_path, None)
        if err:
            raise RuntimeError(f"❌ Failed to load library: {err}")

        self.library = library
        self.kernel_cache = {}
        
        # Pre-allocate reduction buffers (permanent GPU memory)
        TPT = 256
        num_groups = 256
        
        self._partial_buffer = self.device.newBufferWithLength_options_(
            num_groups * 4, Metal.MTLResourceStorageModeShared)
        self._result_buffer = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)
        
        # For argmax/argmin
        self._partial_idx = self.device.newBufferWithLength_options_(
            num_groups * 4, Metal.MTLResourceStorageModeShared)
        self._partial_val = self.device.newBufferWithLength_options_(
            num_groups * 4, Metal.MTLResourceStorageModeShared)
        self._result_idx = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared)

    def array(self, data):
        return MetalArray(self, data)

    # ======================================================
    # ELEMENTWISE OPS
    # ======================================================
    def _launch(self, fname, arrays):
        """Generic launcher for simple elementwise operations."""
        
        # Load PSO
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso

        pso = self.kernel_cache[fname]

        length = arrays[0].length
        for a in arrays:
            if a.length != length:
                raise ValueError("All MetalArrays must have equal length")

        # Output
        out_buf = self.device.newBufferWithLength_options_(
            length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=length, buffer=out_buf)

        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Input buffers
        for i, arr in enumerate(arrays):
            enc.setBuffer_offset_atIndex_(arr.buffer, 0, i)

        # Output buffer
        enc.setBuffer_offset_atIndex_(out_buf, 0, len(arrays))

        # Pass N → next index
        N_c = (ctypes.c_uint * 1)(length)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), len(arrays) + 1)

        # Dispatch
        threads = Metal.MTLSizeMake(length, 1, 1)
        tg = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return out_arr

    # ======================================================
    # OPTIMIZED 2-PASS REDUCTION (for sum/product/max/min)
    # ======================================================
    def _launch_reduce_fast(self, fname, arr):

        # Load PSO
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            if not fn:
                raise RuntimeError(f"Kernel {fname} not found!")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso

        pso = self.kernel_cache[fname]

        N = arr.length
        TPT = 256
        num_groups = 256
        
        # Use pre-allocated buffers (no allocation overhead!)
        partial_buffer = self._partial_buffer
        result_buffer = self._result_buffer
        
        # Single command buffer for both passes
        cmd = self.queue.commandBuffer()
        
        # Pass 1: N -> 256
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(arr.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(partial_buffer, 0, 1)
        
        N_c = (ctypes.c_uint * 1)(N)
        TPT_c = (ctypes.c_uint * 1)(TPT)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 2)
        enc.setBytes_length_atIndex_(TPT_c, ctypes.sizeof(ctypes.c_uint), 3)
        
        enc.setThreadgroupMemoryLength_atIndex_(TPT * 4, 0)
        
        grid = Metal.MTLSizeMake(num_groups, 1, 1)
        threads = Metal.MTLSizeMake(TPT, 1, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, threads)
        enc.endEncoding()
        
        # Pass 2: 256 -> 1
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(partial_buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(result_buffer, 0, 1)
        
        N2_c = (ctypes.c_uint * 1)(num_groups)
        enc.setBytes_length_atIndex_(N2_c, ctypes.sizeof(ctypes.c_uint), 2)
        enc.setBytes_length_atIndex_(TPT_c, ctypes.sizeof(ctypes.c_uint), 3)
        
        enc.setThreadgroupMemoryLength_atIndex_(TPT * 4, 0)
        
        grid = Metal.MTLSizeMake(1, 1, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, threads)
        enc.endEncoding()
        
        # Single wait at the end
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return float((ctypes.c_float * 1).from_buffer(
            result_buffer.contents().as_buffer(4)
        )[0])

    # ======================================================
    # OPTIMIZED 2-PASS ARGMAX/ARGMIN
    # ======================================================
    def _launch_reduce_arg_fast(self, fname, final_fname, arr):
        """Optimized 2-pass argmax/argmin with grid-stride loop."""
        
        # Load PSOs
        for kernel_name in [fname, final_fname]:
            if kernel_name not in self.kernel_cache:
                fn = self.library.newFunctionWithName_(kernel_name)
                if not fn:
                    raise RuntimeError(f"Kernel {kernel_name} not found!")
                pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
                if err: raise RuntimeError(err)
                self.kernel_cache[kernel_name] = pso

        pso1 = self.kernel_cache[fname]
        pso2 = self.kernel_cache[final_fname]

        N = arr.length
        TPT = 256
        num_groups = 256
        
        # Use pre-allocated buffers
        partial_idx = self._partial_idx
        partial_val = self._partial_val
        result_idx = self._result_idx
        
        # Single command buffer
        cmd = self.queue.commandBuffer()
        
        # Pass 1: N -> 256
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso1)
        
        enc.setBuffer_offset_atIndex_(arr.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(partial_idx, 0, 1)
        enc.setBuffer_offset_atIndex_(partial_val, 0, 2)
        
        N_c = (ctypes.c_uint * 1)(N)
        TPT_c = (ctypes.c_uint * 1)(TPT)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 3)
        enc.setBytes_length_atIndex_(TPT_c, ctypes.sizeof(ctypes.c_uint), 4)
        
        enc.setThreadgroupMemoryLength_atIndex_(TPT * 4, 0)
        enc.setThreadgroupMemoryLength_atIndex_(TPT * 4, 1)
        
        grid = Metal.MTLSizeMake(num_groups, 1, 1)
        threads = Metal.MTLSizeMake(TPT, 1, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, threads)
        enc.endEncoding()
        
        # Pass 2: 256 -> 1
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso2)
        
        enc.setBuffer_offset_atIndex_(partial_idx, 0, 0)
        enc.setBuffer_offset_atIndex_(partial_val, 0, 1)
        enc.setBuffer_offset_atIndex_(result_idx, 0, 2)
        
        N2_c = (ctypes.c_uint * 1)(num_groups)
        enc.setBytes_length_atIndex_(N2_c, ctypes.sizeof(ctypes.c_uint), 3)
        enc.setBytes_length_atIndex_(TPT_c, ctypes.sizeof(ctypes.c_uint), 4)
        
        enc.setThreadgroupMemoryLength_atIndex_(TPT * 4, 0)
        enc.setThreadgroupMemoryLength_atIndex_(TPT * 4, 1)
        
        grid = Metal.MTLSizeMake(1, 1, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, threads)
        enc.endEncoding()
        
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return int((ctypes.c_uint * 1).from_buffer(
            result_idx.contents().as_buffer(4)
        )[0])

    # ======================================================
    # PUBLIC API - CATEGORY 1: ELEMENTWISE MATH
    # ======================================================
    def add(self, a, b):       return self._launch("add_kernel", [a, b])
    def sub(self, a, b):       return self._launch("sub_kernel", [a, b])
    def multiply(self, a, b):  return self._launch("multiply_kernel", [a, b])
    def divide(self, a, b):    return self._launch("division_kernel", [a, b])

    def negate(self, a):       return self._launch("negate_kernel", [a])
    def abs(self, a):          return self._launch("abs_kernel", [a])
    
    def pow(self, a, c):
        """Element-wise power function: a^c"""
        fname = "pow_kernel"
        
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            a.length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=a.length, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        
        c_val = (ctypes.c_float * 1)(c)
        enc.setBytes_length_atIndex_(c_val, ctypes.sizeof(ctypes.c_float), 1)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        
        N_c = (ctypes.c_uint * 1)(a.length)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 3)
        
        threads = Metal.MTLSizeMake(a.length, 1, 1)
        tg = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr

    def square(self, a):       return self._launch("square_kernel", [a])
    def sqrt(self, a):         return self._launch("sqrt_kernel", [a])
    def exp(self, a):          return self._launch("exp_kernel", [a])
    def log(self, a):          return self._launch("log_kernel", [a])

    def sin(self, a):          return self._launch("sin_kernel", [a])
    def cos(self, a):          return self._launch("cos_kernel", [a])
    def tan(self, a):          return self._launch("tan_kernel", [a])
    def asin(self, a):         return self._launch("asin_kernel", [a])
    def atan2(self, y, x):     return self._launch("atan2_kernel", [y, x])
    
    def floor(self, a):        return self._launch("floor_kernel", [a])
    def ceil(self, a):         return self._launch("ceil_kernel", [a])
    def sign(self, a):         return self._launch("sign_kernel", [a])
    
    def clip(self, a, low, high):
        """Clip array values between low and high"""
        fname = "clip_kernel"
        
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            a.length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=a.length, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
        
        low_val = (ctypes.c_float * 1)(low)
        enc.setBytes_length_atIndex_(low_val, ctypes.sizeof(ctypes.c_float), 2)
        
        high_val = (ctypes.c_float * 1)(high)
        enc.setBytes_length_atIndex_(high_val, ctypes.sizeof(ctypes.c_float), 3)
        
        N_c = (ctypes.c_uint * 1)(a.length)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 4)
        
        threads = Metal.MTLSizeMake(a.length, 1, 1)
        tg = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    def round(self, a):        return self._launch("kernel_round", [a])
    
    def broadcast_add(self, a, b):
        """Broadcast add: a + b where b is broadcast to match a's length."""
        if a.length % b.length != 0:
            raise ValueError(f"Cannot broadcast: {a.length} % {b.length} != 0")
        
        fname = "broadcast_add_kernel"
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            a.length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=a.length, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(b.buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        
        N_c = (ctypes.c_uint * 1)(a.length)
        M_c = (ctypes.c_uint * 1)(b.length)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 3)
        enc.setBytes_length_atIndex_(M_c, ctypes.sizeof(ctypes.c_uint), 4)
        
        threads = Metal.MTLSizeMake(a.length, 1, 1)
        tg = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    def broadcast_multiply(self, a, b):
        """Broadcast multiply: a * b where b is broadcast to match a's length."""
        if a.length % b.length != 0:
            raise ValueError(f"Cannot broadcast: {a.length} % {b.length} != 0")
        
        fname = "broadcast_multiply_kernel"
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            a.length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=a.length, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(b.buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        
        N_c = (ctypes.c_uint * 1)(a.length)
        M_c = (ctypes.c_uint * 1)(b.length)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 3)
        enc.setBytes_length_atIndex_(M_c, ctypes.sizeof(ctypes.c_uint), 4)
        
        threads = Metal.MTLSizeMake(a.length, 1, 1)
        tg = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr

    # ======================================================
    # CATEGORY 2: ACTIVATIONS
    # ======================================================
    def relu(self, a):         return self._launch("relu_kernel", [a])
    
    def leaky_relu(self, a, alpha=0.01):
        """Leaky ReLU activation"""
        fname = "leaky_relu_kernel"
        
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            a.length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=a.length, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        
        alpha_val = (ctypes.c_float * 1)(alpha)
        enc.setBytes_length_atIndex_(alpha_val, ctypes.sizeof(ctypes.c_float), 1)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        
        N_c = (ctypes.c_uint * 1)(a.length)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 3)
        
        threads = Metal.MTLSizeMake(a.length, 1, 1)
        tg = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    def sigmoid(self, a):      return self._launch("sigmoid_kernel", [a])
    def tanh(self, a):         return self._launch("tanh_kernel", [a])
    def softplus(self, a):     return self._launch("softplus_kernel", [a])
    def swish(self, a):        return self._launch("swish_kernel", [a])
    def gelu(self, a):         return self._launch("gelu_kernel", [a])

    # ======================================================
    # CATEGORY 3: REDUCTIONS
    # ======================================================
    def sum(self, a):          return self._launch_reduce_fast("sum_reduce_kernel", a)
    def product(self, a):      return self._launch_reduce_fast("product_reduce_kernel", a)
    def max(self, a):          return self._launch_reduce_fast("max_reduce_kernel", a)
    def min(self, a):          return self._launch_reduce_fast("min_reduce_kernel", a)
    
    def argmax(self, a):       return self._launch_reduce_arg_fast("argmax_reduce_kernel", "argmax_reduce_final_kernel", a)
    def argmin(self, a):       return self._launch_reduce_arg_fast("argmin_reduce_kernel", "argmin_reduce_final_kernel", a)
    
    # Derived operations
    def mean(self, a):         return self.sum(a) / a.length

    # ======================================================
    # CATEGORY 4: LINEAR ALGEBRA (Matrix Operations)
    # ======================================================
    def hadamard(self, A, B, rows, cols):
        """Element-wise matrix multiplication."""
        return self._launch_matrix("hadamard_mat_kernel", [A, B], rows, cols)
    
    def matmul(self, A, B, M, N, K):
        """Tiled matrix multiplication: C = A @ B
        A: M x K
        B: K x N
        C: M x N
        """
        if A.length != M * K or B.length != K * N:
            raise ValueError(f"Shape mismatch: A={A.length} (need {M}x{K}), B={B.length} (need {K}x{N})")
        
        fname = "tiled_matmul_kernel"
        
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            M * N * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=M * N, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(A.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(B.buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        
        M_c = (ctypes.c_uint * 1)(M)
        N_c = (ctypes.c_uint * 1)(N)
        K_c = (ctypes.c_uint * 1)(K)
        enc.setBytes_length_atIndex_(M_c, ctypes.sizeof(ctypes.c_uint), 3)
        enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 4)
        enc.setBytes_length_atIndex_(K_c, ctypes.sizeof(ctypes.c_uint), 5)
        
        # Calculate threadgroups needed (FIXED)
        TILE_SIZE = 16
        grid_cols = (N + TILE_SIZE - 1) // TILE_SIZE
        grid_rows = (M + TILE_SIZE - 1) // TILE_SIZE
        
        # Dispatch threadgroups (not threads!)
        grid = Metal.MTLSizeMake(grid_cols, grid_rows, 1)
        threads = Metal.MTLSizeMake(TILE_SIZE, TILE_SIZE, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, threads)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    def transpose(self, A, rows, cols):
        """Matrix transpose."""
        return self._launch_matrix_2d("transpose_kernel", [A], rows, cols, out_rows=cols, out_cols=rows)
    
    def row_sum(self, A, rows, cols):
        """Sum each row of a matrix."""
        return self._launch_matrix_1d("row_sum_kernel", [A], rows, cols, output_length=rows)
    
    def col_sum(self, A, rows, cols):
        """Sum each column of a matrix."""
        return self._launch_matrix_1d("col_sum_kernel", [A], rows, cols, output_length=cols)
    
    def row_scale(self, A, s, rows, cols):
        """Multiply each row by corresponding scalar in s."""
        if s.length != rows:
            raise ValueError(f"s.length ({s.length}) must equal rows ({rows})")
        return self._launch_matrix_2d("row_scale_kernel", [A, s], rows, cols)
    
    def col_scale(self, A, s, rows, cols):
        """Multiply each column by corresponding scalar in s."""
        if s.length != cols:
            raise ValueError(f"s.length ({s.length}) must equal cols ({cols})")
        return self._launch_matrix_2d("col_scale_kernel", [A, s], rows, cols)

    # ======================================================
    # CATEGORY 5: MISCELLANEOUS
    # ======================================================
    def slice(self, a, start, end):
        """Extract slice from array: a[start:end]"""
        if start < 0 or end > a.length or start >= end:
            raise ValueError(f"Invalid slice [{start}:{end}] for array of length {a.length}")
        
        fname = "slice_kernel"
        
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        length = end - start
        
        out_buf = self.device.newBufferWithLength_options_(
            length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=length, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
        
        start_c = (ctypes.c_uint * 1)(start)
        end_c = (ctypes.c_uint * 1)(end)
        enc.setBytes_length_atIndex_(start_c, ctypes.sizeof(ctypes.c_uint), 2)
        enc.setBytes_length_atIndex_(end_c, ctypes.sizeof(ctypes.c_uint), 3)
        
        threads = Metal.MTLSizeMake(length, 1, 1)
        tg = Metal.MTLSizeMake(min(length, pso.maxTotalThreadsPerThreadgroup()), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    def bitonic_sort(self, a):
        """Sort array in-place using bitonic sort (N must be power of 2)."""
        N = a.length
        
        # Check if N is power of 2
        if N & (N - 1) != 0:
            raise ValueError(f"Array length {N} must be a power of 2 for bitonic sort")
        
        fname = "bitonic_sort_kernel"
        
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        # Perform bitonic sort passes
        stage = 2
        while stage <= N:
            step = stage >> 1
            while step > 0:
                cmd = self.queue.commandBuffer()
                enc = cmd.computeCommandEncoder()
                enc.setComputePipelineState_(pso)
                
                enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
                
                stage_c = (ctypes.c_uint * 1)(stage)
                step_c = (ctypes.c_uint * 1)(step)
                N_c = (ctypes.c_uint * 1)(N)
                enc.setBytes_length_atIndex_(stage_c, ctypes.sizeof(ctypes.c_uint), 1)
                enc.setBytes_length_atIndex_(step_c, ctypes.sizeof(ctypes.c_uint), 2)
                enc.setBytes_length_atIndex_(N_c, ctypes.sizeof(ctypes.c_uint), 3)
                
                threads = Metal.MTLSizeMake(N, 1, 1)
                tg = Metal.MTLSizeMake(min(N, pso.maxTotalThreadsPerThreadgroup()), 1, 1)
                enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
                
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                
                step >>= 1
            stage <<= 1
        
        return a  # Returns the same array (sorted in-place)

    # ======================================================
    # HELPER METHODS FOR MATRIX OPERATIONS
    # ======================================================
    def _launch_matrix(self, fname, arrays, rows, cols):
        """Launch matrix kernel with 1D output (rows*cols elements)."""
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        for arr in arrays:
            if arr.length != rows * cols:
                raise ValueError(f"Array length {arr.length} doesn't match {rows}x{cols}")
        
        out_buf = self.device.newBufferWithLength_options_(
            rows * cols * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=rows * cols, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        for i, arr in enumerate(arrays):
            enc.setBuffer_offset_atIndex_(arr.buffer, 0, i)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, len(arrays))
        
        rows_c = (ctypes.c_uint * 1)(rows)
        cols_c = (ctypes.c_uint * 1)(cols)
        enc.setBytes_length_atIndex_(rows_c, ctypes.sizeof(ctypes.c_uint), len(arrays) + 1)
        enc.setBytes_length_atIndex_(cols_c, ctypes.sizeof(ctypes.c_uint), len(arrays) + 2)
        
        threads = Metal.MTLSizeMake(rows * cols, 1, 1)
        tg = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    def _launch_matrix_2d(self, fname, arrays, rows, cols, out_rows=None, out_cols=None):
        """Launch matrix kernel with 2D dispatch."""
        if out_rows is None:
            out_rows = rows
        if out_cols is None:
            out_cols = cols
        
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            out_rows * out_cols * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=out_rows * out_cols, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        for i, arr in enumerate(arrays):
            enc.setBuffer_offset_atIndex_(arr.buffer, 0, i)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, len(arrays))
        
        rows_c = (ctypes.c_uint * 1)(rows)
        cols_c = (ctypes.c_uint * 1)(cols)
        enc.setBytes_length_atIndex_(rows_c, ctypes.sizeof(ctypes.c_uint), len(arrays) + 1)
        enc.setBytes_length_atIndex_(cols_c, ctypes.sizeof(ctypes.c_uint), len(arrays) + 2)
        
        threads = Metal.MTLSizeMake(out_cols, out_rows, 1)
        max_threads = pso.maxTotalThreadsPerThreadgroup()
        tg_size = int(max_threads ** 0.5)
        tg = Metal.MTLSizeMake(tg_size, tg_size, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    def _launch_matrix_1d(self, fname, arrays, rows, cols, output_length):
        """Launch matrix kernel with 1D output of specific length."""
        if fname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(fname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err: raise RuntimeError(err)
            self.kernel_cache[fname] = pso
        
        pso = self.kernel_cache[fname]
        
        out_buf = self.device.newBufferWithLength_options_(
            output_length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self, None, length=output_length, buffer=out_buf)
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        for i, arr in enumerate(arrays):
            enc.setBuffer_offset_atIndex_(arr.buffer, 0, i)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, len(arrays))
        
        rows_c = (ctypes.c_uint * 1)(rows)
        cols_c = (ctypes.c_uint * 1)(cols)
        enc.setBytes_length_atIndex_(rows_c, ctypes.sizeof(ctypes.c_uint), len(arrays) + 1)
        enc.setBytes_length_atIndex_(cols_c, ctypes.sizeof(ctypes.c_uint), len(arrays) + 2)
        
        threads = Metal.MTLSizeMake(output_length, 1, 1)
        tg = Metal.MTLSizeMake(min(output_length, pso.maxTotalThreadsPerThreadgroup()), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, tg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr