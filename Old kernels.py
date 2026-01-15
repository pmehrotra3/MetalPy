Old kernels



// Category - 4 Linear Algebra 


// hadamard, 

kernel void hadamard_mat_kernel(device const float *A    [[ buffer(0) ]],
                                device const float *B    [[ buffer(1) ]],
                                device float *C          [[ buffer(2) ]],
                                constant uint &rows      [[ buffer(3) ]],
                                constant uint &cols      [[ buffer(4) ]],
                                uint id                  [[ thread_position_in_grid ]]) 
{
    uint N = rows * cols;

    if (id < N) {
        C[id] = A[id] * B[id];   // elementwise (row-major)
    }
}

// 39) Matrix transpose - with grid-stride for large matrices
kernel void transpose_kernel(device const float *A   [[ buffer(0) ]],
                             device float *B         [[ buffer(1) ]],
                             constant uint &rows     [[ buffer(2) ]],
                             constant uint &cols     [[ buffer(3) ]],
                             uint2 id                [[ thread_position_in_grid ]],
                             uint2 grid_size         [[ threads_per_grid ]]) {
    
    // Grid-stride loop for large matrices

    for (uint row = id.y; row < rows; row += grid_size.y) {
        for (uint col = id.x; col < cols; col += grid_size.x) {
            B[col * rows + row] = A[row * cols + col];
        }
    }
}

// 41) Row sum - sum each row of matrix
kernel void row_sum_kernel(device const float *A     [[ buffer(0) ]],
                           device float *out         [[ buffer(1) ]],
                           constant uint &rows       [[ buffer(2) ]],
                           constant uint &cols       [[ buffer(3) ]],
                           uint id                   [[ thread_position_in_grid ]]) {
    if (id < rows) {
        float sum = 0.0f;
        for (uint j = 0; j < cols; j++) {
            sum += A[id * cols + j];
        }
        out[id] = sum;
    }
}

/ 42) Column sum - sum each column of matrix
kernel void col_sum_kernel(device const float *A     [[ buffer(0) ]],
                           device float *out         [[ buffer(1) ]],
                           constant uint &rows       [[ buffer(2) ]],
                           constant uint &cols       [[ buffer(3) ]],
                           uint id                   [[ thread_position_in_grid ]]) {
    if (id < cols) {
        float sum = 0.0f;
        for (uint i = 0; i < rows; i++) {
            sum += A[i * cols + id];
        }
        out[id] = sum;
    }
}

// 43) Row scale - multiply each row by corresponding scalar
kernel void row_scale_kernel(device const float *A     [[ buffer(0) ]],
                             device const float *s     [[ buffer(1) ]],
                             device float *out         [[ buffer(2) ]],
                             constant uint &rows       [[ buffer(3) ]],
                             constant uint &cols       [[ buffer(4) ]],
                             uint2 id                  [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row < rows && col < cols) {
        out[row * cols + col] = A[row * cols + col] * s[row];
    }
}


// 44) Column scale - multiply each column by corresponding scalar
kernel void col_scale_kernel(device const float *A     [[ buffer(0) ]],
                             device const float *s     [[ buffer(1) ]],
                             device float *out         [[ buffer(2) ]],
                             constant uint &rows       [[ buffer(3) ]],
                             constant uint &cols       [[ buffer(4) ]],
                             uint2 id                  [[ thread_position_in_grid ]]) {
    uint row = id.y;
    uint col = id.x;
    
    if (row < rows && col < cols) {
        out[row * cols + col] = A[row * cols + col] * s[col];
    }
}





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

    def __init__(self, kernel_path="metal_kernels.metal"):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("❌ No Metal GPU detected.")

        self.queue = self.device.newCommandQueue()
        print(f"✅ Using GPU: {self.device.name()}")

        source = Path(kernel_path).read_text()
        library, err = self.device.newLibraryWithSource_options_error_(source, None, None)
        if err:
            raise RuntimeError(f"❌ Kernel compile error:\n{err}")

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
    def _launch(self, fname, arrays, constants=None):

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

        # Extra constants
        if constants:
            for idx, val in constants.items():
                enc.setBytes_length_atIndex_(val, ctypes.sizeof(type(val)), idx)

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
    # PUBLIC API
    # ======================================================
    def add(self, a, b):       return self._launch("add_kernel", [a,b])
    def sub(self, a, b):       return self._launch("sub_kernel", [a,b])
    def multiply(self, a, b):  return self._launch("multiply_kernel", [a,b])
    def divide(self, a, b):    return self._launch("division_kernel", [a,b])

    def negate(self, a):       return self._launch("negate_kernel", [a])
    def abs(self, a):          return self._launch("abs_kernel", [a])
    def pow(self, a, c):       return self._launch("pow_kernel", [a], {1: ctypes.c_float(c)})

    def square(self, a):       return self._launch("square_kernel", [a])
    def sqrt(self, a):         return self._launch("sqrt_kernel", [a])
    def exp(self, a):          return self._launch("exp_kernel", [a])
    def log(self, a):          return self._launch("log_kernel", [a])

    def sin(self, a):          return self._launch("sin_kernel", [a])
    def cos(self, a):          return self._launch("cos_kernel", [a])
    def tan(self, a):          return self._launch("tan_kernel", [a])

    def relu(self, a):         return self._launch("relu_kernel", [a])

    # Reductions - ALL use fast 2-pass with pre-allocated buffers
    def sum(self, a):          return self._launch_reduce_fast("sum_reduce_kernel", a)
    def product(self, a):      return self._launch_reduce_fast("product_reduce_kernel", a)
    def max(self, a):          return self._launch_reduce_fast("max_reduce_kernel", a)
    def min(self, a):          return self._launch_reduce_fast("min_reduce_kernel", a)
    
    # Argmax/Argmin use optimized 2-pass with pre-allocated buffers
    def argmax(self, a):       return self._launch_reduce_arg_fast("argmax_reduce_kernel", "argmax_reduce_final_kernel", a)
    def argmin(self, a):       return self._launch_reduce_arg_fast("argmin_reduce_kernel", "argmin_reduce_final_kernel", a)
    
    # Derived operations
    def mean(self, a):         return self.sum(a) / a.length
    
    
    from MetalPy import MetalPy

mp = MetalPy("metal_kernels.metal")

x = mp.array([1.0, 2.0, 3.0, 4.0])
y = mp.array([10.0, 20.0, 30.0, 40.0])

print("Add:", mp.add(x, y).to_list())
print("Multiply:", mp.multiply(x, y).to_list())
print("Square:", mp.square(x).to_list())

print("ReLU:", mp.relu(x).to_list())
print("Exp:", mp.exp(x).to_list())

print("Sum:", mp.sum(x))
print("Product:", mp.product(x))
print("Max:", mp.max(x))
print("Argmax:", mp.argmax(x))

