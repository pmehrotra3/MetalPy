import Metal
import ctypes
import array
from pathlib import Path
import math


# ==========================================================
# GPU Array Wrapper
# ==========================================================

class MetalArray:
    
    """GPU array wrapper around a Metal buffer."""

    def __init__(self, gpu_device, data, length=None, buffer=None, typecode='f'):
        
        self.gpu_device = gpu_device

        # If kernel already created buffer
        if buffer is not None:
            self.buffer = buffer
            self.length = length
            self.typecode = typecode
            return

        if isinstance(data, list):
            data = array.array('f', data)
            
        elif isinstance(data, array.array):
            if data.typecode not in ('f', 'I'):
                raise TypeError("MetalArray only supports array('f') or array('I') for now.")
        
        else:
            raise TypeError("MetalArray expects list or array('f').")

        self.length = len(data)
        
        self.typecode = data.typecode
        
        self.buffer = gpu_device.newBufferWithBytes_length_options_(
            data.tobytes(), self.length * 4,
            Metal.MTLResourceStorageModeShared
        ) # creates a GPU buffer and copies data into it
        
    def to_array(self):
        
        ptr = self.buffer.contents()
        raw = ptr.as_buffer(self.length * 4)  # view of bytes in shared memory
    
        out = array.array(self.typecode)
        out.frombytes(raw)  # this copies the bytes into a new array('f')
        
        return out
        
        

    def to_list(self):
        
        return self.to_array().tolist()


# ==========================================================
# Main Metal Compute Class
# ==========================================================

class MetalPy:

    def __init__(self, kernel_path="metal_kernels.metallib"):
        
        self.device = Metal.MTLCreateSystemDefaultDevice()
        
        if not self.device:
            raise RuntimeError("Error: No Metal-compatible GPU detected.")

        self.queue = self.device.newCommandQueue()
        
        print(f"Using GPU: {self.device.name()}")

        # Load compiled library instead of source

        lib_path = str(Path(kernel_path).absolute())
        
        library, err = self.device.newLibraryWithFile_error_(lib_path, None)
        if err:
            raise RuntimeError(f"Error: Failed to load library. {err}")

        self.library = library
        self.kernel_cache = {}


    def array(self, data):
        
        return MetalArray(self.device, data)

    # ======================================================
    # OPS LAUNCHERS
    # ======================================================
    
    def launcher_1(self, kname, arrays):
        """
        Launcher for kernels that take one or more MetalArray
        inputs of equal length and a scalar uint input. Matches kernels of the form:

        kernel void <kname>(
            device const float *a0 [[ buffer(0) ]],
            device const float *a1 [[ buffer(1) ]], // optional
            ...
            device float *out      [[ buffer(M) ]],
            constant uint  &N      [[ buffer(M+1) ]],
            uint id                [[ thread_position_in_grid ]]
        );
        """
        
        # Load / cache PSO (Pipeline State Object), which is the compiled GPU kernel.
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso
        
        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution.
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        # Get length of the input arrays and check they all match.
        length = arrays[0].length
        
        for a in arrays:
            if a.length != length:
                raise ValueError("All MetalArrays must have equal length for elementwise ops.")
        
        # Bind input arrays to buffer(0 .. len(arrays)-1).
        for i, arr in enumerate(arrays):
            enc.setBuffer_offset_atIndex_(arr.buffer, 0, i)
        
        # Create output MetalArray and bind it to buffer(len(arrays)).
        out_buf = self.device.newBufferWithLength_options_(length * 4,Metal.MTLResourceStorageModeShared)
        out_arr = MetalArray(self.device, None, length=length, buffer=out_buf)
        enc.setBuffer_offset_atIndex_(out_buf, 0, len(arrays))
        
        # Bind N (length of input arrays) to buffer(len(arrays)+1).
        N = (ctypes.c_uint * 1)(length)
        enc.setBytes_length_atIndex_(N, ctypes.sizeof(ctypes.c_uint), len(arrays) + 1)
        
        # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        threads = Metal.MTLSizeMake(length, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), length),1,1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)
        
        # Finish encoding, submit to the GPU, and wait for completion.
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    
    def launcher_2(self, kname, a, c):
        """
        Launcher for kernels with one MetalArray input and one scalar float input.
        Matches kernels of the form:

        kernel void <kname>(device const float *a [[ buffer(0) ]],
                            device float *out     [[ buffer(1) ]],
                            constant float &c     [[ buffer(2) ]],
                            constant uint  &N     [[ buffer(3) ]],
                            uint id [[ thread_position_in_grid ]])        
        """
        
        # Type checks
        if not isinstance(a, MetalArray):
            raise TypeError(f"{kname} expects a MetalArray as its first argument.")
        
        if not isinstance(c, (int, float)):
            raise TypeError(f"{kname} expects a scalar float as its second argument.")
        c = float(c)
        
        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso
            
        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        # Bind input array to buffer(0)
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        
        # Create output array and bind it to buffer(1)
        out_buf = self.device.newBufferWithLength_options_(a.length * 4,Metal.MTLResourceStorageModeShared)
        out_arr = MetalArray(self.device, None, length=a.length, buffer=out_buf)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
        
        # Bind scalar c to buffer(2): 
        c_val= (ctypes.c_float * 1)(c)
        enc.setBytes_length_atIndex_(c_val, ctypes.sizeof(ctypes.c_float), 2)
        
        # Bind N (length of input array) to buffer(3):
        N = (ctypes.c_uint * 1)(a.length)
        enc.setBytes_length_atIndex_(N, ctypes.sizeof(ctypes.c_uint), 3)
        
        # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        threads = Metal.MTLSizeMake(a.length, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), a.length),1,1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)
        
        # Finish encoding, submit work to the GPU and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    
    def launcher_3(self, kname, a, low, high):
        """
        Launcher for kernels with one MetalArray input and two scalar float inputs.
        Matches kernels of the form:
        
        kernel void <kname>(device const float *a [[ buffer(0) ]],
                            constant float &low   [[ buffer(1) ]],
                            constant float &high  [[ buffer(2) ]],
                            device float *out     [[ buffer(3) ]],
                            constant uint  &N     [[ buffer(4) ]],
                            uint id [[ thread_position_in_grid ]])
        """
        
        # Type checks
        if not isinstance(a, MetalArray):
            raise TypeError(f"{kname} expects a MetalArray as its first argument.")
        
        if not isinstance(low, (int, float)):
            raise TypeError(f"{kname} expects 'low' as a scalar float.")
        
        if not isinstance(high, (int, float)):
            raise TypeError(f"{kname} expects 'high' as a scalar float.")
        
        low = float(low)
        high = float(high)
        
        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso
            
        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        # Bind input array to buffer(0)
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        
        # Bind scalar low to buffer(1)
        low_val = (ctypes.c_float * 1)(low)
        enc.setBytes_length_atIndex_(low_val, ctypes.sizeof(ctypes.c_float), 1)
        
        # Bind scalar high to buffer(2)
        high_val = (ctypes.c_float * 1)(high)
        enc.setBytes_length_atIndex_(high_val, ctypes.sizeof(ctypes.c_float), 2)
        
        # Create output array and bind it to buffer(3)
        length = a.length
        out_buf = self.device.newBufferWithLength_options_(length * 4,Metal.MTLResourceStorageModeShared)
        out_arr = MetalArray(self.device, None, length=length, buffer=out_buf)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
        
        # Bind N (length of input array) to buffer(4)
        N= (ctypes.c_uint * 1)(length)
        enc.setBytes_length_atIndex_(N, ctypes.sizeof(ctypes.c_uint), 4)
        
        # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        threads = Metal.MTLSizeMake(length, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), length),1,1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
    
    
    def launcher_4(self, kname, a, b):
        """
        Launcher for kernels with two MetalArray inputs and two scalar uint inputs.
        Matches kernels of the form:

        kernel void <kname>(device const float *a [[ buffer(0) ]],  // length N
                            device const float *b [[ buffer(1) ]],  // length M
                            device float *out     [[ buffer(2) ]],
                            constant uint &N      [[ buffer(3) ]],
                            constant uint &M      [[ buffer(4) ]],
                            uint id [[ thread_position_in_grid ]])
        """
        
        # Type checks
        if not isinstance(a, MetalArray) or not isinstance(b, MetalArray):
            raise TypeError(f"{kname} expects two MetalArray inputs (a, b).")
        
        if b.length == 0:
            raise ValueError(f"{kname} requires b.length > 0 for broadcasting.")
        
        N = a.length
        M = b.length
        
        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso
        
        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        
        # Bind input array a to buffer(0)
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        
        # Bind input array b to buffer(1)
        enc.setBuffer_offset_atIndex_(b.buffer, 0, 1)
        
        # Create output array and bind it to buffer(2)
        out_buf = self.device.newBufferWithLength_options_(N * 4,Metal.MTLResourceStorageModeShared)
        out_arr = MetalArray(self.device, None, length=N, buffer=out_buf)
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        
        # Bind N (length of input array a) to buffer(3)
        N_val = (ctypes.c_uint * 1)(N)
        enc.setBytes_length_atIndex_(N_val, ctypes.sizeof(ctypes.c_uint), 3)
        
        # Bind M (length of input array b) to buffer(4)
        M_val = (ctypes.c_uint * 1)(M)
        enc.setBytes_length_atIndex_(M_val, ctypes.sizeof(ctypes.c_uint), 4)
        
        # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        threads = Metal.MTLSizeMake(N, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), N),1,1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        return out_arr
     
    # ======================================================
    # LAUNCHER 5: Hadamard product (elementwise A * B)
    # ======================================================
    
    def launcher_5(self, kname, A, B, rows, cols):
        """
        Launcher for matrix Hadamard product (elementwise multiply).
        Matches kernels of the form:

        kernel void <kname>(
            device const float *A [[ buffer(0) ]],  // length = rows * cols
            device const float *B [[ buffer(1) ]],  // length = rows * cols
            device float *C       [[ buffer(2) ]],  // length = rows * cols
            constant uint &rows   [[ buffer(3) ]],
            constant uint &cols   [[ buffer(4) ]],
            uint id [[ thread_position_in_grid ]]
        );
        """

        # Type checks
        if not isinstance(A, MetalArray) or not isinstance(B, MetalArray):
            raise TypeError(f"{kname} expects two MetalArray inputs (A, B).")

        expected_len = rows * cols

        if A.length != expected_len or B.length != expected_len:
            raise ValueError(
                f"{kname} requires A.length == B.length == rows * cols "
                f"(expected {expected_len}, got A={A.length}, B={B.length})."
            )

        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            if fn is None:
                raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso

        pso = self.kernel_cache[kname]

        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Bind A to buffer(0)
        enc.setBuffer_offset_atIndex_(A.buffer, 0, 0)
        
        # Bind B to buffer(1) 
        enc.setBuffer_offset_atIndex_(B.buffer, 0, 1)

        # Create output buffer C (same size as A/B) and bind to buffer(2).
        out_buf = self.device.newBufferWithLength_options_(
            expected_len * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=expected_len, buffer=out_buf)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)

        # Bind rows to buffer(3)
        rows_val = (ctypes.c_uint * 1)(rows)
        enc.setBytes_length_atIndex_(rows_val, ctypes.sizeof(ctypes.c_uint), 3)
        
        # Bind cols to buffer(4)
        cols_val = (ctypes.c_uint * 1)(cols)
        enc.setBytes_length_atIndex_(cols_val, ctypes.sizeof(ctypes.c_uint), 4)
        
        # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        N = rows*cols
        threads = Metal.MTLSizeMake(N, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), N), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)

        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out_arr
    
    
    # ======================================================
    # LAUNCHER 6: Row-wise / column-wise reductions
    # ======================================================
    
    def launcher_6(self, kname, A, rows, cols, out_length):
        """
        Launcher for 1D matrix reductions over rows or columns.
        Matches kernels of the form:

        kernel void <kname>(
            device const float *A [[ buffer(0) ]],  // length = rows * cols
            device float *out     [[ buffer(1) ]],  // length = out_length
            constant uint &rows   [[ buffer(2) ]],
            constant uint &cols   [[ buffer(3) ]],
            uint id               [[ thread_position_in_grid ]]
        );

        where:
          - row_sum: out_length = rows
          - col_sum: out_length = cols
        """

        # Type check
        if not isinstance(A, MetalArray):
            raise TypeError(f"{kname} expects a MetalArray input.")

        expected_len = rows * cols

        if A.length != expected_len:
            raise ValueError(
                f"{kname} requires A.length == rows * cols "
                f"(expected {expected_len}, got {A.length})."
            )

        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            if fn is None:
                raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso

        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Bind A to buffer(0)
        enc.setBuffer_offset_atIndex_(A.buffer, 0, 0)

        # Create output buffer and bind to buffer(1).
        out_buf = self.device.newBufferWithLength_options_(
            out_length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=out_length, buffer=out_buf)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 1)

        # Bind rows to buffer(2)
        rows_val = (ctypes.c_uint * 1)(rows)
        enc.setBytes_length_atIndex_(rows_val, ctypes.sizeof(ctypes.c_uint), 2)
        
        # Bind cols to buffer(3)
        cols_val = (ctypes.c_uint * 1)(cols)
        enc.setBytes_length_atIndex_(cols_val, ctypes.sizeof(ctypes.c_uint), 3)

        # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        threads = Metal.MTLSizeMake(out_length, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), out_length), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out_arr
    
    
    # ======================================================
    # LAUNCHER 7: Row-wise / column-wise scaling (2D grid)
    # ======================================================
    
    def launcher_7(self, kname, A, s, rows, cols):
        """
        Launcher for row-wise / column-wise scaling kernels.

        Matches kernels of the form:

        kernel void <kname>(
            device const float *A [[ buffer(0) ]], // length = rows * cols
            device const float *s [[ buffer(1) ]], // length = rows (row_scale) or cols (col_scale)
            device float *out     [[ buffer(2) ]], // length = rows * cols
            constant uint &rows   [[ buffer(3) ]],
            constant uint &cols   [[ buffer(4) ]],
            uint2 id              [[ thread_position_in_grid ]]
        );

        where:
          - row_scale_kernel: s[i] scales row i
          - col_scale_kernel: s[j] scales column j
        """

        # Type checks
        if not isinstance(A, MetalArray) or not isinstance(s, MetalArray):
            raise TypeError(f"{kname} expects MetalArray inputs A and s.")

        expected_len = rows * cols
        if A.length != expected_len:
            raise ValueError(
                f"{kname} requires A.length == rows * cols "
                f"(expected {expected_len}, got {A.length})."
            )

        # NOTE: s.length is checked in the high-level API (row_scale / col_scale),
        # since row_scale expects s.length == rows, and col_scale expects s.length == cols.

        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            if fn is None:
                raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso

        pso = self.kernel_cache[kname]

        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Bind A to buffer(0)
        enc.setBuffer_offset_atIndex_(A.buffer, 0, 0)
        
        # Bind s to buffer(1)
        enc.setBuffer_offset_atIndex_(s.buffer, 0, 1)

        # Create output buffer and bind to buffer(1).
        out_buf = self.device.newBufferWithLength_options_(
            expected_len * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=expected_len, buffer=out_buf)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)

        # Bind rows to buffer(2)
        rows_val = (ctypes.c_uint * 1)(rows)
        enc.setBytes_length_atIndex_(rows_val, ctypes.sizeof(ctypes.c_uint), 3)

        # Bind cols to buffer(3)
        cols_val = (ctypes.c_uint * 1)(cols)
        enc.setBytes_length_atIndex_(cols_val, ctypes.sizeof(ctypes.c_uint), 4)

        # Encode how many GPU threads to launch (grid size) and 2D threadgroup shape within hardware limits
        grid = Metal.MTLSizeMake(cols, rows, 1)
        
        max_threads = pso.maxTotalThreadsPerThreadgroup()
        tg_w = min(cols, 16)
        tg_h = max_threads // tg_w
        if tg_h == 0:
            tg_h = 1
        tg_h = min(rows, tg_h)
        tpt = Metal.MTLSizeMake(tg_w, tg_h, 1)
        
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpt)
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out_arr
    
    # ======================================================
    # LAUNCHER 8: Matrix transpose (2D grid)
    # ======================================================
    
    def launcher_8(self, kname, A, rows, cols):
        """
        Launcher for matrix transpose.
        Matches kernels of the form:

        kernel void <kname>(
            device const float *A [[ buffer(0) ]],  // length = rows * cols
            device float *B       [[ buffer(1) ]],  // length = rows * cols
            constant uint &rows   [[ buffer(2) ]],
            constant uint &cols   [[ buffer(3) ]],
            uint2 id              [[ thread_position_in_grid ]]
        );

        A is rows x cols, B is cols x rows (both stored row-major).
        """

        if not isinstance(A, MetalArray):
            raise TypeError(f"{kname} expects a MetalArray input.")

        expected_len = rows * cols
        if A.length != expected_len:
            raise ValueError(
                f"{kname} requires A.length == rows * cols "
                f"(expected {expected_len}, got {A.length})."
            )

        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            if fn is None:
                raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso

        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Bind A to buffer(0)
        enc.setBuffer_offset_atIndex_(A.buffer, 0, 0)

        # Create output buffer B and bind it to buffer(1)
        B = self.device.newBufferWithLength_options_(
            expected_len * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=expected_len, buffer=B)
        
        enc.setBuffer_offset_atIndex_(B, 0, 1)
        
        
        # Bind rows to buffer(2)
        rows_val = (ctypes.c_uint * 1)(rows)
        enc.setBytes_length_atIndex_(rows_val, ctypes.sizeof(ctypes.c_uint), 2)
        
        # Bind cols to buffer(3)
        cols_val = (ctypes.c_uint * 1)(cols)
        enc.setBytes_length_atIndex_(cols_val, ctypes.sizeof(ctypes.c_uint), 3)

        # Encode how many GPU threads to launch (grid size) and 2D threadgroup shape within hardware limits
        grid = Metal.MTLSizeMake(cols, rows, 1)
        max_threads = pso.maxTotalThreadsPerThreadgroup()
        tg_w = min(cols, 16)
        tg_h = max_threads // tg_w
        if tg_h == 0:
            tg_h = 1
        tg_h = min(rows, tg_h)
        tpt = Metal.MTLSizeMake(tg_w, tg_h, 1)

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpt)
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out_arr
    
    # ======================================================
    # LAUNCHER 9: Matrix multiplication (A: MxK, B: KxN, C: MxN)
    # ======================================================
    
    def launcher_9(self, kname, A, B, M, N, K):
        """
        Launcher for matrix multiplication:

            C = A * B

        with:
          - A: M x K (row-major), length = M * K
          - B: K x N (row-major), length = K * N
          - C: M x N (row-major), length = M * N

        Matches kernels of the form:

        kernel void <kname>(
            device const float *A [[ buffer(0) ]],
            device const float *B [[ buffer(1) ]],
            device float *C       [[ buffer(2) ]],
            constant uint &M      [[ buffer(3) ]],
            constant uint &N      [[ buffer(4) ]],
            constant uint &K      [[ buffer(5) ]],
            uint2 id              [[ thread_position_in_grid ]]
        );
        """

        if not isinstance(A, MetalArray) or not isinstance(B, MetalArray):
            raise TypeError(f"{kname} expects two MetalArray inputs (A, B).")

        if A.length != M * K:
            raise ValueError(
                f"{kname}: A.length must be M*K "
                f"(expected {M * K}, got {A.length})."
            )
        if B.length != K * N:
            raise ValueError(
                f"{kname}: B.length must be K*N "
                f"(expected {K * N}, got {B.length})."
            )

        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            if fn is None:
                raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso

        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Bind A to buffer(0)
        enc.setBuffer_offset_atIndex_(A.buffer, 0, 0)
        
        # Bind B to buffer(1)
        enc.setBuffer_offset_atIndex_(B.buffer, 0, 1)
        
         # Create output buffer C: M x N and bind it to buffer(2)
        C_len = M * N
        out_buf = self.device.newBufferWithLength_options_(
            C_len * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=C_len, buffer=out_buf)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
        
        # Bind M to buffer(3)
        M_val = (ctypes.c_uint * 1)(M)
        enc.setBytes_length_atIndex_(M_val, ctypes.sizeof(ctypes.c_uint), 3)

        # Bind N to buffer(4)
        N_val = (ctypes.c_uint * 1)(N)
        enc.setBytes_length_atIndex_(N_val, ctypes.sizeof(ctypes.c_uint), 4)
        
        # Bind K to buffer(5)
        K_val = (ctypes.c_uint * 1)(K)
        enc.setBytes_length_atIndex_(K_val, ctypes.sizeof(ctypes.c_uint), 5)
       
       # Encode how many GPU threads to launch (grid size) and 2D threadgroup shape (N columns, M rows) within hardware limits
        grid = Metal.MTLSizeMake(N, M, 1)
        max_threads = pso.maxTotalThreadsPerThreadgroup()
        tg_w = min(N, 16)
        tg_h = max_threads // tg_w
        if tg_h == 0:
            tg_h = 1
        tg_h = min(M, tg_h)
        tpt = Metal.MTLSizeMake(tg_w, tg_h, 1)

        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpt)
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out_arr
    
    # ======================================================
    # LAUNCHER 10: Slice - extract contiguous slice from an array
    # ======================================================

    def launcher_10(self, kname, a, start, end):
        """
        Launcher for contiguous slicing of a 1D array.
        Matches kernels of the form:

        kernel void <kname>(
            device const float *a [[ buffer(0) ]],
            device float *out     [[ buffer(1) ]],
            constant uint &start  [[ buffer(2) ]],
            constant uint &end    [[ buffer(3) ]],
            uint id [[ thread_position_in_grid ]]
        );

        Extracts a[start:end] into out[0 : end-start].
        """

        # Type checks
        if not isinstance(a, MetalArray):
            raise TypeError(f"{kname} expects a MetalArray 'a' as its first argument.")

        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"{kname} expects 'start' and 'end' as Python ints.")

        if start < 0 or end < 0:
            raise ValueError(f"{kname} expects non-negative 'start' and 'end' indices.")
        
        if start > end:
            raise ValueError(f"{kname} requires start <= end (got start={start}, end={end}).")
        
        if end > a.length:
            raise ValueError(
                f"{kname}: 'end' index {end} is out of bounds for array of length {a.length}."
            )

        length = end - start

        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            if fn is None:
                raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso

        pso = self.kernel_cache[kname]

        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Bind input array a to buffer(0)
        enc.setBuffer_offset_atIndex_(a.buffer, 0, 0)

        # Create output buffer of size 'length' and bind it to buffer(1)
        out_buf = self.device.newBufferWithLength_options_(
            length * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=length, buffer=out_buf)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 1)

        # Bind start to buffer(2)
        start_val = (ctypes.c_uint * 1)(start)
        enc.setBytes_length_atIndex_(start_val, ctypes.sizeof(ctypes.c_uint), 2)
        
        # Bind end to buffer(3)
        end_val = (ctypes.c_uint * 1)(end)
        enc.setBytes_length_atIndex_(end_val, ctypes.sizeof(ctypes.c_uint), 3)

        # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        threads = Metal.MTLSizeMake(length, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), length),1,1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)

        # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out_arr
    
    
    # ======================================================
    # LAUNCHER 11: Gather - index-based selection from an array
    # ======================================================

    def launcher_11(self, kname, source, index_arr):
        """
        Launcher for index-based gather from a 1D source array.
        Matches kernels of the form:

        kernel void <kname>(
            device const float *source [[ buffer(0) ]],
            device const uint  *index  [[ buffer(1) ]],
            device float *out          [[ buffer(2) ]],
            constant uint &N           [[ buffer(3) ]],
            uint id [[ thread_position_in_grid ]]
        );

        For each id in [0, N):
            j = index[id]
            out[id] = source[j];
        """

        # Type checks
        if not isinstance(source, MetalArray):
            raise TypeError(f"{kname} expects a MetalArray 'source' as its first argument.")
        
        if not isinstance(index_arr, MetalArray):
            raise TypeError(f"{kname} expects a MetalArray 'index_arr' for indices.")

        N = index_arr.length

        # NOTE: We do not know index bounds here; the kernel assumes
        # that all index_arr elements are valid indices into 'source'.

        # Load / cache PSO
        if kname not in self.kernel_cache:
            fn = self.library.newFunctionWithName_(kname)
            if fn is None:
                raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if err:
                raise RuntimeError(err)
            self.kernel_cache[kname] = pso

        pso = self.kernel_cache[kname]
        
        # Create a GPU command buffer, start a compute encoder, and bind the compiled kernel (PSO) for execution
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)

        # Bind source (float array) to buffer(0)
        enc.setBuffer_offset_atIndex_(source.buffer, 0, 0)

        # Bind index array (uint32 buffer) to buffer(1)
        enc.setBuffer_offset_atIndex_(index_arr.buffer, 0, 1)

        #  Create output buffer (length N) and bind it to buffer(2)
        out_buf = self.device.newBufferWithLength_options_(
            N * 4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=N, buffer=out_buf)
        
        enc.setBuffer_offset_atIndex_(out_buf, 0, 2)

        # Bind N to buffer(3)
        N_val = (ctypes.c_uint * 1)(N)
        enc.setBytes_length_atIndex_(N_val, ctypes.sizeof(ctypes.c_uint), 3)

         # Encode how many GPU threads to launch (grid size) and how large each threadgroup should be
        threads = Metal.MTLSizeMake(N, 1, 1)
        TPT = Metal.MTLSizeMake(min(pso.maxTotalThreadsPerThreadgroup(), N),1,1)
        enc.dispatchThreads_threadsPerThreadgroup_(threads, TPT)
        
         # Finish encoding, submit work to the GPU, and wait for completion
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out_arr
    
    # ======================================================
    # LAUNCHER 12: Scalar reductions (sum / product / max / min)
    # ======================================================

    def launcher_12(self, kname_init, kname_final, a):
        """
        Generic two-pass scalar reduction launcher.

        Handles kernels of the form:

        Initial pass:
        kernel void <kname_init>(
            device const float *a [[ buffer(0) ]],
            device float *out     [[ buffer(1) ]],
            constant uint &N      [[ buffer(2) ]],
            constant uint &TPT    [[ buffer(3) ]],
            uint tid              [[ thread_position_in_threadgroup ]],
            uint gid              [[ threadgroup_position_in_grid ]],
            uint global_id        [[ thread_position_in_grid ]],
            uint num_threadgroups [[ threadgroups_per_grid ]],
            threadgroup float *scratch
        );

        Final pass:
        kernel void <kname_final>(
            device const float *a [[ buffer(0) ]],
            device float *out     [[ buffer(1) ]],
            constant uint &N      [[ buffer(2) ]],
            uint tid              [[ thread_position_in_grid ]]
        );

        Returns: MetalArray of length 1 (scalar result on GPU).
        """
        
        # Type checks
        if not isinstance(a, MetalArray):
            raise TypeError(f"{kname_init} expects a MetalArray 'a' as input.")

        N = a.length
        if N == 0:
            raise ValueError(f"{kname_init} cannot reduce an empty array (N=0).")

        #  Load / cache PSOs 
        for kname in (kname_init, kname_final):
            if kname not in self.kernel_cache:
                fn = self.library.newFunctionWithName_(kname)
                if fn is None:
                    raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
                pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
                if err:
                    raise RuntimeError(err)
                self.kernel_cache[kname] = pso

        pso_init = self.kernel_cache[kname_init]
        pso_final = self.kernel_cache[kname_final]

        # First pass: per-threadgroup partial reduction 
        max_threads = pso_init.maxTotalThreadsPerThreadgroup()
        TPT = min(max_threads, N, 256)  # threads per threadgroup
        TPT = 2 ** int(math.log2(TPT))  # Round down to power of 2
        num_threadgroups = (N + TPT - 1) // TPT

        # Partial results buffer: one float per threadgroup
        partial_buf = self.device.newBufferWithLength_options_(
            num_threadgroups * 4, Metal.MTLResourceStorageModeShared
        )
        partial_arr = MetalArray(self.device, None, length=num_threadgroups, buffer=partial_buf)

        cmd1 = self.queue.commandBuffer()
        enc1 = cmd1.computeCommandEncoder()
        enc1.setComputePipelineState_(pso_init)

        # Bind buffers: a -> 0, partial -> 1
        enc1.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc1.setBuffer_offset_atIndex_(partial_buf, 0, 1)

        # Bind N, TPT
        N_val = (ctypes.c_uint * 1)(N)
        TPT_val = (ctypes.c_uint * 1)(TPT)
        enc1.setBytes_length_atIndex_(N_val, ctypes.sizeof(ctypes.c_uint), 2)
        enc1.setBytes_length_atIndex_(TPT_val, ctypes.sizeof(ctypes.c_uint), 3)
        
        # Allocate threadgroup memory for `scratch` <<<
        #scratch_bytes = TPT * 4  # TPT floats, 4 bytes each
        #enc1.setThreadgroupMemoryLength_atIndex_(scratch_bytes, 0)

        # Dispatch: cover all N elements with num_threadgroups * TPT threads
        threads = Metal.MTLSizeMake(num_threadgroups * TPT, 1, 1)
        tpt     = Metal.MTLSizeMake(TPT, 1, 1)
        enc1.dispatchThreads_threadsPerThreadgroup_(threads, tpt)
        
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc1.endEncoding()
        cmd1.commit()
        cmd1.waitUntilCompleted()

        # ---------- Second pass: final scalar reduction over partials ----------
        out_buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        out_arr = MetalArray(self.device, None, length=1, buffer=out_buf)

        cmd2 = self.queue.commandBuffer()
        enc2 = cmd2.computeCommandEncoder()
        enc2.setComputePipelineState_(pso_final)

        # Input = partials, output = scalar
        enc2.setBuffer_offset_atIndex_(partial_buf, 0, 0)
        enc2.setBuffer_offset_atIndex_(out_buf, 0, 1)

        N2_val = (ctypes.c_uint * 1)(num_threadgroups)
        enc2.setBytes_length_atIndex_(N2_val, ctypes.sizeof(ctypes.c_uint), 2)

        # Only tid == 0 does work in final kernel
        one_thread = Metal.MTLSizeMake(1, 1, 1)
        enc2.dispatchThreads_threadsPerThreadgroup_(one_thread, one_thread)
        
        # Finish encoding, submit work to the GPU, and wait for completion
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()

        return out_arr.to_array()[0]
    
    # ======================================================
    # LAUNCHER 13: Arg reductions (argmax / argmin)
    # ======================================================

    def launcher_13(self, kname_init, kname_final, a):
        """
        Generic two-pass arg reduction launcher.

        Handles kernels of the form:

        Initial pass (argmax / argmin):
        kernel void <kname_init>(
            device const float *a        [[ buffer(0) ]],
            device uint  *out_idx        [[ buffer(1) ]],
            device float *out_val        [[ buffer(2) ]],
            constant uint &N             [[ buffer(3) ]],
            constant uint &TPT           [[ buffer(4) ]],
            uint tid                     [[ thread_position_in_threadgroup ]],
            uint gid                     [[ threadgroup_position_in_grid ]],
            uint num_threadgroups        [[ threadgroups_per_grid ]],
            threadgroup uint  *scratch_idx,
            threadgroup float *scratch_val
        );

        Final pass:
        kernel void <kname_final>(
            device const uint  *in_idx   [[ buffer(0) ]],
            device const float *in_val   [[ buffer(1) ]],
            device uint        *out_idx  [[ buffer(2) ]],
            device float       *out_val  [[ buffer(3) ]],
            constant uint      &N        [[ buffer(4) ]],
            uint tid                      [[ thread_position_in_grid ]]
        );

        Returns:
            (idx_arr, val_arr) as MetalArray objects of length 1.
        """

        if not isinstance(a, MetalArray):
            raise TypeError(f"{kname_init} expects a MetalArray 'a' as input.")

        N = a.length
        if N == 0:
            raise ValueError(f"{kname_init} cannot reduce an empty array (N=0).")

        # ---------- Load / cache PSOs ----------
        for kname in (kname_init, kname_final):
            if kname not in self.kernel_cache:
                fn = self.library.newFunctionWithName_(kname)
                if fn is None:
                    raise ValueError(f"Kernel function '{kname}' not found in Metal library.")
                pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
                if err:
                    raise RuntimeError(err)
                self.kernel_cache[kname] = pso

        pso_init = self.kernel_cache[kname_init]
        pso_final = self.kernel_cache[kname_final]

        # ---------- First pass: per-threadgroup arg reduction ----------
        max_threads = pso_init.maxTotalThreadsPerThreadgroup()
        TPT = min(max_threads, N, 256) # threads per threadgroup
        TPT = 2 ** int(math.log2(TPT))  # Round down to power of 2
        num_threadgroups = (N + TPT - 1) // TPT

        # Partial index + value buffers
        idx_partial_buf = self.device.newBufferWithLength_options_(
            num_threadgroups * 4, Metal.MTLResourceStorageModeShared
        )
        val_partial_buf = self.device.newBufferWithLength_options_(
            num_threadgroups * 4, Metal.MTLResourceStorageModeShared
        )

        idx_partial_arr = MetalArray(self.device, None, length=num_threadgroups,
                             buffer=idx_partial_buf, typecode='I')
        
        val_partial_arr = MetalArray(self.device, None, length=num_threadgroups,
                             buffer=val_partial_buf, typecode='f')

        cmd1 = self.queue.commandBuffer()
        enc1 = cmd1.computeCommandEncoder()
        enc1.setComputePipelineState_(pso_init)

        # Bind: a -> 0, idx_partial -> 1, val_partial -> 2
        enc1.setBuffer_offset_atIndex_(a.buffer, 0, 0)
        enc1.setBuffer_offset_atIndex_(idx_partial_buf, 0, 1)
        enc1.setBuffer_offset_atIndex_(val_partial_buf, 0, 2)

        N_val = (ctypes.c_uint * 1)(N)
        TPT_val = (ctypes.c_uint * 1)(TPT)
        enc1.setBytes_length_atIndex_(N_val,  ctypes.sizeof(ctypes.c_uint), 3)
        enc1.setBytes_length_atIndex_(TPT_val, ctypes.sizeof(ctypes.c_uint), 4)

        threads = Metal.MTLSizeMake(num_threadgroups * TPT, 1, 1)
        tpt     = Metal.MTLSizeMake(TPT, 1, 1)
        enc1.dispatchThreads_threadsPerThreadgroup_(threads, tpt)

        enc1.endEncoding()
        cmd1.commit()
        cmd1.waitUntilCompleted()

        # ---------- Second pass: final arg reduction over partials ----------
        idx_out_buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )
        val_out_buf = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )

        idx_out_arr = MetalArray(self.device, None, length=1,
                         buffer=idx_out_buf, typecode='I')
        
        val_out_arr = MetalArray(self.device, None, length=1,
                         buffer=val_out_buf, typecode='f')
        cmd2 = self.queue.commandBuffer()
        enc2 = cmd2.computeCommandEncoder()
        enc2.setComputePipelineState_(pso_final)

        # Inputs: in_idx, in_val; Outputs: out_idx, out_val
        enc2.setBuffer_offset_atIndex_(idx_partial_buf, 0, 0)
        enc2.setBuffer_offset_atIndex_(val_partial_buf, 0, 1)
        enc2.setBuffer_offset_atIndex_(idx_out_buf,    0, 2)
        enc2.setBuffer_offset_atIndex_(val_out_buf,    0, 3)

        N2_val = (ctypes.c_uint * 1)(num_threadgroups)
        enc2.setBytes_length_atIndex_(N2_val, ctypes.sizeof(ctypes.c_uint), 4)

        one_thread = Metal.MTLSizeMake(1, 1, 1)
        enc2.dispatchThreads_threadsPerThreadgroup_(one_thread, one_thread)

        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()

        return idx_out_arr.to_array()[0], val_out_arr.to_array()[0]


    
    # ======================================================
    # PUBLIC API 
    # ======================================================
    
    # ======================================================
    # CATEGORY 1: ELEMENTWISE MATH OPERATIONS
    # ======================================================
    
    def add(self, a, b): 
        
        if not isinstance(a, MetalArray) or not isinstance(b, MetalArray):
            
            raise TypeError("Error: add() expects MetalArray inputs.")
        
        return self.launcher_1("add_kernel", [a, b])
    
    def sub(self, a, b):       
        
        if not isinstance(a, MetalArray) or not isinstance(b, MetalArray):
            
            raise TypeError("Error: sub() expects MetalArray inputs.")
        
        return self.launcher_1("sub_kernel", [a, b])
    
    def multiply(self, a, b):  
        
        if not isinstance(a, MetalArray) or not isinstance(b, MetalArray):
            
            raise TypeError("Error: multiply() expects MetalArray inputs.")
        
        return self.launcher_1("multiply_kernel", [a, b])
    
    def divide(self, a, b):    
        
        if not isinstance(a, MetalArray) or not isinstance(b, MetalArray):
            
            raise TypeError("Error: divide() expects MetalArray inputs.")
        
        return self.launcher_1("division_kernel", [a, b])

    def negate(self, a):
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: negate() expects MetalArray inputs.")
        
        return self.launcher_1("negate_kernel", [a])
    
    
    def abs(self, a):
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: abs() expects MetalArray inputs.")
        
        return self.launcher_1("abs_kernel", [a])


    def pow(self, a, b):
        
        if not isinstance(a, MetalArray):
            raise TypeError("pow() expects a MetalArray as its first argument.")
        
        if not isinstance(b, (int, float)):
            raise TypeError("pow() expects a scalar float as its second argument.")
        
        b = float(b)
        
        return self.launcher_2("pow_kernel", a, b)
        
    
    def square(self, a):      
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: square() expects MetalArray inputs.")
        
        return self.launcher_1("square_kernel", [a])
    
    
    def sqrt(self, a):    
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: sqrt() expects MetalArray inputs.")
        
        return self.launcher_1("sqrt_kernel", [a])
    
    def exp(self, a):      
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: exp() expects MetalArray inputs.")
        
        return self.launcher_1("exp_kernel", [a])
    
    
    def log(self, a):    
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: log() expects MetalArray inputs.")      
        
        return self.launcher_1("log_kernel", [a])

    
    def sin(self, a):     
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: sin() expects MetalArray inputs.")      
        
        return self.launcher_1("sin_kernel", [a])
    
    
    def cos(self, a):  
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: cos() expects MetalArray inputs.")              
        
        return self.launcher_1("cos_kernel", [a])
    
    
    def tan(self, a):
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: tan() expects MetalArray inputs.")                     
        
        return self.launcher_1("tan_kernel", [a])
    
    
    def asin(self, a): 
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: asin() expects MetalArray inputs.")                      
        
        return self.launcher_1("asin_kernel", [a])
    
    
    def atan2(self, y, x):     
        
        if not isinstance(y, MetalArray) or not isinstance(x, MetalArray):
            
            raise TypeError("Error: atan2() expects MetalArray inputs.")              
        
        return self.launcher_1("atan2_kernel", [y, x])
    
    def floor(self, a):   
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: floor() expects MetalArray inputs.")                   
        
        return self.launcher_1("floor_kernel", [a])
    
    
    def ceil(self, a):      
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: ceil() expects MetalArray inputs.")         
        
        return self.launcher_1("ceil_kernel", [a])
    
    def sign(self, a):         
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: sign() expects MetalArray inputs.")         
        
        return self.launcher_1("sign_kernel", [a])
    
    
    def clip(self, a, low, high):
        
        if not isinstance(a, MetalArray):
            raise TypeError("clip() expects a MetalArray for 'a'.")

        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError("clip() expects scalar floats for 'low' and 'high'.")

        return self.launcher_3("clip_kernel", a, float(low), float(high))

        
        
    def round(self, a):        
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: round() expects MetalArray inputs.")
        
        return self.launcher_1("round_kernel", [a])
    
    
    def broadcast_add(self, a, b):
        
        if not isinstance(a, MetalArray) or not isinstance(b, MetalArray):
            raise TypeError("broadcast_add expects a MetalArray inputs.")
        
        if b.length == 0:
            raise ValueError("broadcast_add requires b.length > 0 for broadcasting.")
        
        return self.launcher_4("broadcast_add_kernel", a, b)
    
    
    def broadcast_multiply(self, a, b):
        
        if not isinstance(a, MetalArray) or not isinstance(b, MetalArray):
            raise TypeError("broadcast_multiply expects a MetalArray inputs.")
        
        if b.length == 0:
            raise ValueError("broadcast_multiply requires b.length > 0 for broadcasting.")
        
        return self.launcher_4("broadcast_multiply_kernel", a, b)

    
    
    # ======================================================
    # CATEGORY 2: ACTIVATIONS
    # ======================================================
    
    def relu(self, a):         
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: relu() expects MetalArray inputs.")
        
        return self.launcher_1("relu_kernel", [a])
    
    
    def leaky_relu(self, a, alpha=0.01):
        
        if not isinstance(a, MetalArray):
            raise TypeError("Error: leaky_relu() expects a MetalArray input for a.")
        
        if not isinstance(alpha, (int, float)):
            raise TypeError("Error: leaky_relu() expects 'alpha' as a scalar float.")

        return self.launcher_2("leaky_relu_kernel", a, float(alpha))

    
    def sigmoid(self, a):   
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: sigmoid() expects MetalArray inputs.")   
        
        return self.launcher_1("sigmoid_kernel", [a])
    
    def tanh(self, a): 
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: tanh() expects MetalArray inputs.")       
        
        return self.launcher_1("tanh_kernel", [a])
    
    def softplus(self, a):     
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: softplus() expects MetalArray inputs.")
        
        return self.launcher_1("softplus_kernel", [a])
    
    def swish(self, a):     
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: swish() expects MetalArray inputs.")   
        
        return self.launcher_1("swish_kernel", [a])
    
    
    def gelu(self, a):    
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("Error: gelu() expects MetalArray inputs.")     
        
        return self.launcher_1("gelu_kernel", [a])


    # ======================================================
    # CATEGORY 3: REDUCTIONS
    # ======================================================

    def sum(self, a):
        
        if not isinstance(a, MetalArray):
            raise TypeError("sum expects a MetalArray 'a'.")

        return self.launcher_12("sum_reduce_kernel", "sum_reduce_final_kernel", a)

    def product(self, a):

        if not isinstance(a, MetalArray):
            raise TypeError("product expects a MetalArray 'a'.")
        
        return self.launcher_12("product_reduce_kernel", "product_reduce_final_kernel", a)

    def max(self, a):
       
        if not isinstance(a, MetalArray):
            raise TypeError("max expects a MetalArray 'a'.")
        
        return self.launcher_12("max_reduce_kernel", "max_reduce_final_kernel", a)

    def min(self, a):

        if not isinstance(a, MetalArray):
            raise TypeError("min expects a MetalArray 'a'.")
        
        return self.launcher_12("min_reduce_kernel", "min_reduce_final_kernel", a)


    def argmax(self, a):
        
        if not isinstance(a, MetalArray):
            raise TypeError("argmax expects a MetalArray 'a'.")
        
        idx, val = self.launcher_13("argmax_reduce_kernel","argmax_reduce_final_kernel",a)
        
        return int(idx), float(val)

    def argmin(self, a):
       
        if not isinstance(a, MetalArray):
            raise TypeError("argmin expects a MetalArray 'a'.")
        
        idx, val = self.launcher_13("argmin_reduce_kernel","argmin_reduce_final_kernel",a)
        return int(idx), float(val)

    def mean(self, a):
  
        if not isinstance(a, MetalArray):
            raise TypeError("mean expects a MetalArray 'a'.")
        
        return self.sum(a) / a.length




    # ======================================================
    # CATEGORY 4: LINEAR ALGEBRA (Matrix Operations)
    # ======================================================
    
    def hadamard(self, A, B, rows, cols):

        if not isinstance(A, MetalArray) or not isinstance(B, MetalArray):
            raise TypeError("hadamard expects MetalArray inputs A and B.")

        expected_len = rows * cols
        if A.length != expected_len or B.length != expected_len:
            raise ValueError(
                f"hadamard: A.length and B.length must both be rows * cols "
                f"(expected {expected_len}, got A={A.length}, B={B.length})."
            )

        return self.launcher_5("hadamard_mat_kernel", A, B, rows, cols)
    

    def row_sum(self, A, rows, cols):
        
        if not isinstance(A, MetalArray):
            raise TypeError("row_sum expects a MetalArray 'A' as input.")

        expected_len = rows * cols
        if A.length != expected_len:
            raise ValueError(
                f"row_sum: A.length must equal rows * cols "
                f"(expected {expected_len}, got {A.length})."
            )

        return self.launcher_6("row_sum_kernel", A, rows, cols, out_length=rows)

    def col_sum(self, A, rows, cols):

        if not isinstance(A, MetalArray):
            raise TypeError("col_sum expects a MetalArray 'A' as input.")

        expected_len = rows * cols
        if A.length != expected_len:
            raise ValueError(
                f"col_sum: A.length must equal rows * cols "
                f"(expected {expected_len}, got {A.length})."
            )

        return self.launcher_6("col_sum_kernel", A, rows, cols, out_length=cols)
    
    def row_scale(self, A, s, rows, cols):
       
        if not isinstance(A, MetalArray) or not isinstance(s, MetalArray):
            raise TypeError("row_scale expects MetalArray inputs A and s.")
        
        if s.length != rows:
            raise ValueError(f"row_scale: s.length ({s.length}) must equal rows ({rows}).")
        
        return self.launcher_7("row_scale_kernel", A, s, rows, cols)

    def col_scale(self, A, s, rows, cols):
    
        if not isinstance(A, MetalArray) or not isinstance(s, MetalArray):
            raise TypeError("col_scale expects MetalArray inputs A and s.")
        
        if s.length != cols:
            raise ValueError(f"col_scale: s.length ({s.length}) must equal cols ({cols}).")
        
        return self.launcher_7("col_scale_kernel", A, s, rows, cols)

    def transpose(self, A, rows, cols):
        
        if not isinstance(A, MetalArray):
            raise TypeError("transpose expects a MetalArray input A.")

        expected_len = rows * cols
        if A.length != expected_len:
            raise ValueError(
                f"transpose: A.length must equal rows * cols "
                f"(expected {expected_len}, got {A.length})."
            )

        return self.launcher_8("transpose_kernel", A, rows, cols)
    
    def matmul(self, A, B, M, K, N):
       
        if not isinstance(A, MetalArray) or not isinstance(B, MetalArray):
            raise TypeError("matmul expects two MetalArray inputs (A, B).")

        if A.length != M * K:
            raise ValueError(
                f"matmul: A.length must be M*K (expected {M * K}, got {A.length})."
            )
        if B.length != K * N:
            raise ValueError(
                f"matmul: B.length must be K*N (expected {K * N}, got {B.length})."
            )

        return self.launcher_9("matmul_kernel", A, B, M, N, K)
    
    
    # ======================================================
    # CATEGORY 5: MISCELLANEOUS
    # ======================================================
    
    def slice(self, a, start, end):
        
        if not isinstance(a, MetalArray):
            
            raise TypeError("slice expects a MetalArray as input.")

        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("slice expects 'start' and 'end' to be ints.")  

        if start < 0 or end < 0:
            raise ValueError("slice expects non-negative indexing.")

        if start > end:
            raise ValueError(f"slice requires start <= end (got {start}, {end}).")

        if end > a.length:
            raise ValueError(
                f"slice: end={end} out of bounds for array of length {a.length}."
            )

        return self.launcher_10("slice_kernel", a, start, end)
    
    
    def gather(self, source, index_arr):
        
        if not isinstance(source, MetalArray):
            
            raise TypeError("gather expects a MetalArray 'source' as input.")
        
        
        if not isinstance(index_arr, MetalArray):
            
            raise TypeError("gather expects a MetalArray 'index_arr' for indices.")
        
        if index_arr.length == 0:
            
            raise ValueError("gather requires index_arr.length > 0.")
        
        #NOTE: We do not bounds-check indices on the CPU; the kernel
        # assumes all index_arr values are valid indices into 'source'.
        
        if index_arr.typecode != 'I':
            raise TypeError("gather expects index_arr to be an array('I') MetalArray (uint32 indices).")
        
        
        return self.launcher_11("gather_kernel", source, index_arr)

