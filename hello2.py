import Metal
import ctypes

class MetalGPU:
    """High-level, Pythonic Metal GPU compute wrapper."""

    def __init__(self):
    
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("❌ No Metal-compatible GPU found.")
            
        self.queue = self.device.newCommandQueue()
        
        print(f"✅ Using Metal device: {self.device.name()}")

    # ---------------------------
    # Buffer creation helper
    # ---------------------------
    def _make_buffer(self, data):
        """Create a shared GPU buffer and copy data into it."""
        import array
        if isinstance(data, list):
            arr = array.array('f', data)
        elif isinstance(data, (tuple, set)):
            arr = array.array('f', list(data))
        elif isinstance(data, array.array):
            arr = data
        else:
            raise TypeError("Data must be list, tuple, or array.array of floats.")

        buf = self.device.newBufferWithBytes_length_options_(
            arr.tobytes(), len(arr) * 4, Metal.MTLResourceStorageModeShared
        )
        return buf, len(arr)

    # ---------------------------
    # Kernel compile helper
    # ---------------------------
    def compile(self, source, func_name):
        """Compile Metal kernel code and return a function handle."""
        library, err = self.device.newLibraryWithSource_options_error_(source, None, None)
        if err:
            raise RuntimeError(f"Kernel compile error: {err}")
        return library.newFunctionWithName_(func_name)

    # ---------------------------
    # Run kernel
    # ---------------------------
    def run_kernel(self, source, func, inputs):
        """Compile + execute a Metal kernel with given inputs."""
        # Compile kernel
        kernel = self.compile(source, func)

        # Create pipeline
        pso, err = self.device.newComputePipelineStateWithFunction_error_(kernel, None)
        if err:
            raise RuntimeError(f"❌ Pipeline error: {err}")

        # Create buffers for inputs and an output
        buffers = []
        n_items = None
        for data in inputs:
            buf, length = self._make_buffer(data)
            buffers.append(buf)
            if n_items is None:
                n_items = length
            elif n_items != length:
                raise ValueError("All input arrays must have same length.")
        out_buf = self.device.newBufferWithLength_options_(n_items * 4, Metal.MTLResourceStorageModeShared)
        buffers.append(out_buf)

        # Encode commands
        cmd_buf = self.queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pso)

        # Bind buffers
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        # Thread layout
        threads = Metal.MTLSizeMake(n_items, 1, 1)
        threadgroup = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(threads, threadgroup)
        encoder.endEncoding()

        # Execute
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Read results back
        output_data = (ctypes.c_float * n_items).from_buffer(out_buf.contents().as_buffer(n_items * 4))
        return list(output_data)



source = """


#include <metal_stdlib>
using namespace metal;


// Elementwise addition
kernel void add(device float *a [[ buffer(0) ]],
                device float *b [[ buffer(1) ]],
                device float *out [[ buffer(2) ]],
                uint id [[ thread_position_in_grid ]]) {
    out[id] = a[id] + b[id];
}


// Elementwise subtraction 

kernel void sub(device float *a [[ buffer(0) ]],
                device float *b [[ buffer(1) ]],
                device float *out [[ buffer(2) ]],
                uint id [[ thread_position_in_grid ]]) {
    out[id] = a[id] - b[id];
}

// Elementwise multiplication
kernel void multiply(device float *a [[ buffer(0) ]],
                     device float *b [[ buffer(1) ]],
                     device float *out [[ buffer(2) ]],
                     uint id [[ thread_position_in_grid ]]) {
    out[id] = a[id] * b[id];
}


// Elementwise division 
kernel void division(device float *a [[ buffer(0) ]],
                     device float *b [[ buffer(1) ]],
                     device float *out [[ buffer(2) ]],
                     uint id [[ thread_position_in_grid ]]) {
    out[id] = a[id] / b[id];
}

// Elementwise negate
kernel void negate(device float *a [[ buffer(0) ]], 
                 device float *out [[buffer(1) ]], 
                 uint id [[thread_position_in_grid]]){
                 
    out[id] = -a[id];                  
                 
}

// Elementwise absolute value
kernel void  abs_kernel(device float *a [[ buffer(0) ]], 
                 device float *out [[buffer(1) ]], 
                 uint id [[thread_position_in_grid]]){
                 
    out[id] = abs(a[id]);                  
                 
}

// Elementwise power function 
kernel void pow_kernel(device float *a [[ buffer(0) ]],
                     constant float &c [[buffer(1) ]],
                     device float *out [[ buffer(2) ]],
                     uint id [[ thread_position_in_grid ]]) {
                     
    out[id] = pow(a[id], c);
}


// Elementwise square root
kernel void kernel_sqrt(device float *a [[ buffer(0) ]],
                     constant float &c [[buffer(1) ]],
                     device float *out [[ buffer(2) ]],
                     uint id [[ thread_position_in_grid ]]) {
                     
    out[id] = pow(a[id], 0.5); 
}


// Elementwise square
kernel void square(device float *a [[ buffer(0) ]],
                   device float *out [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]]) {
    out[id] = a[id] * a[id];
}


// Elementwise ReLU (Rectified Linear Unit)
kernel void relu(device float *a [[ buffer(0) ]],
                 device float *out [[ buffer(1) ]],
                 uint id [[ thread_position_in_grid ]]) {
    out[id] = max(0.0, a[id]);
}
"""


gpu = MetalGPU()


a = [1.0, 2.0, 3.0, 4.0]
b = [10.0, 20.0, 30.0, 40.0] 


out = gpu.run_kernel(source, "add", [a, b])
print("Add:", out)



out_mul = gpu.run_kernel(source, "multiply", [a, b])
print("Multiply:", out_mul)

out_sq = gpu.run_kernel(source, "square", [a])
print("Square:", out_sq)

out_relu = gpu.run_kernel(source, "relu", [a])
print("ReLU:", out_relu)

out_neg = gpu.run_kernel(source, "negate", [a])
print("Negated:", out_neg)
