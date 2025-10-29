"""
"Hello world" example of using Metal from Python.
This script can be run from the command line just like any other Python file. No
need for Xcode or any other IDE. Just make sure you have the latest version of
Python 3 installed, along with the PyObjC and pyobjc-framework-Metal packages.
"""

import Metal
import ctypes
import random
from math import log

#####################################
# 1. Setup the Metal kernel itself.
#####################################

# Define a Metal kernel function
kernel_source = """
#include <metal_stdlib>
using namespace metal;
kernel void log_kernel(device float *in  [[ buffer(0) ]],
                       device float *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = log(in[id]);
}
"""

# Create a Metal device, library, and kernel function
device = Metal.MTLCreateSystemDefaultDevice()
library = device.newLibraryWithSource_options_error_(kernel_source, None, None)[0]
kernel_function = library.newFunctionWithName_("log_kernel")

#########################################
# 2. Setup the input and output buffers.
#########################################

# Create input and output buffers
array_length = 1024
buffer_length = array_length * 4  # 4 bytes per float
input_buffer = device.newBufferWithLength_options_(buffer_length, Metal.MTLResourceStorageModeShared)
output_buffer = device.newBufferWithLength_options_(buffer_length, Metal.MTLResourceStorageModeShared)

# Populate input buffer with random values
input_list = [random.uniform(0.0, 1.0) for _ in range(array_length)]  # Create list of random numbers
input_array = (ctypes.c_float * array_length).from_buffer(input_buffer.contents().as_buffer(buffer_length))  # Map the Metal buffer to a Python array
input_array[:] = input_list  # Populate the arrays with random values

#####################################
# 3. Call the Metal kernel function.
#####################################

# Create a command queue and command buffer
commandQueue = device.newCommandQueue()
commandBuffer = commandQueue.commandBuffer()

# Set the kernel function and buffers
pso = device.newComputePipelineStateWithFunction_error_(kernel_function, None)[0]
computeEncoder = commandBuffer.computeCommandEncoder()
computeEncoder.setComputePipelineState_(pso)
computeEncoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
computeEncoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)

# Define threadgroup size
threadsPerThreadgroup = Metal.MTLSizeMake(1024, 1, 1)
threadgroupSize = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)

# Dispatch the kernel
computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup, threadgroupSize)
computeEncoder.endEncoding()

# Commit the command buffer
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

##################################
# 4. Check the output is correct.
##################################

# Map the Metal buffer to a Python array
output_data = (ctypes.c_float * array_length).from_buffer(output_buffer.contents().as_buffer(buffer_length))
output_list = list(output_data)

# Check the outputs are correct
output_python = [log(x) for x in input_list]
assert all([abs(a - b) < 1e-5 for a, b in zip(output_list, output_python)]), "❌ Output does not match reference!"
print("✅ Reference matches output!")

