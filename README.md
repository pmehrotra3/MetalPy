# MetalPy: GPU-Accelerated Array Computing for Apple Silicon

**Author:** Pranav Mehrotra  
**Platform:** Apple Silicon (M1/M2/M3/M4 series Macs)

----------

## Overview

MetalPy is a GPU-accelerated numerical computing library that brings NumPy-like operations to Apple Silicon's Metal GPU framework. Built from scratch using Apple's Metal Shading Language, it provides 51 optimized GPU kernels for mathematical operations, neural network activations, reductions, and linear algebra—all accessible through a clean Python API.

This project explores the unique advantages of Apple's unified memory architecture, where CPU and GPU share the same memory pool, eliminating traditional data transfer bottlenecks and making GPU acceleration practical even for medium-sized datasets.

----------

## Motivation

When Apple launched its M-series chips in late 2020, they introduced a revolutionary unified memory architecture. Unlike traditional systems where transferring data between CPU and GPU creates significant overhead, Apple Silicon allows both processors to access the same memory seamlessly. This architectural advantage makes GPU computing viable for a much broader range of workloads.

However, the ecosystem for GPU-accelerated computing on Apple Silicon remains underdeveloped. Apple's own MLX library only emerged in late 2023, and many data scientists and researchers using MacBooks lack accessible tools for GPU acceleration of fundamental operations.

**MetalPy addresses this gap by:**

-   Providing a foundational GPU array library specifically designed for Apple Silicon
-   Demonstrating practical Metal compute pipeline implementation from Python
-   Offering 45+ GPU-accelerated operations including math primitives, neural network activations, parallel reductions, and linear algebra
-   Creating an extensible foundation for future machine learning operations

Beyond its practical utility, this project served as a deep exploration of GPU programming concepts: parallel algorithm design, memory coalescing, threadgroup synchronization, tree-based reductions, and performance optimization on modern hardware.

----------

## Technical Architecture

### Core Components

**1. metal_kernels.metal (51 GPU Kernels)**

Written in Metal Shading Language (MSL), this file implements all computational primitives:

-   **Elementwise Operations (23 kernels):** add, subtract, multiply, divide, negate, abs, pow, square, sqrt, exp, log, sin, cos, tan, asin, atan2, floor, ceil, round, sign, clip, broadcast_add, broadcast_multiply
-   **Activation Functions (7 kernels):** ReLU, Leaky ReLU, Sigmoid, Tanh, Softplus, Swish, GELU (approximation)
-   **Parallel Reductions (12 kernels):** Two-pass implementations for sum, product, max, min, argmax, argmin, plus derived operations like mean
-   **Linear Algebra (7 kernels):** Hadamard product, matrix multiplication, transpose, row/column sums, row/column scaling
-   **Utility Operations (2 kernels):** slice, gather (index-based selection)

Each kernel is optimized using techniques like grid-stride loops for arbitrary input sizes, threadgroup memory for efficient inter-thread communication, and parallel tree reduction patterns.

**2. MetalPy.py (Python Interface)**

Contains two primary classes:

-   **MetalArray:** Wraps Metal GPU buffers, providing array-like semantics with support for float32 and uint32 data types. Handles conversions between Python lists, typed arrays, and GPU memory.
-   **MetalPy:** Main API class that manages Metal device/queue initialization, kernel compilation and caching, and provides 13 generic launcher functions that dispatch different kernel categories. The launchers abstract away Metal's verbose command encoding, allowing high-level operations like:

python

```python
  gpu.add(a, b)
  gpu.matmul(A, B, M, N, K)
  gpu.argmax(array)
  gpu.gelu(activations)
```

**3. test.py (Comprehensive Test Suite)**

Validates all 46 operations by comparing GPU results against NumPy's CPU implementations using appropriate tolerance levels for floating-point arithmetic. Outputs timestamped results to ensure correctness and catch regressions.

----------

## Key Technical Challenges Solved

### 1. Parallel Reduction Design

Implementing operations like sum, max, and argmax on the GPU requires careful algorithm design. I implemented a two-pass parallel tree reduction:

-   **Pass 1:** Multiple threadgroups reduce their portions in parallel using shared threadgroup memory
-   **Pass 2:** Final serial reduction over partial results

This approach balances parallelism with communication overhead.

### 2. Grid-Stride Loop Pattern

To handle arrays of arbitrary size with limited GPU threads, kernels use grid-stride loops:

metal

```metal
for (uint i = global_id; i < N; i += grid_size) {
    // Process element i
}
```

This ensures full array coverage regardless of dispatch size.

### 3. Floating-Point Precision Management

GPU and CPU process operations in different orders, leading to different rounding errors. I implemented proper tolerance-based validation using relative error metrics rather than absolute comparisons, essential for operations like product reductions where accumulated errors become significant.

### 4. Generic Launcher Architecture

Rather than writing 51 separate dispatch functions, I designed 13 generic launchers that handle different kernel patterns (elementwise, reductions, 2D operations, etc.). This reduces code duplication while maintaining flexibility.

----------

## Installation & Usage

### Requirements

-   Apple Silicon Mac (M1/M2/M3/M4 series)
-   Xcode Command Line Tools: `xcode-select --install`
-   Python 3.x with NumPy

### Compilation

bash

```bash
# Compile Metal kernels to binary library
xcrun -sdk macosx metal -c metal_kernels.metal -o metal_kernels.air
xcrun -sdk macosx metallib metal_kernels.air -o metal_kernels.metallib
```

### Example Usage

python

```python
from MetalPy import MetalPy

gpu = MetalPy()

# Basic operations
a = gpu.array([1.0, 2.0, 3.0, 4.0])
b = gpu.array([5.0, 6.0, 7.0, 8.0])
c = gpu.add(a, b)
print(c.to_list())  # [6.0, 8.0, 10.0, 12.0]

# Activations
result = gpu.relu(gpu.array([-2.0, -1.0, 0.0, 1.0, 2.0]))

# Reductions
maximum = gpu.max(a)
total = gpu.sum(a)

# Linear algebra (2000×2000 matrix multiply)
A = gpu.array([...])  # 4M elements
B = gpu.array([...])
C = gpu.matmul(A, B, M=2000, N=2000, K=2000)
```

### Running Tests

bash

```bash
python3 test.py
```

Results are written to `output/test_results_[timestamp].txt`

----------

## Performance Characteristics

MetalPy leverages unified memory to minimize overhead, making GPU acceleration practical for:

-   Medium to large arrays (10K+ elements for elementwise ops)
-   Matrix operations (1000×1000 and larger)
-   Compute-intensive operations (transcendental functions, activations)

For very small arrays (<1000 elements), CPU operations may be faster due to kernel launch overhead. The library is designed for scenarios where computational intensity justifies GPU dispatch.

----------

## What I Learned

This project required mastering several advanced concepts:

**GPU Computing Fundamentals:**

-   Metal compute pipeline architecture and command buffer management
-   Threadgroup memory and synchronization primitives
-   Occupancy optimization and memory coalescing
-   Kernel dispatch strategies and grid configuration

**Parallel Algorithms:**

-   Tree-based parallel reductions with logarithmic depth
-   Work-efficient scanning and prefix sum patterns
-   Load balancing across heterogeneous workloads

**Numerical Computing:**

-   Floating-point arithmetic precision and error accumulation
-   Numerically stable algorithm variants (e.g., log-space products)
-   Validation strategies for approximate computations

**Software Engineering:**

-   API design balancing simplicity with control
-   Generic abstractions to reduce code duplication
-   Comprehensive testing and validation frameworks
-   Performance profiling and optimization

----------

## Future Directions

MetalPy provides a foundation for more advanced operations:

1.  **Extended Linear Algebra:** LU decomposition, QR factorization, eigenvalue solvers
2.  **Neural Network Primitives:** Convolution, pooling, attention mechanisms, dropout
3.  **Optimized Matrix Multiplication:** Tiled implementations using threadgroup memory for 10-100× speedup
4.  **Half-Precision Support:** float16 operations for 2× memory bandwidth
5.  **Automatic Differentiation:** Reverse-mode autodiff for gradient computation
6.  **Batched Operations:** Process multiple arrays simultaneously

----------

## Acknowledgments

This project was developed with assistance from AI tools (ChatGPT and Claude) for debugging, learning Metal API specifics, and understanding GPU programming patterns. All architectural decisions, algorithm implementations, and design choices were made independently through this learning process.

----------

## License

MIT License - See LICENSE file for details
