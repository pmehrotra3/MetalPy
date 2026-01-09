#include <metal_stdlib>
using namespace metal;

// Category 1: Elementwise Math

// 1) Elementwise addition
kernel void add_kernel(device const float *a   [[ buffer(0) ]],
                device const float *b   [[ buffer(1) ]],
                device float *out       [[ buffer(2) ]],
                constant uint &N        [[ buffer(3) ]],
                uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] + b[id];
    }
}

// 2) Elementwise subtraction
kernel void sub_kernel(device const float *a   [[ buffer(0) ]],
                device const float *b   [[ buffer(1) ]],
                device float *out       [[ buffer(2) ]],
                constant uint &N        [[ buffer(3) ]],
                uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] - b[id];
    }
}

// 3) Elementwise multiplication
kernel void multiply_kernel(device const float *a   [[ buffer(0) ]],
                     device const float *b   [[ buffer(1) ]],
                     device float *out       [[ buffer(2) ]],
                     constant uint &N        [[ buffer(3) ]],
                     uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] * b[id];
    }
}


// 4) Elementwise division
kernel void division_kernel(device const float *a   [[ buffer(0) ]],
                     device const float *b   [[ buffer(1) ]],
                     device float *out       [[ buffer(2) ]],
                     constant uint &N        [[ buffer(3) ]],
                     uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] / b[id];   // Let IEEE-754 handle all special cases
    }
}

// 5) Elementwise negate
kernel void negate_kernel(device const float *a   [[ buffer(0) ]],
                   device float *out       [[ buffer(1) ]],
                   constant uint &N        [[ buffer(2) ]],
                   uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = -a[id];
    }
}

// 6) Elementwise absolute value
kernel void abs_kernel(device const float *a   [[ buffer(0) ]],
                       device float *out       [[ buffer(1) ]],
                       constant uint &N        [[ buffer(2) ]],
                       uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = fabs(a[id]);   // use fabs for float
    }
}

// 7) Elementwise power function
kernel void pow_kernel(device const float *a   [[ buffer(0) ]],
                       constant float &c       [[ buffer(1) ]],
                       device float *out       [[ buffer(2) ]],
                       constant uint &N        [[ buffer(3) ]],
                       uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = pow(a[id], c);
    }
}

// 8) Elementwise square
kernel void square_kernel(device const float *a   [[ buffer(0) ]],
                   device float *out       [[ buffer(1) ]],
                   constant uint &N        [[ buffer(2) ]],
                   uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] * a[id];
    }
}


// 9) Elementwise square root
kernel void sqrt_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = sqrt(a[id]); 
    }
}

// 10) Elementwise exponent
kernel void exp_kernel(device const float *a   [[ buffer(0) ]],
                       device float *out       [[ buffer(1) ]],
                       constant uint &N        [[ buffer(2) ]],
                       uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = exp(a[id]);
    }
}

// 11) Elementwise log
kernel void log_kernel(device const float *a   [[ buffer(0) ]],
                       device float *out       [[ buffer(1) ]],
                       constant uint &N        [[ buffer(2) ]],
                       uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = log(a[id]); 
    }
}



// 12) Elementwise sin
kernel void sin_kernel(device const float *a   [[ buffer(0) ]],
                       device float *out       [[ buffer(1) ]],
                       constant uint &N        [[ buffer(2) ]],
                       uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = sin(a[id]);
    }
}

// 13) Elementwise cos
kernel void cos_kernel(device const float *a   [[ buffer(0) ]],
                       device float *out       [[ buffer(1) ]],
                       constant uint &N        [[ buffer(2) ]],
                       uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = cos(a[id]);
    }
}

// 14) Elementwise tan
kernel void tan_kernel(device const float *a   [[ buffer(0) ]],
                       device float *out       [[ buffer(1) ]],
                       constant uint &N        [[ buffer(2) ]],
                       uint id                 [[ thread_position_in_grid ]]) 
{
    if (id < N) {
        out[id] = tan(a[id]);
    }
}


// 15) Elementwise asin
kernel void asin_kernel(device const float *a   [[ buffer(0) ]],
                          device float *out       [[ buffer(1) ]],
                          constant uint &N        [[ buffer(2) ]],
                          uint id                 [[ thread_position_in_grid ]]) 
{
    if (id < N) {
        out[id] = asin(a[id]);  
    }
}


// 16) Elementwise atan2
kernel void atan2_kernel(device const float *y   [[ buffer(0) ]],
                         device const float *x   [[ buffer(1) ]],
                         device float *out       [[ buffer(2) ]],
                         constant uint &N        [[ buffer(3) ]],
                         uint id                 [[ thread_position_in_grid ]]) 
{
    if (id < N) {
        out[id] = atan2(y[id], x[id]);
    }
}


// 17) Elementwise floor
kernel void floor_kernel(device const float *a   [[ buffer(0) ]],
                         device float *out       [[ buffer(1) ]],
                         constant uint &N        [[ buffer(2) ]],
                         uint id                 [[ thread_position_in_grid ]]) 
{
    if (id < N) {
        out[id] = floor(a[id]);
    }
}

// 18) Elementwise ceil
kernel void ceil_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) 
{
    if (id < N) {
        out[id] = ceil(a[id]);
    }
}

// 19)  Elementwise sign
kernel void sign_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) 
{
    if (id < N) {

        if (a[id] > 0.0f) {
            out[id] = 1.0f;
        } 
        else if (a[id] < 0.0f) {
            out[id] = -1.0f;
        }
        else {
            out[id] = 0.0f;
        }
    }
}

// 20) Elementwise clipping
kernel void clip_kernel(device const float *a    [[ buffer(0) ]],
                        device float *out        [[ buffer(1) ]],
                        constant float &low      [[ buffer(2) ]],
                        constant float &high     [[ buffer(3) ]],
                        constant uint  &N        [[ buffer(4) ]],
                        uint id                  [[ thread_position_in_grid ]])
{
    if (id < N) {
        out[id] = clamp(a[id], low, high);
    }
}

// 21) Elementwise round
kernel void kernel_round(device const float *a   [[ buffer(0) ]],
                         device float *out       [[ buffer(1) ]],
                         constant uint &N        [[ buffer(2) ]],
                         uint id                 [[ thread_position_in_grid ]])
{
    if (id < N) {
        out[id] = round(a[id]);
    }
}

// Category 2: Activations


// 22) Elementwise ReLU 
kernel void relu_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = fmax(a[id], 0.0f);
    }
}


// 23) Elementwise Leaky ReLU
kernel void leaky_relu_kernel(device const float *a   [[ buffer(0) ]],
                              constant float &alpha    [[ buffer(1) ]],
                              device float *out        [[ buffer(2) ]],
                              constant uint &N         [[ buffer(3) ]],
                              uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {
        float x = a[id];

        if (x > 0.0f) {
            out[id] = x;               
        } else {
            out[id] = alpha * x;       
        }
    }
}


// 24) Elementwise Sigmoid
kernel void sigmoid_kernel(device const float *a   [[ buffer(0) ]],
                           device float *out        [[ buffer(1) ]],
                           constant uint &N         [[ buffer(2) ]],
                           uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = 1.0f / (1.0f + exp(-a[id]));
    }
}

// 25) Elementwise tanh 
kernel void tanh_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = tanh(a[id]);
    }
}




// 26) Elementwise softplus
kernel void softplus_kernel(device const float *a   [[ buffer(0) ]],
                            device float *out        [[ buffer(1) ]],
                            constant uint &N         [[ buffer(2) ]],
                            uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = log(1.0f + exp(a[id]));
    }
}




// 27) Elementwise swish
kernel void swish_kernel(device const float *a   [[ buffer(0) ]],
                         device float *out        [[ buffer(1) ]],
                         constant uint &N         [[ buffer(2) ]],
                         uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] * (1.0f / (1.0f + exp(-a[id])));
    }
}



// 28) Elementwise GELU
kernel void gelu_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out        [[ buffer(1) ]],
                        constant uint &N         [[ buffer(2) ]],
                        uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {

        out[id] =
            0.5f * a[id] *
            (1.0f + tanh(
                sqrt(2.0f / 3.14159265358979f) *
                (a[id] + 0.044715f * a[id] * a[id] * a[id])
            ));
    }
}

// Category 3: Basic Reductions 

// 29) Sum reduction 
kernel void sum_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],   // threads-per-threadgroup
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              threadgroup float *scratch)
{
    // Load element into shared memory

    float val = 0.0f;

    if (global_id < N) {

        val = a[global_id];
    }

    scratch[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Performing reduction

    uint stride = TPT / 4;

    while (stride > 0) {

        if (tid < stride) {

            scratch[tid] += scratch[tid+stride] + scratch[tid+2*stride] + scratch[tid+3*stride]; 

        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 4;
    }

    // First thread writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }


}


// 29B) Simple GPU sum kernel for small N (single-threaded execution)
kernel void small_sum_kernel(device const float *a  [[ buffer(0) ]],
                             device float *out      [[ buffer(1) ]],
                             constant uint &N       [[ buffer(2) ]],
                             uint tid               [[ thread_position_in_grid ]])
{
    if (tid != 0) return;

    float acc = 0.0f;
    for (uint i = 0; i < N; ++i) {
        acc += a[i];
    }

    out[0] = acc;
}


// 30) Product reduction 
kernel void product_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],   // threads-per-threadgroup
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              threadgroup float *scratch)
{

    // Load element into shared memory

    float val = 1.0f;

    if (global_id < N) {

        val = a[global_id];
    }

    scratch[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Performing reduction

    uint stride = TPT / 2;

    while (stride > 0) {

        if (tid < stride) {
            scratch[tid] *= scratch[tid + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 2;
    }

    // First thread writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}


// 31) Max reduction 
kernel void max_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],   // threads-per-threadgroup
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              threadgroup float *scratch)
{

    // Load element into shared memory

    float val = -INFINITY;

    if (global_id < N) {

        val = a[global_id];
    }

    scratch[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Performing reduction

    uint stride = TPT / 2;

    while (stride > 0) {

        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 2;
    }

    // First thread writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}


// 32) Min reduction 
kernel void min_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],   // threads-per-threadgroup
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              threadgroup float *scratch)
{

    // Load element into shared memory

    float val = INFINITY;

    if (global_id < N) {

        val = a[global_id];
    }

    scratch[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Performing reduction

    uint stride = TPT / 2;

    while (stride > 0) {

        if (tid < stride) {
            scratch[tid] = min(scratch[tid], scratch[tid + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 2;
    }

    // First thread writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}

// 33) Argmax reduction 
kernel void argmax_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device uint *out             [[ buffer(1) ]],   // <-- FIXED TYPE
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],   // threads-per-threadgroup
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              threadgroup uint *scratch)
{

    // Load element into shared memory

    uint val = 0;

    if (global_id < N) {
        val = global_id;
    }

    scratch[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Performing reduction

    uint stride = TPT / 2;

    while (stride > 0) {

        if (tid < stride) {

            if (a[scratch[tid]] < a[scratch[tid + stride]]) {

                scratch[tid] = scratch[tid + stride];

            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 2;
    }

    // First thread writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}


// 34) Argmin reduction 
kernel void argmin_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device uint *out             [[ buffer(1) ]],   // <-- FIXED TYPE
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],   // threads-per-threadgroup
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              threadgroup uint *scratch)
{

    // Load element into shared memory

    uint val = 0;

    if (global_id < N) {
        val = global_id;
    }

    scratch[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Performing reduction

    uint stride = TPT / 2;

    while (stride > 0) {

        if (tid < stride) {

            if (a[scratch[tid]] > a[scratch[tid + stride]]) {

                scratch[tid] = scratch[tid + stride];

            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 2;
    }

    // First thread writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}


// Category 4: Broadcasting and Indexing 

// broadcast_add: out[i] = a[i] + b[i % M]
kernel void broadcast_add_kernel(device const float *a   [[ buffer(0) ]],
                                 device const float *b   [[ buffer(1) ]],
                                 device float *out       [[ buffer(2) ]],
                                 constant uint &N        [[ buffer(3) ]],  // length of a
                                 constant uint &M        [[ buffer(4) ]],  // length of b
                                 uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        uint j = id % M;     // broadcast index
        out[id] = a[id] + b[j];
    }
}





// Slicing kernel: copy a[start:end] into out
kernel void slice_kernel(device const float *a     [[ buffer(0) ]],
                         device float *out         [[ buffer(1) ]],
                         constant uint &start      [[ buffer(2) ]],
                         constant uint &end        [[ buffer(3) ]],
                         uint id                   [[ thread_position_in_grid ]]) {

    uint length = end - start;

    if (id < length) {
        out[id] = a[start + id];
    }
}




// Category - 4 Linear Algebra 

kernel void dot_kernel(device const float *a   [[ buffer(0) ]],
                     device const float *b   [[ buffer(1) ]],
                     device float *out       [[ buffer(2) ]],
                     constant uint &N        [[ buffer(3) ]],
                     uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] * b[id];
    }
}


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



