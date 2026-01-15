#include <metal_stdlib>
using namespace metal;


// Category 1: Elementwise Operations


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
        out[id] = a[id] / b[id];   // Let IEEE-754 handle all edge cases.
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
        out[id] = fabs(a[id]);   // fabs() is the Metal standard-library intrinsic for float absolute value.
    }
}

// 7) Elementwise power function
kernel void pow_kernel(device const float *a   [[ buffer(0) ]],
                       device float *out       [[ buffer(1) ]],
                       constant float &c       [[ buffer(2) ]],
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
                        constant float &low      [[ buffer(1) ]],
                        constant float &high     [[ buffer(2) ]],
                        device float *out        [[ buffer(3) ]],
                        constant uint  &N        [[ buffer(4) ]],
                        uint id                  [[ thread_position_in_grid ]])
{
    if (id < N) {
        out[id] = clamp(a[id], low, high);
    }
}

// 21) Elementwise round
kernel void round_kernel(device const float *a   [[ buffer(0) ]],
                         device float *out       [[ buffer(1) ]],
                         constant uint &N        [[ buffer(2) ]],
                         uint id                 [[ thread_position_in_grid ]])
{
    if (id < N) {
        out[id] = round(a[id]);
    }
}

// 22) Elementwise broadcast add
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

// 23) Elementwise broadcast multiply
kernel void broadcast_multiply_kernel(device const float *a   [[ buffer(0) ]],
                                 device const float *b   [[ buffer(1) ]],
                                 device float *out       [[ buffer(2) ]],
                                 constant uint &N        [[ buffer(3) ]],  // length of a
                                 constant uint &M        [[ buffer(4) ]],  // length of b
                                 uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        uint j = id % M;     // broadcast index
        out[id] = a[id] * b[j];
    }
}


// Category 2: Activation operations


// 24) Elementwise ReLU 
kernel void relu_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = fmax(a[id], 0.0f);      // fmax() is a Metal intrinsic that selects the larger of two floats.
    }
}


// 25) Elementwise Leaky ReLU
kernel void leaky_relu_kernel(device const float *a   [[ buffer(0) ]],
                              device float *out        [[ buffer(1) ]],
                              constant float &alpha    [[ buffer(2) ]],
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

// 26) Elementwise Sigmoid
kernel void sigmoid_kernel(device const float *a   [[ buffer(0) ]],
                           device float *out        [[ buffer(1) ]],
                           constant uint &N         [[ buffer(2) ]],
                           uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = 1.0f / (1.0f + exp(-a[id]));
    }
}

// 27) Elementwise tanh 
kernel void tanh_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = tanh(a[id]);
    }
}

// 28) Elementwise softplus
kernel void softplus_kernel(device const float *a   [[ buffer(0) ]],
                            device float *out        [[ buffer(1) ]],
                            constant uint &N         [[ buffer(2) ]],
                            uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = log(1.0f + exp(a[id]));
    }
}

// 29) Elementwise swish
kernel void swish_kernel(device const float *a   [[ buffer(0) ]],
                         device float *out        [[ buffer(1) ]],
                         constant uint &N         [[ buffer(2) ]],
                         uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = a[id] * (1.0f / (1.0f + exp(-a[id])));
    }
}

// 30) Elementwise GELU
kernel void gelu_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out        [[ buffer(1) ]],
                        constant uint &N         [[ buffer(2) ]],
                        uint id                  [[ thread_position_in_grid ]]) {

    if (id < N) {

        float x = a[id];

        float c = 0.7978845608028654f;   // sqrt(2/pi)

        float t = c * (x + 0.044715f * x * x * x);

        out[id] = 0.5f * x * (1.0f + tanh(t));
    }
}

// Category 3: Basic Reductions 


// 31) Sum reduction: Initial pass reduction kernel
kernel void sum_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint num_threadgroups        [[ threadgroups_per_grid ]])
{

    threadgroup float scratch[256];

    // global thread index and total threads in grid
    uint global_id = gid * TPT + tid;
    uint grid_size = num_threadgroups * TPT;

    // Each thread accumulates multiple elements

    float sum = 0.0f;
    
    for (uint i = global_id; i < N; i += grid_size) {

        sum += a[i];
    }
    
    scratch[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction

    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {

            scratch[tid] += scratch[tid + stride];

        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

     // First thread of each threadgroup writes per-group result

    if (tid == 0) {

        out[gid] = scratch[0];
    }
}


// 32) Sum reduction: Final pass reduction kernel
kernel void sum_reduce_final_kernel(device const float *a  [[ buffer(0) ]],
                             device float *out      [[ buffer(1) ]],
                             constant uint &N       [[ buffer(2) ]],
                             uint tid               [[ thread_position_in_grid ]])
{
    if (tid != 0) return;

    float val = 0.0f;

    for (uint i = 0; i < N; ++i) {

        val += a[i];

    }

    out[0] = val;
}


// 33) Product reduction: Initial pass reduction kernel
kernel void product_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],  
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint num_threadgroups        [[ threadgroups_per_grid ]])
{
    threadgroup float scratch[256];

    // global thread index and total threads in grid
    uint global_id = gid * TPT + tid;
    uint grid_size = num_threadgroups * TPT;

     // Each thread accumulates multiple elements

    float product = 1.0f;

    for (uint i = global_id; i <  N; i += grid_size){
        product *= a[i]; 
    }

    scratch[tid] = product;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction 

     for (uint stride = TPT / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            scratch[tid] *= scratch[tid + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread of each threadgroup writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}


// 34) Product reduction: Final pass reduction kernel
kernel void product_reduce_final_kernel(device const float *a  [[ buffer(0) ]],
                             device float *out      [[ buffer(1) ]],
                             constant uint &N       [[ buffer(2) ]],
                             uint tid               [[ thread_position_in_grid ]])
{
    if (tid != 0) return;

    float val = 1.0f;

    for (uint i = 0; i < N; ++i) {
        val *= a[i];

    }

    out[0] = val;
}


// 35) Max reduction - Initial pass reduction kernel
kernel void max_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint num_threadgroups        [[ threadgroups_per_grid ]])
{
    threadgroup float scratch[256];

    // global thread index and total threads in grid
    uint global_id = gid * TPT + tid;
    uint grid_size = num_threadgroups * TPT;

    // Each thread accumulates multiple elements and finds their max.

    float best = -INFINITY;
    
    for (uint i = global_id; i < N; i += grid_size) {
        best = max(best, a[i]);
    }
    
    scratch[tid] = best;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction

    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread of each threadgroup writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}

// 36) Max reduction: Final pass reduction kernel

kernel void max_reduce_final_kernel(device const float *a  [[ buffer(0) ]],
                                    device float *out      [[ buffer(1) ]],
                                    constant uint &N       [[ buffer(2) ]],
                                    uint tid               [[ thread_position_in_grid ]])
{
    if (tid != 0) return;

    float best = -INFINITY;

    for (uint i = 0; i < N; ++i) {
        best = max(best, a[i]);
    }

    out[0] = best;
}


// 37) Min reduction - Initial pass reduction kernel
kernel void min_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint num_threadgroups        [[ threadgroups_per_grid ]])
{
    threadgroup float scratch[256];

    // global thread index and total threads in grid
    uint global_id = gid * TPT + tid;
    uint grid_size = num_threadgroups * TPT;

    // Each thread accumulates multiple elements and finds their min.

    float best = INFINITY;
    
    for (uint i = global_id; i < N; i += grid_size) {
        best = min(best, a[i]);
    }
    
    scratch[tid] = best;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction

    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            scratch[tid] = min(scratch[tid], scratch[tid + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread of each threadgroup writes per-group result

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}

// 38)  Min reduction: Final pass reduction kernel
kernel void min_reduce_final_kernel(device const float *a  [[ buffer(0) ]],
                                    device float *out      [[ buffer(1) ]],
                                    constant uint &N       [[ buffer(2) ]],
                                    uint tid               [[ thread_position_in_grid ]])
{
    if (tid != 0) return;

    float best = INFINITY;

    for (uint i = 0; i < N; ++i) {
        best = min(best, a[i]);
    }

    out[0] = best;
}


// 39) Argmax reduction - Initial pass reduction kernel
kernel void argmax_reduce_kernel(device const float *a        [[ buffer(0) ]],
                                 device uint  *out_idx        [[ buffer(1) ]],
                                 device float *out_val        [[ buffer(2) ]],
                                 constant uint &N             [[ buffer(3) ]],
                                 constant uint &TPT           [[ buffer(4) ]],
                                 uint tid                     [[ thread_position_in_threadgroup ]],
                                 uint gid                     [[ threadgroup_position_in_grid ]],
                                 uint num_threadgroups        [[ threadgroups_per_grid ]])
{

    threadgroup uint scratch_idx[256];
    threadgroup float scratch_val[256];

     // global thread index and total threads in grid
    uint global_id = gid * TPT + tid;
    uint grid_size = num_threadgroups * TPT;
    
    // Each thread scans multiple elements and tracks local argmax.

    uint  best_idx = 0;
    float best_val = -INFINITY;

    for (uint i = global_id; i < N; i += grid_size) {

        float v = a[i];

        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    scratch_idx[tid] = best_idx;
    scratch_val[tid] = best_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction on (value, index) pairs

    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {

            float other_val = scratch_val[tid + stride];

            if (other_val > scratch_val[tid]) {

                scratch_val[tid] = other_val;
                scratch_idx[tid] = scratch_idx[tid + stride];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread of each threadgroup writes per-group result

    if (tid == 0) {
        out_idx[gid] = scratch_idx[0];
        out_val[gid] = scratch_val[0];
    }
}

// 40) Argmax reduction - Final pass reduction kernel
kernel void argmax_reduce_final_kernel(device const uint  *in_idx  [[ buffer(0) ]],
                                       device const float *in_val  [[ buffer(1) ]],
                                       device uint        *out_idx [[ buffer(2) ]],
                                       device float       *out_val [[ buffer(3) ]],
                                       constant uint      &N       [[ buffer(4) ]],
                                       uint tid                    [[ thread_position_in_grid ]])
{
    if (tid != 0) return;

    uint  best_idx = 0;
    float best_val = -INFINITY;

    for (uint i = 0; i < N; ++i) {

        float v = in_val[i];

        if (v > best_val) {
            best_val = v;
            best_idx = in_idx[i];
        }
    }

    out_idx[0] = best_idx;
    out_val[0] = best_val;
}


// 41) Argmin reduction - Initial pass reduction kernel
kernel void argmin_reduce_kernel(device const float *a        [[ buffer(0) ]],
                                 device uint  *out_idx        [[ buffer(1) ]],
                                 device float *out_val        [[ buffer(2) ]],
                                 constant uint &N             [[ buffer(3) ]],
                                 constant uint &TPT           [[ buffer(4) ]],
                                 uint tid                     [[ thread_position_in_threadgroup ]],
                                 uint gid                     [[ threadgroup_position_in_grid ]],
                                 uint num_threadgroups        [[ threadgroups_per_grid ]])
{

    threadgroup uint scratch_idx[256];
    threadgroup float scratch_val[256];

    // global thread index and total threads in grid
    uint global_id = gid * TPT + tid;
    uint grid_size = num_threadgroups * TPT;
    
    // Each thread scans multiple elements and tracks local argmin.

    uint  best_idx = 0;
    float best_val = INFINITY;

    for (uint i = global_id; i < N; i += grid_size) {

        float v = a[i];

        if (v < best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    scratch_idx[tid] = best_idx;
    scratch_val[tid] = best_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction on (value, index) pairs

    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {

            float other_val = scratch_val[tid + stride];

            if (other_val < scratch_val[tid]) {
                scratch_val[tid] = other_val;
                scratch_idx[tid] = scratch_idx[tid + stride];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread of each threadgroup writes per-group result

    if (tid == 0) {
        out_idx[gid] = scratch_idx[0];
        out_val[gid] = scratch_val[0];
    }
}


// 42) Argmin reduction - Final pass reduction kernel
kernel void argmin_reduce_final_kernel(device const uint  *in_idx  [[ buffer(0) ]],
                                       device const float *in_val  [[ buffer(1) ]],
                                       device uint        *out_idx [[ buffer(2) ]],
                                       device float       *out_val [[ buffer(3) ]],
                                       constant uint      &N       [[ buffer(4) ]],
                                       uint tid                    [[ thread_position_in_grid ]])
{
    if (tid != 0) return;

    uint  best_idx = 0;
    float best_val = INFINITY;

    for (uint i = 0; i < N; ++i) {

        float v = in_val[i];

        if (v < best_val) {
            best_val = v;
            best_idx = in_idx[i];
        }
    }

    out_idx[0] = best_idx;
    out_val[0] = best_val;
}


// Category 4: Linear Algebra operations


// 43) Hadamard product 
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


// 44) Row-wise sum of matrix (compute sum of each row)
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

// 45) Column-wise sum of matrix (compute sum of each column)
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

// 46) Row-wise scaling (multiply each row by its scalar)
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

// 47) Column-wise scaling (multiply each column by its scalar)
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



// 48) Matrix transpose
kernel void transpose_kernel(device const float *A   [[ buffer(0) ]],
                             device float *B         [[ buffer(1) ]],
                             constant uint &rows     [[ buffer(2) ]],
                             constant uint &cols     [[ buffer(3) ]],
                             uint2 id                [[ thread_position_in_grid ]]) {
    
    uint row = id.y;
    uint col = id.x;

    if (row < rows && col < cols) {

        B[col * rows + row] = A[row * cols + col];
    }
}


// 49) Matrix multiplication (A * B = C)
// A: M x K, B: K x N, C: M x N  (row-major)
kernel void matmul_kernel(device const float *A    [[ buffer(0) ]],
                          device const float *B    [[ buffer(1) ]],
                          device float *C          [[ buffer(2) ]],
                          constant uint &M         [[ buffer(3) ]],  // rows of A
                          constant uint &N         [[ buffer(4) ]],  // cols of B
                          constant uint &K         [[ buffer(5) ]],  // cols of A / rows of B
                          uint2 id                 [[ thread_position_in_grid ]]) {

    uint row = id.y;
    uint col = id.x;

    // Bounds check: C is M x N

    if (row >= M || col >= N) {
        return;
    }

    float val = 0.0f;

    // Dot product of row 'row' of A and column 'col' of B

    for (uint k = 0; k < K; ++k) {

        float a = A[row * K + k];  // A[row, k]
        float b = B[k * N + col];  // B[k, col]
        val += a * b;
    }

    C[row * N + col] = val;        // C[row, col]
}


// Category 5: Miscellaneous operations

// 50) Slice - extract a contiguous slice from an array
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

// 51) Gather - index-based selection from an array
kernel void gather_kernel(device const float *source [[ buffer(0) ]],
                          device const uint  *index  [[ buffer(1) ]],
                          device float *out         [[ buffer(2) ]],
                          constant uint &N          [[ buffer(3) ]],
                          uint id                   [[ thread_position_in_grid ]])
{
    if (id < N) {
        uint j = index[id];
        out[id] = source[j];
    }
}
