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

// 22) Broadcast_add
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

// 23) Broadcast_multiply
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






// Category 2: Activations


// 24) Elementwise ReLU 
kernel void relu_kernel(device const float *a   [[ buffer(0) ]],
                        device float *out       [[ buffer(1) ]],
                        constant uint &N        [[ buffer(2) ]],
                        uint id                 [[ thread_position_in_grid ]]) {

    if (id < N) {
        out[id] = fmax(a[id], 0.0f);
    }
}


// 25) Elementwise Leaky ReLU
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

        out[id] =
            0.5f * a[id] *
            (1.0f + tanh(
                sqrt(2.0f / 3.14159265358979f) *
                (a[id] + 0.044715f * a[id] * a[id] * a[id])
            ));
    }
}





// Category 3: Basic Reductions 

// 31) Sum reduction 

kernel void sum_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              uint num_threadgroups        [[ threadgroups_per_grid ]],
                              threadgroup float *scratch)
{
    // Each thread accumulates multiple elements
    float sum = 0.0f;
    uint grid_size = num_threadgroups * TPT;
    
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


// 32) Product reduction 
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
// 33) Max reduction - OPTIMIZED with grid-stride loop
kernel void max_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              uint num_threadgroups        [[ threadgroups_per_grid ]],
                              threadgroup float *scratch)
{
    // Each thread finds max of multiple elements
    float best = -INFINITY;
    uint grid_size = num_threadgroups * TPT;
    
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

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}


// 34) Min reduction - OPTIMIZED with grid-stride loop
kernel void min_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              uint num_threadgroups        [[ threadgroups_per_grid ]],
                              threadgroup float *scratch)
{
    // Each thread finds min of multiple elements
    float best = INFINITY;
    uint grid_size = num_threadgroups * TPT;
    
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

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}


// 35) Argmax reduction - Pass 1 (from original array)
kernel void argmax_reduce_kernel(device const float *a        [[ buffer(0) ]],
                                 device uint *out_idx         [[ buffer(1) ]],
                                 device float *out_val        [[ buffer(2) ]],
                                 constant uint &N             [[ buffer(3) ]],
                                 constant uint &TPT           [[ buffer(4) ]],
                                 uint tid                     [[ thread_position_in_threadgroup ]],
                                 uint gid                     [[ threadgroup_position_in_grid ]],
                                 uint global_id               [[ thread_position_in_grid ]],
                                 uint num_threadgroups        [[ threadgroups_per_grid ]],
                                 threadgroup uint *scratch_idx  [[ threadgroup(0) ]],
                                 threadgroup float *scratch_val [[ threadgroup(1) ]])
{
    // Grid-stride loop to find local max
    uint best_idx = 0;
    float best_val = -INFINITY;
    
    uint grid_size = num_threadgroups * TPT;
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

    // Tree reduction
    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (scratch_val[tid + stride] > scratch_val[tid]) {
                scratch_val[tid] = scratch_val[tid + stride];
                scratch_idx[tid] = scratch_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_idx[gid] = scratch_idx[0];
        out_val[gid] = scratch_val[0];
    }
}


// 35b) Argmax reduction - Pass 2 (from partial results)
kernel void argmax_reduce_final_kernel(device const uint *in_idx   [[ buffer(0) ]],
                                       device const float *in_val  [[ buffer(1) ]],
                                       device uint *out_idx        [[ buffer(2) ]],
                                       constant uint &N            [[ buffer(3) ]],
                                       constant uint &TPT          [[ buffer(4) ]],
                                       uint tid                    [[ thread_position_in_threadgroup ]],
                                       uint gid                    [[ threadgroup_position_in_grid ]],
                                       uint global_id              [[ thread_position_in_grid ]],
                                       threadgroup uint *scratch_idx  [[ threadgroup(0) ]],
                                       threadgroup float *scratch_val [[ threadgroup(1) ]])
{
    // Load from partial results
    uint best_idx = 0;
    float best_val = -INFINITY;
    
    if (global_id < N) {
        best_idx = in_idx[global_id];
        best_val = in_val[global_id];
    }
    
    scratch_idx[tid] = best_idx;
    scratch_val[tid] = best_val;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (scratch_val[tid + stride] > scratch_val[tid]) {
                scratch_val[tid] = scratch_val[tid + stride];
                scratch_idx[tid] = scratch_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_idx[gid] = scratch_idx[0];
    }
}


// 36) Argmin reduction - Pass 1 (from original array)
kernel void argmin_reduce_kernel(device const float *a        [[ buffer(0) ]],
                                 device uint *out_idx         [[ buffer(1) ]],
                                 device float *out_val        [[ buffer(2) ]],
                                 constant uint &N             [[ buffer(3) ]],
                                 constant uint &TPT           [[ buffer(4) ]],
                                 uint tid                     [[ thread_position_in_threadgroup ]],
                                 uint gid                     [[ threadgroup_position_in_grid ]],
                                 uint global_id               [[ thread_position_in_grid ]],
                                 uint num_threadgroups        [[ threadgroups_per_grid ]],
                                 threadgroup uint *scratch_idx  [[ threadgroup(0) ]],
                                 threadgroup float *scratch_val [[ threadgroup(1) ]])
{
    // Grid-stride loop to find local min
    uint best_idx = 0;
    float best_val = INFINITY;
    
    uint grid_size = num_threadgroups * TPT;
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

    // Tree reduction
    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (scratch_val[tid + stride] < scratch_val[tid]) {
                scratch_val[tid] = scratch_val[tid + stride];
                scratch_idx[tid] = scratch_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_idx[gid] = scratch_idx[0];
        out_val[gid] = scratch_val[0];
    }
}


// 36b) Argmin reduction - Pass 2 (from partial results)
kernel void argmin_reduce_final_kernel(device const uint *in_idx   [[ buffer(0) ]],
                                       device const float *in_val  [[ buffer(1) ]],
                                       device uint *out_idx        [[ buffer(2) ]],
                                       constant uint &N            [[ buffer(3) ]],
                                       constant uint &TPT          [[ buffer(4) ]],
                                       uint tid                    [[ thread_position_in_threadgroup ]],
                                       uint gid                    [[ threadgroup_position_in_grid ]],
                                       uint global_id              [[ thread_position_in_grid ]],
                                       threadgroup uint *scratch_idx  [[ threadgroup(0) ]],
                                       threadgroup float *scratch_val [[ threadgroup(1) ]])
{
    // Load from partial results
    uint best_idx = 0;
    float best_val = INFINITY;
    
    if (global_id < N) {
        best_idx = in_idx[global_id];
        best_val = in_val[global_id];
    }
    
    scratch_idx[tid] = best_idx;
    scratch_val[tid] = best_val;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = TPT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (scratch_val[tid + stride] < scratch_val[tid]) {
                scratch_val[tid] = scratch_val[tid + stride];
                scratch_idx[tid] = scratch_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_idx[gid] = scratch_idx[0];
    }
}

// Category 4: Linear Algebra (7 kernels)

// 1) Hadamard product (element-wise multiplication)
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

// 2) Tiled matrix multiplication (A * B = C)
kernel void tiled_matmul_kernel(device const float *A    [[ buffer(0) ]],
                                device const float *B    [[ buffer(1) ]],
                                device float *C          [[ buffer(2) ]],
                                constant uint &M         [[ buffer(3) ]],  // rows of A
                                constant uint &N         [[ buffer(4) ]],  // cols of B
                                constant uint &K         [[ buffer(5) ]],  // cols of A / rows of B
                                uint2 gid                [[ thread_position_in_grid ]],
                                uint2 tid                [[ thread_position_in_threadgroup ]],
                                uint2 tg_size            [[ threads_per_threadgroup ]]) 
{
    // Tile size (must match threadgroup size)
    constexpr uint TILE_SIZE = 16;
    
    // Shared memory for tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0f;
    
    // Number of tiles needed
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tile from A
        uint aRow = row;
        uint aCol = t * TILE_SIZE + tid.x;
        if (aRow < M && aCol < K) {
            tileA[tid.y][tid.x] = A[aRow * K + aCol];
        } else {
            tileA[tid.y][tid.x] = 0.0f;
        }
        
        // Load tile from B
        uint bRow = t * TILE_SIZE + tid.y;
        uint bCol = col;
        if (bRow < K && bCol < N) {
            tileB[tid.y][tid.x] = B[bRow * N + bCol];
        } else {
            tileB[tid.y][tid.x] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 3) Matrix transpose - with grid-stride for large matrices
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

// 4) Row sum - sum each row of matrix
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

// 5) Column sum - sum each column of matrix
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

// 6) Row scale - multiply each row by corresponding scalar
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

// 7) Column scale - multiply each column by corresponding scalar
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



// Category 5: Miscellaneous (2 kernels)

// 1) Slice - extract a contiguous slice from an array
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


// 2) Bitonic sort - parallel sorting algorithm
// Note: This is a single step of bitonic sort. Multiple passes are needed for complete sorting.
// The array length must be a power of 2.
kernel void bitonic_sort_kernel(device float *data        [[ buffer(0) ]],
                               constant uint &stage       [[ buffer(1) ]],
                               constant uint &step        [[ buffer(2) ]],
                               constant uint &N           [[ buffer(3) ]],
                               uint id                    [[ thread_position_in_grid ]]) 
{
    if (id >= N) return;
    
    // Calculate the partner index for comparison
    uint partner = id ^ step;
    
    // Only process if this thread's id is less than partner
    // (to avoid duplicate comparisons)
    if (partner > id) {
        float a = data[id];
        float b = data[partner];
        
        // Determine sort direction based on stage
        // If (id & stage) == 0, sort ascending; otherwise descending
        bool ascending = ((id & stage) == 0);
        
        // Swap if needed
        if ((ascending && a > b) || (!ascending && a < b)) {
            data[id] = b;
            data[partner] = a;
        }
    }
}

