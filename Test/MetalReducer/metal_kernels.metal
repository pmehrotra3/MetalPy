#include <metal_stdlib>
using namespace metal;

kernel void sum_reduce_kernel(device const float *a        [[ buffer(0) ]],
                              device float *out            [[ buffer(1) ]],
                              constant uint &N             [[ buffer(2) ]],
                              constant uint &TPT           [[ buffer(3) ]],
                              uint tid                     [[ thread_position_in_threadgroup ]],
                              uint gid                     [[ threadgroup_position_in_grid ]],
                              uint global_id               [[ thread_position_in_grid ]],
                              threadgroup float *scratch)
{
    float v = (global_id < N) ? a[global_id] : 0.0f;
    scratch[tid] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint stride = TPT / 2;

    while (stride > 0) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stride /= 2;
    }

    if (tid == 0) {
        out[gid] = scratch[0];
    }
}

