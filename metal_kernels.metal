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
                   device float *out [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]]) {
    out[id] = -a[id];
}

// Elementwise absolute value
kernel void abs_kernel(device float *a [[ buffer(0) ]],
                       device float *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = abs(a[id]);
}

// Elementwise power function
kernel void pow_kernel(device float *a [[ buffer(0) ]],
                       constant float &c [[ buffer(1) ]],
                       device float *out [[ buffer(2) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = pow(a[id], c);
}

// Elementwise square root
kernel void kernel_sqrt(device float *a [[ buffer(0) ]],
                        constant float &c [[ buffer(1) ]],
                        device float *out [[ buffer(2) ]],
                        uint id [[ thread_position_in_grid ]]) {
    out[id] = pow(a[id], 0.5);
}

// Elementwise exponent
kernel void kernel_exponent(device float *a [[ buffer(0) ]],
                            device float *out [[ buffer(1) ]],
                            uint id [[ thread_position_in_grid ]]) {
    out[id] = exp(a[id]);
}

// Elementwise log
kernel void kernel_exponent(device float *a [[ buffer(0) ]],
                            device float *out [[ buffer(1) ]],
                            uint id [[ thread_position_in_grid ]]) {
    out[id] = log(a[id]);
}

// Elementwise square
kernel void square(device float *a [[ buffer(0) ]],
                   device float *out [[ buffer(1) ]],
                   uint id [[ thread_position_in_grid ]]) {
    out[id] = a[id] * a[id];
}

// Elementwise sin
kernel void kernel_sin(device float *a [[ buffer(0) ]],
                       device float *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = sin(a[id]);
}

// Elementwise cos
kernel void kernel_cos(device float *a [[ buffer(0) ]],
                       device float *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = cos(a[id]);
}

// Elementwise tan
kernel void kernel_tan(device float *a [[ buffer(0) ]],
                       device float *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = tan(a[id]);
}

// Elementwise arcsin
kernel void kernel_arcsin(device float *a [[ buffer(0) ]],
                          device float *out [[ buffer(1) ]],
                          uint id [[ thread_position_in_grid ]]) {
    out[id] = arcsin(a[id]);
}

// Elementwise sign
kernel void kernel_sign(device float *a [[ buffer(0) ]],
                        device float *b [[ buffer(1) ]],
                        device float *out [[ buffer(2) ]],
                        uint id [[ thread_position_in_grid ]]) {
    if (a[id] == 0.0f) {
        out[id] = 0.0f;
    } else if (a[id] < 0.0f) {
        out[id] = -1.0f;
    } else {
        out[id] = 1.0f;
    }
}

// Elementwise atan2
kernel void kernel_atan2(device const float *y [[ buffer(0) ]],
                         device const float *x [[ buffer(1) ]],
                         device float *out [[ buffer(2) ]],
                         uint id [[ thread_position_in_grid ]]) {
    out[id] = atan2(y[id], x[id]);
}

// Elementwise clipping
kernel void kernel_clip(device const float *a [[ buffer(0) ]],
                        device float *out [[ buffer(1) ]],
                        constant float &low [[ buffer(2) ]],
                        constant float &high [[ buffer(3) ]],
                        uint id [[ thread_position_in_grid ]]) {
    out[id] = clamp(a[id], low, high);
}

// Elementwise round
kernel void kernel_round(device const float *a [[ buffer(0) ]],
                         device float *out [[ buffer(1) ]],
                         uint id [[ thread_position_in_grid ]]) {
    out[id] = round(a[id]);
}

// Elementwise floor
kernel void kernel_floor(device const float *a [[ buffer(0) ]],
                         device float *out [[ buffer(1) ]],
                         uint id [[ thread_position_in_grid ]]) {
    out[id] = floor(a[id]);
}

// Elementwise ceil
kernel void kernel_ceil(device const float *a [[ buffer(0) ]],
                        device float *out [[ buffer(1) ]],
                        uint id [[ thread_position_in_grid ]]) {
    out[id] = ceil(a[id]);
}

// Elementwise ReLU (Rectified Linear Unit)
kernel void relu(device float *a [[ buffer(0) ]],
                 device float *out [[ buffer(1) ]],
                 uint id [[ thread_position_in_grid ]]) {
    out[id] = max(0.0, a[id]);
}

