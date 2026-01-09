#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern "C" {

// ============================================================
// Internal GPU buffer object (opaque to Python)
// ============================================================
struct MetalBuffer {
    id<MTLBuffer> buf;
    uint32_t length;
};

// ============================================================
// Global Metal State
// ============================================================
static id<MTLDevice>       g_device        = nil;
static id<MTLCommandQueue> g_queue         = nil;
static id<MTLLibrary>      g_library       = nil;
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pso_cache = nil;

// ============================================================
// Initialization
// ============================================================
void metal_init(const char* metallib_path)
{
    @autoreleasepool {
        if (!g_device) {
            g_device = MTLCreateSystemDefaultDevice();
            if (!g_device) {
                NSLog(@"❌ Failed to create Metal device");
                return;
            }
            g_queue  = [g_device newCommandQueue];
            g_pso_cache = [[NSMutableDictionary alloc] init];
        }

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* err = nil;

        g_library = [g_device newLibraryWithFile:path error:&err];

        if (err || !g_library) {
            NSLog(@"❌ Failed to load metallib: %@", err);
        } else {
            NSLog(@"✅ Loaded Metal library: %@", path);
        }
    }
}

// ============================================================
// Pipeline Loader (Cached)
// ============================================================
static id<MTLComputePipelineState> metal_get_pso(const char* fname)
{
    @autoreleasepool {
        NSString* key = [NSString stringWithUTF8String:fname];

        id<MTLComputePipelineState> cached = g_pso_cache[key];
        if (cached) return cached;

        id<MTLFunction> fn = [g_library newFunctionWithName:key];
        if (!fn) {
            NSLog(@"❌ Function not found: %s", fname);
            return nil;
        }
        
        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [g_device newComputePipelineStateWithFunction:fn error:&err];

        if (err || !pso) {
            NSLog(@"❌ Pipeline creation failed for %s: %@", fname, err);
            return nil;
        }

        g_pso_cache[key] = pso;
        return pso;
    }
}

// ============================================================
// Create GPU Array
// ============================================================
void* metal_create_array(float* data, uint32_t length)
{
    @autoreleasepool {
        if (!g_device || !data || length == 0) {
            NSLog(@"❌ Invalid parameters to metal_create_array");
            return nullptr;
        }
        
        MetalBuffer* mb = new MetalBuffer();
        mb->length = length;
        mb->buf = [g_device newBufferWithBytes:data
                                        length:length * sizeof(float)
                                       options:MTLResourceStorageModeShared];

        if (!mb->buf) {
            delete mb;
            NSLog(@"❌ Failed to create Metal buffer");
            return nullptr;
        }

        return mb;
    }
}

// ============================================================
// Free GPU Array
// ============================================================
void metal_free_array(void* arr)
{
    if (arr) {
        MetalBuffer* mb = (MetalBuffer*)arr;
        mb->buf = nil;
        delete mb;
    }
}

// ============================================================
// Get length of array
// ============================================================
uint32_t metal_get_length(void* arr)
{
    if (!arr) return 0;
    return ((MetalBuffer*)arr)->length;
}

// ============================================================
// Copy GPU → CPU
// ============================================================
void metal_to_cpu(void* arr, float* out)
{
    if (!arr || !out) return;
    MetalBuffer* mb = (MetalBuffer*)arr;
    memcpy(out, [mb->buf contents], mb->length * sizeof(float));
}

// ============================================================
// Elementwise Binary Op Dispatcher
// ============================================================
void metal_binary_op(const char* fname, void* out_ptr, void* a_ptr, void* b_ptr)
{
    @autoreleasepool {
        if (!out_ptr || !a_ptr || !b_ptr) {
            NSLog(@"❌ Null pointer in metal_binary_op");
            return;
        }
        
        MetalBuffer* out = (MetalBuffer*)out_ptr;
        MetalBuffer* a   = (MetalBuffer*)a_ptr;
        MetalBuffer* b   = (MetalBuffer*)b_ptr;

        id<MTLComputePipelineState> pso = metal_get_pso(fname);
        if (!pso) return;

        uint32_t N = a->length;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];

        [enc setBuffer:a->buf offset:0 atIndex:0];
        [enc setBuffer:b->buf offset:0 atIndex:1];
        [enc setBuffer:out->buf offset:0 atIndex:2];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];

        NSUInteger TPT = pso.maxTotalThreadsPerThreadgroup;
        if (TPT > 1024) TPT = 1024;

        MTLSize grid = MTLSizeMake(N, 1, 1);
        MTLSize tg   = MTLSizeMake(TPT, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ============================================================
// Public Elementwise Ops
// ============================================================
void metal_add(void* out, void* a, void* b) { metal_binary_op("add_kernel", out, a, b); }
void metal_sub(void* out, void* a, void* b) { metal_binary_op("sub_kernel", out, a, b); }
void metal_mul(void* out, void* a, void* b) { metal_binary_op("multiply_kernel", out, a, b); }
void metal_div(void* out, void* a, void* b) { metal_binary_op("division_kernel", out, a, b); }

// ============================================================
// Reduction (OPTIMIZED - single command buffer)
// ============================================================
float metal_reduce(const char* fname, void* arr_ptr)
{
    @autoreleasepool {
        if (!arr_ptr) {
            NSLog(@"❌ Null pointer in metal_reduce");
            return 0.0f;
        }
        
        MetalBuffer* input = (MetalBuffer*)arr_ptr;
        id<MTLComputePipelineState> pso = metal_get_pso(fname);
        if (!pso) return 0.0f;

        uint32_t N = input->length;
        NSUInteger TPT = pso.maxTotalThreadsPerThreadgroup;
        if (TPT > 1024) TPT = 1024;
        
        // Create a working copy of the input buffer
        id<MTLBuffer> current_buf = [g_device newBufferWithLength:N * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
        memcpy([current_buf contents], [input->buf contents], N * sizeof(float));

        // Pre-calculate all buffer sizes needed
        NSMutableArray<id<MTLBuffer>>* buffers = [NSMutableArray array];
        [buffers addObject:current_buf];
        
        uint32_t temp_n = N;
        while (temp_n > 1) {
            uint32_t groups = (temp_n + TPT - 1) / TPT;
            id<MTLBuffer> buf = [g_device newBufferWithLength:groups * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
            [buffers addObject:buf];
            temp_n = groups;
        }

        // Single command buffer for ALL passes
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        
        uint32_t buf_idx = 0;
        while (N > 1) {
            uint32_t groups = (N + TPT - 1) / TPT;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:pso];
            [enc setBuffer:buffers[buf_idx] offset:0 atIndex:0];
            [enc setBuffer:buffers[buf_idx + 1] offset:0 atIndex:1];

            uint32_t tpt_u32 = (uint32_t)TPT;
            [enc setBytes:&N       length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&tpt_u32 length:sizeof(uint32_t) atIndex:3];

            [enc setThreadgroupMemoryLength:TPT * sizeof(float) atIndex:0];

            MTLSize grid = MTLSizeMake(N, 1, 1);
            MTLSize tg   = MTLSizeMake(TPT, 1, 1);

            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            buf_idx++;
            N = groups;
        }

        // Single wait at the end
        [cmd commit];
        [cmd waitUntilCompleted];

        return ((float*)[[buffers lastObject] contents])[0];
    }
}

// ============================================================
// Public Reduction APIs
// ============================================================
float metal_sum(void* a) { return metal_reduce("sum_reduce_kernel", a); }
float metal_max(void* a) { return metal_reduce("max_reduce_kernel", a); }
float metal_min(void* a) { return metal_reduce("min_reduce_kernel", a); }
float metal_prod(void* a) { return metal_reduce("product_reduce_kernel", a); }

} // extern "C"
