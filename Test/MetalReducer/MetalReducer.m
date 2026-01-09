#import "MetalReducer.h"
#import <Metal/Metal.h>

@interface MetalReducer () {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso;
    id<MTLCommandQueue> queue;

    id<MTLBuffer> inBuf;
    id<MTLBuffer> outBuf;

    NSUInteger capacity;
    NSUInteger threadsPerTG;
}
@end

@implementation MetalReducer

// ------------------------------------
// INIT (done ONCE, not every call)
// ------------------------------------
- (instancetype)initWithLength:(NSUInteger)length {
    self = [super init];
    if (!self) return nil;

    device = MTLCreateSystemDefaultDevice();
    queue  = [device newCommandQueue];

    NSError *err = nil;

    // Load .metal file once
    NSString *path = [[NSBundle mainBundle] pathForResource:@"metal_kernels"
                                                     ofType:@"metal"];

    NSString *src = [NSString stringWithContentsOfFile:path
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (err) { NSLog(@"Load error: %@", err); return nil; }

    library = [device newLibraryWithSource:src options:nil error:&err];
    if (err) { NSLog(@"Compile error: %@", err); return nil; }

    id<MTLFunction> fn = [library newFunctionWithName:@"sum_reduce_kernel"];
    pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (err) { NSLog(@"Pipeline error: %@", err); return nil; }

    threadsPerTG = pso.maxTotalThreadsPerThreadgroup;
    capacity = length * sizeof(float);

    // Persistent buffers
    inBuf  = [device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
    outBuf = [device newBufferWithLength:(length/threadsPerTG+2)*sizeof(float)
                                 options:MTLResourceStorageModeShared];

    return self;
}


// ------------------------------------
// FAST SUM — your kernel unchanged
// ------------------------------------
- (float)sum:(float *)data length:(NSUInteger)N {

    // Copy input to persistent buffer (shared memory)
    memcpy(inBuf.contents, data, N * sizeof(float));

    NSUInteger groups = (N + threadsPerTG - 1) / threadsPerTG;

    // Zero output buffer
    memset(outBuf.contents, 0, groups * sizeof(float));

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pso];
    [enc setBuffer:inBuf offset:0 atIndex:0];
    [enc setBuffer:outBuf offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&threadsPerTG length:sizeof(uint32_t) atIndex:3];
    [enc setThreadgroupMemoryLength:threadsPerTG * sizeof(float) atIndex:0];

    // Dispatch same as your Python version
    MTLSize grid = MTLSizeMake(groups, 1, 1);
    MTLSize tg   = MTLSizeMake(threadsPerTG, 1, 1);

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    // CPU second pass (your original way)
    float *partials = outBuf.contents;
    float total = 0.0f;
    for (NSUInteger i = 0; i < groups; i++) {
        total += partials[i];
    }

    return total;
}

@end
