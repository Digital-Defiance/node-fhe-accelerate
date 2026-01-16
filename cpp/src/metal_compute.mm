/**
 * Metal GPU Compute Backend Implementation
 * 
 * Provides GPU-accelerated batch operations for FHE.
 * Optimized for M4 Max with 40 GPU cores.
 */

#include "metal_compute.h"
#include <iostream>
#include <chrono>
#include <cstring>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace fhe_accelerate {
namespace metal {

// ============================================================================
// MetalComputeContext Implementation
// ============================================================================

MetalComputeContext::MetalComputeContext()
    : device_(nil)
    , command_queue_(nil)
    , library_(nil)
    , modmul_pipeline_(nil)
    , modadd_pipeline_(nil)
    , ntt_stage_pipeline_(nil)
    , ntt_bitrev_pipeline_(nil)
    , ntt_batch_pipeline_(nil)
    , gpu_cores_(0)
    , max_buffer_size_(0)
    , max_threadgroup_size_(0)
{
#ifdef __APPLE__
    // Get default Metal device
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
        std::cerr << "Metal: No GPU device found" << std::endl;
        return;
    }
    
    // Create command queue
    command_queue_ = [device_ newCommandQueue];
    if (command_queue_ == nil) {
        std::cerr << "Metal: Failed to create command queue" << std::endl;
        device_ = nil;
        return;
    }
    
    // Get device info
    max_buffer_size_ = [device_ maxBufferLength];
    max_threadgroup_size_ = 1024;  // M4 Max supports up to 1024 threads per threadgroup
    
    // Detect GPU cores from device name
    NSString* name = [device_ name];
    if ([name containsString:@"M4 Max"]) {
        gpu_cores_ = 40;
    } else if ([name containsString:@"M4 Pro"]) {
        gpu_cores_ = 20;
    } else if ([name containsString:@"M4"]) {
        gpu_cores_ = 10;
    } else if ([name containsString:@"M3 Max"]) {
        gpu_cores_ = 40;
    } else if ([name containsString:@"M3 Pro"]) {
        gpu_cores_ = 18;
    } else {
        gpu_cores_ = 8;  // Default
    }
    
    std::cout << "Metal: Initialized with " << [name UTF8String] 
              << " (" << gpu_cores_ << " GPU cores)" << std::endl;
    std::cout << "Metal: Max buffer size: " << (max_buffer_size_ / 1024 / 1024) << " MB" << std::endl;
    
    // Try to load shaders
    if (!load_shaders("fhe_shaders.metallib")) {
        // Try alternate paths
        if (!load_shaders("dist/shaders/fhe_shaders.metallib")) {
            std::cerr << "Metal: Warning - shaders not loaded, GPU compute unavailable" << std::endl;
        }
    }
#endif
}

MetalComputeContext::~MetalComputeContext() {
#ifdef __APPLE__
    // ARC handles cleanup
    modmul_pipeline_ = nil;
    modadd_pipeline_ = nil;
    ntt_stage_pipeline_ = nil;
    ntt_bitrev_pipeline_ = nil;
    ntt_batch_pipeline_ = nil;
    library_ = nil;
    command_queue_ = nil;
    device_ = nil;
#endif
}

std::string MetalComputeContext::device_name() const {
#ifdef __APPLE__
    if (device_ == nil) return "No device";
    return [[device_ name] UTF8String];
#else
    return "Metal not available";
#endif
}

size_t MetalComputeContext::max_buffer_size() const {
    return max_buffer_size_;
}

size_t MetalComputeContext::max_threadgroup_size() const {
    return max_threadgroup_size_;
}

void* MetalComputeContext::create_buffer(size_t size) {
#ifdef __APPLE__
    if (device_ == nil) return nullptr;
    
    // Use shared memory for unified memory architecture (Apple Silicon)
    id<MTLBuffer> buffer = [device_ newBufferWithLength:size
                                                options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buffer;
#else
    return nullptr;
#endif
}

void MetalComputeContext::release_buffer(void* buffer) {
#ifdef __APPLE__
    if (buffer != nullptr) {
        id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)buffer;
        mtl_buffer = nil;
    }
#endif
}

void MetalComputeContext::copy_to_buffer(void* buffer, const void* data, size_t size) {
#ifdef __APPLE__
    if (buffer == nullptr) return;
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer;
    memcpy([mtl_buffer contents], data, size);
#endif
}

void MetalComputeContext::copy_from_buffer(const void* buffer, void* data, size_t size) {
#ifdef __APPLE__
    if (buffer == nullptr) return;
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer;
    memcpy(data, [mtl_buffer contents], size);
#endif
}

bool MetalComputeContext::load_shaders(const std::string& metallib_path) {
#ifdef __APPLE__
    if (device_ == nil) return false;
    
    NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
    NSError* error = nil;
    
    // Check if file exists
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        return false;
    }
    
    NSURL* url = [NSURL fileURLWithPath:path];
    library_ = [device_ newLibraryWithURL:url error:&error];
    
    if (library_ == nil) {
        if (error != nil) {
            std::cerr << "Metal: Failed to load library: " 
                      << [[error localizedDescription] UTF8String] << std::endl;
        }
        return false;
    }
    
    std::cout << "Metal: Loaded shaders from " << metallib_path << std::endl;
    
    // List available functions
    NSArray<NSString*>* functions = [library_ functionNames];
    std::cout << "Metal: Available kernels: ";
    for (NSString* name in functions) {
        std::cout << [name UTF8String] << " ";
    }
    std::cout << std::endl;
    
    return create_pipelines();
#else
    return false;
#endif
}

bool MetalComputeContext::create_pipelines() {
#ifdef __APPLE__
    if (library_ == nil) return false;
    
    NSError* error = nil;
    
    // Create modmul pipeline - prefer direct Barrett version
    id<MTLFunction> modmul_func = [library_ newFunctionWithName:@"modmul_direct_batch"];
    if (modmul_func == nil) {
        // Fallback to Montgomery version
        modmul_func = [library_ newFunctionWithName:@"modmul_batch"];
    }
    if (modmul_func != nil) {
        modmul_pipeline_ = [device_ newComputePipelineStateWithFunction:modmul_func error:&error];
        if (modmul_pipeline_ != nil) {
            std::cout << "Metal: Created modmul pipeline (max threads: " 
                      << [modmul_pipeline_ maxTotalThreadsPerThreadgroup] << ")" << std::endl;
        }
    }
    
    // Create modadd pipeline
    id<MTLFunction> modadd_func = [library_ newFunctionWithName:@"modadd_batch"];
    if (modadd_func != nil) {
        modadd_pipeline_ = [device_ newComputePipelineStateWithFunction:modadd_func error:&error];
    }
    
    // Create NTT stage pipeline
    id<MTLFunction> ntt_stage_func = [library_ newFunctionWithName:@"ntt_forward_stage"];
    if (ntt_stage_func != nil) {
        ntt_stage_pipeline_ = [device_ newComputePipelineStateWithFunction:ntt_stage_func error:&error];
        if (ntt_stage_pipeline_ != nil) {
            std::cout << "Metal: Created ntt_forward_stage pipeline" << std::endl;
        }
    }
    
    // Create NTT bit-reversal pipeline
    id<MTLFunction> ntt_bitrev_func = [library_ newFunctionWithName:@"ntt_bit_reverse"];
    if (ntt_bitrev_func != nil) {
        ntt_bitrev_pipeline_ = [device_ newComputePipelineStateWithFunction:ntt_bitrev_func error:&error];
    }
    
    // Create batch NTT pipeline
    id<MTLFunction> ntt_batch_func = [library_ newFunctionWithName:@"ntt_forward_batch"];
    if (ntt_batch_func != nil) {
        ntt_batch_pipeline_ = [device_ newComputePipelineStateWithFunction:ntt_batch_func error:&error];
    }
    
    return modmul_pipeline_ != nil || ntt_stage_pipeline_ != nil;
#else
    return false;
#endif
}

bool MetalComputeContext::has_pipeline(const std::string& name) const {
#ifdef __APPLE__
    if (name == "modmul_batch") return modmul_pipeline_ != nil;
    if (name == "modadd_batch") return modadd_pipeline_ != nil;
    if (name == "ntt_forward_stage") return ntt_stage_pipeline_ != nil;
    if (name == "ntt_bit_reverse") return ntt_bitrev_pipeline_ != nil;
    if (name == "ntt_forward_batch") return ntt_batch_pipeline_ != nil;
#endif
    return false;
}

void MetalComputeContext::synchronize() {
#ifdef __APPLE__
    // Create a command buffer and wait for completion
    if (command_queue_ == nil) return;
    
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    [cmd commit];
    [cmd waitUntilCompleted];
#endif
}

// ============================================================================
// Batch Operations
// ============================================================================

void MetalComputeContext::batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                                        size_t count, uint64_t modulus) {
#ifdef __APPLE__
    if (modmul_pipeline_ == nil || count == 0) return;
    
    size_t buffer_size = count * sizeof(uint64_t);
    
    // Create buffers with shared memory (zero-copy on Apple Silicon)
    id<MTLBuffer> buffer_a = [device_ newBufferWithBytes:a
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_b = [device_ newBufferWithBytes:b
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_result = [device_ newBufferWithLength:buffer_size
                                                       options:MTLResourceStorageModeShared];
    
    // Create Barrett params buffer
    struct {
        uint64_t modulus;
        uint64_t mu;
        uint32_t k;
        uint32_t padding;
    } params;
    
    params.modulus = modulus;
    params.k = 64 - __builtin_clzll(modulus);
    
    // Compute mu = floor(2^(2k) / modulus)
    if (params.k <= 32) {
        params.mu = (1ULL << (2 * params.k)) / modulus;
    } else {
        __uint128_t numerator = static_cast<__uint128_t>(1) << (2 * params.k);
        params.mu = static_cast<uint64_t>(numerator / modulus);
    }
    params.padding = 0;
    
    id<MTLBuffer> buffer_params = [device_ newBufferWithBytes:&params
                                                       length:sizeof(params)
                                                      options:MTLResourceStorageModeShared];
    
    uint32_t size = static_cast<uint32_t>(count);
    id<MTLBuffer> buffer_size_param = [device_ newBufferWithBytes:&size
                                                           length:sizeof(size)
                                                          options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    
    [encoder setComputePipelineState:modmul_pipeline_];
    [encoder setBuffer:buffer_a offset:0 atIndex:0];
    [encoder setBuffer:buffer_b offset:0 atIndex:1];
    [encoder setBuffer:buffer_result offset:0 atIndex:2];
    [encoder setBuffer:buffer_params offset:0 atIndex:3];
    [encoder setBuffer:buffer_size_param offset:0 atIndex:4];
    
    // Dispatch threads - use larger threadgroups for better GPU utilization
    NSUInteger threadgroup_size = std::min((NSUInteger)1024, [modmul_pipeline_ maxTotalThreadsPerThreadgroup]);
    NSUInteger num_threadgroups = (count + threadgroup_size - 1) / threadgroup_size;
    
    [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    // Copy result back
    memcpy(result, [buffer_result contents], buffer_size);
#endif
}

void MetalComputeContext::batch_modadd(const uint64_t* a, const uint64_t* b, uint64_t* result,
                                        size_t count, uint64_t modulus) {
#ifdef __APPLE__
    if (modadd_pipeline_ == nil || count == 0) return;
    
    size_t buffer_size = count * sizeof(uint64_t);
    
    id<MTLBuffer> buffer_a = [device_ newBufferWithBytes:a
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_b = [device_ newBufferWithBytes:b
                                                  length:buffer_size
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_result = [device_ newBufferWithLength:buffer_size
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_modulus = [device_ newBufferWithBytes:&modulus
                                                        length:sizeof(modulus)
                                                       options:MTLResourceStorageModeShared];
    uint32_t size = static_cast<uint32_t>(count);
    id<MTLBuffer> buffer_size_param = [device_ newBufferWithBytes:&size
                                                           length:sizeof(size)
                                                          options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    
    [encoder setComputePipelineState:modadd_pipeline_];
    [encoder setBuffer:buffer_a offset:0 atIndex:0];
    [encoder setBuffer:buffer_b offset:0 atIndex:1];
    [encoder setBuffer:buffer_result offset:0 atIndex:2];
    [encoder setBuffer:buffer_modulus offset:0 atIndex:3];
    [encoder setBuffer:buffer_size_param offset:0 atIndex:4];
    
    NSUInteger threadgroup_size = std::min((NSUInteger)256, [modadd_pipeline_ maxTotalThreadsPerThreadgroup]);
    NSUInteger num_threadgroups = (count + threadgroup_size - 1) / threadgroup_size;
    
    [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    memcpy(result, [buffer_result contents], buffer_size);
#endif
}

void MetalComputeContext::batch_ntt_forward(uint64_t* coeffs, size_t degree, size_t batch_size,
                                             uint64_t modulus, const uint64_t* twiddles) {
#ifdef __APPLE__
    if (ntt_stage_pipeline_ == nil || degree == 0 || batch_size == 0) return;
    
    size_t total_coeffs = degree * batch_size;
    size_t coeff_buffer_size = total_coeffs * sizeof(uint64_t);
    size_t twiddle_buffer_size = degree * sizeof(uint64_t);
    
    // Create buffers
    id<MTLBuffer> buffer_coeffs = [device_ newBufferWithBytes:coeffs
                                                       length:coeff_buffer_size
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_twiddles = [device_ newBufferWithBytes:twiddles
                                                         length:twiddle_buffer_size
                                                        options:MTLResourceStorageModeShared];
    
    // NTT params
    struct {
        uint32_t degree;
        uint32_t log_degree;
        uint64_t modulus;
        uint64_t inv_n;
    } ntt_params;
    
    ntt_params.degree = static_cast<uint32_t>(degree);
    ntt_params.log_degree = 0;
    size_t temp = degree;
    while (temp > 1) { temp >>= 1; ntt_params.log_degree++; }
    ntt_params.modulus = modulus;
    
    // Compute N^(-1) mod q
    uint64_t n_inv = 1;
    uint64_t base = degree;
    uint64_t exp = modulus - 2;
    while (exp > 0) {
        if (exp & 1) {
            n_inv = static_cast<uint64_t>((static_cast<__uint128_t>(n_inv) * base) % modulus);
        }
        base = static_cast<uint64_t>((static_cast<__uint128_t>(base) * base) % modulus);
        exp >>= 1;
    }
    ntt_params.inv_n = n_inv;
    
    id<MTLBuffer> buffer_params = [device_ newBufferWithBytes:&ntt_params
                                                       length:sizeof(ntt_params)
                                                      options:MTLResourceStorageModeShared];
    
    uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
    id<MTLBuffer> buffer_batch = [device_ newBufferWithBytes:&batch_size_u32
                                                      length:sizeof(batch_size_u32)
                                                     options:MTLResourceStorageModeShared];
    
    // First: bit-reversal permutation
    if (ntt_bitrev_pipeline_ != nil) {
        id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        
        [encoder setComputePipelineState:ntt_bitrev_pipeline_];
        [encoder setBuffer:buffer_coeffs offset:0 atIndex:0];
        [encoder setBuffer:buffer_params offset:0 atIndex:1];
        [encoder setBuffer:buffer_batch offset:0 atIndex:2];
        
        NSUInteger threadgroup_size = 256;
        NSUInteger num_threadgroups_x = (degree + threadgroup_size - 1) / threadgroup_size;
        
        [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups_x, 1, batch_size)
                threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
        
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    // Then: NTT stages
    for (uint32_t stage = 0; stage < ntt_params.log_degree; stage++) {
        id<MTLCommandBuffer> cmd = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        
        id<MTLBuffer> buffer_stage = [device_ newBufferWithBytes:&stage
                                                          length:sizeof(stage)
                                                         options:MTLResourceStorageModeShared];
        
        [encoder setComputePipelineState:ntt_stage_pipeline_];
        [encoder setBuffer:buffer_coeffs offset:0 atIndex:0];
        [encoder setBuffer:buffer_twiddles offset:0 atIndex:1];
        [encoder setBuffer:buffer_params offset:0 atIndex:2];
        [encoder setBuffer:buffer_stage offset:0 atIndex:3];
        [encoder setBuffer:buffer_batch offset:0 atIndex:4];
        
        // Each thread handles one butterfly
        NSUInteger butterflies_per_poly = degree / 2;
        NSUInteger threadgroup_size = std::min((NSUInteger)256, 
                                               [ntt_stage_pipeline_ maxTotalThreadsPerThreadgroup]);
        NSUInteger num_threadgroups = (butterflies_per_poly + threadgroup_size - 1) / threadgroup_size;
        
        [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, batch_size)
                threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
        
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    // Copy result back
    memcpy(coeffs, [buffer_coeffs contents], coeff_buffer_size);
#endif
}

void MetalComputeContext::batch_ntt_inverse(uint64_t* coeffs, size_t degree, size_t batch_size,
                                             uint64_t modulus, const uint64_t* inv_twiddles) {
    // Inverse NTT uses same algorithm with inverse twiddles, then scale by N^(-1)
    batch_ntt_forward(coeffs, degree, batch_size, modulus, inv_twiddles);
    
    // Scale by N^(-1)
    uint64_t n_inv = 1;
    uint64_t base = degree;
    uint64_t exp = modulus - 2;
    while (exp > 0) {
        if (exp & 1) {
            n_inv = static_cast<uint64_t>((static_cast<__uint128_t>(n_inv) * base) % modulus);
        }
        base = static_cast<uint64_t>((static_cast<__uint128_t>(base) * base) % modulus);
        exp >>= 1;
    }
    
    size_t total = degree * batch_size;
    for (size_t i = 0; i < total; i++) {
        coeffs[i] = static_cast<uint64_t>((static_cast<__uint128_t>(coeffs[i]) * n_inv) % modulus);
    }
}

void MetalComputeContext::batch_poly_mul(const uint64_t* poly_a, const uint64_t* poly_b,
                                          uint64_t* result, size_t degree, size_t batch_size,
                                          uint64_t modulus) {
    // Pointwise multiplication in NTT domain
    size_t total = degree * batch_size;
    batch_modmul(poly_a, poly_b, result, total, modulus);
}

// ============================================================================
// Global Functions
// ============================================================================

static std::unique_ptr<MetalComputeContext> g_metal_context;

MetalComputeContext& get_metal_context() {
    if (!g_metal_context) {
        g_metal_context = std::make_unique<MetalComputeContext>();
    }
    return *g_metal_context;
}

bool metal_available() {
    return get_metal_context().is_available();
}

void gpu_batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus) {
    auto& ctx = get_metal_context();
    
    if (ctx.is_available() && count >= GPU_DISPATCH_THRESHOLD && ctx.has_pipeline("modmul_batch")) {
        ctx.batch_modmul(a, b, result, count, modulus);
    } else {
        // CPU fallback
        for (size_t i = 0; i < count; i++) {
            __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
            result[i] = product % modulus;
        }
    }
}

void gpu_batch_ntt(uint64_t* coeffs, size_t degree, size_t batch_size,
                   uint64_t modulus, const uint64_t* twiddles, bool inverse) {
    auto& ctx = get_metal_context();
    
    if (ctx.is_available() && batch_size >= 4 && ctx.has_pipeline("ntt_forward_stage")) {
        if (inverse) {
            ctx.batch_ntt_inverse(coeffs, degree, batch_size, modulus, twiddles);
        } else {
            ctx.batch_ntt_forward(coeffs, degree, batch_size, modulus, twiddles);
        }
    } else {
        // CPU fallback - process each polynomial
        // (Would call CPU NTT implementation here)
        std::cerr << "Metal: GPU NTT not available, falling back to CPU" << std::endl;
    }
}

} // namespace metal
} // namespace fhe_accelerate
