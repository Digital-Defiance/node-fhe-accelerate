/**
 * Metal GPU Compute Backend for FHE Operations
 * 
 * Provides GPU-accelerated batch operations for:
 * - Batch NTT (forward and inverse)
 * - Batch modular multiplication
 * - Batch polynomial operations
 * 
 * Optimized for M4 Max with 40 GPU cores.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace fhe_accelerate {
namespace metal {

/**
 * Metal compute context - manages device, command queue, and pipelines
 */
class MetalComputeContext {
public:
    MetalComputeContext();
    ~MetalComputeContext();
    
    // Prevent copying
    MetalComputeContext(const MetalComputeContext&) = delete;
    MetalComputeContext& operator=(const MetalComputeContext&) = delete;
    
    bool is_available() const { return device_ != nullptr; }
    
    // Device info
    std::string device_name() const;
    size_t max_buffer_size() const;
    size_t max_threadgroup_size() const;
    uint32_t gpu_cores() const { return gpu_cores_; }
    
    // Buffer management
    void* create_buffer(size_t size);
    void release_buffer(void* buffer);
    void copy_to_buffer(void* buffer, const void* data, size_t size);
    void copy_from_buffer(const void* buffer, void* data, size_t size);
    
    // Pipeline management
    bool load_shaders(const std::string& metallib_path);
    bool has_pipeline(const std::string& name) const;
    
    // Batch modular multiplication
    // result[i] = (a[i] * b[i]) mod modulus
    void batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus);
    
    // Batch modular addition
    void batch_modadd(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus);
    
    // Batch NTT forward
    // Processes multiple polynomials in parallel
    void batch_ntt_forward(uint64_t* coeffs, size_t degree, size_t batch_size,
                           uint64_t modulus, const uint64_t* twiddles);
    
    // Batch NTT inverse
    void batch_ntt_inverse(uint64_t* coeffs, size_t degree, size_t batch_size,
                           uint64_t modulus, const uint64_t* inv_twiddles);
    
    // Batch polynomial multiplication (via NTT)
    // result[i] = poly_a[i] * poly_b[i] in NTT domain
    void batch_poly_mul(const uint64_t* poly_a, const uint64_t* poly_b,
                        uint64_t* result, size_t degree, size_t batch_size,
                        uint64_t modulus);
    
    // Synchronize - wait for all GPU operations to complete
    void synchronize();
    
private:
#ifdef __APPLE__
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;
    
    // Compute pipelines
    id<MTLComputePipelineState> modmul_pipeline_;
    id<MTLComputePipelineState> modadd_pipeline_;
    id<MTLComputePipelineState> ntt_stage_pipeline_;
    id<MTLComputePipelineState> ntt_bitrev_pipeline_;
    id<MTLComputePipelineState> ntt_batch_pipeline_;
#else
    void* device_;
    void* command_queue_;
    void* library_;
    void* modmul_pipeline_;
    void* modadd_pipeline_;
    void* ntt_stage_pipeline_;
    void* ntt_bitrev_pipeline_;
    void* ntt_batch_pipeline_;
#endif
    
    uint32_t gpu_cores_;
    size_t max_buffer_size_;
    size_t max_threadgroup_size_;
    
    bool create_pipelines();
};

/**
 * GPU-accelerated batch operations
 * 
 * These functions automatically choose between CPU and GPU based on workload size.
 * For small batches, CPU is faster due to GPU dispatch overhead.
 * For large batches (>4096 elements), GPU provides significant speedup.
 */

// Threshold for GPU dispatch (elements)
constexpr size_t GPU_DISPATCH_THRESHOLD = 4096;

// Get global Metal context (singleton)
MetalComputeContext& get_metal_context();

// Check if Metal is available
bool metal_available();

// Batch modular multiplication with automatic dispatch
void gpu_batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus);

// Batch NTT with automatic dispatch
void gpu_batch_ntt(uint64_t* coeffs, size_t degree, size_t batch_size,
                   uint64_t modulus, const uint64_t* twiddles, bool inverse);

} // namespace metal
} // namespace fhe_accelerate
