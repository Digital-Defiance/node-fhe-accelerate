/**
 * Memory Optimization for FHE Operations
 * 
 * Exploits Apple Silicon memory features for FHE acceleration:
 * - NTT-aware prefetch patterns
 * - Hardware memory compression
 * - Zero-copy IOSurface sharing between CPU/GPU/Neural Engine
 * - Cache-optimized memory layouts
 * - Unified memory management
 * 
 * Requirements 14.18, 14.19, 14.20, 14.21, 14.22
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <string>

namespace fhe_accelerate {
namespace memory {

// ============================================================================
// Memory Alignment Constants
// ============================================================================

// M4 Max cache line size
constexpr size_t CACHE_LINE_SIZE = 128;

// L1 cache size per P-core (192KB)
constexpr size_t L1_CACHE_SIZE = 192 * 1024;

// L2 cache size (32MB shared)
constexpr size_t L2_CACHE_SIZE = 32 * 1024 * 1024;

// Page size
constexpr size_t PAGE_SIZE = 16384;  // 16KB on Apple Silicon

// ============================================================================
// Memory Allocator with Cache Alignment
// ============================================================================

/**
 * Allocate cache-line aligned memory
 * 
 * @param size Size in bytes
 * @param alignment Alignment (default: cache line size)
 * @return Aligned pointer (must be freed with aligned_free)
 */
void* aligned_alloc(size_t size, size_t alignment = CACHE_LINE_SIZE);

/**
 * Free aligned memory
 */
void aligned_free(void* ptr);

/**
 * RAII wrapper for aligned memory
 */
template<typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t count, size_t alignment = CACHE_LINE_SIZE)
        : data_(nullptr), count_(count), alignment_(alignment)
    {
        if (count > 0) {
            data_ = static_cast<T*>(aligned_alloc(count * sizeof(T), alignment));
        }
    }
    
    ~AlignedBuffer() {
        if (data_) {
            aligned_free(data_);
        }
    }
    
    // Move semantics
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_), alignment_(other.alignment_)
    {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) aligned_free(data_);
            data_ = other.data_;
            count_ = other.count_;
            alignment_ = other.alignment_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    // No copy
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    
private:
    T* data_;
    size_t count_;
    size_t alignment_;
};

// ============================================================================
// NTT-Aware Prefetch Patterns
// ============================================================================

/**
 * NTT prefetch configuration
 * 
 * NTT has predictable butterfly access patterns that we can prefetch ahead.
 * For stage s of NTT on degree n:
 * - Butterfly pairs are at distance 2^s
 * - We access indices k and k + 2^s together
 */
class NTTPrefetcher {
public:
    NTTPrefetcher();
    ~NTTPrefetcher();
    
    /**
     * Configure prefetch for NTT of given degree
     * 
     * @param degree Polynomial degree (power of 2)
     */
    void configure(size_t degree);
    
    /**
     * Prefetch data for upcoming NTT stage
     * 
     * @param data Coefficient array
     * @param stage Current NTT stage (0 to log2(n)-1)
     * @param lookahead Number of butterflies to prefetch ahead
     */
    void prefetch_stage(const uint64_t* data, size_t stage, size_t lookahead = 8);
    
    /**
     * Prefetch twiddle factors for upcoming stage
     * 
     * @param twiddles Twiddle factor array
     * @param stage Current NTT stage
     */
    void prefetch_twiddles(const uint64_t* twiddles, size_t stage);
    
private:
    size_t degree_;
    size_t log_degree_;
    std::vector<size_t> stage_distances_;
};

// ============================================================================
// Hardware Memory Compression
// ============================================================================

/**
 * Memory compression manager
 * 
 * Apple Silicon has hardware memory compression that can reduce
 * memory bandwidth for compressible data like ciphertexts.
 */
class MemoryCompressor {
public:
    MemoryCompressor();
    ~MemoryCompressor();
    
    /**
     * Check if hardware compression is available
     */
    static bool is_available();
    
    /**
     * Allocate compressed memory region
     * 
     * @param size Size in bytes
     * @return Pointer to compressed region
     */
    void* alloc_compressed(size_t size);
    
    /**
     * Free compressed memory
     */
    void free_compressed(void* ptr);
    
    /**
     * Get compression ratio for a buffer
     * 
     * @param ptr Pointer to buffer
     * @param size Size in bytes
     * @return Compression ratio (1.0 = no compression)
     */
    float get_compression_ratio(const void* ptr, size_t size);
    
    /**
     * Hint to the system that this memory is compressible
     * 
     * @param ptr Pointer to buffer
     * @param size Size in bytes
     */
    void hint_compressible(void* ptr, size_t size);
    
private:
    bool compression_enabled_;
};

// ============================================================================
// Zero-Copy IOSurface Sharing
// ============================================================================

/**
 * Shared buffer for CPU/GPU/Neural Engine
 * 
 * Uses IOSurface for zero-copy sharing between accelerators.
 * All accelerators on Apple Silicon share unified memory.
 */
class SharedBuffer {
public:
    /**
     * Create a shared buffer
     * 
     * @param size Size in bytes
     */
    explicit SharedBuffer(size_t size);
    ~SharedBuffer();
    
    // Move semantics
    SharedBuffer(SharedBuffer&& other) noexcept;
    SharedBuffer& operator=(SharedBuffer&& other) noexcept;
    
    // No copy
    SharedBuffer(const SharedBuffer&) = delete;
    SharedBuffer& operator=(const SharedBuffer&) = delete;
    
    /**
     * Get CPU-accessible pointer
     */
    void* cpu_ptr();
    const void* cpu_ptr() const;
    
    /**
     * Get Metal buffer handle (for GPU access)
     * Returns nullptr if Metal is not available
     */
    void* metal_buffer();
    
    /**
     * Get IOSurface handle (for Neural Engine access)
     */
    void* iosurface();
    
    /**
     * Synchronize after CPU writes (before GPU/ANE reads)
     */
    void sync_for_device();
    
    /**
     * Synchronize after GPU/ANE writes (before CPU reads)
     */
    void sync_for_cpu();
    
    size_t size() const { return size_; }
    
private:
    size_t size_;
    void* iosurface_;
    void* metal_buffer_;
    void* cpu_ptr_;
};

// ============================================================================
// Memory Bandwidth Benchmarking
// ============================================================================

/**
 * Memory bandwidth measurement between accelerator pairs
 */
struct BandwidthResult {
    std::string source;
    std::string destination;
    double bandwidth_gbps;
    double latency_us;
};

/**
 * Measure memory bandwidth between accelerators
 * 
 * @param size Transfer size in bytes
 * @return Bandwidth results for all accelerator pairs
 */
std::vector<BandwidthResult> benchmark_memory_bandwidth(size_t size);

/**
 * Measure cache effects for different data sizes
 * 
 * @param sizes Vector of sizes to test
 * @return Bandwidth for each size
 */
std::vector<std::pair<size_t, double>> benchmark_cache_effects(
    const std::vector<size_t>& sizes);

// ============================================================================
// Memory Pool for FHE Operations
// ============================================================================

/**
 * Memory pool optimized for FHE operations
 * 
 * Pre-allocates memory for common FHE data structures to avoid
 * allocation overhead during computation.
 */
class FHEMemoryPool {
public:
    /**
     * Create a memory pool
     * 
     * @param poly_degree Maximum polynomial degree
     * @param num_polys Number of polynomials to pre-allocate
     */
    FHEMemoryPool(size_t poly_degree, size_t num_polys);
    ~FHEMemoryPool();
    
    /**
     * Allocate a polynomial buffer from the pool
     * 
     * @return Pointer to polynomial buffer (degree coefficients)
     */
    uint64_t* alloc_polynomial();
    
    /**
     * Return a polynomial buffer to the pool
     */
    void free_polynomial(uint64_t* ptr);
    
    /**
     * Allocate a ciphertext buffer (2 polynomials)
     */
    uint64_t* alloc_ciphertext();
    
    /**
     * Return a ciphertext buffer to the pool
     */
    void free_ciphertext(uint64_t* ptr);
    
    /**
     * Get pool statistics
     */
    struct Stats {
        size_t total_bytes;
        size_t used_bytes;
        size_t num_allocations;
        size_t num_frees;
    };
    Stats get_stats() const;
    
private:
    size_t poly_degree_;
    size_t poly_size_bytes_;
    std::vector<uint64_t*> poly_pool_;
    std::vector<uint64_t*> ct_pool_;
    std::vector<bool> poly_in_use_;
    std::vector<bool> ct_in_use_;
    Stats stats_;
};

// ============================================================================
// Memory Layout Optimization
// ============================================================================

/**
 * Optimize memory layout for NTT operations
 * 
 * Rearranges polynomial coefficients for better cache utilization
 * during NTT butterfly operations.
 */
void optimize_ntt_layout(uint64_t* coeffs, size_t degree);

/**
 * Restore standard coefficient order after NTT
 */
void restore_standard_layout(uint64_t* coeffs, size_t degree);

/**
 * Interleave multiple polynomials for batch processing
 * 
 * Converts from [poly0_coeff0, poly0_coeff1, ...][poly1_coeff0, ...]
 * to [poly0_coeff0, poly1_coeff0, ...][poly0_coeff1, poly1_coeff1, ...]
 */
void interleave_polynomials(const uint64_t* const* polys, uint64_t* interleaved,
                            size_t num_polys, size_t degree);

/**
 * De-interleave polynomials after batch processing
 */
void deinterleave_polynomials(const uint64_t* interleaved, uint64_t** polys,
                              size_t num_polys, size_t degree);

} // namespace memory
} // namespace fhe_accelerate
