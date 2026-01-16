/**
 * Comprehensive Hardware Benchmarking Framework
 * 
 * This module benchmarks EVERY available hardware feature on M4 Max
 * to find the fastest path for each FHE operation.
 * 
 * Philosophy: Try everything, benchmark everything, keep what works.
 * We're not going to break the machine - let's push it to its limits.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <map>

namespace fhe_accelerate {
namespace benchmark {

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct BenchmarkResult {
    std::string method_name;
    std::string hardware_used;
    double time_us;           // Microseconds
    double throughput;        // Operations per second
    double speedup_vs_scalar; // Speedup compared to naive scalar
    bool correctness_verified;
    std::string notes;
};

struct BenchmarkSuite {
    std::string operation_name;
    std::vector<BenchmarkResult> results;
    std::string winner;
    double best_time_us;
    
    void print_report() const;
    void export_csv(const std::string& filename) const;
};

// ============================================================================
// Hardware Feature Detection
// ============================================================================

struct HardwareFeatures {
    // CPU Features
    bool has_neon;
    bool has_neon_fp16;
    bool has_neon_bf16;
    bool has_neon_i8mm;      // Int8 matrix multiply
    bool has_neon_dotprod;   // Dot product instructions
    bool has_sve;            // Scalable Vector Extension
    bool has_sve2;
    bool has_sme;            // Scalable Matrix Extension
    bool has_sme2;
    bool has_amx;            // Apple Matrix Coprocessor
    bool has_sha3;           // SHA3 instructions
    bool has_aes;            // AES instructions
    bool has_crc32;
    bool has_atomics;        // LSE atomics
    bool has_fp16;
    bool has_bf16;
    
    // Memory Features
    bool has_mte;            // Memory Tagging Extension
    bool has_lse2;           // Large System Extensions 2
    uint64_t l1_cache_size;
    uint64_t l2_cache_size;
    uint64_t l3_cache_size;
    uint64_t cache_line_size;
    
    // GPU Features
    bool has_metal;
    uint32_t gpu_cores;
    uint64_t gpu_memory;
    bool has_metal3;
    bool has_ray_tracing;
    bool has_mesh_shaders;
    
    // Neural Engine
    bool has_neural_engine;
    uint32_t neural_engine_tops;
    
    // Other
    bool has_secure_enclave;
    bool has_hardware_rng;
    uint32_t num_performance_cores;
    uint32_t num_efficiency_cores;
    
    static HardwareFeatures detect();
    void print_report() const;
};

// ============================================================================
// Benchmark Runner
// ============================================================================

class BenchmarkRunner {
public:
    BenchmarkRunner();
    ~BenchmarkRunner();
    
    // Run all benchmarks for a specific operation
    BenchmarkSuite benchmark_modmul(size_t n, uint64_t modulus, int iterations = 1000);
    BenchmarkSuite benchmark_ntt(size_t degree, uint64_t modulus, int iterations = 100);
    BenchmarkSuite benchmark_poly_mul(size_t degree, uint64_t modulus, int iterations = 100);
    BenchmarkSuite benchmark_batch_add(size_t n, uint64_t modulus, int iterations = 1000);
    
    // Run comprehensive benchmark of all operations
    void run_full_benchmark_suite();
    
    // Export results
    void export_all_results(const std::string& directory);
    
private:
    HardwareFeatures features_;
    std::vector<BenchmarkSuite> all_results_;
    
    // Timing utilities
    template<typename Func>
    double time_operation(Func&& func, int iterations);
    
    // Verification utilities
    bool verify_modmul_result(const uint64_t* a, const uint64_t* b, 
                              const uint64_t* result, size_t n, uint64_t modulus);
};

// ============================================================================
// Individual Hardware Backend Benchmarks
// ============================================================================

namespace backends {

// Scalar (baseline)
void modmul_scalar(const uint64_t* a, const uint64_t* b, 
                   uint64_t* result, size_t n, uint64_t modulus);

// NEON variants
void modmul_neon_basic(const uint64_t* a, const uint64_t* b,
                       uint64_t* result, size_t n, uint64_t modulus);
void modmul_neon_unrolled(const uint64_t* a, const uint64_t* b,
                          uint64_t* result, size_t n, uint64_t modulus);
void modmul_neon_prefetch(const uint64_t* a, const uint64_t* b,
                          uint64_t* result, size_t n, uint64_t modulus);

// Montgomery multiplication (division-free)
void modmul_montgomery(const uint64_t* a, const uint64_t* b,
                       uint64_t* result, size_t n, uint64_t modulus);
void modmul_montgomery_neon(const uint64_t* a, const uint64_t* b,
                            uint64_t* result, size_t n, uint64_t modulus);

// Barrett reduction (division-free)
void modmul_barrett(const uint64_t* a, const uint64_t* b,
                    uint64_t* result, size_t n, uint64_t modulus);
void modmul_barrett_unrolled(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, size_t n, uint64_t modulus);
void modmul_barrett_prefetch(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, size_t n, uint64_t modulus);
void modmul_barrett_parallel(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, size_t n, uint64_t modulus);

// Multi-threaded
void modmul_parallel(const uint64_t* a, const uint64_t* b,
                     uint64_t* result, size_t n, uint64_t modulus);

// AMX via Accelerate
void modmul_amx_accelerate(const uint64_t* a, const uint64_t* b,
                           uint64_t* result, size_t n, uint64_t modulus);

// Metal GPU
void modmul_metal(const uint64_t* a, const uint64_t* b,
                  uint64_t* result, size_t n, uint64_t modulus);

// Hybrid approaches
void modmul_hybrid_neon_metal(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, size_t n, uint64_t modulus);

// NTT variants
void ntt_scalar(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles);
void ntt_neon(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles);
void ntt_barrett(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles);
void ntt_montgomery(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles);
void ntt_neon_unrolled(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles);
void ntt_metal(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles);
void ntt_amx(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles);

} // namespace backends

} // namespace benchmark
} // namespace fhe_accelerate
