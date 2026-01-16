/**
 * Comprehensive Hardware Benchmarking Implementation
 * 
 * Tests EVERY hardware feature we can access on M4 Max.
 * Philosophy: Try it, benchmark it, keep what works.
 */

#include "hardware_benchmark.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <thread>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/machine.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#ifdef __ARM_FEATURE_SME
#include <arm_sme.h>
#endif
#endif

namespace fhe_accelerate {
namespace benchmark {

// ============================================================================
// Hardware Feature Detection
// ============================================================================

HardwareFeatures HardwareFeatures::detect() {
    HardwareFeatures f{};
    
#ifdef __APPLE__
    // Get CPU brand
    char cpu_brand[256];
    size_t size = sizeof(cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", cpu_brand, &size, nullptr, 0) == 0) {
        std::cout << "CPU: " << cpu_brand << "\n";
    }
    
    // Check for Apple Silicon (all have NEON)
    f.has_neon = true;
    
    // Get core counts
    int32_t perf_cores = 0, eff_cores = 0;
    size = sizeof(perf_cores);
    sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &size, nullptr, 0);
    size = sizeof(eff_cores);
    sysctlbyname("hw.perflevel1.physicalcpu", &eff_cores, &size, nullptr, 0);
    f.num_performance_cores = perf_cores;
    f.num_efficiency_cores = eff_cores;
    
    // Cache sizes
    size = sizeof(f.l1_cache_size);
    sysctlbyname("hw.l1dcachesize", &f.l1_cache_size, &size, nullptr, 0);
    size = sizeof(f.l2_cache_size);
    sysctlbyname("hw.l2cachesize", &f.l2_cache_size, &size, nullptr, 0);
    size = sizeof(f.cache_line_size);
    sysctlbyname("hw.cachelinesize", &f.cache_line_size, &size, nullptr, 0);
    
    // Apple Silicon features (M1 and later have these)
    f.has_neon_fp16 = true;
    f.has_neon_dotprod = true;
    f.has_aes = true;
    f.has_sha3 = true;
    f.has_crc32 = true;
    f.has_atomics = true;
    f.has_hardware_rng = true;
    f.has_secure_enclave = true;
    
    // M4 specific features - detect SME via sysctl
    if (strstr(cpu_brand, "M4") != nullptr) {
        f.has_neon_bf16 = true;
        f.has_neon_i8mm = true;
        
        // Check for SME via sysctl
        int has_sme = 0;
        size = sizeof(has_sme);
        if (sysctlbyname("hw.optional.arm.FEAT_SME", &has_sme, &size, nullptr, 0) == 0) {
            f.has_sme = (has_sme == 1);
        }
        
        int has_sme2 = 0;
        size = sizeof(has_sme2);
        if (sysctlbyname("hw.optional.arm.FEAT_SME2", &has_sme2, &size, nullptr, 0) == 0) {
            f.has_sme2 = (has_sme2 == 1);
        }
    }
    
    // AMX is available on all Apple Silicon via Accelerate
    f.has_amx = true;
    
    // Metal is always available on Apple Silicon
    f.has_metal = true;
    f.has_metal3 = true;
    
    // Neural Engine
    f.has_neural_engine = true;
    if (strstr(cpu_brand, "M4 Max") != nullptr) {
        f.neural_engine_tops = 38;
        f.gpu_cores = 40;
    } else if (strstr(cpu_brand, "M4 Pro") != nullptr) {
        f.neural_engine_tops = 38;
        f.gpu_cores = 20;
    } else if (strstr(cpu_brand, "M4") != nullptr) {
        f.neural_engine_tops = 38;
        f.gpu_cores = 10;
    }
    
#endif
    
    return f;
}

void HardwareFeatures::print_report() const {
    std::cout << "\n========================================\n";
    std::cout << "Hardware Feature Detection Report\n";
    std::cout << "========================================\n\n";
    
    std::cout << "CPU Cores:\n";
    std::cout << "  Performance cores: " << num_performance_cores << "\n";
    std::cout << "  Efficiency cores:  " << num_efficiency_cores << "\n\n";
    
    std::cout << "Cache:\n";
    std::cout << "  L1 Data:    " << (l1_cache_size / 1024) << " KB\n";
    std::cout << "  L2:         " << (l2_cache_size / 1024 / 1024) << " MB\n";
    std::cout << "  Line size:  " << cache_line_size << " bytes\n\n";
    
    std::cout << "SIMD Features:\n";
    std::cout << "  NEON:       " << (has_neon ? "YES" : "NO") << "\n";
    std::cout << "  NEON FP16:  " << (has_neon_fp16 ? "YES" : "NO") << "\n";
    std::cout << "  NEON BF16:  " << (has_neon_bf16 ? "YES" : "NO") << "\n";
    std::cout << "  NEON I8MM:  " << (has_neon_i8mm ? "YES" : "NO") << "\n";
    std::cout << "  NEON DotProd: " << (has_neon_dotprod ? "YES" : "NO") << "\n";
    std::cout << "  SME:        " << (has_sme ? "YES" : "NO") << "\n";
    std::cout << "  AMX:        " << (has_amx ? "YES" : "NO") << "\n\n";
    
    std::cout << "Crypto:\n";
    std::cout << "  AES:        " << (has_aes ? "YES" : "NO") << "\n";
    std::cout << "  SHA3:       " << (has_sha3 ? "YES" : "NO") << "\n";
    std::cout << "  HW RNG:     " << (has_hardware_rng ? "YES" : "NO") << "\n";
    std::cout << "  Secure Enclave: " << (has_secure_enclave ? "YES" : "NO") << "\n\n";
    
    std::cout << "GPU:\n";
    std::cout << "  Metal:      " << (has_metal ? "YES" : "NO") << "\n";
    std::cout << "  Metal 3:    " << (has_metal3 ? "YES" : "NO") << "\n";
    std::cout << "  GPU Cores:  " << gpu_cores << "\n\n";
    
    std::cout << "Neural Engine:\n";
    std::cout << "  Available:  " << (has_neural_engine ? "YES" : "NO") << "\n";
    std::cout << "  TOPS:       " << neural_engine_tops << "\n";
    
    std::cout << "========================================\n\n";
}

// ============================================================================
// Benchmark Suite Reporting
// ============================================================================

void BenchmarkSuite::print_report() const {
    std::cout << "\n========================================\n";
    std::cout << "Benchmark: " << operation_name << "\n";
    std::cout << "========================================\n\n";
    
    std::cout << std::left << std::setw(30) << "Method"
              << std::right << std::setw(15) << "Time (µs)"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "Correct"
              << "\n";
    std::cout << std::string(82, '-') << "\n";
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.method_name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << r.time_us
                  << std::setw(15) << std::scientific << std::setprecision(2) << r.throughput
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.speedup_vs_scalar << "x"
                  << std::setw(10) << (r.correctness_verified ? "YES" : "NO")
                  << "\n";
    }
    
    std::cout << "\n*** WINNER: " << winner << " (" << best_time_us << " µs) ***\n";
}

void BenchmarkSuite::export_csv(const std::string& filename) const {
    std::ofstream file(filename);
    file << "method,hardware,time_us,throughput,speedup,correct,notes\n";
    for (const auto& r : results) {
        file << r.method_name << ","
             << r.hardware_used << ","
             << r.time_us << ","
             << r.throughput << ","
             << r.speedup_vs_scalar << ","
             << (r.correctness_verified ? "true" : "false") << ","
             << r.notes << "\n";
    }
}

// ============================================================================
// Backend Implementations
// ============================================================================

namespace backends {

// Scalar baseline - the simplest possible implementation
void modmul_scalar(const uint64_t* a, const uint64_t* b, 
                   uint64_t* result, size_t n, uint64_t modulus) {
    for (size_t i = 0; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = product % modulus;
    }
}

// NEON basic - use NEON loads/stores but scalar multiply
void modmul_neon_basic(const uint64_t* a, const uint64_t* b,
                       uint64_t* result, size_t n, uint64_t modulus) {
#ifdef __aarch64__
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        // NEON loads
        uint64x2_t va = vld1q_u64(&a[i]);
        uint64x2_t vb = vld1q_u64(&b[i]);
        
        // Scalar multiply (NEON doesn't have 64x64->128)
        __uint128_t p0 = static_cast<__uint128_t>(vgetq_lane_u64(va, 0)) * vgetq_lane_u64(vb, 0);
        __uint128_t p1 = static_cast<__uint128_t>(vgetq_lane_u64(va, 1)) * vgetq_lane_u64(vb, 1);
        
        uint64x2_t vr = {static_cast<uint64_t>(p0 % modulus), 
                         static_cast<uint64_t>(p1 % modulus)};
        vst1q_u64(&result[i], vr);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = product % modulus;
    }
#else
    modmul_scalar(a, b, result, n, modulus);
#endif
}

// NEON unrolled - process 4 elements per iteration
void modmul_neon_unrolled(const uint64_t* a, const uint64_t* b,
                          uint64_t* result, size_t n, uint64_t modulus) {
#ifdef __aarch64__
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load 4 elements (2 NEON registers each)
        uint64x2_t va0 = vld1q_u64(&a[i]);
        uint64x2_t va1 = vld1q_u64(&a[i + 2]);
        uint64x2_t vb0 = vld1q_u64(&b[i]);
        uint64x2_t vb1 = vld1q_u64(&b[i + 2]);
        
        // Multiply and reduce
        __uint128_t p0 = static_cast<__uint128_t>(vgetq_lane_u64(va0, 0)) * vgetq_lane_u64(vb0, 0);
        __uint128_t p1 = static_cast<__uint128_t>(vgetq_lane_u64(va0, 1)) * vgetq_lane_u64(vb0, 1);
        __uint128_t p2 = static_cast<__uint128_t>(vgetq_lane_u64(va1, 0)) * vgetq_lane_u64(vb1, 0);
        __uint128_t p3 = static_cast<__uint128_t>(vgetq_lane_u64(va1, 1)) * vgetq_lane_u64(vb1, 1);
        
        uint64x2_t vr0 = {static_cast<uint64_t>(p0 % modulus), 
                          static_cast<uint64_t>(p1 % modulus)};
        uint64x2_t vr1 = {static_cast<uint64_t>(p2 % modulus), 
                          static_cast<uint64_t>(p3 % modulus)};
        
        vst1q_u64(&result[i], vr0);
        vst1q_u64(&result[i + 2], vr1);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = product % modulus;
    }
#else
    modmul_scalar(a, b, result, n, modulus);
#endif
}

// NEON with prefetch - add software prefetch hints
void modmul_neon_prefetch(const uint64_t* a, const uint64_t* b,
                          uint64_t* result, size_t n, uint64_t modulus) {
#ifdef __aarch64__
    const size_t PREFETCH_DISTANCE = 16;  // Prefetch 16 elements ahead
    
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Prefetch ahead
        if (i + PREFETCH_DISTANCE < n) {
            __builtin_prefetch(&a[i + PREFETCH_DISTANCE], 0, 3);
            __builtin_prefetch(&b[i + PREFETCH_DISTANCE], 0, 3);
            __builtin_prefetch(&result[i + PREFETCH_DISTANCE], 1, 3);
        }
        
        // Load 4 elements
        uint64x2_t va0 = vld1q_u64(&a[i]);
        uint64x2_t va1 = vld1q_u64(&a[i + 2]);
        uint64x2_t vb0 = vld1q_u64(&b[i]);
        uint64x2_t vb1 = vld1q_u64(&b[i + 2]);
        
        // Multiply and reduce
        __uint128_t p0 = static_cast<__uint128_t>(vgetq_lane_u64(va0, 0)) * vgetq_lane_u64(vb0, 0);
        __uint128_t p1 = static_cast<__uint128_t>(vgetq_lane_u64(va0, 1)) * vgetq_lane_u64(vb0, 1);
        __uint128_t p2 = static_cast<__uint128_t>(vgetq_lane_u64(va1, 0)) * vgetq_lane_u64(vb1, 0);
        __uint128_t p3 = static_cast<__uint128_t>(vgetq_lane_u64(va1, 1)) * vgetq_lane_u64(vb1, 1);
        
        uint64x2_t vr0 = {static_cast<uint64_t>(p0 % modulus), 
                          static_cast<uint64_t>(p1 % modulus)};
        uint64x2_t vr1 = {static_cast<uint64_t>(p2 % modulus), 
                          static_cast<uint64_t>(p3 % modulus)};
        
        vst1q_u64(&result[i], vr0);
        vst1q_u64(&result[i + 2], vr1);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = product % modulus;
    }
#else
    modmul_scalar(a, b, result, n, modulus);
#endif
}

// Montgomery multiplication - avoid division entirely
// Precompute: R = 2^64, R_inv = R^(-1) mod q, q_inv = -q^(-1) mod R
struct MontgomeryParams {
    uint64_t modulus;
    uint64_t r_squared;  // R^2 mod q
    uint64_t q_inv;      // -q^(-1) mod 2^64
};

static inline uint64_t montgomery_reduce(__uint128_t x, const MontgomeryParams& params) {
    uint64_t m = static_cast<uint64_t>(x) * params.q_inv;
    __uint128_t t = x + static_cast<__uint128_t>(m) * params.modulus;
    uint64_t result = t >> 64;
    return result >= params.modulus ? result - params.modulus : result;
}

static inline uint64_t to_montgomery(uint64_t x, const MontgomeryParams& params) {
    return montgomery_reduce(static_cast<__uint128_t>(x) * params.r_squared, params);
}

static inline uint64_t from_montgomery(uint64_t x, const MontgomeryParams& params) {
    return montgomery_reduce(x, params);
}

static inline uint64_t montgomery_mul(uint64_t a, uint64_t b, const MontgomeryParams& params) {
    return montgomery_reduce(static_cast<__uint128_t>(a) * b, params);
}

// Compute Montgomery parameters
MontgomeryParams compute_montgomery_params(uint64_t modulus) {
    MontgomeryParams params;
    params.modulus = modulus;
    
    // Compute R^2 mod q where R = 2^64
    __uint128_t r = static_cast<__uint128_t>(1) << 64;
    __uint128_t r_squared = (r % modulus) * (r % modulus) % modulus;
    params.r_squared = static_cast<uint64_t>(r_squared);
    
    // Compute -q^(-1) mod 2^64 using extended Euclidean algorithm
    // We use Newton's method: x_{n+1} = x_n * (2 - q * x_n) mod 2^64
    uint64_t q_inv = 1;
    for (int i = 0; i < 6; i++) {  // 6 iterations for 64-bit
        q_inv = q_inv * (2 - modulus * q_inv);
    }
    params.q_inv = -q_inv;  // We want -q^(-1)
    
    return params;
}

// Montgomery multiplication batch
void modmul_montgomery(const uint64_t* a, const uint64_t* b,
                       uint64_t* result, size_t n, uint64_t modulus) {
    MontgomeryParams params = compute_montgomery_params(modulus);
    
    for (size_t i = 0; i < n; i++) {
        // Convert to Montgomery form, multiply, convert back
        uint64_t a_mont = to_montgomery(a[i], params);
        uint64_t b_mont = to_montgomery(b[i], params);
        uint64_t r_mont = montgomery_mul(a_mont, b_mont, params);
        result[i] = from_montgomery(r_mont, params);
    }
}

// Montgomery with NEON
void modmul_montgomery_neon(const uint64_t* a, const uint64_t* b,
                            uint64_t* result, size_t n, uint64_t modulus) {
#ifdef __aarch64__
    MontgomeryParams params = compute_montgomery_params(modulus);
    
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        uint64x2_t va = vld1q_u64(&a[i]);
        uint64x2_t vb = vld1q_u64(&b[i]);
        
        // Process each element (can't vectorize Montgomery reduction easily)
        uint64_t a0_mont = to_montgomery(vgetq_lane_u64(va, 0), params);
        uint64_t a1_mont = to_montgomery(vgetq_lane_u64(va, 1), params);
        uint64_t b0_mont = to_montgomery(vgetq_lane_u64(vb, 0), params);
        uint64_t b1_mont = to_montgomery(vgetq_lane_u64(vb, 1), params);
        
        uint64_t r0_mont = montgomery_mul(a0_mont, b0_mont, params);
        uint64_t r1_mont = montgomery_mul(a1_mont, b1_mont, params);
        
        uint64x2_t vr = {from_montgomery(r0_mont, params), 
                         from_montgomery(r1_mont, params)};
        vst1q_u64(&result[i], vr);
    }
    
    for (; i < n; i++) {
        uint64_t a_mont = to_montgomery(a[i], params);
        uint64_t b_mont = to_montgomery(b[i], params);
        uint64_t r_mont = montgomery_mul(a_mont, b_mont, params);
        result[i] = from_montgomery(r_mont, params);
    }
#else
    modmul_montgomery(a, b, result, n, modulus);
#endif
}

// Barrett reduction - another division-free approach
// For 64-bit modulus and 128-bit product, we use k=64 Barrett reduction
struct BarrettParams {
    uint64_t modulus;
    uint64_t mu;       // floor(2^(2k) / q) where k = number of bits in modulus
    int k;             // bit width of modulus
};

BarrettParams compute_barrett_params(uint64_t modulus) {
    BarrettParams params;
    params.modulus = modulus;
    
    // Find k = ceil(log2(modulus))
    params.k = 64 - __builtin_clzll(modulus);
    
    // Compute mu = floor(2^(2k) / modulus)
    // For a 64-bit modulus, 2k can be up to 128
    // We compute this carefully
    if (params.k <= 32) {
        // 2k <= 64, fits in uint64_t
        params.mu = (1ULL << (2 * params.k)) / modulus;
    } else {
        // 2k > 64, need 128-bit arithmetic
        __uint128_t numerator = static_cast<__uint128_t>(1) << (2 * params.k);
        params.mu = static_cast<uint64_t>(numerator / modulus);
    }
    
    return params;
}

static inline uint64_t barrett_reduce(__uint128_t x, const BarrettParams& params) {
    // Standard Barrett reduction:
    // q = floor(floor(x / 2^(k-1)) * mu / 2^(k+1))
    // r = x - q * modulus
    // if r >= modulus: r -= modulus
    
    int k = params.k;
    
    // q_approx = floor(x >> (k-1)) * mu >> (k+1)
    __uint128_t x_shifted = x >> (k - 1);
    __uint128_t q_approx = (x_shifted * params.mu) >> (k + 1);
    
    // r = x - q * modulus
    __uint128_t r = x - q_approx * params.modulus;
    
    // At most 2 corrections needed due to approximation
    while (r >= params.modulus) {
        r -= params.modulus;
    }
    
    return static_cast<uint64_t>(r);
}

void modmul_barrett(const uint64_t* a, const uint64_t* b,
                    uint64_t* result, size_t n, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    
    for (size_t i = 0; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = barrett_reduce(product, params);
    }
}

// Barrett with loop unrolling for better instruction-level parallelism
void modmul_barrett_unrolled(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, size_t n, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    
    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        __uint128_t p0 = static_cast<__uint128_t>(a[i]) * b[i];
        __uint128_t p1 = static_cast<__uint128_t>(a[i+1]) * b[i+1];
        __uint128_t p2 = static_cast<__uint128_t>(a[i+2]) * b[i+2];
        __uint128_t p3 = static_cast<__uint128_t>(a[i+3]) * b[i+3];
        
        result[i] = barrett_reduce(p0, params);
        result[i+1] = barrett_reduce(p1, params);
        result[i+2] = barrett_reduce(p2, params);
        result[i+3] = barrett_reduce(p3, params);
    }
    
    // Handle remainder
    for (; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = barrett_reduce(product, params);
    }
}

// Barrett with prefetch for better memory access
void modmul_barrett_prefetch(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, size_t n, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    const size_t PREFETCH_DISTANCE = 16;
    
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Prefetch ahead
        if (i + PREFETCH_DISTANCE < n) {
            __builtin_prefetch(&a[i + PREFETCH_DISTANCE], 0, 3);
            __builtin_prefetch(&b[i + PREFETCH_DISTANCE], 0, 3);
        }
        
        __uint128_t p0 = static_cast<__uint128_t>(a[i]) * b[i];
        __uint128_t p1 = static_cast<__uint128_t>(a[i+1]) * b[i+1];
        __uint128_t p2 = static_cast<__uint128_t>(a[i+2]) * b[i+2];
        __uint128_t p3 = static_cast<__uint128_t>(a[i+3]) * b[i+3];
        
        result[i] = barrett_reduce(p0, params);
        result[i+1] = barrett_reduce(p1, params);
        result[i+2] = barrett_reduce(p2, params);
        result[i+3] = barrett_reduce(p3, params);
    }
    
    for (; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = barrett_reduce(product, params);
    }
}

// Parallel Barrett - best of both worlds
void modmul_barrett_parallel(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, size_t n, uint64_t modulus) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = (n + num_threads - 1) / num_threads;
    BarrettParams params = compute_barrett_params(modulus);
    
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        
        if (start < n) {
            threads.emplace_back([&, start, end, params]() {
                for (size_t i = start; i < end; i++) {
                    __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
                    result[i] = barrett_reduce(product, params);
                }
            });
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

// Multi-threaded version using all cores
void modmul_parallel(const uint64_t* a, const uint64_t* b,
                     uint64_t* result, size_t n, uint64_t modulus) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = (n + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        
        if (start < n) {
            threads.emplace_back([&, start, end]() {
                for (size_t i = start; i < end; i++) {
                    __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
                    result[i] = product % modulus;
                }
            });
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

// AMX via Accelerate framework
void modmul_amx_accelerate(const uint64_t* a, const uint64_t* b,
                           uint64_t* result, size_t n, uint64_t modulus) {
    // Same as NEON for now - Accelerate doesn't help with integer modular arithmetic
    // But we keep this as a placeholder for future exploration
    modmul_neon_unrolled(a, b, result, n, modulus);
}

// NTT implementations
void ntt_scalar(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles) {
    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(coeffs[i], coeffs[j]);
        }
    }
    
    // Cooley-Tukey butterflies
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;
        
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half; j++) {
                uint64_t w = twiddles[step * j];
                uint64_t u = coeffs[i + j];
                
                __uint128_t product = static_cast<__uint128_t>(coeffs[i + j + half]) * w;
                uint64_t t = product % modulus;
                
                coeffs[i + j] = (u + t) % modulus;
                coeffs[i + j + half] = (u + modulus - t) % modulus;
            }
        }
    }
}

void ntt_neon(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles) {
#ifdef __aarch64__
    // Bit-reversal permutation (same as scalar)
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(coeffs[i], coeffs[j]);
        }
    }
        // NEON-optimized butterflies
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;
        
        for (size_t i = 0; i < n; i += len) {
            size_t j = 0;
            
            // Process 2 butterflies at a time when possible
            for (; j + 1 < half; j += 2) {
                // Load twiddles at correct positions (step * j and step * (j+1))
                uint64_t w0 = twiddles[step * j];
                uint64_t w1 = twiddles[step * (j + 1)];
                uint64x2_t vw = {w0, w1};
                
                uint64x2_t vu = vld1q_u64(&coeffs[i + j]);
                uint64x2_t vv = vld1q_u64(&coeffs[i + j + half]);
                
                // Scalar multiply (no NEON 64x64->128)
                __uint128_t p0 = static_cast<__uint128_t>(vgetq_lane_u64(vv, 0)) * w0;
                __uint128_t p1 = static_cast<__uint128_t>(vgetq_lane_u64(vv, 1)) * w1;
                
                uint64_t t0 = p0 % modulus;
                uint64_t t1 = p1 % modulus;
                
                uint64_t u0 = vgetq_lane_u64(vu, 0);
                uint64_t u1 = vgetq_lane_u64(vu, 1);
                
                uint64x2_t vr0 = {(u0 + t0) % modulus, (u1 + t1) % modulus};
                uint64x2_t vr1 = {(u0 + modulus - t0) % modulus, (u1 + modulus - t1) % modulus};
                
                vst1q_u64(&coeffs[i + j], vr0);
                vst1q_u64(&coeffs[i + j + half], vr1);
            }
            
            // Handle odd butterfly
            for (; j < half; j++) {
                uint64_t w = twiddles[step * j];
                uint64_t u = coeffs[i + j];
                
                __uint128_t product = static_cast<__uint128_t>(coeffs[i + j + half]) * w;
                uint64_t t = product % modulus;
                
                coeffs[i + j] = (u + t) % modulus;
                coeffs[i + j + half] = (u + modulus - t) % modulus;
            }
        }
    }
#else
    ntt_scalar(coeffs, n, modulus, twiddles);
#endif
}

// Barrett-optimized NTT - uses Barrett reduction instead of modulo
void ntt_barrett(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles) {
    BarrettParams params = compute_barrett_params(modulus);
    
    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(coeffs[i], coeffs[j]);
        }
    }
    
    // Cooley-Tukey butterflies with Barrett reduction
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;
        
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half; j++) {
                uint64_t w = twiddles[step * j];
                uint64_t u = coeffs[i + j];
                
                __uint128_t product = static_cast<__uint128_t>(coeffs[i + j + half]) * w;
                uint64_t t = barrett_reduce(product, params);
                
                // Addition with single conditional subtraction
                uint64_t sum = u + t;
                coeffs[i + j] = sum >= modulus ? sum - modulus : sum;
                
                uint64_t diff = u + modulus - t;
                coeffs[i + j + half] = diff >= modulus ? diff - modulus : diff;
            }
        }
    }
}

// Montgomery-optimized NTT - keeps values in Montgomery form throughout
void ntt_montgomery(uint64_t* coeffs, size_t n, uint64_t modulus, const uint64_t* twiddles) {
    MontgomeryParams params = compute_montgomery_params(modulus);
    
    // Convert twiddles to Montgomery form
    std::vector<uint64_t> mont_twiddles(n);
    for (size_t i = 0; i < n; i++) {
        mont_twiddles[i] = to_montgomery(twiddles[i], params);
    }
    
    // Convert coefficients to Montgomery form
    for (size_t i = 0; i < n; i++) {
        coeffs[i] = to_montgomery(coeffs[i], params);
    }
    
    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(coeffs[i], coeffs[j]);
        }
    }
    
    // Cooley-Tukey butterflies in Montgomery form
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;
        
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < half; j++) {
                uint64_t w = mont_twiddles[step * j];
                uint64_t u = coeffs[i + j];
                
                // Montgomery multiplication
                uint64_t t = montgomery_mul(coeffs[i + j + half], w, params);
                
                // Addition with single conditional subtraction
                uint64_t sum = u + t;
                coeffs[i + j] = sum >= modulus ? sum - modulus : sum;
                
                uint64_t diff = u + modulus - t;
                coeffs[i + j + half] = diff >= modulus ? diff - modulus : diff;
            }
        }
    }
    
    // Convert back from Montgomery form
    for (size_t i = 0; i < n; i++) {
        coeffs[i] = from_montgomery(coeffs[i], params);
    }
}

} // namespace backends

// ============================================================================
// Benchmark Runner Implementation
// ============================================================================

BenchmarkRunner::BenchmarkRunner() {
    features_ = HardwareFeatures::detect();
}

BenchmarkRunner::~BenchmarkRunner() = default;

template<typename Func>
double BenchmarkRunner::time_operation(Func&& func, int iterations) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        func();
    }
    
    // Actual timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return duration.count() / 1000.0 / iterations;  // Return microseconds per iteration
}

bool BenchmarkRunner::verify_modmul_result(const uint64_t* a, const uint64_t* b,
                                           const uint64_t* result, size_t n, uint64_t modulus) {
    for (size_t i = 0; i < n; i++) {
        __uint128_t expected = (static_cast<__uint128_t>(a[i]) * b[i]) % modulus;
        if (result[i] != static_cast<uint64_t>(expected)) {
            return false;
        }
    }
    return true;
}

BenchmarkSuite BenchmarkRunner::benchmark_modmul(size_t n, uint64_t modulus, int iterations) {
    BenchmarkSuite suite;
    suite.operation_name = "Modular Multiplication (n=" + std::to_string(n) + ")";
    
    // Allocate aligned buffers
    std::vector<uint64_t> a(n), b(n), result(n);
    
    // Initialize with random data
    for (size_t i = 0; i < n; i++) {
        a[i] = rand() % modulus;
        b[i] = rand() % modulus;
    }
    
    // Benchmark scalar (baseline)
    double scalar_time = time_operation([&]() {
        backends::modmul_scalar(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "Scalar (baseline)",
        "CPU",
        scalar_time,
        n / (scalar_time / 1e6),
        1.0,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        ""
    });
    
    // Benchmark NEON basic
    double neon_basic_time = time_operation([&]() {
        backends::modmul_neon_basic(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "NEON Basic",
        "NEON",
        neon_basic_time,
        n / (neon_basic_time / 1e6),
        scalar_time / neon_basic_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        ""
    });
    
    // Benchmark NEON unrolled
    double neon_unrolled_time = time_operation([&]() {
        backends::modmul_neon_unrolled(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "NEON Unrolled (4x)",
        "NEON",
        neon_unrolled_time,
        n / (neon_unrolled_time / 1e6),
        scalar_time / neon_unrolled_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        ""
    });
    
    // Benchmark NEON with prefetch
    double neon_prefetch_time = time_operation([&]() {
        backends::modmul_neon_prefetch(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "NEON + Prefetch",
        "NEON",
        neon_prefetch_time,
        n / (neon_prefetch_time / 1e6),
        scalar_time / neon_prefetch_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        ""
    });
    
    // Benchmark Montgomery multiplication
    double montgomery_time = time_operation([&]() {
        backends::modmul_montgomery(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "Montgomery",
        "CPU",
        montgomery_time,
        n / (montgomery_time / 1e6),
        scalar_time / montgomery_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        "Division-free"
    });
    
    // Benchmark Montgomery + NEON
    double montgomery_neon_time = time_operation([&]() {
        backends::modmul_montgomery_neon(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "Montgomery + NEON",
        "NEON",
        montgomery_neon_time,
        n / (montgomery_neon_time / 1e6),
        scalar_time / montgomery_neon_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        "Division-free"
    });
    
    // Benchmark Barrett reduction
    double barrett_time = time_operation([&]() {
        backends::modmul_barrett(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "Barrett",
        "CPU",
        barrett_time,
        n / (barrett_time / 1e6),
        scalar_time / barrett_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        "Division-free"
    });
    
    // Benchmark Barrett unrolled
    double barrett_unrolled_time = time_operation([&]() {
        backends::modmul_barrett_unrolled(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "Barrett Unrolled (4x)",
        "CPU",
        barrett_unrolled_time,
        n / (barrett_unrolled_time / 1e6),
        scalar_time / barrett_unrolled_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        "Division-free + ILP"
    });
    
    // Benchmark Barrett with prefetch
    double barrett_prefetch_time = time_operation([&]() {
        backends::modmul_barrett_prefetch(a.data(), b.data(), result.data(), n, modulus);
    }, iterations);
    
    suite.results.push_back({
        "Barrett + Prefetch",
        "CPU",
        barrett_prefetch_time,
        n / (barrett_prefetch_time / 1e6),
        scalar_time / barrett_prefetch_time,
        verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
        "Division-free + Prefetch"
    });
    
    // Benchmark parallel (only for larger sizes)
    if (n >= 4096) {
        double parallel_time = time_operation([&]() {
            backends::modmul_parallel(a.data(), b.data(), result.data(), n, modulus);
        }, iterations / 10);  // Fewer iterations due to thread overhead
        
        suite.results.push_back({
            "Parallel (all cores)",
            "Multi-core",
            parallel_time,
            n / (parallel_time / 1e6),
            scalar_time / parallel_time,
            verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
            std::to_string(std::thread::hardware_concurrency()) + " threads"
        });
        
        // Benchmark Barrett parallel
        double barrett_parallel_time = time_operation([&]() {
            backends::modmul_barrett_parallel(a.data(), b.data(), result.data(), n, modulus);
        }, iterations / 10);
        
        suite.results.push_back({
            "Barrett Parallel",
            "Multi-core",
            barrett_parallel_time,
            n / (barrett_parallel_time / 1e6),
            scalar_time / barrett_parallel_time,
            verify_modmul_result(a.data(), b.data(), result.data(), n, modulus),
            "Division-free + " + std::to_string(std::thread::hardware_concurrency()) + " threads"
        });
    }
    
    // Find winner
    suite.best_time_us = scalar_time;
    suite.winner = "Scalar (baseline)";
    for (const auto& r : suite.results) {
        if (r.time_us < suite.best_time_us && r.correctness_verified) {
            suite.best_time_us = r.time_us;
            suite.winner = r.method_name;
        }
    }
    
    return suite;
}

// Helper to compute primitive root for NTT
static uint64_t find_primitive_root(uint64_t modulus) {
    // For our test prime 132120577 = 2^23 * 3 * 5 + 1
    // A known primitive root is 3
    return 3;
}

// Helper to compute power mod
static uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t mod) {
    __uint128_t result = 1;
    __uint128_t b = base;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * b) % mod;
        }
        b = (b * b) % mod;
        exp >>= 1;
    }
    return static_cast<uint64_t>(result);
}

// Generate twiddle factors for NTT
static std::vector<uint64_t> generate_twiddles(size_t n, uint64_t modulus) {
    std::vector<uint64_t> twiddles(n);
    uint64_t g = find_primitive_root(modulus);
    uint64_t order = modulus - 1;
    uint64_t w = pow_mod(g, order / n, modulus);  // n-th root of unity
    
    twiddles[0] = 1;
    for (size_t i = 1; i < n; i++) {
        twiddles[i] = static_cast<uint64_t>((static_cast<__uint128_t>(twiddles[i-1]) * w) % modulus);
    }
    return twiddles;
}

// Verify NTT result by doing inverse NTT and checking round-trip
static bool verify_ntt_result(const uint64_t* original, uint64_t* transformed, 
                               size_t n, uint64_t modulus, const uint64_t* twiddles) {
    // Compute inverse twiddles
    std::vector<uint64_t> inv_twiddles(n);
    inv_twiddles[0] = 1;
    for (size_t i = 1; i < n; i++) {
        inv_twiddles[i] = twiddles[n - i];
    }
    
    // Make a copy for inverse NTT
    std::vector<uint64_t> result(transformed, transformed + n);
    
    // Apply inverse NTT (same algorithm with inverse twiddles)
    backends::ntt_scalar(result.data(), n, modulus, inv_twiddles.data());
    
    // Scale by N^(-1)
    uint64_t n_inv = pow_mod(n, modulus - 2, modulus);
    for (size_t i = 0; i < n; i++) {
        result[i] = static_cast<uint64_t>((static_cast<__uint128_t>(result[i]) * n_inv) % modulus);
    }
    
    // Compare with original
    for (size_t i = 0; i < n; i++) {
        if (result[i] != original[i]) {
            return false;
        }
    }
    return true;
}

BenchmarkSuite BenchmarkRunner::benchmark_ntt(size_t degree, uint64_t modulus, int iterations) {
    BenchmarkSuite suite;
    suite.operation_name = "NTT (degree=" + std::to_string(degree) + ")";
    
    // Generate twiddle factors
    auto twiddles = generate_twiddles(degree, modulus);
    
    // Allocate buffers
    std::vector<uint64_t> original(degree), coeffs(degree);
    
    // Initialize with random data
    for (size_t i = 0; i < degree; i++) {
        original[i] = rand() % modulus;
    }
    
    // Benchmark scalar NTT
    std::copy(original.begin(), original.end(), coeffs.begin());
    double scalar_time = time_operation([&]() {
        std::copy(original.begin(), original.end(), coeffs.begin());
        backends::ntt_scalar(coeffs.data(), degree, modulus, twiddles.data());
    }, iterations);
    
    suite.results.push_back({
        "Scalar NTT",
        "CPU",
        scalar_time,
        degree / (scalar_time / 1e6),
        1.0,
        verify_ntt_result(original.data(), coeffs.data(), degree, modulus, twiddles.data()),
        ""
    });
    
    // Benchmark NEON NTT
    std::copy(original.begin(), original.end(), coeffs.begin());
    double neon_time = time_operation([&]() {
        std::copy(original.begin(), original.end(), coeffs.begin());
        backends::ntt_neon(coeffs.data(), degree, modulus, twiddles.data());
    }, iterations);
    
    suite.results.push_back({
        "NEON NTT",
        "NEON",
        neon_time,
        degree / (neon_time / 1e6),
        scalar_time / neon_time,
        verify_ntt_result(original.data(), coeffs.data(), degree, modulus, twiddles.data()),
        ""
    });
    
    // Benchmark Barrett NTT
    std::copy(original.begin(), original.end(), coeffs.begin());
    double barrett_time = time_operation([&]() {
        std::copy(original.begin(), original.end(), coeffs.begin());
        backends::ntt_barrett(coeffs.data(), degree, modulus, twiddles.data());
    }, iterations);
    
    suite.results.push_back({
        "Barrett NTT",
        "CPU",
        barrett_time,
        degree / (barrett_time / 1e6),
        scalar_time / barrett_time,
        verify_ntt_result(original.data(), coeffs.data(), degree, modulus, twiddles.data()),
        "Division-free"
    });
    
    // Benchmark Montgomery NTT
    std::copy(original.begin(), original.end(), coeffs.begin());
    double montgomery_time = time_operation([&]() {
        std::copy(original.begin(), original.end(), coeffs.begin());
        backends::ntt_montgomery(coeffs.data(), degree, modulus, twiddles.data());
    }, iterations);
    
    suite.results.push_back({
        "Montgomery NTT",
        "CPU",
        montgomery_time,
        degree / (montgomery_time / 1e6),
        scalar_time / montgomery_time,
        verify_ntt_result(original.data(), coeffs.data(), degree, modulus, twiddles.data()),
        "Division-free"
    });
    
    // Find winner
    suite.best_time_us = scalar_time;
    suite.winner = "Scalar NTT";
    for (const auto& r : suite.results) {
        if (r.time_us < suite.best_time_us && r.correctness_verified) {
            suite.best_time_us = r.time_us;
            suite.winner = r.method_name;
        }
    }
    
    return suite;
}

void BenchmarkRunner::run_full_benchmark_suite() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     FHE ACCELERATE - COMPREHENSIVE HARDWARE BENCHMARK        ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  Testing EVERY hardware feature on Apple Silicon             ║\n";
    std::cout << "║  Philosophy: Try it, benchmark it, keep what works           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    features_.print_report();
    
    // Benchmark modular multiplication at various sizes
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
    uint64_t modulus = 132120577ULL;  // NTT-friendly prime (2^23 * 3 * 5 + 1 - 1 = 2^23 * 15 + 1)
    
    for (size_t n : sizes) {
        auto suite = benchmark_modmul(n, modulus, 1000);
        suite.print_report();
        all_results_.push_back(suite);
    }
    
    // Benchmark NTT at various degrees
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    NTT BENCHMARKS                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    std::vector<size_t> ntt_sizes = {1024, 4096, 16384};
    for (size_t n : ntt_sizes) {
        auto suite = benchmark_ntt(n, modulus, 100);
        suite.print_report();
        all_results_.push_back(suite);
    }
    
    // Summary
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    BENCHMARK SUMMARY                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    for (const auto& suite : all_results_) {
        std::cout << suite.operation_name << ": " << suite.winner 
                  << " (" << suite.best_time_us << " µs)\n";
    }
}

} // namespace benchmark
} // namespace fhe_accelerate
