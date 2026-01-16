/**
 * Advanced SIMD Optimizations Implementation
 * 
 * Requirements 14.26, 14.27, 14.28, 14.29
 */

#include "advanced_simd.h"
#include <chrono>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace fhe_accelerate {
namespace advanced_simd {

// Barrett reduction parameters
struct BarrettParams {
    uint64_t modulus;
    uint64_t mu;
    int k;
};

static BarrettParams compute_barrett_params(uint64_t modulus) {
    BarrettParams params;
    params.modulus = modulus;
    params.k = 64 - __builtin_clzll(modulus);
    
    if (params.k <= 32) {
        params.mu = (1ULL << (2 * params.k)) / modulus;
    } else {
        __uint128_t numerator = static_cast<__uint128_t>(1) << (2 * params.k);
        params.mu = static_cast<uint64_t>(numerator / modulus);
    }
    
    return params;
}

static inline uint64_t barrett_reduce_scalar(__uint128_t x, const BarrettParams& params) {
    int k = params.k;
    __uint128_t x_shifted = x >> (k - 1);
    __uint128_t q_approx = (x_shifted * params.mu) >> (k + 1);
    __uint128_t r = x - q_approx * params.modulus;
    
    while (r >= params.modulus) {
        r -= params.modulus;
    }
    
    return static_cast<uint64_t>(r);
}

bool sve2_available() {
#ifdef __APPLE__
    int has_sve2 = 0;
    size_t size = sizeof(has_sve2);
    if (sysctlbyname("hw.optional.arm.FEAT_SVE2", &has_sve2, &size, nullptr, 0) == 0) {
        return has_sve2 == 1;
    }
#endif
    return false;
}

bool neon_available() {
#ifdef __aarch64__
    return true;
#else
    return false;
#endif
}

void gather_load(const uint64_t* data, const uint32_t* indices, 
                 uint64_t* output, size_t count) {
    // NEON doesn't have native gather, so we use scalar loads
    // SVE2 would have native gather support
    for (size_t i = 0; i < count; i++) {
        output[i] = data[indices[i]];
    }
}

void scatter_store(const uint64_t* data, const uint32_t* indices,
                   uint64_t* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        output[indices[i]] = data[i];
    }
}

void predicated_reduce(uint64_t* data, size_t count, uint64_t modulus) {
#ifdef __aarch64__
    // Use NEON comparison to create mask, then conditional subtract
    uint64x2_t vmod = vdupq_n_u64(modulus);
    
    size_t i = 0;
    for (; i + 1 < count; i += 2) {
        uint64x2_t vdata = vld1q_u64(&data[i]);
        
        // Compare: mask = (data >= modulus)
        uint64x2_t mask = vcgeq_u64(vdata, vmod);
        
        // Conditional subtract: data = data - (modulus & mask)
        uint64x2_t sub = vandq_u64(vmod, mask);
        vdata = vsubq_u64(vdata, sub);
        
        vst1q_u64(&data[i], vdata);
    }
    
    // Handle remainder
    for (; i < count; i++) {
        if (data[i] >= modulus) {
            data[i] -= modulus;
        }
    }
#else
    for (size_t i = 0; i < count; i++) {
        if (data[i] >= modulus) {
            data[i] -= modulus;
        }
    }
#endif
}

uint64_t horizontal_inner_product(const uint64_t* a, const uint64_t* b,
                                   size_t count, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    
    __uint128_t sum = 0;
    
#ifdef __aarch64__
    // Process 2 elements at a time
    size_t i = 0;
    for (; i + 1 < count; i += 2) {
        uint64x2_t va = vld1q_u64(&a[i]);
        uint64x2_t vb = vld1q_u64(&b[i]);
        
        // Scalar 128-bit multiply and accumulate
        __uint128_t prod0 = static_cast<__uint128_t>(vgetq_lane_u64(va, 0)) * vgetq_lane_u64(vb, 0);
        __uint128_t prod1 = static_cast<__uint128_t>(vgetq_lane_u64(va, 1)) * vgetq_lane_u64(vb, 1);
        
        sum += prod0 + prod1;
    }
    
    // Handle remainder
    for (; i < count; i++) {
        sum += static_cast<__uint128_t>(a[i]) * b[i];
    }
#else
    for (size_t i = 0; i < count; i++) {
        sum += static_cast<__uint128_t>(a[i]) * b[i];
    }
#endif
    
    return barrett_reduce_scalar(sum, params);
}

void montgomery_mul_vec(const uint64_t* a, const uint64_t* b,
                        uint64_t* result, size_t count, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    
#ifdef __aarch64__
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        uint64x2_t va0 = vld1q_u64(&a[i]);
        uint64x2_t va1 = vld1q_u64(&a[i + 2]);
        uint64x2_t vb0 = vld1q_u64(&b[i]);
        uint64x2_t vb1 = vld1q_u64(&b[i + 2]);
        
        __uint128_t p0 = static_cast<__uint128_t>(vgetq_lane_u64(va0, 0)) * vgetq_lane_u64(vb0, 0);
        __uint128_t p1 = static_cast<__uint128_t>(vgetq_lane_u64(va0, 1)) * vgetq_lane_u64(vb0, 1);
        __uint128_t p2 = static_cast<__uint128_t>(vgetq_lane_u64(va1, 0)) * vgetq_lane_u64(vb1, 0);
        __uint128_t p3 = static_cast<__uint128_t>(vgetq_lane_u64(va1, 1)) * vgetq_lane_u64(vb1, 1);
        
        result[i] = barrett_reduce_scalar(p0, params);
        result[i + 1] = barrett_reduce_scalar(p1, params);
        result[i + 2] = barrett_reduce_scalar(p2, params);
        result[i + 3] = barrett_reduce_scalar(p3, params);
    }
    
    for (; i < count; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = barrett_reduce_scalar(product, params);
    }
#else
    for (size_t i = 0; i < count; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = barrett_reduce_scalar(product, params);
    }
#endif
}

void barrett_reduce_vec(uint64_t* data, size_t count, uint64_t modulus) {
    predicated_reduce(data, count, modulus);
}

void ntt_butterfly_gather_scatter(uint64_t* coeffs, size_t degree, size_t stage,
                                   const uint64_t* twiddles, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    
    size_t m = 1ULL << stage;
    size_t step = degree / (2 * m);
    
    for (size_t k = 0; k < degree; k += 2 * m) {
        for (size_t j = 0; j < m; j++) {
            uint64_t w = twiddles[step * j];
            uint64_t u = coeffs[k + j];
            uint64_t v = coeffs[k + j + m];
            
            __uint128_t wv = static_cast<__uint128_t>(v) * w;
            uint64_t t = barrett_reduce_scalar(wv, params);
            
            coeffs[k + j] = (u + t) % modulus;
            coeffs[k + j + m] = (u + modulus - t) % modulus;
        }
    }
}

SIMDBenchmark benchmark_predication_vs_branching(size_t count, uint64_t modulus) {
    SIMDBenchmark result;
    
    std::vector<uint64_t> data(count);
    for (size_t i = 0; i < count; i++) {
        data[i] = rand() % (2 * modulus);
    }
    
    // Benchmark branching
    std::vector<uint64_t> data_copy = data;
    auto start_branch = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        for (size_t i = 0; i < count; i++) {
            if (data_copy[i] >= modulus) {
                data_copy[i] -= modulus;
            }
        }
    }
    auto end_branch = std::chrono::high_resolution_clock::now();
    result.branching_time_us = std::chrono::duration<double, std::micro>(end_branch - start_branch).count() / 1000.0;
    
    // Benchmark predication
    data_copy = data;
    auto start_pred = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        predicated_reduce(data_copy.data(), count, modulus);
    }
    auto end_pred = std::chrono::high_resolution_clock::now();
    result.predicated_time_us = std::chrono::duration<double, std::micro>(end_pred - start_pred).count() / 1000.0;
    
    result.speedup = result.branching_time_us / result.predicated_time_us;
    
    return result;
}

} // namespace advanced_simd
} // namespace fhe_accelerate
