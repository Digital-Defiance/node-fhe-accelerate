/**
 * Adaptive Hardware Dispatcher Implementation
 * 
 * Uses benchmark-winning algorithms:
 * - Barrett Unrolled (4x) for batch modular multiplication
 * - Montgomery NTT for forward/inverse NTT
 */

#include "adaptive_dispatcher.h"
#include <iostream>
#include <cstring>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace fhe_accelerate {

// ============================================================================
// Barrett Reduction (for batch modmul)
// ============================================================================

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

static inline uint64_t barrett_reduce(__uint128_t x, const BarrettParams& params) {
    int k = params.k;
    __uint128_t x_shifted = x >> (k - 1);
    __uint128_t q_approx = (x_shifted * params.mu) >> (k + 1);
    __uint128_t r = x - q_approx * params.modulus;
    
    while (r >= params.modulus) {
        r -= params.modulus;
    }
    
    return static_cast<uint64_t>(r);
}

void fast_modmul_batch(const uint64_t* a, const uint64_t* b,
                       uint64_t* result, size_t n, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    
    size_t i = 0;
    // Process 4 elements at a time (benchmark winner)
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
    
    for (; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = barrett_reduce(product, params);
    }
}

// ============================================================================
// Montgomery Form (for NTT - benchmark winner with 2x speedup)
// ============================================================================

struct MontgomeryParams {
    uint64_t modulus;
    uint64_t r_squared;
    uint64_t q_inv;
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

static MontgomeryParams compute_montgomery_params(uint64_t modulus) {
    MontgomeryParams params;
    params.modulus = modulus;
    
    __uint128_t r = static_cast<__uint128_t>(1) << 64;
    __uint128_t r_squared = (r % modulus) * (r % modulus) % modulus;
    params.r_squared = static_cast<uint64_t>(r_squared);
    
    uint64_t q_inv = 1;
    for (int i = 0; i < 6; i++) {
        q_inv = q_inv * (2 - modulus * q_inv);
    }
    params.q_inv = -q_inv;
    
    return params;
}

void fast_ntt_forward(uint64_t* coeffs, size_t n, uint64_t modulus,
                      const uint64_t* twiddles) {
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
                uint64_t t = montgomery_mul(coeffs[i + j + half], w, params);
                
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

void fast_ntt_inverse(uint64_t* coeffs, size_t n, uint64_t modulus,
                      const uint64_t* inv_twiddles) {
    // Use the same Montgomery-optimized algorithm
    fast_ntt_forward(coeffs, n, modulus, inv_twiddles);
    
    // Scale by N^(-1)
    uint64_t n_inv = 1;
    uint64_t base = n;
    uint64_t exp = modulus - 2;
    while (exp > 0) {
        if (exp & 1) {
            n_inv = static_cast<uint64_t>((static_cast<__uint128_t>(n_inv) * base) % modulus);
        }
        base = static_cast<uint64_t>((static_cast<__uint128_t>(base) * base) % modulus);
        exp >>= 1;
    }
    
    for (size_t i = 0; i < n; i++) {
        coeffs[i] = static_cast<uint64_t>((static_cast<__uint128_t>(coeffs[i]) * n_inv) % modulus);
    }
}

// ============================================================================
// Precomputation Helpers
// ============================================================================

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

std::vector<uint64_t> precompute_twiddles(size_t n, uint64_t modulus,
                                          uint64_t primitive_root) {
    std::vector<uint64_t> twiddles(n);
    uint64_t order = modulus - 1;
    uint64_t w = pow_mod(primitive_root, order / n, modulus);
    
    twiddles[0] = 1;
    for (size_t i = 1; i < n; i++) {
        twiddles[i] = static_cast<uint64_t>((static_cast<__uint128_t>(twiddles[i-1]) * w) % modulus);
    }
    return twiddles;
}

std::vector<uint64_t> precompute_inv_twiddles(size_t n, uint64_t modulus,
                                               uint64_t primitive_root) {
    auto twiddles = precompute_twiddles(n, modulus, primitive_root);
    std::vector<uint64_t> inv_twiddles(n);
    
    inv_twiddles[0] = 1;
    for (size_t i = 1; i < n; i++) {
        inv_twiddles[i] = twiddles[n - i];
    }
    return inv_twiddles;
}

// ============================================================================
// Hardware Detection
// ============================================================================

HardwareCapabilities HardwareCapabilities::detect() {
    HardwareCapabilities caps{};
    
#ifdef __APPLE__
    caps.has_neon = true;
    
    int has_sme = 0;
    size_t size = sizeof(has_sme);
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &has_sme, &size, nullptr, 0) == 0) {
        caps.has_sme = (has_sme == 1);
    }
    
    caps.has_amx = true;  // Available via Accelerate
    caps.has_metal = true;
    
    int32_t perf_cores = 0, eff_cores = 0;
    size = sizeof(perf_cores);
    sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &size, nullptr, 0);
    size = sizeof(eff_cores);
    sysctlbyname("hw.perflevel1.physicalcpu", &eff_cores, &size, nullptr, 0);
    caps.num_cores = perf_cores + eff_cores;
    
    char cpu_brand[256];
    size = sizeof(cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", cpu_brand, &size, nullptr, 0) == 0) {
        if (strstr(cpu_brand, "M4 Max") != nullptr) {
            caps.gpu_cores = 40;
        } else if (strstr(cpu_brand, "M4 Pro") != nullptr) {
            caps.gpu_cores = 20;
        } else if (strstr(cpu_brand, "M4") != nullptr) {
            caps.gpu_cores = 10;
        }
    }
#endif
    
    return caps;
}

void HardwareCapabilities::print() const {
    std::cout << "Hardware Capabilities:\n";
    std::cout << "  NEON: " << (has_neon ? "YES" : "NO") << "\n";
    std::cout << "  SME:  " << (has_sme ? "YES" : "NO") << "\n";
    std::cout << "  AMX:  " << (has_amx ? "YES" : "NO") << "\n";
    std::cout << "  Metal: " << (has_metal ? "YES" : "NO") << "\n";
    std::cout << "  CPU Cores: " << num_cores << "\n";
    std::cout << "  GPU Cores: " << gpu_cores << "\n";
}

} // namespace fhe_accelerate
