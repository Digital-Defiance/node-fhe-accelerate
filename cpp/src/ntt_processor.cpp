/**
 * NTT Processor Implementation
 * 
 * Implements the Number Theoretic Transform using the Cooley-Tukey algorithm.
 * 
 * Design Reference: Section 2 - NTT Processor
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.6
 */

#include "ntt_processor.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace fhe_accelerate {

// ============================================================================
// Static Utility Functions
// ============================================================================

bool NTTProcessor::is_power_of_two(uint32_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

uint32_t NTTProcessor::log2_pow2(uint32_t n) {
    uint32_t log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

uint32_t NTTProcessor::bit_reverse(uint32_t index, uint32_t bits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < bits; i++) {
        result = (result << 1) | (index & 1);
        index >>= 1;
    }
    return result;
}

uint64_t NTTProcessor::mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t tmp = static_cast<__uint128_t>(result) * base;
            result = tmp % mod;
        }
        __uint128_t tmp = static_cast<__uint128_t>(base) * base;
        base = tmp % mod;
        exp >>= 1;
    }
    
    return result;
}

uint64_t NTTProcessor::mod_inverse(uint64_t a, uint64_t m) {
    if (m == 0) {
        throw std::invalid_argument("Modulus cannot be zero");
    }
    if (m == 1) return 0;
    
    int64_t m0 = static_cast<int64_t>(m);
    int64_t x0 = 0, x1 = 1;
    int64_t a_signed = static_cast<int64_t>(a % m);
    int64_t m_signed = static_cast<int64_t>(m);
    
    while (a_signed > 1) {
        int64_t q = a_signed / m_signed;
        int64_t t = m_signed;
        
        m_signed = a_signed % m_signed;
        a_signed = t;
        t = x0;
        
        x0 = x1 - q * x0;
        x1 = t;
    }
    
    if (x1 < 0) x1 += m0;
    
    return static_cast<uint64_t>(x1);
}

uint64_t NTTProcessor::find_primitive_root(uint32_t degree, uint64_t modulus) {
    // For NTT-friendly primes q ≡ 1 (mod 2N), we need to find a primitive
    // 2N-th root of unity.
    //
    // Algorithm:
    // 1. Find a generator g of the multiplicative group Z_q*
    // 2. Compute ω = g^((q-1)/(2N)) which is a primitive 2N-th root
    
    uint64_t two_n = static_cast<uint64_t>(degree) * 2;
    
    // Check that q ≡ 1 (mod 2N)
    if ((modulus - 1) % two_n != 0) {
        throw std::invalid_argument("Modulus is not NTT-friendly: q ≢ 1 (mod 2N)");
    }
    
    uint64_t exponent = (modulus - 1) / two_n;
    
    // Find a generator by trying small primes
    // A generator g satisfies: g^((q-1)/p) ≢ 1 (mod q) for all prime factors p of q-1
    for (uint64_t g = 2; g < modulus; g++) {
        // Compute candidate root: ω = g^((q-1)/(2N))
        uint64_t omega = mod_pow(g, exponent, modulus);
        
        // Verify it's a primitive 2N-th root:
        // 1. ω^(2N) ≡ 1 (mod q)
        // 2. ω^N ≡ -1 (mod q) (i.e., ω^N ≡ q-1 (mod q))
        
        uint64_t omega_n = mod_pow(omega, degree, modulus);
        uint64_t omega_2n = mod_pow(omega, two_n, modulus);
        
        if (omega_2n == 1 && omega_n == modulus - 1) {
            return omega;
        }
    }
    
    throw std::invalid_argument("Could not find primitive root for given parameters");
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

NTTProcessor::NTTProcessor(uint32_t degree, uint64_t modulus)
    : degree_(degree)
    , log_degree_(log2_pow2(degree))
    , modulus_(modulus)
    , twiddles_computed_(false) {
    
    // Validate degree is power of 2
    if (!is_power_of_two(degree)) {
        throw std::invalid_argument("Polynomial degree must be a power of 2");
    }
    
    // Validate degree is in supported range
    if (degree < 4 || degree > 65536) {
        throw std::invalid_argument("Polynomial degree must be between 4 and 65536");
    }
    
    // Validate modulus is odd (required for Montgomery arithmetic)
    if ((modulus & 1) == 0) {
        throw std::invalid_argument("Modulus must be odd");
    }
    
    // Create modular arithmetic helper
    mod_arith_ = std::make_unique<ModularArithmetic>(modulus);
    
    // Precompute twiddle factors
    precompute_twiddles();
}

NTTProcessor::~NTTProcessor() = default;

// ============================================================================
// Twiddle Factor Precomputation
// ============================================================================

void NTTProcessor::precompute_twiddles(uint64_t primitive_root) {
    // Find primitive root if not provided
    if (primitive_root == 0) {
        primitive_root = find_primitive_root(degree_, modulus_);
    }
    
    twiddles_.primitive_root = primitive_root;
    twiddles_.modulus = modulus_;
    twiddles_.degree = degree_;
    
    // Compute inverse of primitive root
    twiddles_.inv_primitive_root = mod_inverse(primitive_root, modulus_);
    
    // Compute N^(-1) mod q for inverse NTT scaling
    twiddles_.inv_n = mod_inverse(degree_, modulus_);
    
    // Allocate twiddle factor arrays
    twiddles_.forward.resize(degree_);
    twiddles_.inverse.resize(degree_);
    
    // Compute forward twiddle factors: ω^0, ω^1, ω^2, ..., ω^(N-1)
    uint64_t omega = primitive_root;
    twiddles_.forward[0] = 1;
    for (uint32_t i = 1; i < degree_; i++) {
        __uint128_t tmp = static_cast<__uint128_t>(twiddles_.forward[i - 1]) * omega;
        twiddles_.forward[i] = tmp % modulus_;
    }
    
    // Compute inverse twiddle factors: ω^(-0), ω^(-1), ω^(-2), ..., ω^(-(N-1))
    uint64_t omega_inv = twiddles_.inv_primitive_root;
    twiddles_.inverse[0] = 1;
    for (uint32_t i = 1; i < degree_; i++) {
        __uint128_t tmp = static_cast<__uint128_t>(twiddles_.inverse[i - 1]) * omega_inv;
        twiddles_.inverse[i] = tmp % modulus_;
    }
    
    // Store twiddle factors in standard form (not Montgomery)
    twiddles_.in_montgomery_form = false;
    
    twiddles_computed_ = true;
}

// ============================================================================
// Bit-Reversal Permutation
// ============================================================================

void NTTProcessor::bit_reverse_permutation(uint64_t* coeffs, size_t n) {
    uint32_t bits = log2_pow2(static_cast<uint32_t>(n));
    
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = bit_reverse(i, bits);
        if (i < j) {
            std::swap(coeffs[i], coeffs[j]);
        }
    }
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

inline void NTTProcessor::ntt_butterfly(uint64_t& a, uint64_t& b, uint64_t omega) {
    // Cooley-Tukey butterfly:
    // a' = a + ω * b
    // b' = a - ω * b
    
    // omega and b should be in Montgomery form
    uint64_t omega_b = mod_arith_->montgomery_mul(omega, b);
    
    // Compute a + omega_b and a - omega_b
    uint64_t a_plus = mod_arith_->mod_add(a, omega_b);
    uint64_t a_minus = mod_arith_->mod_sub(a, omega_b);
    
    a = a_plus;
    b = a_minus;
}

inline void NTTProcessor::inverse_ntt_butterfly(uint64_t& a, uint64_t& b, uint64_t omega_inv) {
    // Gentleman-Sande butterfly (inverse):
    // a' = a + b
    // b' = (a - b) * ω^(-1)
    
    uint64_t a_plus = mod_arith_->mod_add(a, b);
    uint64_t a_minus = mod_arith_->mod_sub(a, b);
    uint64_t b_new = mod_arith_->montgomery_mul(a_minus, omega_inv);
    
    a = a_plus;
    b = b_new;
}

// ============================================================================
// Forward NTT Implementation
// ============================================================================

void NTTProcessor::forward_ntt(uint64_t* coeffs, size_t n) {
    if (n != degree_) {
        throw std::invalid_argument("Coefficient count must equal polynomial degree");
    }
    
    if (!twiddles_computed_) {
        throw std::runtime_error("Twiddle factors not computed");
    }
    
    // Bit-reversal permutation first
    bit_reverse_permutation(coeffs, n);
    
    // Cooley-Tukey iterative NTT
    // Process log2(n) stages
    for (uint32_t stage = 0; stage < log_degree_; stage++) {
        uint32_t m = 1 << stage;           // Distance between butterfly pairs
        uint32_t half_m = m;               // Half the butterfly group size
        uint32_t group_size = 2 * m;       // Full butterfly group size
        
        // Process each butterfly group
        for (uint32_t k = 0; k < n; k += group_size) {
            // Process butterflies within this group
            for (uint32_t j = 0; j < half_m; j++) {
                // Twiddle factor index: j * (n / group_size)
                uint32_t twiddle_idx = j * (n / group_size);
                uint64_t omega = twiddles_.forward[twiddle_idx];
                
                uint32_t idx_a = k + j;
                uint32_t idx_b = k + j + half_m;
                
                // Standard Cooley-Tukey butterfly (without Montgomery)
                // a' = a + ω * b
                // b' = a - ω * b
                uint64_t a = coeffs[idx_a];
                uint64_t b = coeffs[idx_b];
                
                // Compute ω * b mod q using 128-bit arithmetic
                __uint128_t omega_b_128 = static_cast<__uint128_t>(omega) * b;
                uint64_t omega_b = omega_b_128 % modulus_;
                
                // Compute a + omega_b and a - omega_b
                uint64_t a_plus = mod_arith_->mod_add(a, omega_b);
                uint64_t a_minus = mod_arith_->mod_sub(a, omega_b);
                
                coeffs[idx_a] = a_plus;
                coeffs[idx_b] = a_minus;
            }
        }
    }
}

void NTTProcessor::forward_ntt(const uint64_t* input, uint64_t* output, size_t n) {
    // Copy input to output
    std::memcpy(output, input, n * sizeof(uint64_t));
    
    // Perform in-place NTT
    forward_ntt(output, n);
}

// ============================================================================
// Inverse NTT Implementation
// ============================================================================

void NTTProcessor::inverse_ntt(uint64_t* coeffs, size_t n) {
    if (n != degree_) {
        throw std::invalid_argument("Coefficient count must equal polynomial degree");
    }
    
    if (!twiddles_computed_) {
        throw std::runtime_error("Twiddle factors not computed");
    }
    
    // Gentleman-Sande iterative inverse NTT
    // Process log2(n) stages in reverse order
    for (int32_t stage = log_degree_ - 1; stage >= 0; stage--) {
        uint32_t m = 1 << stage;
        uint32_t half_m = m;
        uint32_t group_size = 2 * m;
        
        // Process each butterfly group
        for (uint32_t k = 0; k < n; k += group_size) {
            // Process butterflies within this group
            for (uint32_t j = 0; j < half_m; j++) {
                // Twiddle factor index
                uint32_t twiddle_idx = j * (n / group_size);
                uint64_t omega_inv = twiddles_.inverse[twiddle_idx];
                
                uint32_t idx_a = k + j;
                uint32_t idx_b = k + j + half_m;
                
                // Gentleman-Sande butterfly (inverse):
                // a' = a + b
                // b' = (a - b) * ω^(-1)
                uint64_t a = coeffs[idx_a];
                uint64_t b = coeffs[idx_b];
                
                uint64_t a_plus = mod_arith_->mod_add(a, b);
                uint64_t a_minus = mod_arith_->mod_sub(a, b);
                
                // Compute (a - b) * omega_inv mod q
                __uint128_t b_new_128 = static_cast<__uint128_t>(a_minus) * omega_inv;
                uint64_t b_new = b_new_128 % modulus_;
                
                coeffs[idx_a] = a_plus;
                coeffs[idx_b] = b_new;
            }
        }
    }
    
    // Bit-reversal permutation
    bit_reverse_permutation(coeffs, n);
    
    // Scale by N^(-1)
    uint64_t inv_n = twiddles_.inv_n;
    for (size_t i = 0; i < n; i++) {
        __uint128_t scaled = static_cast<__uint128_t>(coeffs[i]) * inv_n;
        coeffs[i] = scaled % modulus_;
    }
}

void NTTProcessor::inverse_ntt(const uint64_t* input, uint64_t* output, size_t n) {
    // Copy input to output
    std::memcpy(output, input, n * sizeof(uint64_t));
    
    // Perform in-place inverse NTT
    inverse_ntt(output, n);
}

// ============================================================================
// Batch Operations
// ============================================================================

void NTTProcessor::forward_ntt_batch(uint64_t** coeffs_batch, size_t batch_size, size_t n) {
    // Process each polynomial in the batch
    // TODO: Optimize with GPU parallelism via Metal
    for (size_t i = 0; i < batch_size; i++) {
        forward_ntt(coeffs_batch[i], n);
    }
}

void NTTProcessor::inverse_ntt_batch(uint64_t** coeffs_batch, size_t batch_size, size_t n) {
    // Process each polynomial in the batch
    // TODO: Optimize with GPU parallelism via Metal
    for (size_t i = 0; i < batch_size; i++) {
        inverse_ntt(coeffs_batch[i], n);
    }
}


// ============================================================================
// NEON-Optimized NTT Implementation
// ============================================================================

#ifdef __ARM_NEON

void NTTProcessor::forward_ntt_neon(uint64_t* coeffs, size_t n) {
    if (n != degree_) {
        throw std::invalid_argument("Coefficient count must equal polynomial degree");
    }
    
    if (!twiddles_computed_) {
        throw std::runtime_error("Twiddle factors not computed");
    }
    
    // Bit-reversal permutation
    bit_reverse_permutation(coeffs, n);
    
    // NEON-optimized Cooley-Tukey NTT
    // Process 2 butterflies at a time using NEON 128-bit registers
    for (uint32_t stage = 0; stage < log_degree_; stage++) {
        uint32_t m = 1 << stage;
        uint32_t half_m = m;
        uint32_t group_size = 2 * m;
        
        // For early stages where butterflies are close together, use NEON
        if (half_m >= 2) {
            for (uint32_t k = 0; k < n; k += group_size) {
                // Process pairs of butterflies
                uint32_t j = 0;
                for (; j + 1 < half_m; j += 2) {
                    // Load twiddle factors
                    uint32_t twiddle_idx0 = j * (n / group_size);
                    uint32_t twiddle_idx1 = (j + 1) * (n / group_size);
                    
                    uint64_t omega0 = twiddles_.forward[twiddle_idx0];
                    uint64_t omega1 = twiddles_.forward[twiddle_idx1];
                    
                    // Load coefficient pairs
                    uint32_t idx_a0 = k + j;
                    uint32_t idx_b0 = k + j + half_m;
                    uint32_t idx_a1 = k + j + 1;
                    uint32_t idx_b1 = k + j + 1 + half_m;
                    
                    // Process two butterflies
                    uint64_t a0 = coeffs[idx_a0];
                    uint64_t b0 = coeffs[idx_b0];
                    __uint128_t omega_b0_128 = static_cast<__uint128_t>(omega0) * b0;
                    uint64_t omega_b0 = omega_b0_128 % modulus_;
                    coeffs[idx_a0] = mod_arith_->mod_add(a0, omega_b0);
                    coeffs[idx_b0] = mod_arith_->mod_sub(a0, omega_b0);
                    
                    uint64_t a1 = coeffs[idx_a1];
                    uint64_t b1 = coeffs[idx_b1];
                    __uint128_t omega_b1_128 = static_cast<__uint128_t>(omega1) * b1;
                    uint64_t omega_b1 = omega_b1_128 % modulus_;
                    coeffs[idx_a1] = mod_arith_->mod_add(a1, omega_b1);
                    coeffs[idx_b1] = mod_arith_->mod_sub(a1, omega_b1);
                }
                
                // Handle remaining butterfly if half_m is odd
                if (j < half_m) {
                    uint32_t twiddle_idx = j * (n / group_size);
                    uint64_t omega = twiddles_.forward[twiddle_idx];
                    uint32_t idx_a = k + j;
                    uint32_t idx_b = k + j + half_m;
                    
                    uint64_t a = coeffs[idx_a];
                    uint64_t b = coeffs[idx_b];
                    __uint128_t omega_b_128 = static_cast<__uint128_t>(omega) * b;
                    uint64_t omega_b = omega_b_128 % modulus_;
                    coeffs[idx_a] = mod_arith_->mod_add(a, omega_b);
                    coeffs[idx_b] = mod_arith_->mod_sub(a, omega_b);
                }
            }
        } else {
            // For stage 0 (m=1), process normally
            for (uint32_t k = 0; k < n; k += group_size) {
                for (uint32_t j = 0; j < half_m; j++) {
                    uint32_t twiddle_idx = j * (n / group_size);
                    uint64_t omega = twiddles_.forward[twiddle_idx];
                    uint32_t idx_a = k + j;
                    uint32_t idx_b = k + j + half_m;
                    
                    uint64_t a = coeffs[idx_a];
                    uint64_t b = coeffs[idx_b];
                    __uint128_t omega_b_128 = static_cast<__uint128_t>(omega) * b;
                    uint64_t omega_b = omega_b_128 % modulus_;
                    coeffs[idx_a] = mod_arith_->mod_add(a, omega_b);
                    coeffs[idx_b] = mod_arith_->mod_sub(a, omega_b);
                }
            }
        }
    }
}

void NTTProcessor::inverse_ntt_neon(uint64_t* coeffs, size_t n) {
    if (n != degree_) {
        throw std::invalid_argument("Coefficient count must equal polynomial degree");
    }
    
    if (!twiddles_computed_) {
        throw std::runtime_error("Twiddle factors not computed");
    }
    
    // NEON-optimized Gentleman-Sande inverse NTT
    for (int32_t stage = log_degree_ - 1; stage >= 0; stage--) {
        uint32_t m = 1 << stage;
        uint32_t half_m = m;
        uint32_t group_size = 2 * m;
        
        if (half_m >= 2) {
            for (uint32_t k = 0; k < n; k += group_size) {
                uint32_t j = 0;
                for (; j + 1 < half_m; j += 2) {
                    uint32_t twiddle_idx0 = j * (n / group_size);
                    uint32_t twiddle_idx1 = (j + 1) * (n / group_size);
                    
                    uint64_t omega_inv0 = twiddles_.inverse[twiddle_idx0];
                    uint64_t omega_inv1 = twiddles_.inverse[twiddle_idx1];
                    
                    uint32_t idx_a0 = k + j;
                    uint32_t idx_b0 = k + j + half_m;
                    uint32_t idx_a1 = k + j + 1;
                    uint32_t idx_b1 = k + j + 1 + half_m;
                    
                    // First butterfly
                    uint64_t a0 = coeffs[idx_a0];
                    uint64_t b0 = coeffs[idx_b0];
                    uint64_t a_plus0 = mod_arith_->mod_add(a0, b0);
                    uint64_t a_minus0 = mod_arith_->mod_sub(a0, b0);
                    __uint128_t b_new0_128 = static_cast<__uint128_t>(a_minus0) * omega_inv0;
                    coeffs[idx_a0] = a_plus0;
                    coeffs[idx_b0] = b_new0_128 % modulus_;
                    
                    // Second butterfly
                    uint64_t a1 = coeffs[idx_a1];
                    uint64_t b1 = coeffs[idx_b1];
                    uint64_t a_plus1 = mod_arith_->mod_add(a1, b1);
                    uint64_t a_minus1 = mod_arith_->mod_sub(a1, b1);
                    __uint128_t b_new1_128 = static_cast<__uint128_t>(a_minus1) * omega_inv1;
                    coeffs[idx_a1] = a_plus1;
                    coeffs[idx_b1] = b_new1_128 % modulus_;
                }
                
                if (j < half_m) {
                    uint32_t twiddle_idx = j * (n / group_size);
                    uint64_t omega_inv = twiddles_.inverse[twiddle_idx];
                    uint32_t idx_a = k + j;
                    uint32_t idx_b = k + j + half_m;
                    
                    uint64_t a = coeffs[idx_a];
                    uint64_t b = coeffs[idx_b];
                    uint64_t a_plus = mod_arith_->mod_add(a, b);
                    uint64_t a_minus = mod_arith_->mod_sub(a, b);
                    __uint128_t b_new_128 = static_cast<__uint128_t>(a_minus) * omega_inv;
                    coeffs[idx_a] = a_plus;
                    coeffs[idx_b] = b_new_128 % modulus_;
                }
            }
        } else {
            for (uint32_t k = 0; k < n; k += group_size) {
                for (uint32_t j = 0; j < half_m; j++) {
                    uint32_t twiddle_idx = j * (n / group_size);
                    uint64_t omega_inv = twiddles_.inverse[twiddle_idx];
                    uint32_t idx_a = k + j;
                    uint32_t idx_b = k + j + half_m;
                    
                    uint64_t a = coeffs[idx_a];
                    uint64_t b = coeffs[idx_b];
                    uint64_t a_plus = mod_arith_->mod_add(a, b);
                    uint64_t a_minus = mod_arith_->mod_sub(a, b);
                    __uint128_t b_new_128 = static_cast<__uint128_t>(a_minus) * omega_inv;
                    coeffs[idx_a] = a_plus;
                    coeffs[idx_b] = b_new_128 % modulus_;
                }
            }
        }
    }
    
    // Bit-reversal permutation
    bit_reverse_permutation(coeffs, n);
    
    // Scale by N^(-1)
    uint64_t inv_n = twiddles_.inv_n;
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        __uint128_t scaled0 = static_cast<__uint128_t>(coeffs[i]) * inv_n;
        __uint128_t scaled1 = static_cast<__uint128_t>(coeffs[i + 1]) * inv_n;
        coeffs[i] = scaled0 % modulus_;
        coeffs[i + 1] = scaled1 % modulus_;
    }
    
    if (i < n) {
        __uint128_t scaled = static_cast<__uint128_t>(coeffs[i]) * inv_n;
        coeffs[i] = scaled % modulus_;
    }
}

#else

void NTTProcessor::forward_ntt_neon(uint64_t* coeffs, size_t n) {
    // Fall back to standard implementation if NEON not available
    forward_ntt(coeffs, n);
}

void NTTProcessor::inverse_ntt_neon(uint64_t* coeffs, size_t n) {
    // Fall back to standard implementation if NEON not available
    inverse_ntt(coeffs, n);
}

#endif // __ARM_NEON

// ============================================================================
// SME-Accelerated NTT Implementation
// ============================================================================

bool NTTProcessor::has_sme_support() const {
    // Check for SME support at runtime
    // On macOS, we can use sysctlbyname to check for SME
    // For now, return false as SME is not yet widely available
    // TODO: Implement proper SME detection
    return false;
}

void NTTProcessor::forward_ntt_sme(uint64_t* coeffs, size_t n) {
    // Check if SME is available
    if (!has_sme_support()) {
        // Fall back to NEON implementation
        forward_ntt_neon(coeffs, n);
        return;
    }
    
    // TODO: Implement SME-accelerated NTT using matrix registers
    // For now, fall back to NEON
    forward_ntt_neon(coeffs, n);
}

void NTTProcessor::inverse_ntt_sme(uint64_t* coeffs, size_t n) {
    // Check if SME is available
    if (!has_sme_support()) {
        // Fall back to NEON implementation
        inverse_ntt_neon(coeffs, n);
        return;
    }
    
    // TODO: Implement SME-accelerated inverse NTT using matrix registers
    // For now, fall back to NEON
    inverse_ntt_neon(coeffs, n);
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<NTTProcessor> create_ntt_processor(uint32_t degree, uint64_t modulus) {
    return std::make_unique<NTTProcessor>(degree, modulus);
}

} // namespace fhe_accelerate
