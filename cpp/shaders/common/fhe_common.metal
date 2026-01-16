//
// fhe_common.metal
// Common utilities and types for FHE Metal shaders
//
// This file contains shared functions used across all FHE compute shaders.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Type Definitions
// ============================================================================

/// 64-bit unsigned integer for polynomial coefficients
typedef ulong coeff_t;

/// 32-bit unsigned integer for indices and sizes
typedef uint index_t;

/// Structure for passing modular arithmetic parameters
struct ModularParams {
    coeff_t modulus;           // Prime modulus q
    coeff_t inv_modulus;       // Precomputed inverse for Montgomery reduction
    coeff_t r_squared;         // R^2 mod q for Montgomery conversion
    uint32_t modulus_bits;     // Bit length of modulus
};

/// Structure for NTT parameters
struct NTTParams {
    uint32_t degree;           // Polynomial degree N (power of 2)
    uint32_t log_degree;       // log2(N)
    coeff_t modulus;           // Prime modulus q
    coeff_t inv_n;             // N^(-1) mod q for inverse NTT
};

// ============================================================================
// Modular Arithmetic Functions
// ============================================================================

/// Add two coefficients modulo q
/// @param a First coefficient
/// @param b Second coefficient
/// @param q Modulus
/// @return (a + b) mod q
inline coeff_t mod_add(coeff_t a, coeff_t b, coeff_t q) {
    coeff_t sum = a + b;
    // Conditional subtraction to avoid branching
    return sum >= q ? sum - q : sum;
}

/// Subtract two coefficients modulo q
/// @param a First coefficient
/// @param b Second coefficient
/// @param q Modulus
/// @return (a - b) mod q
inline coeff_t mod_sub(coeff_t a, coeff_t b, coeff_t q) {
    // Add q to handle negative results
    return a >= b ? a - b : a + q - b;
}

/// Montgomery reduction: compute (a * R^(-1)) mod q
/// @param a Input value (must be < q * R)
/// @param q Modulus
/// @param q_inv Precomputed -q^(-1) mod R
/// @return (a * R^(-1)) mod q
inline coeff_t montgomery_reduce(coeff_t a, coeff_t q, coeff_t q_inv) {
    // Montgomery reduction algorithm
    // T = a
    // m = (T * q_inv) mod R  (R = 2^64)
    // t = (T + m * q) / R
    // if t >= q then t = t - q
    
    coeff_t m = a * q_inv;  // Lower 64 bits
    coeff_t t = (a + m * q) >> 64;  // Upper 64 bits after division by R
    return t >= q ? t - q : t;
}

/// Montgomery multiplication: compute (a * b * R^(-1)) mod q
/// @param a First coefficient in Montgomery form
/// @param b Second coefficient in Montgomery form
/// @param q Modulus
/// @param q_inv Precomputed -q^(-1) mod R
/// @return (a * b * R^(-1)) mod q in Montgomery form
inline coeff_t montgomery_mul(coeff_t a, coeff_t b, coeff_t q, coeff_t q_inv) {
    // Compute full 128-bit product
    // For Metal, we need to handle this carefully as there's no native 128-bit type
    
    // Split into high and low 32-bit parts
    uint32_t a_lo = a & 0xFFFFFFFF;
    uint32_t a_hi = a >> 32;
    uint32_t b_lo = b & 0xFFFFFFFF;
    uint32_t b_hi = b >> 32;
    
    // Compute partial products
    coeff_t p_ll = (coeff_t)a_lo * b_lo;
    coeff_t p_lh = (coeff_t)a_lo * b_hi;
    coeff_t p_hl = (coeff_t)a_hi * b_lo;
    coeff_t p_hh = (coeff_t)a_hi * b_hi;
    
    // Combine partial products
    coeff_t mid = p_lh + p_hl;
    coeff_t carry = (mid < p_lh) ? (1UL << 32) : 0;
    
    coeff_t low = p_ll + (mid << 32);
    coeff_t high = p_hh + (mid >> 32) + carry + ((low < p_ll) ? 1 : 0);
    
    // Montgomery reduction on 128-bit product
    coeff_t m = low * q_inv;
    coeff_t t_low = low + m * q;
    coeff_t t_high = high + ((t_low < low) ? 1 : 0);
    
    // Extract upper 64 bits
    coeff_t result = t_high;
    return result >= q ? result - q : result;
}

/// Barrett reduction: compute a mod q using precomputed constants
/// @param a Input value
/// @param q Modulus
/// @param mu Precomputed floor(2^(2k) / q) where k = ceil(log2(q))
/// @param k Bit length parameter
/// @return a mod q
inline coeff_t barrett_reduce(coeff_t a, coeff_t q, coeff_t mu, uint32_t k) {
    // Barrett reduction algorithm
    // q_hat = floor((a * mu) / 2^(2k))
    // r = a - q_hat * q
    // if r >= q then r = r - q
    
    coeff_t q_hat = (a * mu) >> (2 * k);
    coeff_t r = a - q_hat * q;
    return r >= q ? r - q : r;
}

/// Modular negation: compute -a mod q
/// @param a Input coefficient
/// @param q Modulus
/// @return (-a) mod q
inline coeff_t mod_neg(coeff_t a, coeff_t q) {
    return a == 0 ? 0 : q - a;
}

// ============================================================================
// Bit Manipulation Functions
// ============================================================================

/// Reverse the bits of an index for NTT bit-reversal permutation
/// @param x Input index
/// @param bits Number of bits to reverse
/// @return Bit-reversed index
inline index_t bit_reverse(index_t x, uint32_t bits) {
    index_t result = 0;
    for (uint32_t i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/// Compute log2 of a power of 2
/// @param n Input value (must be power of 2)
/// @return log2(n)
inline uint32_t log2_pow2(uint32_t n) {
    uint32_t log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

// ============================================================================
// Memory Access Helpers
// ============================================================================

/// Load coefficient with bounds checking (debug builds)
/// @param buffer Coefficient buffer
/// @param index Index to load
/// @param size Buffer size
/// @return Coefficient value
inline coeff_t safe_load(device const coeff_t* buffer, index_t index, index_t size) {
#ifdef DEBUG
    return index < size ? buffer[index] : 0;
#else
    return buffer[index];
#endif
}

/// Store coefficient with bounds checking (debug builds)
/// @param buffer Coefficient buffer
/// @param index Index to store
/// @param value Value to store
/// @param size Buffer size
inline void safe_store(device coeff_t* buffer, index_t index, coeff_t value, index_t size) {
#ifdef DEBUG
    if (index < size) {
        buffer[index] = value;
    }
#else
    buffer[index] = value;
#endif
}

// ============================================================================
// Thread Group Utilities
// ============================================================================

/// Synchronize threads within a threadgroup
inline void threadgroup_barrier() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Get the number of threads in the threadgroup
inline uint32_t get_threadgroup_size(uint3 threads_per_threadgroup) {
    return threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z;
}

/// Get the linear thread index within the threadgroup
inline uint32_t get_threadgroup_linear_index(uint3 thread_position_in_threadgroup,
                                             uint3 threads_per_threadgroup) {
    return thread_position_in_threadgroup.z * threads_per_threadgroup.x * threads_per_threadgroup.y +
           thread_position_in_threadgroup.y * threads_per_threadgroup.x +
           thread_position_in_threadgroup.x;
}
