/**
 * Adaptive Hardware Dispatcher
 * 
 * Automatically selects the fastest backend for each operation
 * based on benchmarking results on M4 Max.
 * 
 * Benchmark-driven selection:
 * - ModMul: Barrett Unrolled (4x) for most sizes
 * - NTT: Montgomery form (2x speedup)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fhe_accelerate {

// ============================================================================
// Optimized Modular Multiplication
// ============================================================================

/**
 * Batch modular multiplication using the fastest available method.
 * Uses Barrett reduction with loop unrolling (benchmark winner).
 * 
 * @param a First operand array
 * @param b Second operand array  
 * @param result Output array (can alias a or b)
 * @param n Number of elements
 * @param modulus The modulus (must be < 2^63)
 */
void fast_modmul_batch(const uint64_t* a, const uint64_t* b,
                       uint64_t* result, size_t n, uint64_t modulus);

/**
 * Single modular multiplication.
 * For single operations, direct computation is faster than Barrett setup.
 */
inline uint64_t fast_modmul(uint64_t a, uint64_t b, uint64_t modulus) {
    __uint128_t product = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(product % modulus);
}

// ============================================================================
// Optimized NTT
// ============================================================================

/**
 * Forward NTT using Montgomery form (benchmark winner - 2x speedup).
 * 
 * @param coeffs Coefficient array (modified in-place)
 * @param n Polynomial degree (must be power of 2)
 * @param modulus NTT-friendly prime
 * @param twiddles Precomputed twiddle factors
 */
void fast_ntt_forward(uint64_t* coeffs, size_t n, uint64_t modulus, 
                      const uint64_t* twiddles);

/**
 * Inverse NTT using Montgomery form.
 * 
 * @param coeffs Coefficient array (modified in-place)
 * @param n Polynomial degree (must be power of 2)
 * @param modulus NTT-friendly prime
 * @param inv_twiddles Precomputed inverse twiddle factors
 */
void fast_ntt_inverse(uint64_t* coeffs, size_t n, uint64_t modulus,
                      const uint64_t* inv_twiddles);

// ============================================================================
// Precomputation Helpers
// ============================================================================

/**
 * Precompute twiddle factors for NTT.
 * 
 * @param n Polynomial degree
 * @param modulus NTT-friendly prime
 * @param primitive_root Primitive root of the modulus
 * @return Vector of twiddle factors
 */
std::vector<uint64_t> precompute_twiddles(size_t n, uint64_t modulus, 
                                          uint64_t primitive_root);

/**
 * Precompute inverse twiddle factors for inverse NTT.
 */
std::vector<uint64_t> precompute_inv_twiddles(size_t n, uint64_t modulus,
                                               uint64_t primitive_root);

// ============================================================================
// Hardware Info
// ============================================================================

struct HardwareCapabilities {
    bool has_neon;
    bool has_sme;
    bool has_amx;
    bool has_metal;
    int num_cores;
    int gpu_cores;
    
    static HardwareCapabilities detect();
    void print() const;
};

} // namespace fhe_accelerate
