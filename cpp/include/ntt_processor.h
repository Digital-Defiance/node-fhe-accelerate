/**
 * NTT Processor - Number Theoretic Transform Implementation
 * 
 * This module implements the Number Theoretic Transform (NTT) for polynomial
 * multiplication in FHE schemes. The NTT is the finite field analog of the FFT
 * and is used to achieve O(N log N) polynomial multiplication.
 * 
 * Design Reference: Section 2 - NTT Processor
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.6
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include "modular_arithmetic.h"
#include "fhe_types.h"

namespace fhe_accelerate {

/**
 * Twiddle factor tables for NTT operations
 * 
 * Stores precomputed twiddle factors (powers of primitive root) for both
 * forward and inverse NTT. Twiddle factors are stored in Montgomery form
 * for efficient modular multiplication.
 */
struct TwiddleFactors {
    std::vector<uint64_t> forward;      // Forward NTT twiddle factors
    std::vector<uint64_t> inverse;      // Inverse NTT twiddle factors
    uint64_t primitive_root;            // Primitive 2N-th root of unity
    uint64_t inv_primitive_root;        // Inverse of primitive root
    uint64_t inv_n;                     // N^(-1) mod q for inverse NTT scaling
    uint64_t modulus;                   // The modulus q
    uint32_t degree;                    // Polynomial degree N
    bool in_montgomery_form;            // Whether twiddles are in Montgomery form
    
    TwiddleFactors() : primitive_root(0), inv_primitive_root(0), inv_n(0),
                       modulus(0), degree(0), in_montgomery_form(false) {}
};

/**
 * NTT Processor
 * 
 * Implements the Number Theoretic Transform using the Cooley-Tukey algorithm.
 * Supports multiple hardware backends (NEON, SME, Metal GPU) for acceleration.
 */
class NTTProcessor {
public:
    /**
     * Construct NTT processor for given degree and modulus
     * 
     * @param degree Polynomial degree N (must be power of 2)
     * @param modulus Prime modulus q (must be NTT-friendly: q ≡ 1 (mod 2N))
     * @throws std::invalid_argument if degree is not power of 2 or modulus is not NTT-friendly
     */
    NTTProcessor(uint32_t degree, uint64_t modulus);
    
    /**
     * Destructor
     */
    ~NTTProcessor();
    
    // ========================================================================
    // Twiddle Factor Management
    // ========================================================================
    
    /**
     * Precompute twiddle factors for NTT operations
     * 
     * Computes powers of the primitive 2N-th root of unity for forward NTT
     * and powers of its inverse for inverse NTT. Twiddle factors are stored
     * in bit-reversed order for efficient Cooley-Tukey access patterns.
     * 
     * @param primitive_root Optional: specify primitive root (auto-computed if 0)
     */
    void precompute_twiddles(uint64_t primitive_root = 0);
    
    /**
     * Get the precomputed twiddle factors
     * @return Reference to twiddle factor tables
     */
    const TwiddleFactors& get_twiddles() const { return twiddles_; }
    
    // ========================================================================
    // Core NTT Operations
    // ========================================================================
    
    /**
     * Forward NTT (in-place)
     * 
     * Transforms polynomial from coefficient representation to NTT domain.
     * Uses iterative Cooley-Tukey algorithm with O(N log N) complexity.
     * 
     * @param coeffs Coefficient array (modified in-place)
     * @param n Number of coefficients (must equal degree)
     */
    void forward_ntt(uint64_t* coeffs, size_t n);
    
    /**
     * Inverse NTT (in-place)
     * 
     * Transforms polynomial from NTT domain back to coefficient representation.
     * Includes scaling by N^(-1) mod q.
     * 
     * @param coeffs NTT coefficient array (modified in-place)
     * @param n Number of coefficients (must equal degree)
     */
    void inverse_ntt(uint64_t* coeffs, size_t n);
    
    /**
     * Forward NTT (out-of-place)
     * 
     * @param input Input coefficient array
     * @param output Output NTT coefficient array
     * @param n Number of coefficients
     */
    void forward_ntt(const uint64_t* input, uint64_t* output, size_t n);
    
    /**
     * Inverse NTT (out-of-place)
     * 
     * @param input Input NTT coefficient array
     * @param output Output coefficient array
     * @param n Number of coefficients
     */
    void inverse_ntt(const uint64_t* input, uint64_t* output, size_t n);
    
    // ========================================================================
    // Batch Operations (for GPU acceleration)
    // ========================================================================
    
    /**
     * Batch forward NTT
     * 
     * Process multiple polynomials in parallel. Optimal for GPU execution.
     * 
     * @param coeffs_batch Array of coefficient arrays
     * @param batch_size Number of polynomials
     * @param n Degree of each polynomial
     */
    void forward_ntt_batch(uint64_t** coeffs_batch, size_t batch_size, size_t n);
    
    /**
     * Batch inverse NTT
     * 
     * @param coeffs_batch Array of NTT coefficient arrays
     * @param batch_size Number of polynomials
     * @param n Degree of each polynomial
     */
    void inverse_ntt_batch(uint64_t** coeffs_batch, size_t batch_size, size_t n);
    
    // ========================================================================
    // NEON-Optimized Operations
    // ========================================================================
    
    /**
     * Forward NTT using NEON SIMD (in-place)
     * 
     * Vectorized implementation processing 2 butterflies per SIMD operation.
     * 
     * @param coeffs Coefficient array (modified in-place)
     * @param n Number of coefficients
     */
    void forward_ntt_neon(uint64_t* coeffs, size_t n);
    
    /**
     * Inverse NTT using NEON SIMD (in-place)
     * 
     * @param coeffs NTT coefficient array (modified in-place)
     * @param n Number of coefficients
     */
    void inverse_ntt_neon(uint64_t* coeffs, size_t n);
    
    // ========================================================================
    // SME-Accelerated Operations
    // ========================================================================
    
    /**
     * Forward NTT using SME matrix operations (in-place)
     * 
     * Uses SME matrix registers for butterfly stages. Falls back to NEON
     * if SME is unavailable.
     * 
     * @param coeffs Coefficient array (modified in-place)
     * @param n Number of coefficients
     */
    void forward_ntt_sme(uint64_t* coeffs, size_t n);
    
    /**
     * Inverse NTT using SME matrix operations (in-place)
     * 
     * @param coeffs NTT coefficient array (modified in-place)
     * @param n Number of coefficients
     */
    void inverse_ntt_sme(uint64_t* coeffs, size_t n);
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /**
     * Perform bit-reversal permutation on coefficient array
     * 
     * @param coeffs Coefficient array (modified in-place)
     * @param n Number of coefficients
     */
    void bit_reverse_permutation(uint64_t* coeffs, size_t n);
    
    /**
     * Get the polynomial degree
     * @return Polynomial degree N
     */
    uint32_t get_degree() const { return degree_; }
    
    /**
     * Get the modulus
     * @return Prime modulus q
     */
    uint64_t get_modulus() const { return modulus_; }
    
    /**
     * Check if SME is available for acceleration
     * @return true if SME can be used
     */
    bool has_sme_support() const;
    
    // ========================================================================
    // Static Utility Functions
    // ========================================================================
    
    /**
     * Find a primitive 2N-th root of unity modulo q
     * 
     * For NTT-friendly primes q ≡ 1 (mod 2N), there exists a primitive
     * 2N-th root of unity ω such that ω^(2N) ≡ 1 (mod q) and ω^k ≢ 1
     * for 0 < k < 2N.
     * 
     * @param degree Polynomial degree N
     * @param modulus Prime modulus q
     * @return Primitive 2N-th root of unity
     * @throws std::invalid_argument if no primitive root exists
     */
    static uint64_t find_primitive_root(uint32_t degree, uint64_t modulus);
    
    /**
     * Compute modular inverse using extended Euclidean algorithm
     * 
     * @param a Value to invert
     * @param m Modulus
     * @return a^(-1) mod m
     * @throws std::invalid_argument if inverse doesn't exist
     */
    static uint64_t mod_inverse(uint64_t a, uint64_t m);
    
    /**
     * Compute modular exponentiation: base^exp mod m
     * 
     * @param base Base value
     * @param exp Exponent
     * @param mod Modulus
     * @return base^exp mod m
     */
    static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod);
    
    /**
     * Check if a number is a power of 2
     * 
     * @param n Number to check
     * @return true if n is a power of 2
     */
    static bool is_power_of_two(uint32_t n);
    
    /**
     * Compute log2 of a power of 2
     * 
     * @param n Power of 2
     * @return log2(n)
     */
    static uint32_t log2_pow2(uint32_t n);
    
    /**
     * Compute bit-reversed index
     * 
     * @param index Original index
     * @param bits Number of bits
     * @return Bit-reversed index
     */
    static uint32_t bit_reverse(uint32_t index, uint32_t bits);
    
private:
    uint32_t degree_;                           // Polynomial degree N
    uint32_t log_degree_;                       // log2(N)
    uint64_t modulus_;                          // Prime modulus q
    TwiddleFactors twiddles_;                   // Precomputed twiddle factors
    std::unique_ptr<ModularArithmetic> mod_arith_;  // Modular arithmetic helper
    bool twiddles_computed_;                    // Whether twiddles are ready
    
    // Internal helper functions
    void ntt_butterfly(uint64_t& a, uint64_t& b, uint64_t omega);
    void inverse_ntt_butterfly(uint64_t& a, uint64_t& b, uint64_t omega_inv);
};

// Factory function for creating NTT processors
std::unique_ptr<NTTProcessor> create_ntt_processor(uint32_t degree, uint64_t modulus);

} // namespace fhe_accelerate
