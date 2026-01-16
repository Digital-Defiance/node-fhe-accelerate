/**
 * Polynomial Ring Operations
 * 
 * This module implements polynomial arithmetic in the cyclotomic ring Z_q[X]/(X^N + 1).
 * Polynomials can be represented in either coefficient form or NTT (evaluation) form.
 * 
 * Design Reference: Section 3 - Polynomial Ring
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 14.2
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <cstdlib>
#include "modular_arithmetic.h"
#include "ntt_processor.h"

namespace fhe_accelerate {

// Cache line size for alignment (128 bytes on M4 Max)
constexpr size_t CACHE_LINE_SIZE = 128;

/**
 * Custom allocator for cache-aligned memory
 * 
 * Ensures coefficient arrays are aligned to cache line boundaries
 * for optimal memory access patterns on M4 Max.
 */
template <typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        void* ptr = nullptr;
        size_t size = n * sizeof(T);
        
        // Use aligned_alloc for cache-aligned memory
        #if defined(_WIN32)
        ptr = _aligned_malloc(size, Alignment);
        #else
        if (posix_memalign(&ptr, Alignment, size) != 0) {
            ptr = nullptr;
        }
        #endif
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) {
            #if defined(_WIN32)
            _aligned_free(p);
            #else
            free(p);
            #endif
        }
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

/**
 * Polynomial representation in Z_q[X]/(X^N + 1)
 * 
 * Supports both coefficient and NTT (evaluation) representations.
 * Uses cache-aligned memory layout for optimal performance on M4 Max.
 */
class Polynomial {
public:
    // Type alias for cache-aligned coefficient storage
    using CoeffVector = std::vector<uint64_t, AlignedAllocator<uint64_t>>;
    
    /**
     * Construct a zero polynomial
     * 
     * @param degree Polynomial degree N (must be power of 2)
     * @param modulus Prime modulus q
     */
    Polynomial(uint32_t degree, uint64_t modulus);
    
    /**
     * Construct polynomial from coefficient vector
     * 
     * @param coeffs Coefficient vector (will be copied)
     * @param modulus Prime modulus q
     * @param is_ntt Whether coefficients are in NTT form
     */
    Polynomial(const std::vector<uint64_t>& coeffs, uint64_t modulus, bool is_ntt = false);
    
    /**
     * Construct polynomial from coefficient vector (move)
     * 
     * @param coeffs Coefficient vector (will be moved)
     * @param modulus Prime modulus q
     * @param is_ntt Whether coefficients are in NTT form
     */
    Polynomial(std::vector<uint64_t>&& coeffs, uint64_t modulus, bool is_ntt = false);
    
    /**
     * Copy constructor
     */
    Polynomial(const Polynomial& other);
    
    /**
     * Move constructor
     */
    Polynomial(Polynomial&& other) noexcept;
    
    /**
     * Copy assignment
     */
    Polynomial& operator=(const Polynomial& other);
    
    /**
     * Move assignment
     */
    Polynomial& operator=(Polynomial&& other) noexcept;
    
    /**
     * Destructor
     */
    ~Polynomial() = default;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    /**
     * Get polynomial degree N
     */
    uint32_t degree() const { return degree_; }
    
    /**
     * Get number of coefficients (same as degree for this ring)
     */
    size_t size() const { return coeffs_.size(); }
    
    /**
     * Get modulus q
     */
    uint64_t modulus() const { return modulus_; }
    
    /**
     * Check if polynomial is in NTT form
     */
    bool is_ntt() const { return is_ntt_; }
    
    /**
     * Get coefficient at index i
     */
    uint64_t operator[](size_t i) const { return coeffs_[i]; }
    
    /**
     * Get mutable reference to coefficient at index i
     */
    uint64_t& operator[](size_t i) { return coeffs_[i]; }
    
    /**
     * Get raw coefficient pointer (const)
     */
    const uint64_t* data() const { return coeffs_.data(); }
    
    /**
     * Get raw coefficient pointer (mutable)
     */
    uint64_t* data() { return coeffs_.data(); }
    
    /**
     * Get coefficient vector (const reference)
     */
    const CoeffVector& coefficients() const { return coeffs_; }
    
    /**
     * Get coefficient vector (mutable reference)
     */
    CoeffVector& coefficients() { return coeffs_; }
    
    // ========================================================================
    // Representation Conversion
    // ========================================================================
    
    /**
     * Convert to NTT form (in-place)
     * 
     * @param ntt NTT processor to use for transformation
     */
    void to_ntt(NTTProcessor& ntt);
    
    /**
     * Convert from NTT form to coefficient form (in-place)
     * 
     * @param ntt NTT processor to use for transformation
     */
    void from_ntt(NTTProcessor& ntt);
    
    /**
     * Set NTT flag (use with caution - only when you know the representation)
     */
    void set_ntt_flag(bool is_ntt) { is_ntt_ = is_ntt; }
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /**
     * Set all coefficients to zero
     */
    void set_zero();
    
    /**
     * Set to multiplicative identity (constant polynomial 1)
     */
    void set_identity();
    
    /**
     * Check if polynomial is zero
     */
    bool is_zero() const;
    
    /**
     * Check if polynomial is the multiplicative identity
     */
    bool is_identity() const;
    
    /**
     * Check equality with another polynomial
     */
    bool operator==(const Polynomial& other) const;
    
    /**
     * Check inequality with another polynomial
     */
    bool operator!=(const Polynomial& other) const;
    
    /**
     * Create a copy of this polynomial
     */
    Polynomial clone() const;
    
    // ========================================================================
    // Static Factory Methods
    // ========================================================================
    
    /**
     * Create a zero polynomial
     */
    static Polynomial zero(uint32_t degree, uint64_t modulus);
    
    /**
     * Create the multiplicative identity polynomial (constant 1)
     */
    static Polynomial identity(uint32_t degree, uint64_t modulus);
    
    /**
     * Create a random polynomial with coefficients in [0, modulus)
     */
    static Polynomial random(uint32_t degree, uint64_t modulus);
    
    /**
     * Create a polynomial with small coefficients (ternary: -1, 0, 1)
     * Used for secret key generation
     */
    static Polynomial random_ternary(uint32_t degree, uint64_t modulus);
    
private:
    CoeffVector coeffs_;    // Coefficient storage (cache-aligned)
    uint64_t modulus_;      // Prime modulus q
    uint32_t degree_;       // Polynomial degree N
    bool is_ntt_;           // True if in NTT (evaluation) form
};

/**
 * Polynomial Ring Operations
 * 
 * Provides arithmetic operations on polynomials in Z_q[X]/(X^N + 1).
 * Operations are optimized using NTT for multiplication and NEON SIMD
 * for addition/subtraction.
 */
class PolynomialRing {
public:
    /**
     * Construct polynomial ring for given degree and modulus
     * 
     * @param degree Polynomial degree N (must be power of 2)
     * @param modulus Prime modulus q (must be NTT-friendly: q ≡ 1 (mod 2N))
     */
    PolynomialRing(uint32_t degree, uint64_t modulus);
    
    /**
     * Construct polynomial ring with multiple moduli (RNS representation)
     * 
     * @param degree Polynomial degree N
     * @param moduli Vector of prime moduli for RNS
     */
    PolynomialRing(uint32_t degree, const std::vector<uint64_t>& moduli);
    
    /**
     * Destructor
     */
    ~PolynomialRing();
    
    // ========================================================================
    // Ring Operations
    // ========================================================================
    
    /**
     * Add two polynomials: result = a + b (mod q)
     * 
     * @param a First polynomial
     * @param b Second polynomial
     * @return Sum polynomial
     */
    Polynomial add(const Polynomial& a, const Polynomial& b) const;
    
    /**
     * Add two polynomials in-place: a += b (mod q)
     * 
     * @param a Polynomial to modify (result stored here)
     * @param b Polynomial to add
     */
    void add_inplace(Polynomial& a, const Polynomial& b) const;
    
    /**
     * Subtract two polynomials: result = a - b (mod q)
     * 
     * @param a First polynomial
     * @param b Second polynomial
     * @return Difference polynomial
     */
    Polynomial subtract(const Polynomial& a, const Polynomial& b) const;
    
    /**
     * Subtract two polynomials in-place: a -= b (mod q)
     * 
     * @param a Polynomial to modify (result stored here)
     * @param b Polynomial to subtract
     */
    void subtract_inplace(Polynomial& a, const Polynomial& b) const;
    
    /**
     * Negate polynomial: result = -a (mod q)
     * 
     * @param a Polynomial to negate
     * @return Negated polynomial
     */
    Polynomial negate(const Polynomial& a) const;
    
    /**
     * Negate polynomial in-place: a = -a (mod q)
     * 
     * @param a Polynomial to negate (result stored here)
     */
    void negate_inplace(Polynomial& a) const;
    
    /**
     * Multiply two polynomials: result = a * b (mod X^N + 1, mod q)
     * 
     * Uses NTT-based multiplication for O(N log N) complexity.
     * 
     * @param a First polynomial
     * @param b Second polynomial
     * @return Product polynomial
     */
    Polynomial multiply(const Polynomial& a, const Polynomial& b);
    
    /**
     * Multiply two polynomials in-place: a *= b (mod X^N + 1, mod q)
     * 
     * @param a Polynomial to modify (result stored here)
     * @param b Polynomial to multiply by
     */
    void multiply_inplace(Polynomial& a, const Polynomial& b);
    
    /**
     * Multiply polynomial by scalar: result = a * scalar (mod q)
     * 
     * @param a Polynomial
     * @param scalar Scalar value
     * @return Scaled polynomial
     */
    Polynomial multiply_scalar(const Polynomial& a, uint64_t scalar) const;
    
    /**
     * Multiply polynomial by scalar in-place: a *= scalar (mod q)
     * 
     * @param a Polynomial to modify
     * @param scalar Scalar value
     */
    void multiply_scalar_inplace(Polynomial& a, uint64_t scalar) const;
    
    // ========================================================================
    // NTT Operations
    // ========================================================================
    
    /**
     * Pointwise multiply two polynomials in NTT form
     * 
     * Both polynomials must be in NTT form. Result is also in NTT form.
     * 
     * @param a First polynomial (NTT form)
     * @param b Second polynomial (NTT form)
     * @return Pointwise product (NTT form)
     */
    Polynomial pointwise_multiply(const Polynomial& a, const Polynomial& b) const;
    
    /**
     * Pointwise multiply in-place
     * 
     * @param a Polynomial to modify (NTT form)
     * @param b Polynomial to multiply by (NTT form)
     */
    void pointwise_multiply_inplace(Polynomial& a, const Polynomial& b) const;
    
    /**
     * Convert polynomial to NTT form
     * 
     * @param p Polynomial to convert (modified in-place)
     */
    void to_ntt(Polynomial& p);
    
    /**
     * Convert polynomial from NTT form
     * 
     * @param p Polynomial to convert (modified in-place)
     */
    void from_ntt(Polynomial& p);
    
    // ========================================================================
    // NEON-Optimized Operations
    // ========================================================================
    
    /**
     * Add two polynomials using NEON SIMD
     */
    void add_neon(const Polynomial& a, const Polynomial& b, Polynomial& result) const;
    
    /**
     * Subtract two polynomials using NEON SIMD
     */
    void subtract_neon(const Polynomial& a, const Polynomial& b, Polynomial& result) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    /**
     * Get polynomial degree
     */
    uint32_t degree() const { return degree_; }
    
    /**
     * Get primary modulus
     */
    uint64_t modulus() const { return moduli_[0]; }
    
    /**
     * Get all moduli (for RNS)
     */
    const std::vector<uint64_t>& moduli() const { return moduli_; }
    
    /**
     * Get NTT processor for primary modulus
     */
    NTTProcessor& ntt_processor() { return *ntt_processors_[0]; }
    
    /**
     * Get modular arithmetic helper
     */
    const ModularArithmetic& mod_arith() const { return *mod_arith_[0]; }
    
private:
    uint32_t degree_;                                       // Polynomial degree N
    std::vector<uint64_t> moduli_;                          // Moduli for RNS
    std::vector<std::unique_ptr<NTTProcessor>> ntt_processors_;  // NTT processors
    std::vector<std::unique_ptr<ModularArithmetic>> mod_arith_;  // Modular arithmetic
    
    // Validate polynomial parameters match ring parameters
    void validate_polynomial(const Polynomial& p) const;
    void validate_polynomials(const Polynomial& a, const Polynomial& b) const;
};

// Factory function for creating polynomial rings
std::unique_ptr<PolynomialRing> create_polynomial_ring(uint32_t degree, uint64_t modulus);

} // namespace fhe_accelerate
