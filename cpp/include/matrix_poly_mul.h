/**
 * Matrix-Centric Polynomial Multiplication
 * 
 * Express polynomial multiplication as matrix operations:
 * - Toeplitz matrix-vector product for linear convolution
 * - Circulant matrix for cyclic convolution (mod X^n - 1)
 * - Negacyclic matrix for negacyclic convolution (mod X^n + 1)
 * 
 * Requirements 22.2, 22.8
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fhe_accelerate {
namespace matrix_poly {

// ============================================================================
// Toeplitz Matrix Polynomial Multiplication
// ============================================================================

/**
 * Toeplitz matrix polynomial multiplication
 * 
 * For polynomials a, b of degree n-1:
 * result = a * b is computed as Toeplitz(a) * b
 * 
 * Toeplitz(a) is a (2n-1) x n matrix where:
 * T[i][j] = a[i-j] if 0 <= i-j < n, else 0
 */
class ToeplitzPolyMul {
public:
    /**
     * Create Toeplitz polynomial multiplier
     * 
     * @param degree Polynomial degree
     * @param modulus Prime modulus
     */
    ToeplitzPolyMul(size_t degree, uint64_t modulus);
    ~ToeplitzPolyMul();
    
    /**
     * Multiply two polynomials using Toeplitz matrix
     * 
     * @param a First polynomial (degree coefficients)
     * @param b Second polynomial (degree coefficients)
     * @param result Output polynomial (2*degree-1 coefficients)
     */
    void multiply(const uint64_t* a, const uint64_t* b, uint64_t* result);
    
    /**
     * Multiply and reduce mod X^n + 1 (negacyclic)
     * 
     * @param a First polynomial
     * @param b Second polynomial
     * @param result Output polynomial (degree coefficients)
     */
    void multiply_negacyclic(const uint64_t* a, const uint64_t* b, uint64_t* result);
    
    /**
     * Batch multiply multiple polynomial pairs
     * 
     * @param a_batch Array of first polynomials
     * @param b_batch Array of second polynomials
     * @param result_batch Array of output polynomials
     * @param batch_size Number of polynomial pairs
     */
    void multiply_batch(const uint64_t* const* a_batch, const uint64_t* const* b_batch,
                        uint64_t** result_batch, size_t batch_size);
    
private:
    size_t degree_;
    uint64_t modulus_;
    
    // Precomputed Toeplitz matrix (stored as float for BLAS)
    std::vector<float> toeplitz_float_;
    
    // Build Toeplitz matrix from polynomial
    void build_toeplitz(const uint64_t* poly);
};

// ============================================================================
// Circulant Matrix Polynomial Multiplication
// ============================================================================

/**
 * Circulant matrix polynomial multiplication
 * 
 * For cyclic convolution (mod X^n - 1):
 * result = a * b mod (X^n - 1) is computed as Circulant(a) * b
 * 
 * Circulant(a) is an n x n matrix where:
 * C[i][j] = a[(i-j) mod n]
 */
class CirculantPolyMul {
public:
    /**
     * Create circulant polynomial multiplier
     * 
     * @param degree Polynomial degree
     * @param modulus Prime modulus
     */
    CirculantPolyMul(size_t degree, uint64_t modulus);
    ~CirculantPolyMul();
    
    /**
     * Multiply two polynomials mod X^n - 1 using circulant matrix
     * 
     * @param a First polynomial
     * @param b Second polynomial
     * @param result Output polynomial
     */
    void multiply_cyclic(const uint64_t* a, const uint64_t* b, uint64_t* result);
    
    /**
     * Batch multiply multiple polynomial pairs
     * 
     * @param a_batch Array of first polynomials
     * @param b_batch Array of second polynomials
     * @param result_batch Array of output polynomials
     * @param batch_size Number of polynomial pairs
     */
    void multiply_cyclic_batch(const uint64_t* const* a_batch, const uint64_t* const* b_batch,
                               uint64_t** result_batch, size_t batch_size);
    
private:
    size_t degree_;
    uint64_t modulus_;
    
    // Build circulant matrix from polynomial
    void build_circulant(const uint64_t* poly, float* matrix);
};

// ============================================================================
// Negacyclic Matrix Polynomial Multiplication
// ============================================================================

/**
 * Negacyclic matrix polynomial multiplication
 * 
 * For negacyclic convolution (mod X^n + 1):
 * result = a * b mod (X^n + 1)
 * 
 * This is the most common case in FHE (RLWE uses X^n + 1).
 * 
 * The negacyclic matrix is similar to circulant but with sign changes:
 * N[i][j] = a[(i-j) mod n] if i >= j
 * N[i][j] = -a[(i-j+n) mod n] if i < j
 */
class NegacyclicPolyMul {
public:
    /**
     * Create negacyclic polynomial multiplier
     * 
     * @param degree Polynomial degree
     * @param modulus Prime modulus
     */
    NegacyclicPolyMul(size_t degree, uint64_t modulus);
    ~NegacyclicPolyMul();
    
    /**
     * Multiply two polynomials mod X^n + 1 using negacyclic matrix
     * 
     * @param a First polynomial
     * @param b Second polynomial
     * @param result Output polynomial
     */
    void multiply(const uint64_t* a, const uint64_t* b, uint64_t* result);
    
    /**
     * Batch multiply multiple polynomial pairs
     * 
     * @param a_batch Array of first polynomials
     * @param b_batch Array of second polynomials
     * @param result_batch Array of output polynomials
     * @param batch_size Number of polynomial pairs
     */
    void multiply_batch(const uint64_t* const* a_batch, const uint64_t* const* b_batch,
                        uint64_t** result_batch, size_t batch_size);
    
private:
    size_t degree_;
    uint64_t modulus_;
    
    // Build negacyclic matrix from polynomial
    void build_negacyclic(const uint64_t* poly, float* matrix);
};

// ============================================================================
// Batch Polynomial Multiplication as Matrix-Matrix Multiply
// ============================================================================

/**
 * Batch polynomial multiplication using matrix-matrix multiply
 * 
 * For batch processing, we can stack polynomials as rows of matrices
 * and perform the multiplication as a single matrix-matrix multiply.
 * 
 * This is efficient when:
 * - Processing many polynomial pairs
 * - Using BLAS which leverages AMX on Apple Silicon
 */
class BatchMatrixPolyMul {
public:
    /**
     * Create batch polynomial multiplier
     * 
     * @param degree Polynomial degree
     * @param modulus Prime modulus
     */
    BatchMatrixPolyMul(size_t degree, uint64_t modulus);
    ~BatchMatrixPolyMul();
    
    /**
     * Multiply batches of polynomials using matrix-matrix multiply
     * 
     * Input: A is batch_size x degree, B is batch_size x degree
     * Output: C is batch_size x degree (negacyclic product)
     * 
     * @param a_batch First polynomial batch (batch_size * degree elements)
     * @param b_batch Second polynomial batch (batch_size * degree elements)
     * @param result_batch Output polynomial batch (batch_size * degree elements)
     * @param batch_size Number of polynomial pairs
     */
    void multiply_batch(const uint64_t* a_batch, const uint64_t* b_batch,
                        uint64_t* result_batch, size_t batch_size);
    
    /**
     * Check if batch multiplication is beneficial
     * 
     * @param batch_size Number of polynomial pairs
     * @return true if batch multiplication is faster than individual
     */
    bool is_beneficial(size_t batch_size) const;
    
private:
    size_t degree_;
    uint64_t modulus_;
    
    // Minimum batch size for matrix multiply to be beneficial
    static constexpr size_t MIN_BATCH_SIZE = 4;
};

// ============================================================================
// Benchmarking
// ============================================================================

/**
 * Benchmark results for matrix polynomial multiplication
 */
struct MatrixPolyMulBenchmark {
    double direct_time_us;      // Direct convolution
    double toeplitz_time_us;    // Toeplitz matrix
    double circulant_time_us;   // Circulant matrix
    double negacyclic_time_us;  // Negacyclic matrix
    double ntt_time_us;         // NTT-based
    double speedup_toeplitz;
    double speedup_circulant;
    double speedup_negacyclic;
    double speedup_ntt;
};

/**
 * Run polynomial multiplication benchmarks
 * 
 * @param degree Polynomial degree
 * @param modulus Prime modulus
 * @param num_iterations Number of iterations
 * @return Benchmark results
 */
MatrixPolyMulBenchmark benchmark_matrix_poly_mul(size_t degree, uint64_t modulus, 
                                                  size_t num_iterations = 100);

} // namespace matrix_poly
} // namespace fhe_accelerate
