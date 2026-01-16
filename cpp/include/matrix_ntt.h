/**
 * Matrix-Centric NTT Implementation
 * 
 * Express NTT as sparse butterfly matrix multiplications for SME/AMX acceleration.
 * Each NTT stage is a matrix multiply: Y = W_stage * X
 * 
 * For batched NTT, we can use dense matrix multiply:
 * - Stack polynomials as rows of a matrix
 * - Apply NTT as matrix-matrix multiply
 * 
 * Requirements 22.1, 22.6, 22.7
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

namespace fhe_accelerate {
namespace matrix_ntt {

// ============================================================================
// Butterfly Matrix Representation
// ============================================================================

/**
 * Sparse butterfly matrix for a single NTT stage
 * 
 * Each butterfly matrix has exactly 2 non-zeros per row:
 * - Row i has non-zeros at columns i and i XOR (1 << stage)
 * - Values are 1 and twiddle factor
 */
struct ButterflyMatrix {
    size_t degree;
    size_t stage;
    std::vector<uint64_t> twiddles;  // Twiddle factors for this stage
    uint64_t modulus;
};

/**
 * Create butterfly matrix for a specific NTT stage
 * 
 * @param degree Polynomial degree
 * @param stage NTT stage (0 to log2(degree)-1)
 * @param modulus Prime modulus
 * @param primitive_root Primitive root of unity
 * @return Butterfly matrix
 */
ButterflyMatrix create_butterfly_matrix(size_t degree, size_t stage,
                                        uint64_t modulus, uint64_t primitive_root);

/**
 * Apply butterfly matrix to polynomial (sparse matrix-vector multiply)
 * 
 * @param matrix Butterfly matrix
 * @param coeffs Polynomial coefficients (modified in place)
 */
void apply_butterfly_matrix(const ButterflyMatrix& matrix, uint64_t* coeffs);

// ============================================================================
// Matrix-Form NTT
// ============================================================================

/**
 * Matrix-form NTT processor
 * 
 * Expresses NTT as a sequence of sparse matrix multiplications.
 * Can use SME/AMX for the matrix operations.
 */
class MatrixNTT {
public:
    /**
     * Create matrix NTT processor
     * 
     * @param degree Polynomial degree (power of 2)
     * @param modulus Prime modulus (must be NTT-friendly)
     */
    MatrixNTT(size_t degree, uint64_t modulus);
    ~MatrixNTT();
    
    /**
     * Compute forward NTT using matrix formulation
     * 
     * @param coeffs Polynomial coefficients (modified in place)
     */
    void forward_ntt(uint64_t* coeffs);
    
    /**
     * Compute inverse NTT using matrix formulation
     * 
     * @param coeffs NTT coefficients (modified in place)
     */
    void inverse_ntt(uint64_t* coeffs);
    
    /**
     * Compute batched forward NTT using dense matrix multiply
     * 
     * @param polys Array of polynomial coefficient arrays
     * @param num_polys Number of polynomials
     */
    void forward_ntt_batch(uint64_t** polys, size_t num_polys);
    
    /**
     * Compute batched inverse NTT using dense matrix multiply
     * 
     * @param polys Array of NTT coefficient arrays
     * @param num_polys Number of polynomials
     */
    void inverse_ntt_batch(uint64_t** polys, size_t num_polys);
    
    /**
     * Get polynomial degree
     */
    size_t degree() const { return degree_; }
    
    /**
     * Get modulus
     */
    uint64_t modulus() const { return modulus_; }
    
private:
    size_t degree_;
    size_t log_degree_;
    uint64_t modulus_;
    uint64_t primitive_root_;
    uint64_t inv_degree_;  // Multiplicative inverse of degree
    
    // Precomputed butterfly matrices for each stage
    std::vector<ButterflyMatrix> forward_matrices_;
    std::vector<ButterflyMatrix> inverse_matrices_;
    
    // Precomputed twiddle factors
    std::vector<uint64_t> forward_twiddles_;
    std::vector<uint64_t> inverse_twiddles_;
    
    // Find primitive root of unity
    uint64_t find_primitive_root();
    
    // Compute modular inverse
    uint64_t mod_inverse(uint64_t a);
    
    // Compute modular exponentiation
    uint64_t mod_pow(uint64_t base, uint64_t exp);
};

// ============================================================================
// Dense Matrix NTT for Batching
// ============================================================================

/**
 * Dense matrix representation for batched NTT
 * 
 * For batch processing, we can express NTT as:
 * - Input: N x D matrix (N polynomials, D coefficients each)
 * - Output: N x D matrix (N NTT results)
 * - Transform: Apply NTT to each row
 * 
 * This can be done as matrix-matrix multiply with the DFT matrix.
 */
class DenseMatrixNTT {
public:
    /**
     * Create dense matrix NTT processor
     * 
     * @param degree Polynomial degree
     * @param modulus Prime modulus
     */
    DenseMatrixNTT(size_t degree, uint64_t modulus);
    ~DenseMatrixNTT();
    
    /**
     * Compute batched NTT using dense matrix multiply
     * 
     * Uses BLAS for the matrix multiply, which leverages AMX on Apple Silicon.
     * 
     * @param input Input matrix (num_polys x degree)
     * @param output Output matrix (num_polys x degree)
     * @param num_polys Number of polynomials
     */
    void forward_ntt_dense(const uint64_t* input, uint64_t* output, size_t num_polys);
    
    /**
     * Compute batched inverse NTT using dense matrix multiply
     * 
     * @param input Input matrix (num_polys x degree)
     * @param output Output matrix (num_polys x degree)
     * @param num_polys Number of polynomials
     */
    void inverse_ntt_dense(const uint64_t* input, uint64_t* output, size_t num_polys);
    
private:
    size_t degree_;
    uint64_t modulus_;
    
    // Precomputed DFT matrix (degree x degree)
    // DFT[i][j] = w^(i*j) where w is primitive root
    std::vector<float> dft_matrix_float_;  // For BLAS
    std::vector<float> idft_matrix_float_; // Inverse DFT matrix
    
    // Build DFT matrix
    void build_dft_matrix();
};

// ============================================================================
// SME Tile-Based NTT
// ============================================================================

/**
 * SME tile-based NTT
 * 
 * Uses SME's tile registers for NTT butterfly operations.
 * Processes data in tiles that fit in SME registers.
 */
class SMETileNTT {
public:
    /**
     * Create SME tile NTT processor
     * 
     * @param degree Polynomial degree
     * @param modulus Prime modulus
     * @param tile_size SME tile size (default: 8 for 512-bit tiles)
     */
    SMETileNTT(size_t degree, uint64_t modulus, size_t tile_size = 8);
    ~SMETileNTT();
    
    /**
     * Check if SME is available
     */
    static bool is_available();
    
    /**
     * Compute forward NTT using SME tiles
     * 
     * @param coeffs Polynomial coefficients (modified in place)
     */
    void forward_ntt(uint64_t* coeffs);
    
    /**
     * Compute inverse NTT using SME tiles
     * 
     * @param coeffs NTT coefficients (modified in place)
     */
    void inverse_ntt(uint64_t* coeffs);
    
    /**
     * Benchmark different tile sizes
     * 
     * @param degree Polynomial degree
     * @param modulus Prime modulus
     * @return Optimal tile size
     */
    static size_t benchmark_tile_sizes(size_t degree, uint64_t modulus);
    
private:
    size_t degree_;
    size_t log_degree_;
    uint64_t modulus_;
    size_t tile_size_;
    
    std::vector<uint64_t> twiddles_;
    
    // Process a tile of butterflies
    void process_tile(uint64_t* coeffs, size_t stage, size_t tile_start);
};

// ============================================================================
// Benchmarking
// ============================================================================

/**
 * Benchmark matrix NTT vs scalar NTT
 */
struct MatrixNTTBenchmark {
    double scalar_time_us;
    double matrix_time_us;
    double dense_time_us;
    double sme_time_us;
    double speedup_matrix;
    double speedup_dense;
    double speedup_sme;
};

/**
 * Run matrix NTT benchmarks
 * 
 * @param degree Polynomial degree
 * @param modulus Prime modulus
 * @param num_iterations Number of iterations
 * @return Benchmark results
 */
MatrixNTTBenchmark benchmark_matrix_ntt(size_t degree, uint64_t modulus, size_t num_iterations = 100);

} // namespace matrix_ntt
} // namespace fhe_accelerate
