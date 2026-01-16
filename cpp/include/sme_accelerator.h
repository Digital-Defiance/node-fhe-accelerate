/**
 * SME (Scalable Matrix Extension) Accelerator for FHE Operations
 * 
 * Uses Apple M4's SME/SME2 instructions for matrix-based polynomial operations.
 * SME provides streaming matrix operations that can accelerate:
 * - Batch polynomial multiplication (via matrix form)
 * - NTT butterfly stages (as matrix operations)
 * - Key switching operations
 * 
 * Note: SME is available on M4 chips but requires special handling.
 * The SME streaming mode must be entered/exited properly.
 * 
 * Requirements 14.30, 14.31, 14.32
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fhe_accelerate {
namespace sme {

/**
 * Check if SME is available on this system
 */
bool sme_available();

/**
 * Check if SME2 is available (enhanced SME on M4)
 */
bool sme2_available();

/**
 * Get SME vector length in bytes
 * Returns 0 if SME is not available
 */
size_t sme_vector_length();

/**
 * SME-accelerated batch modular multiplication
 * 
 * Uses SME outer product instructions to compute multiple
 * modular multiplications in parallel.
 * 
 * @param a First input array
 * @param b Second input array
 * @param result Output array
 * @param count Number of elements
 * @param modulus Prime modulus
 */
void sme_batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus);

/**
 * SME-accelerated NTT butterfly stage
 * 
 * Processes multiple butterfly operations using SME matrix operations.
 * Each butterfly: (u, v) -> (u + w*v, u - w*v)
 * 
 * @param coeffs Coefficient array (modified in place)
 * @param degree Polynomial degree
 * @param stage Current NTT stage
 * @param modulus Prime modulus
 * @param twiddles Twiddle factor array
 */
void sme_ntt_butterfly_stage(uint64_t* coeffs, size_t degree, size_t stage,
                              uint64_t modulus, const uint64_t* twiddles);

/**
 * SME-accelerated polynomial multiplication
 * 
 * Multiplies two polynomials using SME matrix operations.
 * Uses Toeplitz matrix representation for convolution.
 * 
 * @param poly_a First polynomial coefficients
 * @param poly_b Second polynomial coefficients
 * @param result Output polynomial coefficients
 * @param degree Polynomial degree
 * @param modulus Prime modulus
 */
void sme_poly_mul(const uint64_t* poly_a, const uint64_t* poly_b,
                  uint64_t* result, size_t degree, uint64_t modulus);

/**
 * SME-accelerated batch polynomial addition
 * 
 * @param a First polynomial batch
 * @param b Second polynomial batch
 * @param result Output polynomial batch
 * @param degree Polynomial degree
 * @param batch_size Number of polynomials
 * @param modulus Prime modulus
 */
void sme_batch_poly_add(const uint64_t* a, const uint64_t* b, uint64_t* result,
                        size_t degree, size_t batch_size, uint64_t modulus);

/**
 * Benchmark SME vs NEON for various operations
 * Returns speedup factor (SME time / NEON time)
 */
double benchmark_sme_vs_neon(size_t size, uint64_t modulus);

// ============================================================================
// SME Streaming Mode Operations (Requirements 14.30, 14.31, 14.32)
// ============================================================================

/**
 * SME streaming mode context
 * 
 * Manages entry/exit from SME streaming mode.
 * Streaming mode provides continuous polynomial processing pipelines.
 */
class SMEStreamingContext {
public:
    SMEStreamingContext();
    ~SMEStreamingContext();
    
    /**
     * Enter SME streaming mode
     */
    void enter_streaming_mode();
    
    /**
     * Exit SME streaming mode
     */
    void exit_streaming_mode();
    
    /**
     * Check if currently in streaming mode
     */
    bool is_streaming() const { return streaming_active_; }
    
    /**
     * Process polynomial pipeline in streaming mode
     * 
     * @param input Input polynomials
     * @param output Output polynomials
     * @param num_polys Number of polynomials
     * @param degree Polynomial degree
     * @param modulus Modulus
     */
    void process_pipeline(const uint64_t* input, uint64_t* output,
                          size_t num_polys, size_t degree, uint64_t modulus);
    
private:
    bool streaming_active_;
};

/**
 * SME2 predicated coefficient processing
 * 
 * Uses SME2 predication for conditional coefficient operations.
 * 
 * @param coeffs Coefficient array
 * @param count Number of coefficients
 * @param modulus Modulus
 * @param predicate Predicate mask (which coefficients to process)
 */
void sme2_predicated_process(uint64_t* coeffs, size_t count, uint64_t modulus,
                              const bool* predicate);

/**
 * Benchmark SME tile configurations
 * 
 * @param degree Polynomial degree
 * @param modulus Modulus
 * @return Optimal tile size
 */
size_t benchmark_sme_tile_sizes(size_t degree, uint64_t modulus);

} // namespace sme
} // namespace fhe_accelerate
