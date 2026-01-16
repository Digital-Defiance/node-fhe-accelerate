/**
 * Hardware-Accelerated Zero-Knowledge Proof Operations
 * 
 * Implements hardware acceleration for ZK proof systems:
 * - Metal GPU MSM (Multi-Scalar Multiplication) using Pippenger's algorithm
 * - Metal GPU FFT for ZK fields (BLS12-381, BN254)
 * - Neural Engine hash acceleration for Poseidon
 * - AMX-accelerated constraint evaluation
 * - Batch proof generation
 * 
 * Optimized for M4 Max with 40-core GPU, 38 TOPS Neural Engine.
 * Target: 10x speedup over CPU for MSM, 100x for hash tree construction.
 * 
 * Requirements: 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8
 */

#pragma once

#include "zk_field_arithmetic.h"
#include "zk_elliptic_curve.h"
#include "zk_hash.h"
#include "groth16.h"
#include "plonk.h"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Metal GPU MSM (Multi-Scalar Multiplication)
// ============================================================================

/**
 * Configuration for GPU MSM
 */
struct MSMConfig {
    size_t window_bits;           // Window size for Pippenger (auto-selected if 0)
    size_t max_batch_size;        // Maximum points per batch
    bool use_precomputation;      // Use precomputed point tables
    bool use_mixed_addition;      // Use mixed Jacobian-Affine addition
    
    MSMConfig() 
        : window_bits(0)
        , max_batch_size(1 << 20)
        , use_precomputation(true)
        , use_mixed_addition(true) {}
    
    /**
     * Auto-select optimal window size based on point count
     */
    static size_t optimal_window_bits(size_t num_points);
};

/**
 * Metal GPU MSM Engine
 * 
 * Implements Pippenger's algorithm with parallel bucket accumulation on GPU.
 * Optimized for M4 Max 40-core GPU.
 */
class MetalMSM {
public:
    MetalMSM();
    ~MetalMSM();
    
    /**
     * Check if Metal MSM is available
     */
    bool is_available() const;
    
    /**
     * Compute MSM: sum(scalars[i] * points[i])
     * 
     * @param points Array of affine points
     * @param scalars Array of scalars
     * @param count Number of point-scalar pairs
     * @param config MSM configuration
     * @return Result point in Jacobian coordinates
     */
    JacobianPoint256 msm_256(const AffinePoint256* points,
                              const FieldElement256* scalars,
                              size_t count,
                              const MSMConfig& config = MSMConfig()) const;
    
    /**
     * Compute MSM for 384-bit field (BLS12-381)
     */
    JacobianPoint384 msm_384(const AffinePoint384* points,
                              const FieldElement256* scalars,
                              size_t count,
                              const MSMConfig& config = MSMConfig()) const;
    
    /**
     * Batch MSM: compute multiple MSMs in parallel
     * 
     * @param points_batches Array of point arrays
     * @param scalars_batches Array of scalar arrays
     * @param counts Array of counts for each batch
     * @param num_batches Number of batches
     * @param results Output array for results
     */
    void batch_msm_256(const AffinePoint256* const* points_batches,
                       const FieldElement256* const* scalars_batches,
                       const size_t* counts,
                       size_t num_batches,
                       JacobianPoint256* results,
                       const MSMConfig& config = MSMConfig()) const;
    
    /**
     * Get estimated speedup over CPU
     */
    double estimated_speedup(size_t num_points) const;
    
    /**
     * Benchmark MSM performance
     */
    struct BenchmarkResult {
        double cpu_time_ms;
        double gpu_time_ms;
        double speedup;
        size_t num_points;
    };
    
    BenchmarkResult benchmark(size_t num_points, size_t iterations = 10) const;
    
private:
#ifdef __APPLE__
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;
    
    // Compute pipelines for MSM stages
    id<MTLComputePipelineState> bucket_accumulate_pipeline_;
    id<MTLComputePipelineState> bucket_reduce_pipeline_;
    id<MTLComputePipelineState> window_combine_pipeline_;
    id<MTLComputePipelineState> point_add_pipeline_;
    id<MTLComputePipelineState> point_double_pipeline_;
#else
    void* device_;
    void* command_queue_;
    void* library_;
    void* bucket_accumulate_pipeline_;
    void* bucket_reduce_pipeline_;
    void* window_combine_pipeline_;
    void* point_add_pipeline_;
    void* point_double_pipeline_;
#endif
    
    bool available_;
    
    /**
     * Initialize Metal pipelines
     */
    bool initialize_pipelines();
    
    /**
     * CPU fallback implementation
     */
    JacobianPoint256 msm_cpu_256(const AffinePoint256* points,
                                  const FieldElement256* scalars,
                                  size_t count,
                                  const MSMConfig& config) const;
    
    JacobianPoint384 msm_cpu_384(const AffinePoint384* points,
                                  const FieldElement256* scalars,
                                  size_t count,
                                  const MSMConfig& config) const;
};

// ============================================================================
// Metal GPU FFT for ZK Fields
// ============================================================================

/**
 * Configuration for GPU FFT
 */
struct FFTConfig {
    bool use_radix4;              // Use radix-4 FFT when possible
    bool in_place;                // Perform in-place FFT
    size_t max_batch_size;        // Maximum polynomials per batch
    
    FFTConfig()
        : use_radix4(true)
        , in_place(true)
        , max_batch_size(1024) {}
};

/**
 * Metal GPU FFT Engine for ZK Fields
 * 
 * Reuses NTT infrastructure for FFT over ZK fields.
 * Supports BLS12-381 and BN254 scalar fields.
 */
class MetalZKFFT {
public:
    MetalZKFFT();
    ~MetalZKFFT();
    
    /**
     * Check if Metal FFT is available
     */
    bool is_available() const;
    
    /**
     * Forward FFT over BN254 scalar field
     * 
     * @param coeffs Polynomial coefficients (modified in place)
     * @param degree Polynomial degree (must be power of 2)
     */
    void fft_bn254(FieldElement256* coeffs, size_t degree,
                   const FFTConfig& config = FFTConfig()) const;
    
    /**
     * Inverse FFT over BN254 scalar field
     */
    void ifft_bn254(FieldElement256* coeffs, size_t degree,
                    const FFTConfig& config = FFTConfig()) const;
    
    /**
     * Forward FFT over BLS12-381 scalar field
     */
    void fft_bls12_381(FieldElement256* coeffs, size_t degree,
                       const FFTConfig& config = FFTConfig()) const;
    
    /**
     * Inverse FFT over BLS12-381 scalar field
     */
    void ifft_bls12_381(FieldElement256* coeffs, size_t degree,
                        const FFTConfig& config = FFTConfig()) const;
    
    /**
     * Batch FFT: process multiple polynomials in parallel
     * 
     * @param coeffs_batch Array of coefficient arrays
     * @param degrees Array of degrees
     * @param num_polys Number of polynomials
     * @param inverse true for inverse FFT
     */
    void batch_fft_bn254(FieldElement256** coeffs_batch,
                         const size_t* degrees,
                         size_t num_polys,
                         bool inverse,
                         const FFTConfig& config = FFTConfig()) const;
    
    /**
     * Coset FFT for PLONK
     * Evaluates polynomial at coset ω^i * g for i = 0..n-1
     */
    void coset_fft_bn254(FieldElement256* coeffs, size_t degree,
                         const FieldElement256& coset_gen,
                         const FFTConfig& config = FFTConfig()) const;
    
    /**
     * Get precomputed twiddle factors
     */
    const std::vector<FieldElement256>& get_twiddles_bn254(size_t degree) const;
    const std::vector<FieldElement256>& get_twiddles_bls12_381(size_t degree) const;
    
private:
#ifdef __APPLE__
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;
    
    id<MTLComputePipelineState> fft_butterfly_pipeline_;
    id<MTLComputePipelineState> fft_batch_pipeline_;
    id<MTLComputePipelineState> bit_reverse_pipeline_;
#else
    void* device_;
    void* command_queue_;
    void* library_;
    void* fft_butterfly_pipeline_;
    void* fft_batch_pipeline_;
    void* bit_reverse_pipeline_;
#endif
    
    bool available_;
    
    // Cached twiddle factors
    mutable std::unordered_map<size_t, std::vector<FieldElement256>> twiddles_bn254_;
    mutable std::unordered_map<size_t, std::vector<FieldElement256>> twiddles_bls12_381_;
    
    /**
     * Initialize Metal pipelines
     */
    bool initialize_pipelines();
    
    /**
     * Precompute twiddle factors for given degree
     */
    void precompute_twiddles_bn254(size_t degree) const;
    void precompute_twiddles_bls12_381(size_t degree) const;
    
    /**
     * CPU fallback
     */
    void fft_cpu(FieldElement256* coeffs, size_t degree,
                 const std::vector<FieldElement256>& twiddles,
                 const Field256& field, bool inverse) const;
};

// ============================================================================
// Neural Engine Hash Acceleration
// ============================================================================

/**
 * Configuration for Neural Engine hashing
 */
struct NeuralHashConfig {
    size_t batch_size;            // Elements per batch
    bool use_quantization;        // Use INT8 quantization
    size_t num_threads;           // ANE threads to use
    
    NeuralHashConfig()
        : batch_size(1024)
        , use_quantization(true)
        , num_threads(4) {}
};

/**
 * Neural Engine Poseidon Hash Accelerator
 * 
 * Implements Poseidon hash as a quantized neural network.
 * Target: 100x speedup for hash tree construction.
 */
class NeuralPoseidon {
public:
    NeuralPoseidon();
    ~NeuralPoseidon();
    
    /**
     * Check if Neural Engine is available
     */
    bool is_available() const;
    
    /**
     * Hash two field elements
     */
    FieldElement256 hash2(const FieldElement256& left,
                          const FieldElement256& right,
                          const NeuralHashConfig& config = NeuralHashConfig()) const;
    
    /**
     * Batch hash pairs of elements
     * 
     * @param lefts Array of left inputs
     * @param rights Array of right inputs
     * @param outputs Output array
     * @param count Number of pairs
     */
    void batch_hash2(const FieldElement256* lefts,
                     const FieldElement256* rights,
                     FieldElement256* outputs,
                     size_t count,
                     const NeuralHashConfig& config = NeuralHashConfig()) const;
    
    /**
     * Build Merkle tree using Neural Engine
     * 
     * @param leaves Leaf values
     * @param tree Output tree layers
     * @return Root hash
     */
    FieldElement256 build_merkle_tree(const std::vector<FieldElement256>& leaves,
                                       std::vector<std::vector<FieldElement256>>& tree,
                                       const NeuralHashConfig& config = NeuralHashConfig()) const;
    
    /**
     * Parallel hash tree construction
     * Processes multiple trees simultaneously
     */
    void batch_merkle_trees(const std::vector<std::vector<FieldElement256>>& leaves_batch,
                            std::vector<FieldElement256>& roots,
                            const NeuralHashConfig& config = NeuralHashConfig()) const;
    
    /**
     * Get estimated speedup over CPU
     */
    double estimated_speedup(size_t num_hashes) const;
    
private:
    bool available_;
    PoseidonHash cpu_fallback_;
    
    // Neural Engine model handle (opaque)
    void* model_handle_;
    
    /**
     * Initialize Neural Engine model
     */
    bool initialize_model();
    
    /**
     * CPU fallback implementation
     */
    void batch_hash2_cpu(const FieldElement256* lefts,
                         const FieldElement256* rights,
                         FieldElement256* outputs,
                         size_t count) const;
};

// ============================================================================
// AMX-Accelerated Constraint Evaluation
// ============================================================================

/**
 * Configuration for AMX constraint evaluation
 */
struct AMXConstraintConfig {
    size_t tile_size;             // Matrix tile size
    bool use_blas;                // Use Accelerate BLAS
    size_t num_threads;           // Threads for parallel evaluation
    
    AMXConstraintConfig()
        : tile_size(32)
        , use_blas(true)
        , num_threads(8) {}
};

/**
 * AMX-Accelerated R1CS Constraint Evaluator
 * 
 * Uses AMX for R1CS matrix-vector products.
 * Optimizes witness generation with BLAS.
 */
class AMXConstraintEvaluator {
public:
    AMXConstraintEvaluator();
    ~AMXConstraintEvaluator();
    
    /**
     * Check if AMX acceleration is available
     */
    bool is_available() const;
    
    /**
     * Evaluate R1CS constraints: A*w ⊙ B*w = C*w
     * 
     * @param A Sparse matrix A (num_constraints x num_variables)
     * @param B Sparse matrix B
     * @param C Sparse matrix C
     * @param witness Witness vector
     * @return true if all constraints satisfied
     */
    bool evaluate_r1cs(const std::vector<SparseVector>& A,
                       const std::vector<SparseVector>& B,
                       const std::vector<SparseVector>& C,
                       const std::vector<FieldElement256>& witness,
                       const AMXConstraintConfig& config = AMXConstraintConfig()) const;
    
    /**
     * Compute A*w, B*w, C*w products
     * 
     * @param matrix Sparse matrix
     * @param witness Witness vector
     * @param result Output vector
     */
    void sparse_matvec(const std::vector<SparseVector>& matrix,
                       const std::vector<FieldElement256>& witness,
                       std::vector<FieldElement256>& result,
                       const AMXConstraintConfig& config = AMXConstraintConfig()) const;
    
    /**
     * Batch evaluate multiple constraint systems
     */
    void batch_evaluate(const std::vector<std::vector<SparseVector>>& A_batch,
                        const std::vector<std::vector<SparseVector>>& B_batch,
                        const std::vector<std::vector<SparseVector>>& C_batch,
                        const std::vector<std::vector<FieldElement256>>& witnesses,
                        std::vector<bool>& results,
                        const AMXConstraintConfig& config = AMXConstraintConfig()) const;
    
    /**
     * Optimize witness generation using BLAS
     * Computes intermediate values efficiently
     */
    void optimize_witness_generation(std::vector<FieldElement256>& witness,
                                     const std::vector<SparseVector>& constraints,
                                     const AMXConstraintConfig& config = AMXConstraintConfig()) const;
    
private:
    bool available_;
    
    /**
     * Convert sparse matrix to dense for BLAS
     */
    void to_dense_matrix(const std::vector<SparseVector>& sparse,
                         std::vector<double>& dense,
                         size_t num_cols) const;
    
    /**
     * CPU fallback
     */
    void sparse_matvec_cpu(const std::vector<SparseVector>& matrix,
                           const std::vector<FieldElement256>& witness,
                           std::vector<FieldElement256>& result) const;
};

// ============================================================================
// Batch Proof Generation
// ============================================================================

/**
 * Configuration for batch proof generation
 */
struct BatchProofConfig {
    size_t max_parallel_proofs;   // Maximum proofs to generate in parallel
    bool share_setup;             // Share setup data across proofs
    bool use_gpu;                 // Use GPU acceleration
    bool use_neural_engine;       // Use Neural Engine for hashing
    size_t gpu_memory_limit;      // GPU memory limit in bytes
    
    BatchProofConfig()
        : max_parallel_proofs(128)
        , share_setup(true)
        , use_gpu(true)
        , use_neural_engine(true)
        , gpu_memory_limit(8ULL * 1024 * 1024 * 1024) {}  // 8GB default
};

/**
 * Batch Proof Generator
 * 
 * Parallelizes proof generation across GPU cores.
 * Supports generating 100+ proofs simultaneously.
 */
class BatchProofGenerator {
public:
    BatchProofGenerator();
    ~BatchProofGenerator();
    
    /**
     * Check if batch generation is available
     */
    bool is_available() const;
    
    /**
     * Generate multiple Groth16 proofs in parallel
     * 
     * @param pk Proving key (shared across all proofs)
     * @param witnesses Array of witness vectors
     * @param num_proofs Number of proofs to generate
     * @param proofs Output array for proofs
     * @param config Batch configuration
     */
    void batch_groth16(const Groth16ProvingKey& pk,
                       const std::vector<std::vector<FieldElement256>>& witnesses,
                       std::vector<Groth16Proof>& proofs,
                       const BatchProofConfig& config = BatchProofConfig()) const;
    
    /**
     * Generate multiple PLONK proofs in parallel
     */
    void batch_plonk(const PLONKSetup& setup,
                     const PLONKProvingKey& pk,
                     const std::vector<std::vector<FieldElement256>>& witnesses,
                     const std::vector<std::vector<FieldElement256>>& public_inputs,
                     std::vector<PLONKProof>& proofs,
                     const BatchProofConfig& config = BatchProofConfig()) const;
    
    /**
     * Generate multiple Bulletproofs in parallel
     */
    void batch_bulletproofs(const std::vector<uint64_t>& values,
                            const std::vector<FieldElement256>& blindings,
                            size_t num_bits,
                            std::vector<std::vector<uint8_t>>& proofs,
                            const BatchProofConfig& config = BatchProofConfig()) const;
    
    /**
     * Get optimal batch size for given proof type
     */
    size_t optimal_batch_size(const std::string& proof_type,
                              size_t circuit_size) const;
    
    /**
     * Estimate time for batch proof generation
     */
    double estimate_time_ms(const std::string& proof_type,
                            size_t num_proofs,
                            size_t circuit_size) const;
    
private:
    std::unique_ptr<MetalMSM> msm_;
    std::unique_ptr<MetalZKFFT> fft_;
    std::unique_ptr<NeuralPoseidon> poseidon_;
    std::unique_ptr<AMXConstraintEvaluator> amx_;
    
    bool available_;
    
    /**
     * Parallel MSM for batch proofs
     */
    void parallel_msm(const std::vector<std::pair<const AffinePoint256*, const FieldElement256*>>& msm_inputs,
                      const std::vector<size_t>& counts,
                      std::vector<JacobianPoint256>& results) const;
    
    /**
     * Parallel FFT for batch proofs
     */
    void parallel_fft(std::vector<std::vector<FieldElement256>>& polynomials,
                      bool inverse) const;
};

// ============================================================================
// Global Hardware Acceleration Context
// ============================================================================

/**
 * Get global Metal MSM engine (singleton)
 */
MetalMSM& get_metal_msm();

/**
 * Get global Metal FFT engine (singleton)
 */
MetalZKFFT& get_metal_zk_fft();

/**
 * Get global Neural Engine Poseidon (singleton)
 */
NeuralPoseidon& get_neural_poseidon();

/**
 * Get global AMX constraint evaluator (singleton)
 */
AMXConstraintEvaluator& get_amx_evaluator();

/**
 * Get global batch proof generator (singleton)
 */
BatchProofGenerator& get_batch_proof_generator();

/**
 * Check overall hardware acceleration availability
 */
struct HardwareAccelStatus {
    bool metal_available;
    bool neural_engine_available;
    bool amx_available;
    uint32_t gpu_cores;
    double neural_engine_tops;
    size_t unified_memory_gb;
};

HardwareAccelStatus get_hardware_accel_status();

} // namespace zk
} // namespace fhe_accelerate
