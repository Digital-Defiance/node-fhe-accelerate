/**
 * Texture Sampling for Polynomial Evaluation
 * 
 * Exploits GPU texture sampling hardware for:
 * - Fast polynomial evaluation at multiple points
 * - Twiddle factor lookup with hardware interpolation
 * - LUT evaluation for bootstrapping
 * 
 * Requirements 14.15, 14.16, 14.17
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fhe_accelerate {
namespace texture_sampling {

/**
 * Check if texture sampling is available
 */
bool texture_sampling_available();

/**
 * Polynomial texture for fast evaluation
 * 
 * Encodes polynomial coefficients as a 1D texture.
 * Uses hardware interpolation for evaluation.
 */
class PolynomialTexture {
public:
    /**
     * Create polynomial texture
     * 
     * @param degree Maximum polynomial degree
     */
    explicit PolynomialTexture(size_t degree);
    ~PolynomialTexture();
    
    /**
     * Load polynomial coefficients into texture
     * 
     * @param coeffs Polynomial coefficients
     * @param degree Polynomial degree
     */
    void load(const uint64_t* coeffs, size_t degree);
    
    /**
     * Evaluate polynomial at multiple points using texture sampling
     * 
     * @param points Evaluation points (normalized to [0, 1])
     * @param results Output values
     * @param count Number of points
     * @param modulus Modulus for reduction
     */
    void evaluate(const float* points, uint64_t* results, size_t count, uint64_t modulus);
    
    /**
     * Check if texture sampling is beneficial
     */
    bool is_beneficial(size_t num_points) const;
    
private:
    size_t max_degree_;
    void* texture_;  // MTLTexture
    void* sampler_;  // MTLSamplerState
    void* pipeline_; // MTLComputePipelineState
};

/**
 * Twiddle factor texture for NTT
 * 
 * Stores precomputed twiddle factors as a 2D texture.
 * Indexed by (stage, butterfly_index).
 */
class TwiddleTexture {
public:
    /**
     * Create twiddle texture
     * 
     * @param max_degree Maximum polynomial degree
     * @param modulus Prime modulus
     */
    TwiddleTexture(size_t max_degree, uint64_t modulus);
    ~TwiddleTexture();
    
    /**
     * Precompute and load twiddle factors
     * 
     * @param primitive_root Primitive root of unity
     */
    void precompute(uint64_t primitive_root);
    
    /**
     * Sample twiddle factor using texture hardware
     * 
     * @param stage NTT stage
     * @param index Butterfly index
     * @return Twiddle factor
     */
    uint64_t sample(size_t stage, size_t index);
    
    /**
     * Batch sample twiddle factors
     * 
     * @param stage NTT stage
     * @param indices Butterfly indices
     * @param twiddles Output twiddle factors
     * @param count Number of samples
     */
    void sample_batch(size_t stage, const size_t* indices, uint64_t* twiddles, size_t count);
    
private:
    size_t max_degree_;
    size_t log_degree_;
    uint64_t modulus_;
    std::vector<uint64_t> twiddles_;  // CPU fallback
    void* texture_;
};

/**
 * LUT texture for bootstrapping
 * 
 * Stores lookup table as texture for fast evaluation.
 */
class LUTTexture {
public:
    /**
     * Create LUT texture
     * 
     * @param lut_size Size of lookup table
     */
    explicit LUTTexture(size_t lut_size);
    ~LUTTexture();
    
    /**
     * Load lookup table into texture
     * 
     * @param lut Lookup table values
     */
    void load(const uint64_t* lut);
    
    /**
     * Evaluate LUT at multiple indices
     * 
     * @param indices Input indices
     * @param results Output values
     * @param count Number of evaluations
     */
    void evaluate(const size_t* indices, uint64_t* results, size_t count);
    
    /**
     * Evaluate LUT with interpolation (for approximate values)
     * 
     * @param positions Positions (can be fractional)
     * @param results Output values
     * @param count Number of evaluations
     */
    void evaluate_interpolated(const float* positions, uint64_t* results, size_t count);
    
private:
    size_t lut_size_;
    std::vector<uint64_t> lut_;  // CPU fallback
    void* texture_;
};

/**
 * Benchmark texture sampling vs direct computation
 */
struct TextureSamplingBenchmark {
    double direct_time_us;
    double texture_time_us;
    double speedup;
    std::string operation;
};

TextureSamplingBenchmark benchmark_polynomial_eval(size_t degree, size_t num_points);
TextureSamplingBenchmark benchmark_twiddle_lookup(size_t degree, size_t num_lookups);
TextureSamplingBenchmark benchmark_lut_eval(size_t lut_size, size_t num_evals);

} // namespace texture_sampling
} // namespace fhe_accelerate
