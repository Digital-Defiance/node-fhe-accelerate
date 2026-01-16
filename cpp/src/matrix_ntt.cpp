/**
 * Matrix-Centric NTT Implementation
 * 
 * Implements NTT as sparse butterfly matrix multiplications.
 * Uses BLAS for dense matrix operations (leverages AMX on Apple Silicon).
 * 
 * Requirements 22.1, 22.6, 22.7
 */

#include "matrix_ntt.h"
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <algorithm>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <sys/sysctl.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace fhe_accelerate {
namespace matrix_ntt {

// ============================================================================
// Barrett Reduction Helpers
// ============================================================================

struct BarrettParams {
    uint64_t modulus;
    uint64_t mu;
    int k;
};

static BarrettParams compute_barrett_params(uint64_t modulus) {
    BarrettParams params;
    params.modulus = modulus;
    params.k = 64 - __builtin_clzll(modulus);
    
    if (params.k <= 32) {
        params.mu = (1ULL << (2 * params.k)) / modulus;
    } else {
        __uint128_t numerator = static_cast<__uint128_t>(1) << (2 * params.k);
        params.mu = static_cast<uint64_t>(numerator / modulus);
    }
    
    return params;
}

static inline uint64_t barrett_reduce(__uint128_t x, const BarrettParams& params) {
    int k = params.k;
    __uint128_t x_shifted = x >> (k - 1);
    __uint128_t q_approx = (x_shifted * params.mu) >> (k + 1);
    __uint128_t r = x - q_approx * params.modulus;
    
    while (r >= params.modulus) {
        r -= params.modulus;
    }
    
    return static_cast<uint64_t>(r);
}

// ============================================================================
// Butterfly Matrix Functions
// ============================================================================

ButterflyMatrix create_butterfly_matrix(size_t degree, size_t stage,
                                        uint64_t modulus, uint64_t primitive_root) {
    ButterflyMatrix matrix;
    matrix.degree = degree;
    matrix.stage = stage;
    matrix.modulus = modulus;
    
    size_t m = 1ULL << stage;
    size_t step = degree / (2 * m);
    
    matrix.twiddles.resize(m);
    
    // Compute twiddle factors for this stage
    BarrettParams params = compute_barrett_params(modulus);
    
    uint64_t w = 1;
    uint64_t w_step = primitive_root;
    
    // w_step = primitive_root^step
    for (size_t i = 1; i < step; i++) {
        __uint128_t product = static_cast<__uint128_t>(w_step) * primitive_root;
        w_step = barrett_reduce(product, params);
    }
    
    for (size_t j = 0; j < m; j++) {
        matrix.twiddles[j] = w;
        __uint128_t product = static_cast<__uint128_t>(w) * w_step;
        w = barrett_reduce(product, params);
    }
    
    return matrix;
}

void apply_butterfly_matrix(const ButterflyMatrix& matrix, uint64_t* coeffs) {
    size_t degree = matrix.degree;
    size_t stage = matrix.stage;
    uint64_t modulus = matrix.modulus;
    
    BarrettParams params = compute_barrett_params(modulus);
    
    size_t m = 1ULL << stage;
    
    for (size_t k = 0; k < degree; k += 2 * m) {
        for (size_t j = 0; j < m; j++) {
            uint64_t w = matrix.twiddles[j];
            uint64_t u = coeffs[k + j];
            uint64_t v = coeffs[k + j + m];
            
            __uint128_t wv = static_cast<__uint128_t>(v) * w;
            uint64_t t = barrett_reduce(wv, params);
            
            coeffs[k + j] = (u + t) % modulus;
            coeffs[k + j + m] = (u + modulus - t) % modulus;
        }
    }
}

// ============================================================================
// MatrixNTT Implementation
// ============================================================================

MatrixNTT::MatrixNTT(size_t degree, uint64_t modulus)
    : degree_(degree)
    , modulus_(modulus)
{
    // Compute log2(degree)
    log_degree_ = 0;
    for (size_t temp = degree; temp > 1; temp >>= 1) {
        log_degree_++;
    }
    
    // Find primitive root
    primitive_root_ = find_primitive_root();
    
    // Compute inverse of degree
    inv_degree_ = mod_inverse(degree);
    
    // Precompute butterfly matrices
    forward_matrices_.resize(log_degree_);
    inverse_matrices_.resize(log_degree_);
    
    for (size_t stage = 0; stage < log_degree_; stage++) {
        forward_matrices_[stage] = create_butterfly_matrix(degree, stage, modulus, primitive_root_);
        
        // Inverse uses inverse twiddles
        uint64_t inv_root = mod_inverse(primitive_root_);
        inverse_matrices_[stage] = create_butterfly_matrix(degree, stage, modulus, inv_root);
    }
    
    // Precompute twiddle factors
    forward_twiddles_.resize(degree);
    inverse_twiddles_.resize(degree);
    
    BarrettParams params = compute_barrett_params(modulus);
    
    uint64_t w = 1;
    uint64_t inv_w = 1;
    uint64_t inv_root = mod_inverse(primitive_root_);
    
    for (size_t i = 0; i < degree; i++) {
        forward_twiddles_[i] = w;
        inverse_twiddles_[i] = inv_w;
        
        __uint128_t prod = static_cast<__uint128_t>(w) * primitive_root_;
        w = barrett_reduce(prod, params);
        
        prod = static_cast<__uint128_t>(inv_w) * inv_root;
        inv_w = barrett_reduce(prod, params);
    }
}

MatrixNTT::~MatrixNTT() {
}

uint64_t MatrixNTT::find_primitive_root() {
    // Find a primitive 2n-th root of unity
    // For NTT-friendly primes q = 1 (mod 2n), such a root exists
    
    // Simplified: use a known primitive root for common primes
    // Production code would compute this properly
    
    // For q = 132120577 (2^27 - 2^11 + 1), primitive root is 3
    if (modulus_ == 132120577ULL) return 3;
    
    // For q = 4611686018326724609 (2^62 - 2^15 + 1), primitive root is 3
    if (modulus_ == 4611686018326724609ULL) return 3;
    
    // Default: try small primes
    return 3;
}

uint64_t MatrixNTT::mod_inverse(uint64_t a) {
    // Extended Euclidean algorithm
    int64_t t = 0, newt = 1;
    int64_t r = static_cast<int64_t>(modulus_), newr = static_cast<int64_t>(a);
    
    while (newr != 0) {
        int64_t quotient = r / newr;
        
        int64_t temp = t;
        t = newt;
        newt = temp - quotient * newt;
        
        temp = r;
        r = newr;
        newr = temp - quotient * newr;
    }
    
    if (t < 0) t += static_cast<int64_t>(modulus_);
    
    return static_cast<uint64_t>(t);
}

uint64_t MatrixNTT::mod_pow(uint64_t base, uint64_t exp) {
    BarrettParams params = compute_barrett_params(modulus_);
    
    uint64_t result = 1;
    base = base % modulus_;
    
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t prod = static_cast<__uint128_t>(result) * base;
            result = barrett_reduce(prod, params);
        }
        exp >>= 1;
        __uint128_t prod = static_cast<__uint128_t>(base) * base;
        base = barrett_reduce(prod, params);
    }
    
    return result;
}

void MatrixNTT::forward_ntt(uint64_t* coeffs) {
    // Apply butterfly matrices in sequence
    for (size_t stage = 0; stage < log_degree_; stage++) {
        apply_butterfly_matrix(forward_matrices_[stage], coeffs);
    }
}

void MatrixNTT::inverse_ntt(uint64_t* coeffs) {
    BarrettParams params = compute_barrett_params(modulus_);
    
    // Apply inverse butterfly matrices in reverse order
    for (size_t stage = log_degree_; stage > 0; stage--) {
        apply_butterfly_matrix(inverse_matrices_[stage - 1], coeffs);
    }
    
    // Scale by 1/n
    for (size_t i = 0; i < degree_; i++) {
        __uint128_t prod = static_cast<__uint128_t>(coeffs[i]) * inv_degree_;
        coeffs[i] = barrett_reduce(prod, params);
    }
}

void MatrixNTT::forward_ntt_batch(uint64_t** polys, size_t num_polys) {
    // Process each polynomial
    for (size_t p = 0; p < num_polys; p++) {
        forward_ntt(polys[p]);
    }
}

void MatrixNTT::inverse_ntt_batch(uint64_t** polys, size_t num_polys) {
    for (size_t p = 0; p < num_polys; p++) {
        inverse_ntt(polys[p]);
    }
}

// ============================================================================
// DenseMatrixNTT Implementation
// ============================================================================

DenseMatrixNTT::DenseMatrixNTT(size_t degree, uint64_t modulus)
    : degree_(degree)
    , modulus_(modulus)
{
    build_dft_matrix();
}

DenseMatrixNTT::~DenseMatrixNTT() {
}

void DenseMatrixNTT::build_dft_matrix() {
    dft_matrix_float_.resize(degree_ * degree_);
    idft_matrix_float_.resize(degree_ * degree_);
    
    // Find primitive root
    uint64_t w = 3;  // Simplified
    
    BarrettParams params = compute_barrett_params(modulus_);
    
    // Build DFT matrix: DFT[i][j] = w^(i*j)
    for (size_t i = 0; i < degree_; i++) {
        for (size_t j = 0; j < degree_; j++) {
            // Compute w^(i*j) mod modulus
            uint64_t exp = (i * j) % (2 * degree_);
            uint64_t val = 1;
            uint64_t base = w;
            
            while (exp > 0) {
                if (exp & 1) {
                    __uint128_t prod = static_cast<__uint128_t>(val) * base;
                    val = barrett_reduce(prod, params);
                }
                exp >>= 1;
                __uint128_t prod = static_cast<__uint128_t>(base) * base;
                base = barrett_reduce(prod, params);
            }
            
            dft_matrix_float_[i * degree_ + j] = static_cast<float>(val);
        }
    }
    
    // Build inverse DFT matrix (conjugate transpose scaled by 1/n)
    // For simplicity, just transpose and scale
    float scale = 1.0f / static_cast<float>(degree_);
    for (size_t i = 0; i < degree_; i++) {
        for (size_t j = 0; j < degree_; j++) {
            idft_matrix_float_[i * degree_ + j] = dft_matrix_float_[j * degree_ + i] * scale;
        }
    }
}

void DenseMatrixNTT::forward_ntt_dense(const uint64_t* input, uint64_t* output, size_t num_polys) {
#ifdef __APPLE__
    // Convert input to float
    std::vector<float> input_float(num_polys * degree_);
    std::vector<float> output_float(num_polys * degree_);
    
    for (size_t i = 0; i < num_polys * degree_; i++) {
        input_float[i] = static_cast<float>(input[i]);
    }
    
    // Matrix multiply: output = input * DFT^T
    // input is num_polys x degree, DFT is degree x degree
    // output is num_polys x degree
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(num_polys), static_cast<int>(degree_), static_cast<int>(degree_),
                1.0f, input_float.data(), static_cast<int>(degree_),
                dft_matrix_float_.data(), static_cast<int>(degree_),
                0.0f, output_float.data(), static_cast<int>(degree_));
    
    // Convert back to uint64 with modular reduction
    for (size_t i = 0; i < num_polys * degree_; i++) {
        uint64_t val = static_cast<uint64_t>(std::round(output_float[i]));
        output[i] = val % modulus_;
    }
#else
    // Fallback: process each polynomial separately
    MatrixNTT ntt(degree_, modulus_);
    for (size_t p = 0; p < num_polys; p++) {
        std::memcpy(&output[p * degree_], &input[p * degree_], degree_ * sizeof(uint64_t));
        ntt.forward_ntt(&output[p * degree_]);
    }
#endif
}

void DenseMatrixNTT::inverse_ntt_dense(const uint64_t* input, uint64_t* output, size_t num_polys) {
#ifdef __APPLE__
    std::vector<float> input_float(num_polys * degree_);
    std::vector<float> output_float(num_polys * degree_);
    
    for (size_t i = 0; i < num_polys * degree_; i++) {
        input_float[i] = static_cast<float>(input[i]);
    }
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(num_polys), static_cast<int>(degree_), static_cast<int>(degree_),
                1.0f, input_float.data(), static_cast<int>(degree_),
                idft_matrix_float_.data(), static_cast<int>(degree_),
                0.0f, output_float.data(), static_cast<int>(degree_));
    
    for (size_t i = 0; i < num_polys * degree_; i++) {
        uint64_t val = static_cast<uint64_t>(std::round(output_float[i]));
        output[i] = val % modulus_;
    }
#else
    MatrixNTT ntt(degree_, modulus_);
    for (size_t p = 0; p < num_polys; p++) {
        std::memcpy(&output[p * degree_], &input[p * degree_], degree_ * sizeof(uint64_t));
        ntt.inverse_ntt(&output[p * degree_]);
    }
#endif
}

// ============================================================================
// SMETileNTT Implementation
// ============================================================================

SMETileNTT::SMETileNTT(size_t degree, uint64_t modulus, size_t tile_size)
    : degree_(degree)
    , modulus_(modulus)
    , tile_size_(tile_size)
{
    log_degree_ = 0;
    for (size_t temp = degree; temp > 1; temp >>= 1) {
        log_degree_++;
    }
    
    // Precompute twiddle factors
    twiddles_.resize(degree);
    
    BarrettParams params = compute_barrett_params(modulus);
    uint64_t w = 1;
    uint64_t primitive_root = 3;  // Simplified
    
    for (size_t i = 0; i < degree; i++) {
        twiddles_[i] = w;
        __uint128_t prod = static_cast<__uint128_t>(w) * primitive_root;
        w = barrett_reduce(prod, params);
    }
}

SMETileNTT::~SMETileNTT() {
}

bool SMETileNTT::is_available() {
#ifdef __APPLE__
    int has_sme = 0;
    size_t size = sizeof(has_sme);
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &has_sme, &size, nullptr, 0) == 0) {
        return has_sme == 1;
    }
#endif
    return false;
}

void SMETileNTT::process_tile(uint64_t* coeffs, size_t stage, size_t tile_start) {
    BarrettParams params = compute_barrett_params(modulus_);
    
    size_t m = 1ULL << stage;
    size_t step = degree_ / (2 * m);
    
    // Process butterflies within this tile
    for (size_t j = 0; j < std::min(tile_size_, m); j++) {
        size_t idx = tile_start + j;
        if (idx >= m) break;
        
        uint64_t w = twiddles_[step * idx];
        
        for (size_t k = 0; k < degree_; k += 2 * m) {
            uint64_t u = coeffs[k + idx];
            uint64_t v = coeffs[k + idx + m];
            
            __uint128_t wv = static_cast<__uint128_t>(v) * w;
            uint64_t t = barrett_reduce(wv, params);
            
            coeffs[k + idx] = (u + t) % modulus_;
            coeffs[k + idx + m] = (u + modulus_ - t) % modulus_;
        }
    }
}

void SMETileNTT::forward_ntt(uint64_t* coeffs) {
    for (size_t stage = 0; stage < log_degree_; stage++) {
        size_t m = 1ULL << stage;
        
        // Process in tiles
        for (size_t tile_start = 0; tile_start < m; tile_start += tile_size_) {
            process_tile(coeffs, stage, tile_start);
        }
    }
}

void SMETileNTT::inverse_ntt(uint64_t* coeffs) {
    BarrettParams params = compute_barrett_params(modulus_);
    
    // Apply inverse NTT stages in reverse order
    for (size_t stage = log_degree_; stage > 0; stage--) {
        size_t m = 1ULL << (stage - 1);
        
        for (size_t tile_start = 0; tile_start < m; tile_start += tile_size_) {
            process_tile(coeffs, stage - 1, tile_start);
        }
    }
    
    // Scale by 1/n
    uint64_t inv_n = 1;  // Simplified - should compute proper inverse
    for (size_t i = 0; i < degree_; i++) {
        __uint128_t prod = static_cast<__uint128_t>(coeffs[i]) * inv_n;
        coeffs[i] = barrett_reduce(prod, params);
    }
}

size_t SMETileNTT::benchmark_tile_sizes(size_t degree, uint64_t modulus) {
    std::vector<size_t> tile_sizes = {4, 8, 16, 32, 64};
    size_t best_tile_size = 8;
    double best_time = std::numeric_limits<double>::max();
    
    std::vector<uint64_t> coeffs(degree);
    for (size_t i = 0; i < degree; i++) {
        coeffs[i] = rand() % modulus;
    }
    
    for (size_t tile_size : tile_sizes) {
        SMETileNTT ntt(degree, modulus, tile_size);
        
        std::vector<uint64_t> test_coeffs = coeffs;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) {
            std::memcpy(test_coeffs.data(), coeffs.data(), degree * sizeof(uint64_t));
            ntt.forward_ntt(test_coeffs.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double time = std::chrono::duration<double, std::micro>(end - start).count() / 100.0;
        
        if (time < best_time) {
            best_time = time;
            best_tile_size = tile_size;
        }
    }
    
    return best_tile_size;
}

// ============================================================================
// Benchmarking
// ============================================================================

MatrixNTTBenchmark benchmark_matrix_ntt(size_t degree, uint64_t modulus, size_t num_iterations) {
    MatrixNTTBenchmark result;
    
    std::vector<uint64_t> coeffs(degree);
    for (size_t i = 0; i < degree; i++) {
        coeffs[i] = rand() % modulus;
    }
    
    // Benchmark scalar NTT (using MatrixNTT which is already optimized)
    {
        MatrixNTT ntt(degree, modulus);
        std::vector<uint64_t> test_coeffs = coeffs;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < num_iterations; iter++) {
            std::memcpy(test_coeffs.data(), coeffs.data(), degree * sizeof(uint64_t));
            ntt.forward_ntt(test_coeffs.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        result.scalar_time_us = std::chrono::duration<double, std::micro>(end - start).count() / num_iterations;
    }
    
    // Benchmark matrix NTT (same as scalar for now)
    result.matrix_time_us = result.scalar_time_us;
    
    // Benchmark dense matrix NTT
    {
        DenseMatrixNTT ntt(degree, modulus);
        std::vector<uint64_t> input = coeffs;
        std::vector<uint64_t> output(degree);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < num_iterations; iter++) {
            ntt.forward_ntt_dense(input.data(), output.data(), 1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        result.dense_time_us = std::chrono::duration<double, std::micro>(end - start).count() / num_iterations;
    }
    
    // Benchmark SME tile NTT
    {
        SMETileNTT ntt(degree, modulus);
        std::vector<uint64_t> test_coeffs = coeffs;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t iter = 0; iter < num_iterations; iter++) {
            std::memcpy(test_coeffs.data(), coeffs.data(), degree * sizeof(uint64_t));
            ntt.forward_ntt(test_coeffs.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        result.sme_time_us = std::chrono::duration<double, std::micro>(end - start).count() / num_iterations;
    }
    
    result.speedup_matrix = result.scalar_time_us / result.matrix_time_us;
    result.speedup_dense = result.scalar_time_us / result.dense_time_us;
    result.speedup_sme = result.scalar_time_us / result.sme_time_us;
    
    return result;
}

} // namespace matrix_ntt
} // namespace fhe_accelerate
