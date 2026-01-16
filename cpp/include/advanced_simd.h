/**
 * Advanced SIMD Optimizations for FHE
 * 
 * Implements advanced NEON/SVE2 optimizations:
 * - Gather/scatter loads for NTT butterflies
 * - Predicated modular reduction
 * - Horizontal reductions for inner products
 * 
 * Requirements 14.26, 14.27, 14.28, 14.29
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace fhe_accelerate {
namespace advanced_simd {

/**
 * Check if advanced SIMD features are available
 */
bool sve2_available();
bool neon_available();

/**
 * Gather load for NTT butterfly patterns
 * 
 * Loads non-contiguous elements based on butterfly indices.
 * 
 * @param data Source array
 * @param indices Indices to gather
 * @param output Output array
 * @param count Number of elements
 */
void gather_load(const uint64_t* data, const uint32_t* indices, 
                 uint64_t* output, size_t count);

/**
 * Scatter store for inverse NTT
 * 
 * Stores elements to non-contiguous locations.
 * 
 * @param data Source array
 * @param indices Destination indices
 * @param output Output array
 * @param count Number of elements
 */
void scatter_store(const uint64_t* data, const uint32_t* indices,
                   uint64_t* output, size_t count);

/**
 * Predicated modular reduction
 * 
 * Only reduces elements that exceed the modulus.
 * Uses predication to avoid branches.
 * 
 * @param data Array to reduce (modified in place)
 * @param count Number of elements
 * @param modulus Modulus
 */
void predicated_reduce(uint64_t* data, size_t count, uint64_t modulus);

/**
 * Horizontal reduction for inner products
 * 
 * Computes sum of products: sum(a[i] * b[i]) mod modulus
 * 
 * @param a First array
 * @param b Second array
 * @param count Number of elements
 * @param modulus Modulus
 * @return Inner product
 */
uint64_t horizontal_inner_product(const uint64_t* a, const uint64_t* b,
                                   size_t count, uint64_t modulus);

/**
 * Vectorized Montgomery multiplication
 * 
 * @param a First array
 * @param b Second array
 * @param result Output array
 * @param count Number of elements
 * @param modulus Modulus
 */
void montgomery_mul_vec(const uint64_t* a, const uint64_t* b,
                        uint64_t* result, size_t count, uint64_t modulus);

/**
 * Vectorized Barrett reduction
 * 
 * @param data Array to reduce (modified in place)
 * @param count Number of elements
 * @param modulus Modulus
 */
void barrett_reduce_vec(uint64_t* data, size_t count, uint64_t modulus);

/**
 * NTT butterfly with gather/scatter
 * 
 * Processes NTT butterflies using gather/scatter for optimal memory access.
 * 
 * @param coeffs Coefficient array
 * @param degree Polynomial degree
 * @param stage NTT stage
 * @param twiddles Twiddle factors
 * @param modulus Modulus
 */
void ntt_butterfly_gather_scatter(uint64_t* coeffs, size_t degree, size_t stage,
                                   const uint64_t* twiddles, uint64_t modulus);

/**
 * Benchmark predication vs branching
 */
struct SIMDBenchmark {
    double branching_time_us;
    double predicated_time_us;
    double speedup;
};

SIMDBenchmark benchmark_predication_vs_branching(size_t count, uint64_t modulus);

} // namespace advanced_simd
} // namespace fhe_accelerate
