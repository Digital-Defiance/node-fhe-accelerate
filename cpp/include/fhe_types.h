#pragma once

#include <cstdint>
#include <vector>
#include <memory>

namespace fhe_accelerate {

/// Hardware backend types
enum class HardwareBackend {
    SME,      // ARM Scalable Matrix Extension
    Metal,    // Apple GPU compute
    NEON,     // ARM SIMD
    AMX,      // Apple Matrix Coprocessor (via Accelerate)
    Fallback  // Pure C++ implementation
};

/// Hardware capabilities structure
struct HardwareCapabilities {
    bool has_sme;
    bool has_metal;
    bool has_neon;
    bool has_amx;
    uint32_t metal_gpu_cores;
    uint64_t unified_memory_size;
};

/// Security levels
enum class SecurityLevel {
    Bits128 = 128,
    Bits192 = 192,
    Bits256 = 256
};

/// Polynomial representation (simple struct for basic use)
struct PolynomialData {
    std::vector<uint64_t> coeffs;
    uint64_t modulus;
    bool is_ntt;
    
    size_t degree() const { return coeffs.size(); }
};

/// Parameter set for FHE operations
struct ParameterSet {
    uint32_t poly_degree;
    std::vector<uint64_t> moduli;
    uint32_t lwe_dimension;
    double lwe_noise_std;
    uint32_t glwe_dimension;
    uint32_t decomp_base_log;
    uint32_t decomp_level;
    SecurityLevel security;
    uint64_t plaintext_modulus;
    double noise_budget;
    uint32_t max_mult_depth;
};

} // namespace fhe_accelerate
