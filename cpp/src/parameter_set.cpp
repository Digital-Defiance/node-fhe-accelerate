/**
 * @file parameter_set.cpp
 * @brief Implementation of FHE Parameter Set presets and utilities
 * 
 * This file implements the preset parameter configurations for various
 * FHE schemes and security levels, along with derived parameter calculations.
 * 
 * Requirements: 10.1, 10.2, 10.3
 */

#include "parameter_set.h"
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace fhe_accelerate {

// ========== Common NTT-Friendly Primes ==========
// These primes are of the form q = 1 (mod 2N) for efficient NTT

namespace primes {
    // 60-bit primes for high precision
    constexpr uint64_t Q_60_1 = 1152921504606584833ULL;  // 2^60 - 2^14 + 1
    constexpr uint64_t Q_60_2 = 1152921504598720513ULL;  // Another 60-bit NTT prime
    constexpr uint64_t Q_60_3 = 1152921504597016577ULL;  // Another 60-bit NTT prime
    
    // 50-bit primes for medium precision
    constexpr uint64_t Q_50_1 = 1125899906826241ULL;     // 2^50 - 2^13 + 1
    constexpr uint64_t Q_50_2 = 1125899906793473ULL;     // Another 50-bit NTT prime
    
    // 40-bit primes for TFHE
    constexpr uint64_t Q_40_1 = 1099511627777ULL;        // 2^40 - 2^13 + 1
    constexpr uint64_t Q_40_2 = 1099511562241ULL;        // Another 40-bit NTT prime
    
    // 30-bit primes for fast operations
    constexpr uint64_t Q_30_1 = 1073479681ULL;           // 2^30 - 2^14 + 1
    constexpr uint64_t Q_30_2 = 1073217537ULL;           // Another 30-bit NTT prime
    
    // Special prime for TFHE bootstrapping
    constexpr uint64_t Q_TFHE_BOOT = 4294967296ULL;      // 2^32 (power of 2 for torus)
}

void ParameterSet::calculate_derived_parameters() {
    // Calculate noise budget based on modulus and parameters
    // Noise budget ≈ log2(q/t) - noise_growth_per_operation
    
    double log_q = get_log_modulus();
    double log_t = std::log2(static_cast<double>(plaintext_modulus));
    
    // Initial noise budget estimate
    // For TFHE, this is related to the precision of the torus representation
    if (scheme == FHEScheme::TFHE) {
        // TFHE noise budget is determined by the LWE parameters
        // Approximate formula: budget ≈ log2(q) - log2(σ * sqrt(n))
        double noise_term = std::log2(lwe_noise_std * std::sqrt(static_cast<double>(lwe_dimension)));
        noise_budget = log_q - noise_term - 10.0; // Safety margin
    } else {
        // BFV/CKKS noise budget
        noise_budget = log_q - log_t - 20.0; // Conservative estimate
    }
    
    // Ensure noise budget is non-negative
    if (noise_budget < 0) {
        noise_budget = 0;
    }
    
    // Calculate maximum multiplication depth
    // Each multiplication roughly doubles the noise
    // depth ≈ noise_budget / noise_per_mult
    double noise_per_mult = 10.0; // Approximate bits consumed per multiplication
    max_mult_depth = static_cast<uint32_t>(noise_budget / noise_per_mult);
    
    // For TFHE with bootstrapping, depth is unlimited (but we set a practical limit)
    if (scheme == FHEScheme::TFHE && decomp_level > 0) {
        max_mult_depth = 1000; // Effectively unlimited with bootstrapping
    }
}

std::string ParameterSet::to_string() const {
    std::ostringstream oss;
    oss << "ParameterSet {\n";
    oss << "  scheme: " << (scheme == FHEScheme::TFHE ? "TFHE" : 
                           scheme == FHEScheme::BFV ? "BFV" : "CKKS") << "\n";
    oss << "  security: " << static_cast<uint32_t>(security) << " bits\n";
    oss << "  poly_degree: " << poly_degree << "\n";
    oss << "  moduli: [";
    for (size_t i = 0; i < moduli.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << moduli[i];
    }
    oss << "]\n";
    oss << "  log2(q): " << get_log_modulus() << " bits\n";
    oss << "  lwe_dimension: " << lwe_dimension << "\n";
    oss << "  lwe_noise_std: " << lwe_noise_std << "\n";
    oss << "  glwe_dimension: " << glwe_dimension << "\n";
    oss << "  decomp_base_log: " << decomp_base_log << "\n";
    oss << "  decomp_level: " << decomp_level << "\n";
    oss << "  plaintext_modulus: " << plaintext_modulus << "\n";
    oss << "  noise_budget: " << noise_budget << " bits\n";
    oss << "  max_mult_depth: " << max_mult_depth << "\n";
    oss << "}";
    return oss.str();
}

// ========== Preset Implementations ==========

ParameterSet TFHE_128_FAST() {
    ParameterSet params;
    
    // Core parameters for fast bootstrapping
    params.scheme = FHEScheme::TFHE;
    params.security = SecurityLevel::Bits128;
    params.poly_degree = 1024;  // Smaller degree for speed
    
    // Single modulus for TFHE (torus representation)
    params.moduli = { primes::Q_40_1 };
    
    // LWE parameters (based on TFHE-rs defaults for 128-bit security)
    params.lwe_dimension = 742;
    params.lwe_noise_std = 3.2e-11;  // Standard deviation for Gaussian noise
    
    // GLWE parameters
    params.glwe_dimension = 1;  // k=1 for GLWE
    
    // Gadget decomposition for bootstrapping
    params.decomp_base_log = 23;  // log2(base) for decomposition
    params.decomp_level = 1;      // Number of decomposition levels
    
    // Plaintext space (typically small for TFHE)
    params.plaintext_modulus = 4;  // 2-bit plaintexts
    
    // Calculate derived parameters
    params.calculate_derived_parameters();
    
    return params;
}

ParameterSet TFHE_128_BALANCED() {
    ParameterSet params;
    
    params.scheme = FHEScheme::TFHE;
    params.security = SecurityLevel::Bits128;
    params.poly_degree = 2048;  // Larger degree for better noise
    
    params.moduli = { primes::Q_50_1 };
    
    // LWE parameters
    params.lwe_dimension = 830;
    params.lwe_noise_std = 2.9e-11;
    
    // GLWE parameters
    params.glwe_dimension = 1;
    
    // Gadget decomposition
    params.decomp_base_log = 15;
    params.decomp_level = 2;
    
    params.plaintext_modulus = 8;  // 3-bit plaintexts
    
    params.calculate_derived_parameters();
    
    return params;
}

ParameterSet TFHE_256_SECURE() {
    ParameterSet params;
    
    params.scheme = FHEScheme::TFHE;
    params.security = SecurityLevel::Bits256;
    params.poly_degree = 4096;  // Larger degree for higher security
    
    params.moduli = { primes::Q_60_1 };
    
    // LWE parameters for 256-bit security
    params.lwe_dimension = 1024;
    params.lwe_noise_std = 2.0e-12;
    
    // GLWE parameters
    params.glwe_dimension = 1;
    
    // Gadget decomposition
    params.decomp_base_log = 10;
    params.decomp_level = 3;
    
    params.plaintext_modulus = 16;  // 4-bit plaintexts
    
    params.calculate_derived_parameters();
    
    return params;
}

ParameterSet BFV_128_SIMD() {
    ParameterSet params;
    
    params.scheme = FHEScheme::BFV;
    params.security = SecurityLevel::Bits128;
    params.poly_degree = 8192;  // Larger degree to support the modulus chain
    
    // RNS modulus chain for BFV
    params.moduli = { 
        primes::Q_60_1, 
        primes::Q_60_2, 
        primes::Q_60_3 
    };
    
    // LWE parameters (not used in BFV, but set for consistency)
    params.lwe_dimension = 0;
    params.lwe_noise_std = 3.2;  // Standard deviation for RLWE noise
    
    // GLWE/RLWE parameters
    params.glwe_dimension = 1;
    
    // Decomposition for relinearization
    params.decomp_base_log = 60;
    params.decomp_level = 3;
    
    // Plaintext modulus for integer arithmetic
    params.plaintext_modulus = 65537;  // Common choice: 2^16 + 1 (prime)
    
    params.calculate_derived_parameters();
    
    return params;
}

ParameterSet CKKS_128_ML() {
    ParameterSet params;
    
    params.scheme = FHEScheme::CKKS;
    params.security = SecurityLevel::Bits128;
    params.poly_degree = 16384;  // Larger degree for ML workloads
    
    // RNS modulus chain for CKKS (more levels for rescaling)
    params.moduli = { 
        primes::Q_60_1, 
        primes::Q_50_1, 
        primes::Q_50_2,
        primes::Q_40_1,
        primes::Q_40_2
    };
    
    // RLWE parameters
    params.lwe_dimension = 0;
    params.lwe_noise_std = 3.2;
    
    params.glwe_dimension = 1;
    
    // Decomposition for relinearization
    params.decomp_base_log = 40;
    params.decomp_level = 5;
    
    // CKKS uses scaling factor instead of plaintext modulus
    // We store the initial scale here
    params.plaintext_modulus = 1ULL << 40;  // 2^40 scale
    
    params.calculate_derived_parameters();
    
    return params;
}

ParameterSet TFHE_128_VOTING() {
    ParameterSet params;
    
    // Optimized for voting: fast encryption, compact ciphertexts
    params.scheme = FHEScheme::TFHE;
    params.security = SecurityLevel::Bits128;
    params.poly_degree = 1024;  // Compact for network transmission
    
    params.moduli = { primes::Q_40_1 };
    
    // LWE parameters optimized for voting
    params.lwe_dimension = 742;
    params.lwe_noise_std = 3.2e-11;
    
    params.glwe_dimension = 1;
    
    // Fast bootstrapping parameters
    params.decomp_base_log = 23;
    params.decomp_level = 1;
    
    // Small plaintext space for vote choices (0-9 candidates)
    params.plaintext_modulus = 16;  // 4-bit plaintexts
    
    params.calculate_derived_parameters();
    
    return params;
}

// ========== Factory Functions ==========

ParameterSet create_parameter_set(const std::string& preset_name) {
    if (preset_name == "tfhe-128-fast") {
        return TFHE_128_FAST();
    } else if (preset_name == "tfhe-128-balanced") {
        return TFHE_128_BALANCED();
    } else if (preset_name == "tfhe-256-secure") {
        return TFHE_256_SECURE();
    } else if (preset_name == "bfv-128-simd") {
        return BFV_128_SIMD();
    } else if (preset_name == "ckks-128-ml") {
        return CKKS_128_ML();
    } else if (preset_name == "tfhe-128-voting") {
        return TFHE_128_VOTING();
    } else {
        throw std::invalid_argument("Unknown parameter preset: " + preset_name);
    }
}

std::vector<std::string> get_available_presets() {
    return {
        "tfhe-128-fast",
        "tfhe-128-balanced",
        "tfhe-256-secure",
        "bfv-128-simd",
        "ckks-128-ml",
        "tfhe-128-voting"
    };
}

} // namespace fhe_accelerate
