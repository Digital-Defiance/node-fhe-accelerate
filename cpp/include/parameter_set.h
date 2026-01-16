/**
 * @file parameter_set.h
 * @brief FHE Parameter Set definitions and presets
 * 
 * This file defines the ParameterSet structure and preset configurations
 * for various FHE schemes (TFHE, BFV, CKKS) at different security levels.
 * 
 * Requirements: 10.1, 10.2, 10.3
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <cmath>

namespace fhe_accelerate {

/**
 * Security levels supported by the library
 */
enum class SecurityLevel : uint32_t {
    Bits128 = 128,
    Bits192 = 192,
    Bits256 = 256
};

/**
 * FHE scheme types
 */
enum class FHEScheme {
    TFHE,   // Torus FHE - fast bootstrapping
    BFV,    // Brakerski/Fan-Vercauteren - integer arithmetic
    CKKS    // Cheon-Kim-Kim-Song - approximate arithmetic
};

/**
 * Parameter validation result
 */
struct ValidationResult {
    bool is_valid;
    std::vector<std::string> violations;
    
    ValidationResult() : is_valid(true) {}
    
    void add_violation(const std::string& msg) {
        is_valid = false;
        violations.push_back(msg);
    }
    
    std::string get_error_message() const {
        if (is_valid) return "";
        std::string result = "Parameter validation failed:\n";
        for (const auto& v : violations) {
            result += "  - " + v + "\n";
        }
        return result;
    }
};

/**
 * Complete FHE parameter set
 * 
 * Contains all parameters needed to configure an FHE context,
 * including polynomial parameters, TFHE-specific settings,
 * and derived values.
 */
struct ParameterSet {
    // ========== Core Polynomial Parameters ==========
    
    /** Polynomial degree N (must be power of 2: 1024, 2048, 4096, etc.) */
    uint32_t poly_degree;
    
    /** Coefficient modulus chain q (for RNS representation) */
    std::vector<uint64_t> moduli;
    
    // ========== TFHE-Specific Parameters ==========
    
    /** LWE dimension n */
    uint32_t lwe_dimension;
    
    /** LWE noise standard deviation σ */
    double lwe_noise_std;
    
    /** GLWE dimension k */
    uint32_t glwe_dimension;
    
    /** Gadget decomposition base log (log2 of base) */
    uint32_t decomp_base_log;
    
    /** Number of decomposition levels */
    uint32_t decomp_level;
    
    // ========== Security Configuration ==========
    
    /** Target security level in bits */
    SecurityLevel security;
    
    /** FHE scheme type */
    FHEScheme scheme;
    
    // ========== Derived Parameters ==========
    
    /** Plaintext modulus t */
    uint64_t plaintext_modulus;
    
    /** Estimated initial noise budget in bits */
    double noise_budget;
    
    /** Maximum supported multiplication depth */
    uint32_t max_mult_depth;
    
    // ========== Constructors ==========
    
    ParameterSet() 
        : poly_degree(0)
        , lwe_dimension(0)
        , lwe_noise_std(0.0)
        , glwe_dimension(0)
        , decomp_base_log(0)
        , decomp_level(0)
        , security(SecurityLevel::Bits128)
        , scheme(FHEScheme::TFHE)
        , plaintext_modulus(0)
        , noise_budget(0.0)
        , max_mult_depth(0)
    {}
    
    // ========== Utility Methods ==========
    
    /** Get total coefficient modulus (product of all moduli) */
    __uint128_t get_total_modulus() const {
        __uint128_t total = 1;
        for (uint64_t q : moduli) {
            total *= q;
        }
        return total;
    }
    
    /** Get log2 of total coefficient modulus */
    double get_log_modulus() const {
        double log_q = 0.0;
        for (uint64_t q : moduli) {
            log_q += std::log2(static_cast<double>(q));
        }
        return log_q;
    }
    
    /** Check if polynomial degree is valid (power of 2) */
    bool is_valid_degree() const {
        return poly_degree > 0 && (poly_degree & (poly_degree - 1)) == 0;
    }
    
    /** Get the number of coefficient moduli */
    size_t get_modulus_count() const {
        return moduli.size();
    }
    
    /** Calculate derived parameters from base parameters */
    void calculate_derived_parameters();
    
    /** Get a human-readable description of the parameter set */
    std::string to_string() const;
};

// ========== Preset Parameter Sets ==========

/**
 * TFHE-128-FAST: Fast bootstrapping with 128-bit security
 * 
 * Optimized for low-latency bootstrapping operations.
 * Suitable for applications requiring frequent bootstrapping.
 * 
 * Requirements: 10.1, 10.2
 */
ParameterSet TFHE_128_FAST();

/**
 * TFHE-128-BALANCED: Balanced performance/security for 128-bit
 * 
 * Good balance between bootstrapping speed and noise budget.
 * Recommended for general-purpose TFHE applications.
 * 
 * Requirements: 10.1, 10.2
 */
ParameterSet TFHE_128_BALANCED();

/**
 * TFHE-256-SECURE: Maximum security with 256-bit level
 * 
 * Highest security level, suitable for long-term security requirements.
 * Slower bootstrapping but maximum protection.
 * 
 * Requirements: 10.1, 10.2
 */
ParameterSet TFHE_256_SECURE();

/**
 * BFV-128-SIMD: BFV scheme with SIMD packing for 128-bit security
 * 
 * Optimized for batched integer arithmetic using SIMD slots.
 * Good for applications processing many values in parallel.
 * 
 * Requirements: 10.1, 10.2
 */
ParameterSet BFV_128_SIMD();

/**
 * CKKS-128-ML: CKKS scheme optimized for ML workloads
 * 
 * Approximate arithmetic suitable for machine learning inference.
 * Supports floating-point-like operations on encrypted data.
 * 
 * Requirements: 10.1, 10.2
 */
ParameterSet CKKS_128_ML();

/**
 * TFHE-128-VOTING: Optimized for voting applications
 * 
 * Fast, compact parameters suitable for encrypted voting.
 * Optimized for ballot encryption and homomorphic tallying.
 * 
 * Requirements: 10.1, 10.2
 */
ParameterSet TFHE_128_VOTING();

// ========== Parameter Set Factory ==========

/**
 * Create a parameter set from a preset name
 * 
 * @param preset_name Name of the preset (e.g., "tfhe-128-fast")
 * @return ParameterSet for the given preset
 * @throws std::invalid_argument if preset name is unknown
 */
ParameterSet create_parameter_set(const std::string& preset_name);

/**
 * Get list of available preset names
 */
std::vector<std::string> get_available_presets();

} // namespace fhe_accelerate
