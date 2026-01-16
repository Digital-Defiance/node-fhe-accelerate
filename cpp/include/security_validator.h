/**
 * @file security_validator.h
 * @brief Security constraint validation for FHE parameters
 * 
 * This file implements lattice security estimator checks and validates
 * that parameter sets meet the required security levels.
 * 
 * Requirements: 10.4, 10.5
 */

#pragma once

#include "parameter_set.h"
#include <string>
#include <vector>
#include <cmath>

namespace fhe_accelerate {

/**
 * Security validation error codes
 */
enum class SecurityViolation {
    NONE = 0,
    POLY_DEGREE_TOO_SMALL,
    POLY_DEGREE_NOT_POWER_OF_TWO,
    MODULUS_TOO_LARGE,
    MODULUS_TOO_SMALL,
    LWE_DIMENSION_TOO_SMALL,
    NOISE_STD_TOO_SMALL,
    NOISE_STD_TOO_LARGE,
    DECOMP_LEVEL_INVALID,
    PLAINTEXT_MODULUS_INVALID,
    SECURITY_LEVEL_NOT_MET,
    MODULUS_NOT_NTT_FRIENDLY,
    INVALID_GLWE_DIMENSION
};

/**
 * Detailed security violation information
 */
struct SecurityViolationInfo {
    SecurityViolation code;
    std::string message;
    std::string parameter_name;
    double actual_value;
    double required_value;
    
    SecurityViolationInfo(
        SecurityViolation c,
        const std::string& msg,
        const std::string& param = "",
        double actual = 0.0,
        double required = 0.0
    ) : code(c), message(msg), parameter_name(param),
        actual_value(actual), required_value(required) {}
};

/**
 * Security validation result with detailed information
 */
struct SecurityValidationResult {
    bool is_secure;
    std::vector<SecurityViolationInfo> violations;
    double estimated_security_bits;
    
    SecurityValidationResult() 
        : is_secure(true), estimated_security_bits(0.0) {}
    
    void add_violation(const SecurityViolationInfo& info) {
        is_secure = false;
        violations.push_back(info);
    }
    
    std::string get_error_message() const;
};

/**
 * Security constraint validator for FHE parameters
 * 
 * Implements lattice security estimator checks based on the
 * LWE/RLWE hardness assumptions. Uses conservative estimates
 * based on the lattice estimator tool.
 */
class SecurityValidator {
public:
    /**
     * Validate a complete parameter set
     * 
     * @param params The parameter set to validate
     * @return Validation result with any violations
     */
    static SecurityValidationResult validate(const ParameterSet& params);
    
    /**
     * Estimate the security level in bits for given parameters
     * 
     * Uses a simplified lattice security estimator based on:
     * - LWE dimension n
     * - Modulus q
     * - Noise standard deviation σ
     * 
     * @param n LWE dimension
     * @param log_q Log2 of the modulus
     * @param noise_std Noise standard deviation
     * @return Estimated security in bits
     */
    static double estimate_security_bits(
        uint32_t n,
        double log_q,
        double noise_std
    );
    
    /**
     * Estimate RLWE security for polynomial ring parameters
     * 
     * @param poly_degree Polynomial degree N
     * @param log_q Log2 of the modulus
     * @param noise_std Noise standard deviation
     * @return Estimated security in bits
     */
    static double estimate_rlwe_security_bits(
        uint32_t poly_degree,
        double log_q,
        double noise_std
    );
    
    /**
     * Check if a modulus is NTT-friendly for a given degree
     * 
     * A modulus q is NTT-friendly for degree N if:
     * - q is prime
     * - q ≡ 1 (mod 2N)
     * 
     * @param modulus The modulus to check
     * @param degree The polynomial degree
     * @return true if NTT-friendly
     */
    static bool is_ntt_friendly(uint64_t modulus, uint32_t degree);
    
    /**
     * Get minimum polynomial degree for a security level
     */
    static uint32_t get_min_poly_degree(SecurityLevel security);
    
    /**
     * Get maximum log2(q) for a given degree and security level
     */
    static double get_max_log_modulus(uint32_t degree, SecurityLevel security);
    
    /**
     * Get minimum LWE dimension for a security level
     */
    static uint32_t get_min_lwe_dimension(SecurityLevel security);

private:
    // Validation helper methods
    static void validate_poly_degree(
        const ParameterSet& params,
        SecurityValidationResult& result
    );
    
    static void validate_modulus(
        const ParameterSet& params,
        SecurityValidationResult& result
    );
    
    static void validate_lwe_parameters(
        const ParameterSet& params,
        SecurityValidationResult& result
    );
    
    static void validate_decomposition(
        const ParameterSet& params,
        SecurityValidationResult& result
    );
    
    static void validate_security_level(
        const ParameterSet& params,
        SecurityValidationResult& result
    );
    
    // Miller-Rabin primality test
    static bool is_prime(uint64_t n);
    static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod);
};

} // namespace fhe_accelerate
