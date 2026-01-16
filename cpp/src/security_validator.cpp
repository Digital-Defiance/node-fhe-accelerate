/**
 * @file security_validator.cpp
 * @brief Implementation of security constraint validation
 * 
 * Implements lattice security estimator checks and parameter validation
 * based on conservative security estimates from the lattice estimator.
 * 
 * Requirements: 10.4, 10.5
 */

#include "security_validator.h"
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>

namespace fhe_accelerate {

// ========== Security Bounds Tables ==========
// Based on conservative estimates from the lattice estimator
// https://github.com/malb/lattice-estimator

// Minimum polynomial degrees for each security level
// These are conservative lower bounds
constexpr uint32_t MIN_DEGREE_128 = 1024;
constexpr uint32_t MIN_DEGREE_192 = 2048;
constexpr uint32_t MIN_DEGREE_256 = 4096;

// Maximum log2(q) for each (degree, security) pair
// Format: max_log_q[degree_index][security_index]
// degree_index: 0=1024, 1=2048, 2=4096, 3=8192, 4=16384, 5=32768
// security_index: 0=128-bit, 1=192-bit, 2=256-bit
const double MAX_LOG_Q[6][3] = {
    // N=1024:  128-bit, 192-bit, 256-bit
    {27.0, 19.0, 14.0},
    // N=2048
    {54.0, 37.0, 29.0},
    // N=4096
    {109.0, 75.0, 58.0},
    // N=8192
    {218.0, 152.0, 118.0},
    // N=16384
    {438.0, 305.0, 237.0},
    // N=32768
    {881.0, 611.0, 476.0}
};

// Minimum LWE dimensions for each security level
constexpr uint32_t MIN_LWE_DIM_128 = 630;
constexpr uint32_t MIN_LWE_DIM_192 = 880;
constexpr uint32_t MIN_LWE_DIM_256 = 1024;

// ========== SecurityValidationResult Implementation ==========

std::string SecurityValidationResult::get_error_message() const {
    if (is_secure) {
        return "";
    }
    
    std::ostringstream oss;
    oss << "Security validation failed with " << violations.size() << " violation(s):\n";
    
    for (const auto& v : violations) {
        oss << "  - " << v.message;
        if (!v.parameter_name.empty()) {
            oss << " (parameter: " << v.parameter_name;
            if (v.actual_value != 0.0 || v.required_value != 0.0) {
                oss << ", actual: " << v.actual_value 
                    << ", required: " << v.required_value;
            }
            oss << ")";
        }
        oss << "\n";
    }
    
    oss << "Estimated security: " << estimated_security_bits << " bits\n";
    
    return oss.str();
}

// ========== SecurityValidator Implementation ==========

SecurityValidationResult SecurityValidator::validate(const ParameterSet& params) {
    SecurityValidationResult result;
    
    // Run all validation checks
    validate_poly_degree(params, result);
    validate_modulus(params, result);
    validate_lwe_parameters(params, result);
    validate_decomposition(params, result);
    validate_security_level(params, result);
    
    return result;
}

void SecurityValidator::validate_poly_degree(
    const ParameterSet& params,
    SecurityValidationResult& result
) {
    // Check if degree is a power of 2
    if (params.poly_degree == 0 || 
        (params.poly_degree & (params.poly_degree - 1)) != 0) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::POLY_DEGREE_NOT_POWER_OF_TWO,
            "Polynomial degree must be a power of 2",
            "poly_degree",
            static_cast<double>(params.poly_degree),
            0.0
        ));
        return;
    }
    
    // Check minimum degree for security level
    uint32_t min_degree = get_min_poly_degree(params.security);
    if (params.poly_degree < min_degree) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::POLY_DEGREE_TOO_SMALL,
            "Polynomial degree too small for target security level",
            "poly_degree",
            static_cast<double>(params.poly_degree),
            static_cast<double>(min_degree)
        ));
    }
}

void SecurityValidator::validate_modulus(
    const ParameterSet& params,
    SecurityValidationResult& result
) {
    if (params.moduli.empty()) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::MODULUS_TOO_SMALL,
            "At least one modulus must be specified",
            "moduli"
        ));
        return;
    }
    
    // Calculate total log2(q)
    double log_q = 0.0;
    for (uint64_t q : params.moduli) {
        if (q < 2) {
            result.add_violation(SecurityViolationInfo(
                SecurityViolation::MODULUS_TOO_SMALL,
                "Modulus must be at least 2",
                "moduli",
                static_cast<double>(q),
                2.0
            ));
            return;
        }
        log_q += std::log2(static_cast<double>(q));
    }
    
    // Check maximum modulus for security
    double max_log_q = get_max_log_modulus(params.poly_degree, params.security);
    if (log_q > max_log_q) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::MODULUS_TOO_LARGE,
            "Total modulus too large for polynomial degree and security level",
            "moduli (log2)",
            log_q,
            max_log_q
        ));
    }
    
    // Check NTT-friendliness for each modulus
    for (uint64_t q : params.moduli) {
        if (!is_ntt_friendly(q, params.poly_degree)) {
            // This is a warning, not a hard failure
            // Some applications may use non-NTT-friendly moduli
            // result.add_violation(SecurityViolationInfo(
            //     SecurityViolation::MODULUS_NOT_NTT_FRIENDLY,
            //     "Modulus is not NTT-friendly for the given degree",
            //     "moduli",
            //     static_cast<double>(q),
            //     0.0
            // ));
        }
    }
}

void SecurityValidator::validate_lwe_parameters(
    const ParameterSet& params,
    SecurityValidationResult& result
) {
    // Only validate LWE parameters for TFHE scheme
    if (params.scheme != FHEScheme::TFHE) {
        return;
    }
    
    // Check minimum LWE dimension
    uint32_t min_dim = get_min_lwe_dimension(params.security);
    if (params.lwe_dimension < min_dim) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::LWE_DIMENSION_TOO_SMALL,
            "LWE dimension too small for target security level",
            "lwe_dimension",
            static_cast<double>(params.lwe_dimension),
            static_cast<double>(min_dim)
        ));
    }
    
    // Check noise standard deviation bounds
    // Too small noise makes the scheme insecure
    // Too large noise causes decryption failures
    if (params.lwe_noise_std <= 0) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::NOISE_STD_TOO_SMALL,
            "Noise standard deviation must be positive",
            "lwe_noise_std",
            params.lwe_noise_std,
            1e-15
        ));
    }
    
    // Upper bound on noise (relative to modulus)
    // Noise should be small enough to allow correct decryption
    double log_q = 0.0;
    for (uint64_t q : params.moduli) {
        log_q += std::log2(static_cast<double>(q));
    }
    double max_noise_std = std::pow(2.0, log_q / 4.0);  // Conservative bound
    
    if (params.lwe_noise_std > max_noise_std) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::NOISE_STD_TOO_LARGE,
            "Noise standard deviation too large for correct decryption",
            "lwe_noise_std",
            params.lwe_noise_std,
            max_noise_std
        ));
    }
}

void SecurityValidator::validate_decomposition(
    const ParameterSet& params,
    SecurityValidationResult& result
) {
    // Validate decomposition parameters for bootstrapping
    if (params.scheme != FHEScheme::TFHE) {
        return;
    }
    
    // Decomposition base log should be reasonable
    if (params.decomp_base_log == 0 || params.decomp_base_log > 64) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::DECOMP_LEVEL_INVALID,
            "Decomposition base log must be between 1 and 64",
            "decomp_base_log",
            static_cast<double>(params.decomp_base_log),
            23.0  // Typical value
        ));
    }
    
    // Decomposition level should be at least 1
    if (params.decomp_level == 0) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::DECOMP_LEVEL_INVALID,
            "Decomposition level must be at least 1",
            "decomp_level",
            static_cast<double>(params.decomp_level),
            1.0
        ));
    }
    
    // GLWE dimension should be at least 1
    if (params.glwe_dimension == 0) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::INVALID_GLWE_DIMENSION,
            "GLWE dimension must be at least 1",
            "glwe_dimension",
            static_cast<double>(params.glwe_dimension),
            1.0
        ));
    }
}

void SecurityValidator::validate_security_level(
    const ParameterSet& params,
    SecurityValidationResult& result
) {
    // Estimate actual security and compare to target
    double log_q = 0.0;
    for (uint64_t q : params.moduli) {
        log_q += std::log2(static_cast<double>(q));
    }
    
    double estimated_bits;
    if (params.scheme == FHEScheme::TFHE) {
        // Use LWE security estimate
        estimated_bits = estimate_security_bits(
            params.lwe_dimension,
            log_q,
            params.lwe_noise_std
        );
    } else {
        // Use RLWE security estimate
        estimated_bits = estimate_rlwe_security_bits(
            params.poly_degree,
            log_q,
            params.lwe_noise_std > 0 ? params.lwe_noise_std : 3.2
        );
    }
    
    result.estimated_security_bits = estimated_bits;
    
    double target_bits = static_cast<double>(params.security);
    if (estimated_bits < target_bits) {
        result.add_violation(SecurityViolationInfo(
            SecurityViolation::SECURITY_LEVEL_NOT_MET,
            "Estimated security does not meet target level",
            "security",
            estimated_bits,
            target_bits
        ));
    }
}

double SecurityValidator::estimate_security_bits(
    uint32_t n,
    double log_q,
    double noise_std
) {
    // Simplified security estimate based on the BKZ algorithm
    // This is a conservative estimate using the core-SVP methodology
    //
    // Security ≈ n * log2(q/σ) / log2(q) * constant
    //
    // More accurate estimates require the full lattice estimator,
    // but this gives a reasonable approximation.
    
    if (n == 0 || log_q <= 0 || noise_std <= 0) {
        return 0.0;
    }
    
    // Estimate based on BKZ block size required to solve LWE
    // Using the formula from [Albrecht et al., 2015]
    double log_sigma = std::log2(noise_std);
    
    // Hermite factor required
    double delta = std::pow(2.0, (log_q - log_sigma) / (4.0 * n));
    
    // Convert Hermite factor to BKZ block size
    // Using the approximation: delta ≈ (πb)^(1/b) * b / (2πe)
    // Simplified: b ≈ 2 * log2(1/delta) / log2(log2(1/delta))
    
    if (delta <= 1.0) {
        return 256.0;  // Very secure
    }
    
    double log_delta = std::log2(delta);
    if (log_delta <= 0) {
        return 256.0;
    }
    
    // Estimate BKZ block size
    double b = 2.0 * n * log_delta;
    
    // Convert block size to security bits
    // Using the core-SVP model: cost ≈ 2^(0.292 * b)
    double security = 0.292 * b;
    
    // Apply a safety margin
    security *= 0.9;
    
    return std::max(0.0, std::min(256.0, security));
}

double SecurityValidator::estimate_rlwe_security_bits(
    uint32_t poly_degree,
    double log_q,
    double noise_std
) {
    // RLWE security is related to LWE security
    // For ring dimension N, RLWE is at least as hard as LWE with dimension N
    // (under standard assumptions)
    
    return estimate_security_bits(poly_degree, log_q, noise_std);
}

bool SecurityValidator::is_ntt_friendly(uint64_t modulus, uint32_t degree) {
    // Check if modulus is prime
    if (!is_prime(modulus)) {
        return false;
    }
    
    // Check if q ≡ 1 (mod 2N)
    uint64_t two_n = 2ULL * degree;
    return (modulus % two_n) == 1;
}

uint32_t SecurityValidator::get_min_poly_degree(SecurityLevel security) {
    switch (security) {
        case SecurityLevel::Bits128:
            return MIN_DEGREE_128;
        case SecurityLevel::Bits192:
            return MIN_DEGREE_192;
        case SecurityLevel::Bits256:
            return MIN_DEGREE_256;
        default:
            return MIN_DEGREE_128;
    }
}

double SecurityValidator::get_max_log_modulus(uint32_t degree, SecurityLevel security) {
    // Find the degree index
    int degree_idx = -1;
    uint32_t d = 1024;
    for (int i = 0; i < 6; ++i) {
        if (degree == d) {
            degree_idx = i;
            break;
        }
        d *= 2;
    }
    
    if (degree_idx < 0) {
        // Interpolate for non-standard degrees
        double log_degree = std::log2(static_cast<double>(degree));
        double base_idx = log_degree - 10.0;  // log2(1024) = 10
        
        if (base_idx < 0) base_idx = 0;
        if (base_idx > 5) base_idx = 5;
        
        int lower_idx = static_cast<int>(base_idx);
        int upper_idx = std::min(lower_idx + 1, 5);
        double frac = base_idx - lower_idx;
        
        int sec_idx = (security == SecurityLevel::Bits128) ? 0 :
                      (security == SecurityLevel::Bits192) ? 1 : 2;
        
        return MAX_LOG_Q[lower_idx][sec_idx] * (1.0 - frac) +
               MAX_LOG_Q[upper_idx][sec_idx] * frac;
    }
    
    int sec_idx = (security == SecurityLevel::Bits128) ? 0 :
                  (security == SecurityLevel::Bits192) ? 1 : 2;
    
    return MAX_LOG_Q[degree_idx][sec_idx];
}

uint32_t SecurityValidator::get_min_lwe_dimension(SecurityLevel security) {
    switch (security) {
        case SecurityLevel::Bits128:
            return MIN_LWE_DIM_128;
        case SecurityLevel::Bits192:
            return MIN_LWE_DIM_192;
        case SecurityLevel::Bits256:
            return MIN_LWE_DIM_256;
        default:
            return MIN_LWE_DIM_128;
    }
}

// Miller-Rabin primality test
bool SecurityValidator::is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    
    // Write n-1 as 2^r * d
    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }
    
    // Witnesses to test (sufficient for 64-bit numbers)
    const uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    
    for (uint64_t a : witnesses) {
        if (a >= n) continue;
        
        uint64_t x = mod_pow(a, d, n);
        
        if (x == 1 || x == n - 1) continue;
        
        bool composite = true;
        for (int i = 0; i < r - 1; ++i) {
            x = mod_pow(x, 2, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        
        if (composite) return false;
    }
    
    return true;
}

uint64_t SecurityValidator::mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    __uint128_t result = 1;
    __uint128_t b = base % mod;
    
    while (exp > 0) {
        if (exp & 1) {
            result = (result * b) % mod;
        }
        b = (b * b) % mod;
        exp >>= 1;
    }
    
    return static_cast<uint64_t>(result);
}

} // namespace fhe_accelerate
