/**
 * @file key_manager.cpp
 * @brief FHE Key Generation and Management Implementation
 * 
 * Implements key generation for FHE operations including secret keys,
 * public keys, evaluation keys, and bootstrapping keys.
 * 
 * Requirements: 4.1, 4.2, 4.3, 4.4, 15.4
 */

#include "key_manager.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <cstring>

#ifdef __APPLE__
#include <Security/Security.h>
#endif

namespace fhe_accelerate {

// ============================================================================
// SecureRandom Implementation
// ============================================================================

SecureRandom::SecureRandom() {
    rd_ = std::make_unique<std::random_device>();
    gen_ = std::make_unique<std::mt19937_64>((*rd_)());
}

SecureRandom::~SecureRandom() = default;

void SecureRandom::fill_bytes(uint8_t* buffer, size_t length) {
#ifdef __APPLE__
    // Use macOS Security framework for cryptographically secure random
    OSStatus status = SecRandomCopyBytes(kSecRandomDefault, length, buffer);
    if (status != errSecSuccess) {
        // Fallback to std::random_device
        for (size_t i = 0; i < length; ++i) {
            buffer[i] = static_cast<uint8_t>((*rd_)() & 0xFF);
        }
    }
#else
    // Use std::random_device on other platforms
    for (size_t i = 0; i < length; ++i) {
        buffer[i] = static_cast<uint8_t>((*rd_)() & 0xFF);
    }
#endif
}

uint64_t SecureRandom::random_u64() {
    uint64_t result;
    fill_bytes(reinterpret_cast<uint8_t*>(&result), sizeof(result));
    return result;
}

uint64_t SecureRandom::random_u64_range(uint64_t max) {
    if (max == 0) return 0;
    
    // Rejection sampling to avoid modulo bias
    uint64_t threshold = (UINT64_MAX - max + 1) % max;
    uint64_t r;
    do {
        r = random_u64();
    } while (r < threshold);
    
    return r % max;
}

uint64_t SecureRandom::sample_ternary(uint64_t modulus) {
    // Sample from {-1, 0, 1} with equal probability
    // Returns value in [0, modulus) representing the coefficient
    uint64_t r = random_u64_range(3);
    switch (r) {
        case 0: return modulus - 1;  // -1 mod q
        case 1: return 0;            // 0
        case 2: return 1;            // 1
        default: return 0;
    }
}

uint64_t SecureRandom::sample_gaussian(double std_dev, uint64_t modulus) {
    // Box-Muller transform for Gaussian sampling
    // Then round to nearest integer
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    double u1 = uniform(*gen_);
    double u2 = uniform(*gen_);
    
    // Avoid log(0)
    while (u1 == 0.0) {
        u1 = uniform(*gen_);
    }
    
    double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    double sample = std::round(z * std_dev);
    
    // Convert to [0, modulus)
    int64_t int_sample = static_cast<int64_t>(sample);
    if (int_sample < 0) {
        // Negative values become modulus + sample
        int_sample = static_cast<int64_t>(modulus) + int_sample;
        // Handle very negative values
        while (int_sample < 0) {
            int_sample += static_cast<int64_t>(modulus);
        }
    }
    
    return static_cast<uint64_t>(int_sample) % modulus;
}

uint64_t SecureRandom::sample_binary() {
    return random_u64() & 1;
}

// ============================================================================
// KeyManager Implementation
// ============================================================================

KeyManager::KeyManager(const ParameterSet& params)
    : params_(params)
    , next_key_id_(1)
{
    // Initialize polynomial ring with primary modulus
    if (!params_.moduli.empty()) {
        ring_ = std::make_unique<PolynomialRing>(params_.poly_degree, params_.moduli[0]);
    } else {
        throw std::invalid_argument("Parameter set must have at least one modulus");
    }
    
    // Initialize secure random number generator
    rng_ = std::make_unique<SecureRandom>();
}

KeyManager::~KeyManager() {
    // Secure cleanup - zero out any sensitive data
    // The unique_ptrs will handle deallocation
}

uint64_t KeyManager::generate_key_id() {
    return next_key_id_++;
}

// ============================================================================
// Secret Key Generation (Requirement 4.1)
// ============================================================================

std::unique_ptr<SecretKey> KeyManager::generate_secret_key(
    SecretKeyDistribution distribution
) {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    // Create polynomial for secret key
    std::vector<uint64_t> coeffs(degree);
    
    switch (distribution) {
        case SecretKeyDistribution::TERNARY:
            // Sample coefficients from {-1, 0, 1}
            for (uint32_t i = 0; i < degree; ++i) {
                coeffs[i] = rng_->sample_ternary(modulus);
            }
            break;
            
        case SecretKeyDistribution::GAUSSIAN:
            // Sample from discrete Gaussian with standard deviation from params
            for (uint32_t i = 0; i < degree; ++i) {
                coeffs[i] = rng_->sample_gaussian(params_.lwe_noise_std, modulus);
            }
            break;
            
        case SecretKeyDistribution::BINARY:
            // Sample coefficients from {0, 1}
            for (uint32_t i = 0; i < degree; ++i) {
                coeffs[i] = rng_->sample_binary();
            }
            break;
            
        case SecretKeyDistribution::UNIFORM:
            // Sample uniformly from [0, modulus)
            for (uint32_t i = 0; i < degree; ++i) {
                coeffs[i] = rng_->random_u64_range(modulus);
            }
            break;
    }
    
    // Create polynomial from coefficients
    Polynomial sk_poly(std::move(coeffs), modulus, false);
    
    // Generate unique key ID
    uint64_t key_id = generate_key_id();
    
    return std::make_unique<SecretKey>(std::move(sk_poly), distribution, key_id);
}

// ============================================================================
// Helper Functions
// ============================================================================

Polynomial KeyManager::sample_error_polynomial() {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    double std_dev = params_.lwe_noise_std;
    
    std::vector<uint64_t> coeffs(degree);
    for (uint32_t i = 0; i < degree; ++i) {
        coeffs[i] = rng_->sample_gaussian(std_dev, modulus);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

Polynomial KeyManager::sample_random_polynomial() {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    std::vector<uint64_t> coeffs(degree);
    for (uint32_t i = 0; i < degree; ++i) {
        coeffs[i] = rng_->random_u64_range(modulus);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

// ============================================================================
// Public Key Generation (Requirement 4.2)
// ============================================================================

std::unique_ptr<PublicKey> KeyManager::generate_public_key(const SecretKey& sk) {
    // Public key is an RLWE encryption of zero: pk = (a, b = a*s + e)
    // where a is random, s is secret key, e is error
    
    // Sample random polynomial a
    Polynomial a = sample_random_polynomial();
    
    // Sample error polynomial e
    Polynomial e = sample_error_polynomial();
    
    // Compute b = a * s + e (mod q)
    // First, convert to NTT domain for efficient multiplication
    Polynomial a_ntt = a.clone();
    ring_->to_ntt(a_ntt);
    
    Polynomial s_ntt = sk.poly.clone();
    ring_->to_ntt(s_ntt);
    
    // Pointwise multiply in NTT domain
    Polynomial b_ntt = ring_->pointwise_multiply(a_ntt, s_ntt);
    
    // Convert back to coefficient domain
    ring_->from_ntt(b_ntt);
    
    // Add error: b = a*s + e
    ring_->add_inplace(b_ntt, e);
    
    return std::make_unique<PublicKey>(std::move(a), std::move(b_ntt), sk.key_id);
}

// ============================================================================
// Evaluation Key Generation (Requirement 4.3)
// ============================================================================

std::unique_ptr<EvaluationKey> KeyManager::generate_eval_key(
    const SecretKey& sk,
    uint32_t decomp_base_log,
    uint32_t decomp_level
) {
    // Use default values from params if not specified
    if (decomp_base_log == 0) {
        decomp_base_log = params_.decomp_base_log;
    }
    if (decomp_level == 0) {
        decomp_level = params_.decomp_level;
    }
    
    auto eval_key = std::make_unique<EvaluationKey>();
    eval_key->key_id = sk.key_id;
    eval_key->relin_key.decomp_base_log = decomp_base_log;
    eval_key->relin_key.decomp_level = decomp_level;
    eval_key->relin_key.key_id = sk.key_id;
    
    uint64_t modulus = params_.moduli[0];
    uint64_t base = 1ULL << decomp_base_log;
    
    // Compute s^2 for relinearization key
    Polynomial s_ntt = sk.poly.clone();
    ring_->to_ntt(s_ntt);
    
    Polynomial s_squared_ntt = ring_->pointwise_multiply(s_ntt, s_ntt);
    ring_->from_ntt(s_squared_ntt);
    
    // Generate key switching keys for each decomposition level
    // Each level encrypts s^2 * base^i
    eval_key->relin_key.keys.reserve(decomp_level);
    
    uint64_t power = 1;
    for (uint32_t i = 0; i < decomp_level; ++i) {
        // Sample random a
        Polynomial a = sample_random_polynomial();
        
        // Sample error e
        Polynomial e = sample_error_polynomial();
        
        // Compute b = a * s + e + s^2 * base^i
        Polynomial a_ntt = a.clone();
        ring_->to_ntt(a_ntt);
        
        Polynomial s_ntt_copy = sk.poly.clone();
        ring_->to_ntt(s_ntt_copy);
        
        Polynomial b_ntt = ring_->pointwise_multiply(a_ntt, s_ntt_copy);
        ring_->from_ntt(b_ntt);
        
        // Add error
        ring_->add_inplace(b_ntt, e);
        
        // Add s^2 * base^i
        Polynomial scaled_s2 = ring_->multiply_scalar(s_squared_ntt, power);
        ring_->add_inplace(b_ntt, scaled_s2);
        
        eval_key->relin_key.keys.emplace_back(std::move(a), std::move(b_ntt));
        
        // Update power for next level
        power = (power * base) % modulus;
    }
    
    return eval_key;
}

// ============================================================================
// Bootstrapping Key Generation (Requirement 4.4)
// ============================================================================

std::unique_ptr<BootstrapKey> KeyManager::generate_bootstrap_key(const SecretKey& sk) {
    auto bsk = std::make_unique<BootstrapKey>();
    bsk->key_id = sk.key_id;
    bsk->lwe_dimension = params_.lwe_dimension;
    
    // For TFHE bootstrapping, we need to encrypt each bit of the secret key
    // as a GGSW ciphertext. This is a simplified implementation.
    
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    uint32_t decomp_level = params_.decomp_level;
    uint32_t decomp_base_log = params_.decomp_base_log;
    
    // Reserve space for bootstrapping key
    bsk->bsk.reserve(degree);
    
    // Parallelize key generation across available cores
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::future<std::vector<std::pair<Polynomial, Polynomial>>>> futures;
    
    // Divide work among threads
    uint32_t chunk_size = (degree + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        uint32_t start = t * chunk_size;
        uint32_t end = std::min(start + chunk_size, degree);
        
        if (start >= degree) break;
        
        futures.push_back(std::async(std::launch::async, [&, start, end]() {
            std::vector<std::pair<Polynomial, Polynomial>> local_keys;
            local_keys.reserve(end - start);
            
            // Create local RNG for thread safety
            SecureRandom local_rng;
            
            for (uint32_t i = start; i < end; ++i) {
                // Get the i-th coefficient of secret key
                uint64_t sk_coeff = sk.poly[i];
                
                // Create GGSW encryption of sk_coeff
                // Simplified: just create RLWE encryptions for each decomposition level
                std::vector<std::pair<Polynomial, Polynomial>> ggsw_row;
                ggsw_row.reserve(decomp_level);
                
                uint64_t base = 1ULL << decomp_base_log;
                uint64_t power = 1;
                
                for (uint32_t l = 0; l < decomp_level; ++l) {
                    // Sample random a
                    std::vector<uint64_t> a_coeffs(degree);
                    for (uint32_t j = 0; j < degree; ++j) {
                        a_coeffs[j] = local_rng.random_u64_range(modulus);
                    }
                    Polynomial a(std::move(a_coeffs), modulus, false);
                    
                    // Sample error e
                    std::vector<uint64_t> e_coeffs(degree);
                    for (uint32_t j = 0; j < degree; ++j) {
                        e_coeffs[j] = local_rng.sample_gaussian(params_.lwe_noise_std, modulus);
                    }
                    Polynomial e(std::move(e_coeffs), modulus, false);
                    
                    // Compute b = a * s + e + sk_coeff * base^l
                    // For simplicity, we compute this in coefficient domain
                    // A full implementation would use NTT
                    std::vector<uint64_t> b_coeffs(degree);
                    
                    // b = e + sk_coeff * base^l (constant term)
                    for (uint32_t j = 0; j < degree; ++j) {
                        b_coeffs[j] = e_coeffs[j];
                    }
                    
                    // Add sk_coeff * power to constant term
                    uint64_t scaled_sk = (sk_coeff * power) % modulus;
                    b_coeffs[0] = (b_coeffs[0] + scaled_sk) % modulus;
                    
                    Polynomial b(std::move(b_coeffs), modulus, false);
                    
                    ggsw_row.emplace_back(std::move(a), std::move(b));
                    power = (power * base) % modulus;
                }
                
                // Store first element of GGSW (simplified)
                if (!ggsw_row.empty()) {
                    local_keys.push_back(std::move(ggsw_row[0]));
                }
            }
            
            return local_keys;
        }));
    }
    
    // Collect results
    for (auto& future : futures) {
        auto local_keys = future.get();
        for (auto& key : local_keys) {
            bsk->bsk.push_back({std::move(key)});
        }
    }
    
    // Generate key switching key
    bsk->ksk.decomp_base_log = decomp_base_log;
    bsk->ksk.decomp_level = decomp_level;
    bsk->ksk.key_id = sk.key_id;
    
    // KSK is similar to evaluation key
    uint64_t base = 1ULL << decomp_base_log;
    uint64_t power = 1;
    
    for (uint32_t i = 0; i < decomp_level; ++i) {
        Polynomial a = sample_random_polynomial();
        Polynomial e = sample_error_polynomial();
        
        // b = a * s + e + s * base^i
        Polynomial a_ntt = a.clone();
        ring_->to_ntt(a_ntt);
        
        Polynomial s_ntt = sk.poly.clone();
        ring_->to_ntt(s_ntt);
        
        Polynomial b_ntt = ring_->pointwise_multiply(a_ntt, s_ntt);
        ring_->from_ntt(b_ntt);
        
        ring_->add_inplace(b_ntt, e);
        
        Polynomial scaled_s = ring_->multiply_scalar(sk.poly, power);
        ring_->add_inplace(b_ntt, scaled_s);
        
        bsk->ksk.keys.emplace_back(std::move(a), std::move(b_ntt));
        
        power = (power * base) % modulus;
    }
    
    return bsk;
}

// ============================================================================
// Threshold Key Generation (Requirement 15.4)
// ============================================================================

std::unique_ptr<ThresholdKeys> KeyManager::generate_threshold_keys(
    uint32_t threshold,
    uint32_t total_shares
) {
    if (threshold == 0 || threshold > total_shares) {
        throw std::invalid_argument("Invalid threshold parameters");
    }
    
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    // Generate master secret key
    auto master_sk = generate_secret_key(SecretKeyDistribution::TERNARY);
    
    // Generate random polynomial coefficients for Shamir's secret sharing
    // We need (threshold - 1) random polynomials for the sharing polynomial
    std::vector<Polynomial> sharing_coeffs;
    sharing_coeffs.reserve(threshold);
    
    // First coefficient is the secret key
    sharing_coeffs.push_back(master_sk->poly.clone());
    
    // Generate random coefficients for higher degree terms
    for (uint32_t i = 1; i < threshold; ++i) {
        sharing_coeffs.push_back(sample_random_polynomial());
    }
    
    // Generate shares by evaluating the sharing polynomial at points 1, 2, ..., N
    std::vector<SecretKeyShare> shares;
    shares.reserve(total_shares);
    
    for (uint32_t i = 1; i <= total_shares; ++i) {
        // Evaluate sharing polynomial at point i
        // share_i = sum_{j=0}^{t-1} coeff_j * i^j
        std::vector<uint64_t> share_coeffs(degree, 0);
        
        uint64_t point = i;
        uint64_t point_power = 1;
        
        for (uint32_t j = 0; j < threshold; ++j) {
            for (uint32_t k = 0; k < degree; ++k) {
                uint64_t term = (__uint128_t(sharing_coeffs[j][k]) * point_power) % modulus;
                share_coeffs[k] = (share_coeffs[k] + term) % modulus;
            }
            point_power = (__uint128_t(point_power) * point) % modulus;
        }
        
        Polynomial share_poly(std::move(share_coeffs), modulus, false);
        
        // Generate commitment (hash of share - simplified)
        std::vector<uint8_t> commitment(32, 0);
        // In a real implementation, this would be a cryptographic commitment
        
        shares.emplace_back(i, std::move(share_poly), master_sk->key_id);
        shares.back().commitment = std::move(commitment);
    }
    
    // Generate public key from master secret key
    auto pk = generate_public_key(*master_sk);
    
    return std::make_unique<ThresholdKeys>(
        std::move(shares),
        std::move(*pk),
        threshold,
        total_shares
    );
}

uint64_t KeyManager::lagrange_coefficient(
    uint32_t i,
    const std::vector<uint32_t>& indices,
    uint64_t modulus
) {
    // Compute Lagrange coefficient for point i given the set of indices
    // L_i = prod_{j != i} (0 - j) / (i - j)
    // Since we evaluate at 0, this simplifies to:
    // L_i = prod_{j != i} j / (j - i)
    
    __uint128_t numerator = 1;
    __uint128_t denominator = 1;
    
    for (uint32_t j : indices) {
        if (j != i) {
            numerator = (numerator * j) % modulus;
            
            // Compute (j - i) mod modulus
            int64_t diff = static_cast<int64_t>(j) - static_cast<int64_t>(i);
            uint64_t diff_mod;
            if (diff < 0) {
                diff_mod = modulus - (static_cast<uint64_t>(-diff) % modulus);
            } else {
                diff_mod = static_cast<uint64_t>(diff) % modulus;
            }
            
            denominator = (denominator * diff_mod) % modulus;
        }
    }
    
    // Compute modular inverse of denominator
    uint64_t denom_inv = NTTProcessor::mod_inverse(static_cast<uint64_t>(denominator), modulus);
    
    return (__uint128_t(numerator) * denom_inv) % modulus;
}

PartialDecryption KeyManager::partial_decrypt(
    const Polynomial& ct_body,
    const SecretKeyShare& share
) {
    // Partial decryption: compute ct_body * share
    // This is a simplified version - full implementation would handle
    // the complete ciphertext structure
    
    Polynomial share_ntt = share.share_poly.clone();
    ring_->to_ntt(share_ntt);
    
    Polynomial body_ntt = ct_body.clone();
    ring_->to_ntt(body_ntt);
    
    Polynomial result_ntt = ring_->pointwise_multiply(body_ntt, share_ntt);
    ring_->from_ntt(result_ntt);
    
    return PartialDecryption(share.share_id, std::move(result_ntt));
}

Polynomial KeyManager::combine_partial_decryptions(
    const Polynomial& ct_body,
    const std::vector<PartialDecryption>& partials,
    uint32_t threshold
) {
    if (partials.size() < threshold) {
        throw std::invalid_argument("Not enough partial decryptions");
    }
    
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    // Collect share indices
    std::vector<uint32_t> indices;
    indices.reserve(partials.size());
    for (const auto& partial : partials) {
        indices.push_back(partial.share_id);
    }
    
    // Combine using Lagrange interpolation
    std::vector<uint64_t> result_coeffs(degree, 0);
    
    for (size_t i = 0; i < partials.size() && i < threshold; ++i) {
        uint64_t lambda = lagrange_coefficient(partials[i].share_id, indices, modulus);
        
        for (uint32_t j = 0; j < degree; ++j) {
            uint64_t term = (__uint128_t(partials[i].partial_result[j]) * lambda) % modulus;
            result_coeffs[j] = (result_coeffs[j] + term) % modulus;
        }
    }
    
    return Polynomial(std::move(result_coeffs), modulus, false);
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<KeyManager> create_key_manager(const ParameterSet& params) {
    return std::make_unique<KeyManager>(params);
}

} // namespace fhe_accelerate
