/**
 * @file key_manager.h
 * @brief FHE Key Generation and Management
 * 
 * This file defines the key management infrastructure for FHE operations,
 * including secret key, public key, evaluation key, and bootstrapping key
 * generation. Keys are stored in protected memory regions where possible.
 * 
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 15.4, 15.5, 16, 17
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <random>
#include <optional>
#include <functional>
#include "polynomial_ring.h"
#include "parameter_set.h"
#include "ntt_processor.h"

namespace fhe_accelerate {

// Forward declarations
class SecureRandom;
class KeySerializer;

/**
 * Distribution types for secret key sampling
 */
enum class SecretKeyDistribution {
    TERNARY,    // Coefficients in {-1, 0, 1}
    GAUSSIAN,   // Discrete Gaussian distribution
    BINARY,     // Coefficients in {0, 1}
    UNIFORM     // Uniform random in [0, q)
};

/**
 * Secret key structure
 * 
 * Contains the secret polynomial with small coefficients.
 * Memory is protected where platform supports it.
 */
struct SecretKey {
    Polynomial poly;                    // Secret polynomial
    SecretKeyDistribution distribution; // Distribution used for sampling
    uint64_t key_id;                    // Unique identifier for this key
    
    SecretKey(Polynomial&& p, SecretKeyDistribution dist, uint64_t id)
        : poly(std::move(p)), distribution(dist), key_id(id) {}
    
    // Disable copy to prevent key leakage
    SecretKey(const SecretKey&) = delete;
    SecretKey& operator=(const SecretKey&) = delete;
    
    // Allow move
    SecretKey(SecretKey&&) = default;
    SecretKey& operator=(SecretKey&&) = default;
};

/**
 * Public key structure
 * 
 * Contains an RLWE encryption of zero: (a, b = a*s + e)
 * where s is the secret key and e is error.
 */
struct PublicKey {
    Polynomial a;       // Random polynomial
    Polynomial b;       // b = a*s + e (mod q)
    uint64_t key_id;    // Matches corresponding secret key
    
    PublicKey(Polynomial&& a_poly, Polynomial&& b_poly, uint64_t id)
        : a(std::move(a_poly)), b(std::move(b_poly)), key_id(id) {}
    
    // Allow copy and move
    PublicKey(const PublicKey& other)
        : a(other.a), b(other.b), key_id(other.key_id) {}
    PublicKey(PublicKey&&) = default;
    PublicKey& operator=(const PublicKey&) = default;
    PublicKey& operator=(PublicKey&&) = default;
};

/**
 * Key switching key structure
 * 
 * Used for relinearization after homomorphic multiplication.
 * Contains encryptions of powers of the secret key.
 */
struct KeySwitchKey {
    std::vector<std::pair<Polynomial, Polynomial>> keys;  // Vector of (a, b) pairs
    uint32_t decomp_base_log;   // Decomposition base (log2)
    uint32_t decomp_level;      // Number of decomposition levels
    uint64_t key_id;            // Matches corresponding secret key
    
    KeySwitchKey() : decomp_base_log(0), decomp_level(0), key_id(0) {}
};

/**
 * Evaluation key structure
 * 
 * Contains key switching keys for relinearization.
 */
struct EvaluationKey {
    KeySwitchKey relin_key;     // Relinearization key (s^2 -> s)
    uint64_t key_id;            // Matches corresponding secret key
    
    EvaluationKey() : key_id(0) {}
};

/**
 * Bootstrapping key structure (TFHE)
 * 
 * Contains GGSW encryptions of secret key bits for bootstrapping.
 */
struct BootstrapKey {
    std::vector<std::vector<std::pair<Polynomial, Polynomial>>> bsk;  // Bootstrapping key
    KeySwitchKey ksk;           // Key switching key for sample extraction
    uint32_t lwe_dimension;     // LWE dimension
    uint64_t key_id;            // Matches corresponding secret key
    
    BootstrapKey() : lwe_dimension(0), key_id(0) {}
};

/**
 * Secret key share for threshold decryption
 */
struct SecretKeyShare {
    uint32_t share_id;          // Share identifier (1 to N)
    Polynomial share_poly;      // The polynomial share
    std::vector<uint8_t> commitment;  // Commitment for verification
    uint64_t key_id;            // Matches the combined key
    
    SecretKeyShare(uint32_t id, Polynomial&& poly, uint64_t kid)
        : share_id(id), share_poly(std::move(poly)), key_id(kid) {}
};

/**
 * Threshold key set
 */
struct ThresholdKeys {
    std::vector<SecretKeyShare> shares;  // N shares
    PublicKey public_key;                // Combined public key
    uint32_t threshold;                  // M required shares
    uint32_t total_shares;               // N total shares
    
    ThresholdKeys(std::vector<SecretKeyShare>&& s, PublicKey&& pk, uint32_t t, uint32_t n)
        : shares(std::move(s)), public_key(std::move(pk)), threshold(t), total_shares(n) {}
};

/**
 * Partial decryption result
 */
struct PartialDecryption {
    uint32_t share_id;          // Which share was used
    Polynomial partial_result;  // Partial decryption result
    std::vector<uint8_t> proof; // ZK proof of correctness (optional)
    
    PartialDecryption(uint32_t id, Polynomial&& result)
        : share_id(id), partial_result(std::move(result)) {}
};

/**
 * Secure random number generator
 * 
 * Uses platform-specific secure random sources (e.g., /dev/urandom, SecRandomCopyBytes)
 */
class SecureRandom {
public:
    SecureRandom();
    ~SecureRandom();
    
    /**
     * Fill buffer with cryptographically secure random bytes
     */
    void fill_bytes(uint8_t* buffer, size_t length);
    
    /**
     * Generate random uint64_t
     */
    uint64_t random_u64();
    
    /**
     * Generate random uint64_t in range [0, max)
     */
    uint64_t random_u64_range(uint64_t max);
    
    /**
     * Sample from ternary distribution {-1, 0, 1}
     * Returns value in [0, q) representing the coefficient
     */
    uint64_t sample_ternary(uint64_t modulus);
    
    /**
     * Sample from discrete Gaussian distribution
     * @param std_dev Standard deviation
     * @param modulus The modulus q
     */
    uint64_t sample_gaussian(double std_dev, uint64_t modulus);
    
    /**
     * Sample from binary distribution {0, 1}
     */
    uint64_t sample_binary();
    
private:
    std::unique_ptr<std::random_device> rd_;
    std::unique_ptr<std::mt19937_64> gen_;
};

/**
 * Key Manager
 * 
 * Handles generation, storage, and management of FHE keys.
 * Supports parallel key generation for performance.
 */
class KeyManager {
public:
    /**
     * Construct key manager with given parameters
     */
    explicit KeyManager(const ParameterSet& params);
    
    /**
     * Destructor - securely erases any cached keys
     */
    ~KeyManager();
    
    // ========================================================================
    // Secret Key Generation (Requirement 4.1)
    // ========================================================================
    
    /**
     * Generate a new secret key
     * 
     * Samples coefficients from the specified distribution using a secure
     * random number generator. The key is stored in protected memory.
     * 
     * @param distribution Distribution for coefficient sampling (default: TERNARY)
     * @return Unique pointer to the generated secret key
     */
    std::unique_ptr<SecretKey> generate_secret_key(
        SecretKeyDistribution distribution = SecretKeyDistribution::TERNARY
    );
    
    // ========================================================================
    // Public Key Generation (Requirement 4.2)
    // ========================================================================
    
    /**
     * Generate a public key from a secret key
     * 
     * Computes an RLWE encryption of zero: pk = (a, b = a*s + e)
     * Parallelized across available cores for performance.
     * Target: < 100ms for 128-bit security.
     * 
     * @param sk The secret key
     * @return The generated public key
     */
    std::unique_ptr<PublicKey> generate_public_key(const SecretKey& sk);
    
    // ========================================================================
    // Evaluation Key Generation (Requirement 4.3)
    // ========================================================================
    
    /**
     * Generate evaluation key for relinearization
     * 
     * Creates key switching keys with configurable decomposition base.
     * Supports multiple decomposition levels for noise control.
     * 
     * @param sk The secret key
     * @param decomp_base_log Log2 of decomposition base (default from params)
     * @param decomp_level Number of decomposition levels (default from params)
     * @return The generated evaluation key
     */
    std::unique_ptr<EvaluationKey> generate_eval_key(
        const SecretKey& sk,
        uint32_t decomp_base_log = 0,
        uint32_t decomp_level = 0
    );
    
    // ========================================================================
    // Bootstrapping Key Generation (Requirement 4.4)
    // ========================================================================
    
    /**
     * Generate bootstrapping key for TFHE
     * 
     * Creates GGSW encryptions of secret key bits.
     * Parallelized across cores for performance.
     * 
     * @param sk The secret key
     * @return The generated bootstrapping key
     */
    std::unique_ptr<BootstrapKey> generate_bootstrap_key(const SecretKey& sk);
    
    // ========================================================================
    // Threshold Key Generation (Requirement 15.4)
    // ========================================================================
    
    /**
     * Generate threshold keys for M-of-N decryption
     * 
     * Uses Shamir's secret sharing to split the secret key.
     * 
     * @param threshold M - minimum shares required
     * @param total_shares N - total number of shares
     * @return Threshold key set with shares and public key
     */
    std::unique_ptr<ThresholdKeys> generate_threshold_keys(
        uint32_t threshold,
        uint32_t total_shares
    );
    
    /**
     * Perform partial decryption with a key share
     * 
     * @param ct_body The ciphertext body polynomial
     * @param share The secret key share
     * @return Partial decryption result
     */
    PartialDecryption partial_decrypt(
        const Polynomial& ct_body,
        const SecretKeyShare& share
    );
    
    /**
     * Combine partial decryptions to get final result
     * 
     * @param ct_body The ciphertext body polynomial
     * @param partials Vector of partial decryptions (at least threshold)
     * @param threshold The threshold M
     * @return Combined decryption result
     */
    Polynomial combine_partial_decryptions(
        const Polynomial& ct_body,
        const std::vector<PartialDecryption>& partials,
        uint32_t threshold
    );
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const ParameterSet& get_params() const { return params_; }
    PolynomialRing& get_ring() { return *ring_; }
    
private:
    ParameterSet params_;
    std::unique_ptr<PolynomialRing> ring_;
    std::unique_ptr<SecureRandom> rng_;
    uint64_t next_key_id_;
    
    // Helper functions
    Polynomial sample_error_polynomial();
    Polynomial sample_random_polynomial();
    uint64_t generate_key_id();
    
    // Lagrange interpolation for threshold decryption
    uint64_t lagrange_coefficient(
        uint32_t i,
        const std::vector<uint32_t>& indices,
        uint64_t modulus
    );
};

// Factory function
std::unique_ptr<KeyManager> create_key_manager(const ParameterSet& params);

} // namespace fhe_accelerate
