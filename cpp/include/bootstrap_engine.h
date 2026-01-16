/**
 * @file bootstrap_engine.h
 * @brief TFHE Bootstrapping Engine
 * 
 * This file defines the bootstrapping infrastructure for TFHE operations,
 * including bootstrapping key generation, blind rotate, sample extract,
 * key switching, and programmable bootstrapping.
 * 
 * Requirements: 4.4, 8.1, 8.3, 8.5, 8.6
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include "polynomial_ring.h"
#include "key_manager.h"
#include "parameter_set.h"
#include "ntt_processor.h"
#include "encryption.h"

namespace fhe_accelerate {

// Forward declarations
class HardwareDispatcher;

/**
 * LWE Ciphertext structure for TFHE
 * 
 * An LWE ciphertext consists of:
 * - a: mask vector of n elements
 * - b: body (scalar)
 * 
 * Decryption: m = round((b - <a, s>) / delta)
 */
struct LWECiphertext {
    std::vector<uint64_t> a;    // Mask vector (n elements)
    uint64_t b;                 // Body
    uint64_t modulus;           // Ciphertext modulus
    uint64_t key_id;            // ID of the key used for encryption
    
    LWECiphertext() : b(0), modulus(0), key_id(0) {}
    
    LWECiphertext(std::vector<uint64_t>&& mask, uint64_t body, uint64_t mod, uint64_t kid)
        : a(std::move(mask)), b(body), modulus(mod), key_id(kid) {}
    
    LWECiphertext(size_t dimension, uint64_t mod, uint64_t kid)
        : a(dimension, 0), b(0), modulus(mod), key_id(kid) {}
    
    // Copy and move constructors
    LWECiphertext(const LWECiphertext&) = default;
    LWECiphertext(LWECiphertext&&) = default;
    LWECiphertext& operator=(const LWECiphertext&) = default;
    LWECiphertext& operator=(LWECiphertext&&) = default;
    
    size_t dimension() const { return a.size(); }
};

/**
 * GLWE Ciphertext structure for TFHE
 * 
 * A GLWE ciphertext consists of:
 * - mask: k polynomials
 * - body: 1 polynomial
 * 
 * Decryption: m = body - sum(mask[i] * s[i])
 */
struct GLWECiphertext {
    std::vector<Polynomial> mask;   // k polynomials
    Polynomial body;                // 1 polynomial
    uint64_t key_id;                // ID of the key used for encryption
    bool is_ntt;                    // Whether polynomials are in NTT form
    
    GLWECiphertext(std::vector<Polynomial>&& m, Polynomial&& b, uint64_t kid, bool ntt = false)
        : mask(std::move(m)), body(std::move(b)), key_id(kid), is_ntt(ntt) {}
    
    GLWECiphertext(uint32_t k, uint32_t degree, uint64_t modulus, uint64_t kid)
        : key_id(kid), is_ntt(false), body(degree, modulus) {
        mask.reserve(k);
        for (uint32_t i = 0; i < k; ++i) {
            mask.emplace_back(degree, modulus);
        }
    }
    
    // Copy and move constructors
    GLWECiphertext(const GLWECiphertext& other)
        : mask(other.mask), body(other.body), key_id(other.key_id), is_ntt(other.is_ntt) {}
    GLWECiphertext(GLWECiphertext&&) = default;
    GLWECiphertext& operator=(const GLWECiphertext&) = default;
    GLWECiphertext& operator=(GLWECiphertext&&) = default;
    
    uint32_t glwe_dimension() const { return static_cast<uint32_t>(mask.size()); }
    uint32_t poly_degree() const { return body.degree(); }
};

/**
 * GGSW Ciphertext structure for TFHE bootstrapping
 * 
 * A GGSW ciphertext is a matrix of GLWE ciphertexts used for
 * external product in blind rotate.
 */
struct GGSWCiphertext {
    // Matrix of GLWE ciphertexts: (k+1) * decomp_level rows
    // Each row is a GLWE ciphertext
    std::vector<GLWECiphertext> matrix;
    uint32_t decomp_base_log;   // Log2 of decomposition base
    uint32_t decomp_level;      // Number of decomposition levels
    uint64_t key_id;            // ID of the key used for encryption
    
    GGSWCiphertext() : decomp_base_log(0), decomp_level(0), key_id(0) {}
    
    GGSWCiphertext(std::vector<GLWECiphertext>&& m, uint32_t base_log, uint32_t level, uint64_t kid)
        : matrix(std::move(m)), decomp_base_log(base_log), decomp_level(level), key_id(kid) {}
    
    size_t num_rows() const { return matrix.size(); }
};

/**
 * Lookup table for programmable bootstrapping
 * 
 * Encodes a function f: Z_t -> Z_t as a test polynomial.
 */
struct LookupTable {
    Polynomial table;           // Encoded function values
    uint64_t input_modulus;     // Input plaintext modulus
    uint64_t output_modulus;    // Output plaintext modulus
    std::string name;           // Optional name for debugging
    
    LookupTable(Polynomial&& t, uint64_t in_mod, uint64_t out_mod, const std::string& n = "")
        : table(std::move(t)), input_modulus(in_mod), output_modulus(out_mod), name(n) {}
    
    LookupTable(uint32_t degree, uint64_t modulus)
        : table(degree, modulus), input_modulus(0), output_modulus(0) {}
};

/**
 * Extended Bootstrapping Key structure for TFHE
 * 
 * Contains GGSW encryptions of secret key bits for bootstrapping,
 * plus key switching keys for sample extraction.
 */
struct ExtendedBootstrapKey {
    std::vector<GGSWCiphertext> bsk;    // GGSW encryptions of LWE secret key bits
    KeySwitchKey ksk;                    // Key switching key (GLWE -> LWE)
    uint32_t lwe_dimension;              // LWE dimension n
    uint32_t glwe_dimension;             // GLWE dimension k
    uint32_t poly_degree;                // Polynomial degree N
    uint32_t decomp_base_log;            // Decomposition base log
    uint32_t decomp_level;               // Decomposition levels
    uint64_t key_id;                     // Key identifier
    
    ExtendedBootstrapKey() 
        : lwe_dimension(0), glwe_dimension(0), poly_degree(0),
          decomp_base_log(0), decomp_level(0), key_id(0) {}
};

/**
 * Progress callback for bootstrapping operations
 */
using BootstrapProgressCallback = std::function<void(size_t completed, size_t total, const std::string& stage)>;

/**
 * Bootstrap Engine
 * 
 * Implements TFHE bootstrapping operations including:
 * - Bootstrapping key generation (GGSW encryptions)
 * - Blind rotate operation
 * - Sample extract and key switching
 * - Programmable bootstrapping with lookup tables
 * 
 * Requirements: 4.4, 8.1, 8.3, 8.5, 8.6
 */
class BootstrapEngine {
public:
    /**
     * Construct bootstrap engine with given parameters
     * 
     * @param params FHE parameter set
     * @param dispatcher Optional hardware dispatcher for acceleration
     */
    explicit BootstrapEngine(const ParameterSet& params, HardwareDispatcher* dispatcher = nullptr);
    ~BootstrapEngine();
    
    // ========================================================================
    // Bootstrapping Key Generation (Requirement 4.4)
    // ========================================================================
    
    /**
     * Generate bootstrapping key from secret key
     * 
     * Creates GGSW encryptions of each bit of the LWE secret key.
     * Parallelized across available cores for performance.
     * 
     * @param lwe_sk LWE secret key (vector of small coefficients)
     * @param glwe_sk GLWE secret key (polynomial)
     * @param progress Optional progress callback
     * @return Extended bootstrapping key
     */
    std::unique_ptr<ExtendedBootstrapKey> generate_bootstrap_key(
        const std::vector<int64_t>& lwe_sk,
        const SecretKey& glwe_sk,
        BootstrapProgressCallback progress = nullptr
    );
    
    /**
     * Generate GGSW encryption of a single value
     * 
     * @param value The value to encrypt (typically 0 or 1 for key bits)
     * @param glwe_sk GLWE secret key
     * @return GGSW ciphertext
     */
    GGSWCiphertext encrypt_ggsw(int64_t value, const SecretKey& glwe_sk);
    
    /**
     * Generate key switching key
     * 
     * Creates key switching key from GLWE key to LWE key.
     * 
     * @param glwe_sk GLWE secret key
     * @param lwe_sk Target LWE secret key
     * @return Key switching key
     */
    KeySwitchKey generate_key_switch_key(
        const SecretKey& glwe_sk,
        const std::vector<int64_t>& lwe_sk
    );
    
    // ========================================================================
    // Blind Rotate Operation (Requirement 8.1, 8.3)
    // ========================================================================
    
    /**
     * Perform blind rotate operation
     * 
     * Rotates the accumulator polynomial by an encrypted amount.
     * This is the core operation of TFHE bootstrapping.
     * 
     * @param acc Accumulator (GLWE ciphertext, modified in-place)
     * @param lwe_ct Input LWE ciphertext
     * @param bsk Bootstrapping key
     */
    void blind_rotate(
        GLWECiphertext& acc,
        const LWECiphertext& lwe_ct,
        const ExtendedBootstrapKey& bsk
    );
    
    /**
     * Perform blind rotate with SME acceleration
     * 
     * Uses SME matrix operations for polynomial multiplication.
     * Falls back to NEON if SME unavailable.
     * 
     * @param acc Accumulator (GLWE ciphertext, modified in-place)
     * @param lwe_ct Input LWE ciphertext
     * @param bsk Bootstrapping key
     */
    void blind_rotate_sme(
        GLWECiphertext& acc,
        const LWECiphertext& lwe_ct,
        const ExtendedBootstrapKey& bsk
    );
    
    /**
     * Compute external product: GLWE * GGSW -> GLWE
     * 
     * @param glwe Input GLWE ciphertext
     * @param ggsw GGSW ciphertext
     * @return Result GLWE ciphertext
     */
    GLWECiphertext external_product(
        const GLWECiphertext& glwe,
        const GGSWCiphertext& ggsw
    );
    
    /**
     * Compute CMux: select between two GLWE ciphertexts based on GGSW bit
     * 
     * CMux(ggsw, ct0, ct1) = ct0 + ggsw * (ct1 - ct0)
     * If ggsw encrypts 0, returns ct0; if ggsw encrypts 1, returns ct1.
     * 
     * @param ggsw GGSW ciphertext encrypting selector bit
     * @param ct0 GLWE ciphertext for selector = 0
     * @param ct1 GLWE ciphertext for selector = 1
     * @return Selected GLWE ciphertext
     */
    GLWECiphertext cmux(
        const GGSWCiphertext& ggsw,
        const GLWECiphertext& ct0,
        const GLWECiphertext& ct1
    );
    
    // ========================================================================
    // Sample Extract and Key Switching (Requirement 8.1)
    // ========================================================================
    
    /**
     * Extract LWE sample from GLWE ciphertext
     * 
     * Extracts the constant coefficient as an LWE ciphertext.
     * 
     * @param glwe Input GLWE ciphertext
     * @return Extracted LWE ciphertext
     */
    LWECiphertext sample_extract(const GLWECiphertext& glwe);
    
    /**
     * Apply key switching to LWE ciphertext
     * 
     * Switches from one LWE key to another using key switching key.
     * 
     * @param lwe Input LWE ciphertext
     * @param ksk Key switching key
     * @return Key-switched LWE ciphertext
     */
    LWECiphertext key_switch(
        const LWECiphertext& lwe,
        const KeySwitchKey& ksk
    );
    
    // ========================================================================
    // Bootstrapping Operations (Requirement 8.5)
    // ========================================================================
    
    /**
     * Bootstrap an LWE ciphertext
     * 
     * Refreshes the noise to initial levels while preserving the encrypted value.
     * Target: < 20ms on M4 Max hardware.
     * 
     * @param lwe Input LWE ciphertext
     * @param bsk Bootstrapping key
     * @return Bootstrapped LWE ciphertext with refreshed noise
     */
    LWECiphertext bootstrap(
        const LWECiphertext& lwe,
        const ExtendedBootstrapKey& bsk
    );
    
    /**
     * Bootstrap with custom test polynomial
     * 
     * @param lwe Input LWE ciphertext
     * @param bsk Bootstrapping key
     * @param test_poly Custom test polynomial
     * @return Bootstrapped LWE ciphertext
     */
    LWECiphertext bootstrap_with_test_poly(
        const LWECiphertext& lwe,
        const ExtendedBootstrapKey& bsk,
        const Polynomial& test_poly
    );
    
    // ========================================================================
    // Programmable Bootstrapping (Requirement 8.6)
    // ========================================================================
    
    /**
     * Programmable bootstrapping with lookup table
     * 
     * Applies an arbitrary function encoded as a lookup table during bootstrapping.
     * 
     * @param lwe Input LWE ciphertext
     * @param bsk Bootstrapping key
     * @param lut Lookup table encoding the function
     * @return LWE ciphertext encrypting f(plaintext)
     */
    LWECiphertext programmable_bootstrap(
        const LWECiphertext& lwe,
        const ExtendedBootstrapKey& bsk,
        const LookupTable& lut
    );
    
    /**
     * Create lookup table for a function
     * 
     * Encodes function f: Z_t -> Z_t as a test polynomial.
     * 
     * @param func Function to encode
     * @param input_modulus Input plaintext modulus
     * @param output_modulus Output plaintext modulus
     * @param name Optional name for debugging
     * @return Lookup table
     */
    LookupTable create_lookup_table(
        std::function<uint64_t(uint64_t)> func,
        uint64_t input_modulus,
        uint64_t output_modulus,
        const std::string& name = ""
    );
    
    /**
     * Create negation lookup table
     * 
     * @param modulus Plaintext modulus
     * @return Lookup table for negation
     */
    LookupTable create_negation_lut(uint64_t modulus);
    
    /**
     * Create threshold lookup table
     * 
     * Returns 1 if input >= threshold, 0 otherwise.
     * 
     * @param threshold Threshold value
     * @param modulus Plaintext modulus
     * @return Lookup table for threshold function
     */
    LookupTable create_threshold_lut(uint64_t threshold, uint64_t modulus);
    
    /**
     * Create identity lookup table (for standard bootstrapping)
     * 
     * @param modulus Plaintext modulus
     * @return Lookup table for identity function
     */
    LookupTable create_identity_lut(uint64_t modulus);
    
    // ========================================================================
    // LWE Operations
    // ========================================================================
    
    /**
     * Encrypt a value as LWE ciphertext
     * 
     * @param value Plaintext value
     * @param sk LWE secret key
     * @return LWE ciphertext
     */
    LWECiphertext encrypt_lwe(uint64_t value, const std::vector<int64_t>& sk);
    
    /**
     * Decrypt LWE ciphertext
     * 
     * @param lwe LWE ciphertext
     * @param sk LWE secret key
     * @return Decrypted plaintext value
     */
    uint64_t decrypt_lwe(const LWECiphertext& lwe, const std::vector<int64_t>& sk);
    
    /**
     * Add two LWE ciphertexts
     * 
     * @param lwe1 First LWE ciphertext
     * @param lwe2 Second LWE ciphertext
     * @return Sum LWE ciphertext
     */
    LWECiphertext add_lwe(const LWECiphertext& lwe1, const LWECiphertext& lwe2);
    
    /**
     * Negate LWE ciphertext
     * 
     * @param lwe LWE ciphertext
     * @return Negated LWE ciphertext
     */
    LWECiphertext negate_lwe(const LWECiphertext& lwe);
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const ParameterSet& get_params() const { return params_; }
    PolynomialRing& get_ring() { return *ring_; }
    
    /**
     * Get the default test polynomial for bootstrapping
     */
    const Polynomial& get_default_test_poly() const { return *default_test_poly_; }
    
private:
    ParameterSet params_;
    std::unique_ptr<PolynomialRing> ring_;
    std::unique_ptr<SecureRandom> rng_;
    HardwareDispatcher* dispatcher_;
    std::unique_ptr<Polynomial> default_test_poly_;
    
    // Precomputed values
    uint64_t delta_;            // Scaling factor q / t
    uint64_t lwe_modulus_;      // LWE ciphertext modulus
    uint64_t glwe_modulus_;     // GLWE ciphertext modulus
    
    // Helper functions
    Polynomial sample_random_polynomial();
    Polynomial sample_error_polynomial();
    std::vector<int64_t> sample_lwe_error(size_t dimension);
    
    // Decomposition helpers
    std::vector<Polynomial> decompose_polynomial(
        const Polynomial& poly,
        uint32_t base_log,
        uint32_t level
    );
    
    // Initialize default test polynomial
    void init_default_test_poly();
    
    // Polynomial rotation: X^k * poly mod (X^N + 1)
    Polynomial rotate_polynomial(const Polynomial& poly, int32_t rotation);
    
    // GLWE operations
    GLWECiphertext encrypt_glwe_zero(const SecretKey& sk);
    void add_glwe_inplace(GLWECiphertext& ct1, const GLWECiphertext& ct2);
    void subtract_glwe_inplace(GLWECiphertext& ct1, const GLWECiphertext& ct2);
    GLWECiphertext multiply_glwe_by_monomial(const GLWECiphertext& ct, int32_t exponent);
};

// Factory function
std::unique_ptr<BootstrapEngine> create_bootstrap_engine(
    const ParameterSet& params,
    HardwareDispatcher* dispatcher = nullptr
);

} // namespace fhe_accelerate
