/**
 * @file encryption.h
 * @brief RLWE Encryption and Decryption for FHE
 * 
 * This file defines the encryption and decryption infrastructure for FHE operations,
 * including RLWE encryption, SIMD packing, batch encryption with Metal GPU,
 * and noise budget management.
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 15.1, 15.2
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

namespace fhe_accelerate {

// Forward declarations
class MetalBatchEncryptor;

/**
 * Ciphertext structure for RLWE encryption
 * 
 * A ciphertext consists of two or three polynomials:
 * - Standard (degree-1): (c0, c1) where:
 *   - c0 = pk.b * u + e1 + m * delta (for encryption)
 *   - c1 = pk.a * u + e2
 * - After multiplication (degree-2): (c0, c1, c2) where c2 is the high-degree term
 * 
 * Decryption: m = round((c0 + c1 * sk + c2 * sk^2) / delta)
 */
struct Ciphertext {
    Polynomial c0;              // First ciphertext component
    Polynomial c1;              // Second ciphertext component
    std::optional<Polynomial> c2;  // Third component (only for degree-2 ciphertexts)
    double noise_budget;        // Estimated remaining noise budget in bits
    uint64_t key_id;            // ID of the key used for encryption
    bool is_ntt;                // Whether polynomials are in NTT form
    
    Ciphertext(Polynomial&& c0_poly, Polynomial&& c1_poly, 
               double budget, uint64_t kid, bool ntt = false)
        : c0(std::move(c0_poly)), c1(std::move(c1_poly)),
          c2(std::nullopt), noise_budget(budget), key_id(kid), is_ntt(ntt) {}
    
    // Constructor for degree-2 ciphertexts
    Ciphertext(Polynomial&& c0_poly, Polynomial&& c1_poly, Polynomial&& c2_poly,
               double budget, uint64_t kid, bool ntt = false)
        : c0(std::move(c0_poly)), c1(std::move(c1_poly)),
          c2(std::move(c2_poly)), noise_budget(budget), key_id(kid), is_ntt(ntt) {}
    
    // Copy constructor
    Ciphertext(const Ciphertext& other)
        : c0(other.c0), c1(other.c1),
          c2(other.c2), noise_budget(other.noise_budget), key_id(other.key_id),
          is_ntt(other.is_ntt) {}
    
    // Move constructor
    Ciphertext(Ciphertext&&) = default;
    
    // Copy assignment
    Ciphertext& operator=(const Ciphertext& other) {
        if (this != &other) {
            c0 = other.c0;
            c1 = other.c1;
            c2 = other.c2;
            noise_budget = other.noise_budget;
            key_id = other.key_id;
            is_ntt = other.is_ntt;
        }
        return *this;
    }
    
    // Move assignment
    Ciphertext& operator=(Ciphertext&&) = default;
    
    // Check if this is a degree-2 ciphertext
    bool is_degree_2() const { return c2.has_value(); }
    
    // Get the degree of the ciphertext (1 or 2)
    int degree() const { return c2.has_value() ? 2 : 1; }
};

/**
 * Plaintext structure for FHE
 * 
 * Plaintexts can be:
 * - Single values (scalar)
 * - Vectors of values (SIMD packed)
 */
struct Plaintext {
    std::vector<uint64_t> values;   // Plaintext values
    uint64_t plaintext_modulus;     // Plaintext modulus t
    bool is_packed;                 // Whether values are SIMD packed
    
    Plaintext() : plaintext_modulus(0), is_packed(false) {}
    
    Plaintext(uint64_t value, uint64_t mod)
        : values({value}), plaintext_modulus(mod), is_packed(false) {}
    
    Plaintext(std::vector<uint64_t>&& vals, uint64_t mod, bool packed = true)
        : values(std::move(vals)), plaintext_modulus(mod), is_packed(packed) {}
    
    // Get single value (for non-packed plaintexts)
    uint64_t value() const { return values.empty() ? 0 : values[0]; }
    
    // Get number of slots
    size_t slot_count() const { return values.size(); }
};

/**
 * Encryption result with optional proof
 */
struct EncryptionResult {
    Ciphertext ciphertext;
    std::optional<std::vector<uint8_t>> proof;  // Optional ZK proof
    
    EncryptionResult(Ciphertext&& ct)
        : ciphertext(std::move(ct)) {}
    
    EncryptionResult(Ciphertext&& ct, std::vector<uint8_t>&& p)
        : ciphertext(std::move(ct)), proof(std::move(p)) {}
};

/**
 * Decryption result with noise information
 */
struct DecryptionResult {
    Plaintext plaintext;
    double remaining_noise_budget;
    bool success;
    std::string error_message;
    
    DecryptionResult() : remaining_noise_budget(0), success(false) {}
    
    DecryptionResult(Plaintext&& pt, double budget)
        : plaintext(std::move(pt)), remaining_noise_budget(budget), success(true) {}
    
    static DecryptionResult failure(const std::string& msg) {
        DecryptionResult result;
        result.success = false;
        result.error_message = msg;
        return result;
    }
};

/**
 * Batch encryption request
 */
struct BatchEncryptionRequest {
    std::vector<Plaintext> plaintexts;
    const PublicKey* public_key;
    bool generate_proofs;
    
    BatchEncryptionRequest(std::vector<Plaintext>&& pts, const PublicKey* pk, bool proofs = false)
        : plaintexts(std::move(pts)), public_key(pk), generate_proofs(proofs) {}
};

/**
 * Batch encryption result
 */
struct BatchEncryptionResult {
    std::vector<Ciphertext> ciphertexts;
    std::vector<std::vector<uint8_t>> proofs;  // Optional proofs
    double elapsed_ms;
    size_t throughput_per_second;
    
    BatchEncryptionResult() : elapsed_ms(0), throughput_per_second(0) {}
};

/**
 * Progress callback for batch operations
 */
using EncryptionProgressCallback = std::function<void(size_t completed, size_t total)>;

/**
 * Encryption Engine
 * 
 * Handles RLWE encryption and decryption operations with support for:
 * - Single value encryption/decryption
 * - SIMD packing for multiple values
 * - Batch encryption with Metal GPU acceleration
 * - Noise budget tracking
 */
class EncryptionEngine {
public:
    /**
     * Construct encryption engine with given parameters
     */
    explicit EncryptionEngine(const ParameterSet& params);
    ~EncryptionEngine();
    
    // ========================================================================
    // RLWE Encryption (Requirement 5.1, 5.3)
    // ========================================================================
    
    /**
     * Encrypt a plaintext value using RLWE encryption
     * 
     * Computes ciphertext (c0, c1) where:
     * - c0 = pk.b * u + e1 + m * delta
     * - c1 = pk.a * u + e2
     * 
     * @param plaintext The plaintext to encrypt
     * @param pk The public key
     * @return Encrypted ciphertext
     */
    Ciphertext encrypt(const Plaintext& plaintext, const PublicKey& pk);
    
    /**
     * Encrypt a single integer value
     * 
     * @param value The value to encrypt
     * @param pk The public key
     * @return Encrypted ciphertext
     */
    Ciphertext encrypt_value(uint64_t value, const PublicKey& pk);
    
    /**
     * Encrypt with SIMD packing (multiple values in one ciphertext)
     * 
     * Packs multiple plaintext values into a single ciphertext using
     * the polynomial's coefficient slots.
     * 
     * @param values Vector of values to pack and encrypt
     * @param pk The public key
     * @return Encrypted ciphertext containing all values
     */
    Ciphertext encrypt_packed(const std::vector<uint64_t>& values, const PublicKey& pk);
    
    // ========================================================================
    // Decryption (Requirement 5.2, 5.4)
    // ========================================================================
    
    /**
     * Decrypt a ciphertext
     * 
     * Computes: m = round((c0 - c1 * sk) / delta)
     * 
     * @param ciphertext The ciphertext to decrypt
     * @param sk The secret key
     * @return Decryption result with plaintext and noise budget
     */
    DecryptionResult decrypt(const Ciphertext& ciphertext, const SecretKey& sk);
    
    /**
     * Decrypt to a single integer value
     * 
     * @param ciphertext The ciphertext to decrypt
     * @param sk The secret key
     * @return The decrypted value, or nullopt if decryption failed
     */
    std::optional<uint64_t> decrypt_value(const Ciphertext& ciphertext, const SecretKey& sk);
    
    /**
     * Decrypt SIMD-packed ciphertext to multiple values
     * 
     * @param ciphertext The packed ciphertext
     * @param sk The secret key
     * @param num_values Number of values to extract
     * @return Vector of decrypted values
     */
    std::vector<uint64_t> decrypt_packed(const Ciphertext& ciphertext, 
                                         const SecretKey& sk,
                                         size_t num_values);
    
    // ========================================================================
    // Batch Encryption with Metal GPU (Requirement 5.6, 15.2)
    // ========================================================================
    
    /**
     * Batch encrypt multiple plaintexts using Metal GPU
     * 
     * Parallelizes encryption across GPU threads for high throughput.
     * Target: 10,000+ ballots/second encryption.
     * 
     * @param request Batch encryption request
     * @param progress Optional progress callback
     * @return Batch encryption result with all ciphertexts
     */
    BatchEncryptionResult batch_encrypt(
        const BatchEncryptionRequest& request,
        EncryptionProgressCallback progress = nullptr
    );
    
    /**
     * Batch encrypt with streaming input
     * 
     * Supports streaming ballot ingestion for real-time processing.
     * 
     * @param plaintext_stream Iterator over plaintexts
     * @param pk The public key
     * @param batch_size Number of plaintexts to process per batch
     * @param output Callback to receive encrypted ciphertexts
     */
    template<typename Iterator>
    void batch_encrypt_streaming(
        Iterator begin, Iterator end,
        const PublicKey& pk,
        size_t batch_size,
        std::function<void(std::vector<Ciphertext>&&)> output
    );
    
    // ========================================================================
    // Homomorphic Addition (Requirement 6.1, 6.2, 6.3, 6.4, 6.5, 6.6)
    // ========================================================================
    
    /**
     * Add two ciphertexts homomorphically
     * 
     * Computes: ct_result = ct1 + ct2 where decrypt(ct_result) = decrypt(ct1) + decrypt(ct2)
     * 
     * The addition is performed component-wise on the polynomial pairs:
     * - result.c0 = ct1.c0 + ct2.c0 (mod q)
     * - result.c1 = ct1.c1 + ct2.c1 (mod q)
     * 
     * Noise growth: The noise budget decreases slightly (by about 1 bit) due to
     * the addition of noise terms.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @return Ciphertext encrypting the sum of plaintexts
     * @throws std::invalid_argument if ciphertexts have different key IDs
     * 
     * Requirements: 6.1, 6.4
     */
    Ciphertext add(const Ciphertext& ct1, const Ciphertext& ct2);
    
    /**
     * Add two ciphertexts in-place
     * 
     * Modifies ct1 to contain the sum: ct1 = ct1 + ct2
     * 
     * @param ct1 Ciphertext to modify (result stored here)
     * @param ct2 Ciphertext to add
     * @throws std::invalid_argument if ciphertexts have different key IDs
     * 
     * Requirements: 6.1
     */
    void add_inplace(Ciphertext& ct1, const Ciphertext& ct2);
    
    /**
     * Add a plaintext to a ciphertext homomorphically
     * 
     * Computes: ct_result = ct + pt where decrypt(ct_result) = decrypt(ct) + pt
     * 
     * The plaintext is encoded and added to the ciphertext body (c0):
     * - result.c0 = ct.c0 + encode(pt) (mod q)
     * - result.c1 = ct.c1 (unchanged)
     * 
     * This operation has minimal noise growth since no multiplication is involved.
     * 
     * @param ct Ciphertext
     * @param pt Plaintext to add
     * @return Ciphertext encrypting the sum
     * 
     * Requirements: 6.2
     */
    Ciphertext add_plain(const Ciphertext& ct, const Plaintext& pt);
    
    /**
     * Add a plaintext to a ciphertext in-place
     * 
     * Modifies ct to contain the sum: ct = ct + pt
     * 
     * @param ct Ciphertext to modify
     * @param pt Plaintext to add
     * 
     * Requirements: 6.2
     */
    void add_plain_inplace(Ciphertext& ct, const Plaintext& pt);
    
    /**
     * Add a scalar value to a ciphertext
     * 
     * Convenience method for adding a single integer value.
     * 
     * @param ct Ciphertext
     * @param value Scalar value to add
     * @return Ciphertext encrypting the sum
     * 
     * Requirements: 6.2
     */
    Ciphertext add_scalar(const Ciphertext& ct, uint64_t value);
    
    /**
     * Subtract two ciphertexts homomorphically
     * 
     * Computes: ct_result = ct1 - ct2 where decrypt(ct_result) = decrypt(ct1) - decrypt(ct2)
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext to subtract
     * @return Ciphertext encrypting the difference
     * @throws std::invalid_argument if ciphertexts have different key IDs
     * 
     * Requirements: 6.1
     */
    Ciphertext subtract(const Ciphertext& ct1, const Ciphertext& ct2);
    
    /**
     * Negate a ciphertext homomorphically
     * 
     * Computes: ct_result = -ct where decrypt(ct_result) = -decrypt(ct)
     * 
     * @param ct Ciphertext to negate
     * @return Negated ciphertext
     * 
     * Requirements: 6.1
     */
    Ciphertext negate(const Ciphertext& ct);
    
    /**
     * Get the encryption of zero (additive identity)
     * 
     * Returns a fresh encryption of zero that can be used as the identity
     * element for homomorphic addition.
     * 
     * @param pk Public key to use for encryption
     * @return Ciphertext encrypting zero
     * 
     * Requirements: 6.5
     */
    Ciphertext get_zero_ciphertext(const PublicKey& pk);
    
    // ========================================================================
    // Batch Homomorphic Operations (Requirement 15.1, 15.2)
    // ========================================================================
    
    /**
     * Batch add thousands of ciphertexts
     * 
     * Efficiently accumulates ciphertexts using parallel processing.
     * Target: Process 10,000+ ballots in under 5 seconds.
     * 
     * @param ciphertexts Vector of ciphertexts to add
     * @return Sum ciphertext
     */
    Ciphertext batch_add(const std::vector<Ciphertext>& ciphertexts);
    
    /**
     * Batch add with tree reduction (more efficient for large batches)
     * 
     * @param ciphertexts Vector of ciphertexts to add
     * @param progress Optional progress callback
     * @return Sum ciphertext
     */
    Ciphertext batch_add_tree(
        const std::vector<Ciphertext>& ciphertexts,
        EncryptionProgressCallback progress = nullptr
    );
    
    // ========================================================================
    // Homomorphic Multiplication (Requirement 7.1, 7.2, 7.3, 7.4, 7.5, 7.6)
    // ========================================================================
    
    /**
     * Multiply two ciphertexts homomorphically
     * 
     * Computes: ct_result = ct1 * ct2 where decrypt(ct_result) = decrypt(ct1) * decrypt(ct2)
     * 
     * The multiplication is performed using tensor product of ciphertext polynomials:
     * - result.c0 = ct1.c0 * ct2.c0 (mod q)
     * - result.c1 = ct1.c0 * ct2.c1 + ct1.c1 * ct2.c0 (mod q)
     * - result.c2 = ct1.c1 * ct2.c1 (mod q)
     * 
     * Note: This produces a degree-2 ciphertext that requires relinearization.
     * 
     * Noise growth: The noise budget decreases significantly due to multiplication
     * of noise terms. Approximately halves the noise budget.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @return Ciphertext encrypting the product of plaintexts (degree-2)
     * @throws std::invalid_argument if ciphertexts have different key IDs
     * 
     * Requirements: 7.1, 7.5
     */
    Ciphertext multiply(const Ciphertext& ct1, const Ciphertext& ct2);
    
    /**
     * Multiply two ciphertexts and relinearize in one step
     * 
     * Combines multiplication and relinearization for efficiency.
     * Returns a standard degree-1 ciphertext.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param ek Evaluation key for relinearization
     * @return Ciphertext encrypting the product (degree-1, relinearized)
     * @throws std::invalid_argument if ciphertexts have different key IDs
     * 
     * Requirements: 7.1, 7.4
     */
    Ciphertext multiply_relin(const Ciphertext& ct1, const Ciphertext& ct2, 
                               const EvaluationKey& ek);
    
    /**
     * Multiply a ciphertext by a plaintext homomorphically
     * 
     * Computes: ct_result = ct * pt where decrypt(ct_result) = decrypt(ct) * pt
     * 
     * The plaintext is encoded and multiplied with ciphertext components:
     * - result.c0 = ct.c0 * encode(pt) (mod q)
     * - result.c1 = ct.c1 * encode(pt) (mod q)
     * 
     * This operation has less noise growth than ciphertext-ciphertext multiplication.
     * 
     * @param ct Ciphertext
     * @param pt Plaintext to multiply
     * @return Ciphertext encrypting the product
     * 
     * Requirements: 7.2
     */
    Ciphertext multiply_plain(const Ciphertext& ct, const Plaintext& pt);
    
    /**
     * Multiply a ciphertext by a plaintext in-place
     * 
     * Modifies ct to contain the product: ct = ct * pt
     * 
     * @param ct Ciphertext to modify
     * @param pt Plaintext to multiply
     * 
     * Requirements: 7.2
     */
    void multiply_plain_inplace(Ciphertext& ct, const Plaintext& pt);
    
    /**
     * Multiply a ciphertext by a scalar value
     * 
     * Convenience method for multiplying by a single integer value.
     * Optimized for scalar multiplication.
     * 
     * @param ct Ciphertext
     * @param scalar Scalar value to multiply
     * @return Ciphertext encrypting the product
     * 
     * Requirements: 7.2
     */
    Ciphertext multiply_scalar(const Ciphertext& ct, uint64_t scalar);
    
    /**
     * Relinearize a degree-2 ciphertext to degree-1
     * 
     * Applies the evaluation key to reduce ciphertext size after multiplication.
     * Uses decomposition for noise control.
     * 
     * @param ct Degree-2 ciphertext (with c2 component)
     * @param ek Evaluation key
     * @return Degree-1 ciphertext
     * 
     * Requirements: 7.4
     */
    Ciphertext relinearize(const Ciphertext& ct, const EvaluationKey& ek);
    
    /**
     * Relinearize a ciphertext in-place
     * 
     * @param ct Ciphertext to relinearize (modified in-place)
     * @param ek Evaluation key
     * 
     * Requirements: 7.4
     */
    void relinearize_inplace(Ciphertext& ct, const EvaluationKey& ek);
    
    /**
     * Square a ciphertext homomorphically
     * 
     * Computes: ct_result = ct^2 where decrypt(ct_result) = decrypt(ct)^2
     * Optimized compared to multiply(ct, ct).
     * 
     * @param ct Ciphertext to square
     * @return Ciphertext encrypting the square (degree-2)
     * 
     * Requirements: 7.1
     */
    Ciphertext square(const Ciphertext& ct);
    
    /**
     * Square a ciphertext and relinearize
     * 
     * @param ct Ciphertext to square
     * @param ek Evaluation key for relinearization
     * @return Ciphertext encrypting the square (degree-1, relinearized)
     * 
     * Requirements: 7.1, 7.4
     */
    Ciphertext square_relin(const Ciphertext& ct, const EvaluationKey& ek);
    
    // ========================================================================
    // Ballot Aggregation Primitives (Requirement 15.1, 15.8)
    // ========================================================================
    
    /**
     * Homomorphic vote tallying (encrypted sum)
     * 
     * Efficiently aggregates encrypted ballots into a running total.
     * Target: Process 10,000+ ballots in under 5 seconds.
     * 
     * @param ballots Vector of encrypted ballots (each encrypting a vote)
     * @param progress Optional progress callback
     * @return Ciphertext encrypting the sum of all votes
     * 
     * Requirements: 15.1
     */
    Ciphertext tally_votes(
        const std::vector<Ciphertext>& ballots,
        EncryptionProgressCallback progress = nullptr
    );
    
    /**
     * Weighted vote tallying
     * 
     * Aggregates encrypted ballots with encrypted weights.
     * Each ballot is multiplied by its weight before summing.
     * 
     * @param ballots Vector of encrypted ballots
     * @param weights Vector of encrypted weights (same size as ballots)
     * @param ek Evaluation key for relinearization after multiplication
     * @param progress Optional progress callback
     * @return Ciphertext encrypting the weighted sum
     * 
     * Requirements: 15.1
     */
    Ciphertext tally_weighted_votes(
        const std::vector<Ciphertext>& ballots,
        const std::vector<Ciphertext>& weights,
        const EvaluationKey& ek,
        EncryptionProgressCallback progress = nullptr
    );
    
    /**
     * Threshold detection without decryption
     * 
     * Checks if the encrypted tally exceeds a threshold without revealing
     * the actual count. Returns an encrypted indicator (0 or 1).
     * 
     * Note: This requires programmable bootstrapping for comparison.
     * For now, returns a placeholder that can be used with PBS.
     * 
     * @param tally Encrypted tally
     * @param threshold Plaintext threshold value
     * @return Ciphertext encrypting 1 if tally >= threshold, 0 otherwise
     * 
     * Requirements: 15.8
     */
    Ciphertext check_threshold(
        const Ciphertext& tally,
        uint64_t threshold
    );
    
    /**
     * Incremental tally update
     * 
     * Adds a single ballot to an existing tally efficiently.
     * Useful for real-time vote counting.
     * 
     * @param current_tally Current encrypted tally
     * @param new_ballot New encrypted ballot to add
     * @return Updated tally ciphertext
     * 
     * Requirements: 15.1
     */
    Ciphertext update_tally(
        const Ciphertext& current_tally,
        const Ciphertext& new_ballot
    );
    
    /**
     * Multi-candidate vote tallying
     * 
     * Tallies votes for multiple candidates simultaneously using SIMD packing.
     * Each ballot encodes votes for all candidates in different slots.
     * 
     * @param ballots Vector of packed ballots (each slot = one candidate)
     * @param num_candidates Number of candidates
     * @param progress Optional progress callback
     * @return Packed ciphertext with tallies for all candidates
     * 
     * Requirements: 15.1
     */
    Ciphertext tally_multi_candidate(
        const std::vector<Ciphertext>& ballots,
        size_t num_candidates,
        EncryptionProgressCallback progress = nullptr
    );
    
    // ========================================================================
    // Comparison Operations for Fraud Detection (Requirement 15.3, 15.7)
    // ========================================================================
    
    /**
     * Encrypted greater-than comparison
     * 
     * Computes ct1 > ct2 on encrypted values using programmable bootstrapping.
     * Returns an encrypted indicator (0 or 1).
     * 
     * Note: This requires bootstrapping keys for PBS. Without them, this
     * returns a prepared ciphertext that can be used with external PBS.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param bk Bootstrapping key (optional, for full PBS)
     * @return Ciphertext encrypting 1 if ct1 > ct2, 0 otherwise
     * 
     * Requirements: 15.3
     */
    Ciphertext compare_greater_than(
        const Ciphertext& ct1,
        const Ciphertext& ct2,
        const BootstrapKey* bk = nullptr
    );
    
    /**
     * Encrypted less-than comparison
     * 
     * Computes ct1 < ct2 on encrypted values.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param bk Bootstrapping key (optional)
     * @return Ciphertext encrypting 1 if ct1 < ct2, 0 otherwise
     * 
     * Requirements: 15.3
     */
    Ciphertext compare_less_than(
        const Ciphertext& ct1,
        const Ciphertext& ct2,
        const BootstrapKey* bk = nullptr
    );
    
    /**
     * Encrypted equality test
     * 
     * Tests if two encrypted values are equal. Useful for duplicate detection.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param bk Bootstrapping key (optional)
     * @return Ciphertext encrypting 1 if ct1 == ct2, 0 otherwise
     * 
     * Requirements: 15.3
     */
    Ciphertext compare_equal(
        const Ciphertext& ct1,
        const Ciphertext& ct2,
        const BootstrapKey* bk = nullptr
    );
    
    /**
     * Range check on encrypted value
     * 
     * Checks if an encrypted value is within a specified range [min, max].
     * 
     * @param ct Ciphertext to check
     * @param min_value Minimum value (plaintext)
     * @param max_value Maximum value (plaintext)
     * @param bk Bootstrapping key (optional)
     * @return Ciphertext encrypting 1 if min <= ct <= max, 0 otherwise
     * 
     * Requirements: 15.7
     */
    Ciphertext check_range(
        const Ciphertext& ct,
        uint64_t min_value,
        uint64_t max_value,
        const BootstrapKey* bk = nullptr
    );
    
    /**
     * Duplicate detection on encrypted ballots
     * 
     * Checks if a ballot matches any in a list of existing ballots.
     * Uses equality testing on encrypted data.
     * 
     * @param new_ballot New ballot to check
     * @param existing_ballots List of existing ballots
     * @param bk Bootstrapping key (optional)
     * @return Ciphertext encrypting 1 if duplicate found, 0 otherwise
     * 
     * Requirements: 15.3
     */
    Ciphertext detect_duplicate(
        const Ciphertext& new_ballot,
        const std::vector<Ciphertext>& existing_ballots,
        const BootstrapKey* bk = nullptr
    );
    
    /**
     * Anomaly score computation
     * 
     * Computes an encrypted anomaly score based on deviation from expected
     * voting patterns. Higher scores indicate potential fraud.
     * 
     * @param ballot Encrypted ballot to analyze
     * @param expected_distribution Expected vote distribution (encrypted)
     * @param ek Evaluation key for multiplication
     * @return Ciphertext encrypting the anomaly score
     * 
     * Requirements: 15.7
     */
    Ciphertext compute_anomaly_score(
        const Ciphertext& ballot,
        const Ciphertext& expected_distribution,
        const EvaluationKey& ek
    );
    
    // ========================================================================
    // Noise Budget Management (Requirement 5.4)
    // ========================================================================
    
    /**
     * Get the remaining noise budget of a ciphertext
     * 
     * @param ciphertext The ciphertext to check
     * @param sk The secret key (needed for accurate measurement)
     * @return Noise budget in bits
     */
    double get_noise_budget(const Ciphertext& ciphertext, const SecretKey& sk);
    
    /**
     * Estimate noise budget without secret key (less accurate)
     * 
     * @param ciphertext The ciphertext to check
     * @return Estimated noise budget in bits
     */
    double estimate_noise_budget(const Ciphertext& ciphertext) const;
    
    // ========================================================================
    // Plaintext Encoding
    // ========================================================================
    
    /**
     * Encode a plaintext value as a polynomial
     * 
     * @param value The value to encode
     * @return Encoded polynomial
     */
    Polynomial encode_plaintext(uint64_t value) const;
    
    /**
     * Encode multiple values as a packed polynomial (SIMD)
     * 
     * @param values The values to encode
     * @return Encoded polynomial with values in slots
     */
    Polynomial encode_packed(const std::vector<uint64_t>& values) const;
    
    /**
     * Decode a polynomial to a plaintext value
     * 
     * @param poly The polynomial to decode
     * @return Decoded value
     */
    uint64_t decode_plaintext(const Polynomial& poly) const;
    
    /**
     * Decode a packed polynomial to multiple values
     * 
     * @param poly The polynomial to decode
     * @param num_values Number of values to extract
     * @return Vector of decoded values
     */
    std::vector<uint64_t> decode_packed(const Polynomial& poly, size_t num_values) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const ParameterSet& get_params() const { return params_; }
    PolynomialRing& get_ring() { return *ring_; }
    
    /**
     * Get the scaling factor delta = q / t
     */
    uint64_t get_delta() const { return delta_; }
    
    /**
     * Get maximum number of SIMD slots
     */
    size_t get_slot_count() const { return params_.poly_degree; }
    
private:
    ParameterSet params_;
    std::unique_ptr<PolynomialRing> ring_;
    std::unique_ptr<SecureRandom> rng_;
    std::unique_ptr<MetalBatchEncryptor> metal_encryptor_;
    
    uint64_t delta_;            // Scaling factor q / t
    double initial_noise_budget_;
    
    // Helper functions
    Polynomial sample_random_polynomial();
    Polynomial sample_error_polynomial();
    Polynomial sample_ternary_polynomial();
    
    // Internal encryption helper
    Ciphertext encrypt_internal(const Polynomial& encoded_plaintext, const PublicKey& pk);
    
    // Noise estimation helpers
    double compute_noise_budget(const Polynomial& noise, uint64_t modulus) const;
};

// Factory function
std::unique_ptr<EncryptionEngine> create_encryption_engine(const ParameterSet& params);

} // namespace fhe_accelerate
