/**
 * @file verifiable_encryption.h
 * @brief Verifiable Encryption with Zero-Knowledge Proofs
 * 
 * This file defines the infrastructure for generating zero-knowledge proofs
 * of correct encryption, supporting public verifiability of ballots and
 * receipt generation for voters.
 * 
 * Requirements: 15.5, 17
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include "polynomial_ring.h"
#include "key_manager.h"

namespace fhe_accelerate {

/**
 * Proof types for verifiable encryption
 */
enum class ProofType {
    ENCRYPTION_CORRECTNESS,  // Proves ciphertext is valid encryption
    BALLOT_VALIDITY,         // Proves ballot contains valid choice
    DECRYPTION_CORRECTNESS   // Proves partial decryption is correct
};

/**
 * Zero-knowledge proof of correct encryption
 * 
 * Proves that a ciphertext is a valid RLWE encryption without
 * revealing the plaintext or randomness used.
 */
struct EncryptionProof {
    // Commitment to randomness
    Polynomial commitment_a;
    Polynomial commitment_b;
    
    // Challenge (Fiat-Shamir)
    std::vector<uint8_t> challenge;
    
    // Response
    Polynomial response_r;  // Response for randomness
    Polynomial response_e;  // Response for error
    
    // Metadata
    uint64_t timestamp;
    ProofType type;
    
    EncryptionProof() : timestamp(0), type(ProofType::ENCRYPTION_CORRECTNESS) {}
};

/**
 * Ballot validity proof
 * 
 * Proves that an encrypted ballot contains a valid choice
 * (e.g., choice ∈ {0, 1, ..., num_candidates-1}) without revealing the choice.
 */
struct BallotValidityProof {
    // Range proof components
    std::vector<Polynomial> commitments;
    std::vector<uint8_t> challenge;
    std::vector<Polynomial> responses;
    
    // Metadata
    uint32_t num_candidates;
    uint64_t timestamp;
    
    BallotValidityProof() : num_candidates(0), timestamp(0) {}
};

/**
 * Voter receipt
 * 
 * A receipt that allows voters to verify their vote was recorded
 * without revealing how they voted.
 */
struct VoterReceipt {
    // Receipt identifier (hash of ballot)
    std::vector<uint8_t> receipt_id;
    
    // Commitment to the vote
    std::vector<uint8_t> vote_commitment;
    
    // Timestamp
    uint64_t timestamp;
    
    // Signature from election authority
    std::vector<uint8_t> authority_signature;
    
    // Verification data
    std::vector<uint8_t> verification_data;
};

/**
 * Decryption correctness proof
 * 
 * Proves that a partial decryption was computed correctly
 * using the correct key share.
 */
struct DecryptionProof {
    // Commitment
    Polynomial commitment;
    
    // Challenge
    std::vector<uint8_t> challenge;
    
    // Response
    Polynomial response;
    
    // Share identifier
    uint32_t share_id;
    
    DecryptionProof() : share_id(0) {}
};

/**
 * Verifiable Encryption Engine
 * 
 * Generates and verifies zero-knowledge proofs for FHE operations.
 */
class VerifiableEncryption {
public:
    /**
     * Construct verifiable encryption engine
     */
    explicit VerifiableEncryption(const ParameterSet& params);
    ~VerifiableEncryption();
    
    // ========================================================================
    // Encryption Proofs (Requirement 15.5)
    // ========================================================================
    
    /**
     * Generate proof of correct encryption
     * 
     * Proves that ciphertext (a, b) is a valid RLWE encryption of plaintext
     * under public key pk, without revealing the plaintext or randomness.
     * 
     * @param ct_a First component of ciphertext
     * @param ct_b Second component of ciphertext
     * @param pk Public key used for encryption
     * @param plaintext The encrypted plaintext (prover's witness)
     * @param randomness The randomness used (prover's witness)
     * @return Zero-knowledge proof of correct encryption
     */
    EncryptionProof prove_encryption(
        const Polynomial& ct_a,
        const Polynomial& ct_b,
        const PublicKey& pk,
        const Polynomial& plaintext,
        const Polynomial& randomness
    );
    
    /**
     * Verify proof of correct encryption
     * 
     * @param ct_a First component of ciphertext
     * @param ct_b Second component of ciphertext
     * @param pk Public key
     * @param proof The proof to verify
     * @return true if proof is valid
     */
    bool verify_encryption(
        const Polynomial& ct_a,
        const Polynomial& ct_b,
        const PublicKey& pk,
        const EncryptionProof& proof
    );
    
    // ========================================================================
    // Ballot Validity Proofs (Requirement 15.5)
    // ========================================================================
    
    /**
     * Generate proof of ballot validity
     * 
     * Proves that the encrypted ballot contains a valid choice
     * in the range [0, num_candidates).
     * 
     * @param encrypted_choice The encrypted choice
     * @param choice The actual choice (prover's witness)
     * @param num_candidates Number of valid choices
     * @param pk Public key
     * @return Zero-knowledge proof of ballot validity
     */
    BallotValidityProof prove_ballot_validity(
        const std::pair<Polynomial, Polynomial>& encrypted_choice,
        uint64_t choice,
        uint32_t num_candidates,
        const PublicKey& pk
    );
    
    /**
     * Verify proof of ballot validity
     * 
     * @param encrypted_choice The encrypted choice
     * @param num_candidates Number of valid choices
     * @param pk Public key
     * @param proof The proof to verify
     * @return true if proof is valid
     */
    bool verify_ballot_validity(
        const std::pair<Polynomial, Polynomial>& encrypted_choice,
        uint32_t num_candidates,
        const PublicKey& pk,
        const BallotValidityProof& proof
    );
    
    // ========================================================================
    // Voter Receipts (Requirement 17)
    // ========================================================================
    
    /**
     * Generate voter receipt
     * 
     * Creates a receipt that allows the voter to verify their vote
     * was recorded without revealing how they voted.
     * 
     * @param encrypted_ballot The encrypted ballot
     * @param ballot_id Unique ballot identifier
     * @return Voter receipt
     */
    VoterReceipt generate_receipt(
        const std::vector<std::pair<Polynomial, Polynomial>>& encrypted_ballot,
        uint64_t ballot_id
    );
    
    /**
     * Verify voter receipt
     * 
     * @param receipt The receipt to verify
     * @param encrypted_ballot The encrypted ballot
     * @return true if receipt is valid
     */
    bool verify_receipt(
        const VoterReceipt& receipt,
        const std::vector<std::pair<Polynomial, Polynomial>>& encrypted_ballot
    );
    
    // ========================================================================
    // Decryption Proofs (Requirement 15.5)
    // ========================================================================
    
    /**
     * Generate proof of correct partial decryption
     * 
     * @param ct_body Ciphertext body polynomial
     * @param partial_result Partial decryption result
     * @param share The key share used
     * @return Zero-knowledge proof of correct decryption
     */
    DecryptionProof prove_partial_decryption(
        const Polynomial& ct_body,
        const Polynomial& partial_result,
        const SecretKeyShare& share
    );
    
    /**
     * Verify proof of correct partial decryption
     * 
     * @param ct_body Ciphertext body polynomial
     * @param partial_result Partial decryption result
     * @param commitment Public commitment to the share
     * @param proof The proof to verify
     * @return true if proof is valid
     */
    bool verify_partial_decryption(
        const Polynomial& ct_body,
        const Polynomial& partial_result,
        const std::vector<uint8_t>& commitment,
        const DecryptionProof& proof
    );
    
private:
    ParameterSet params_;
    std::unique_ptr<PolynomialRing> ring_;
    std::unique_ptr<SecureRandom> rng_;
    
    // Helper functions
    std::vector<uint8_t> compute_challenge(
        const std::vector<Polynomial>& commitments,
        const std::vector<uint8_t>& public_data
    );
    
    std::vector<uint8_t> hash_polynomial(const Polynomial& poly);
    std::vector<uint8_t> hash_ballot(
        const std::vector<std::pair<Polynomial, Polynomial>>& ballot
    );
};

// Factory function
std::unique_ptr<VerifiableEncryption> create_verifiable_encryption(const ParameterSet& params);

} // namespace fhe_accelerate
