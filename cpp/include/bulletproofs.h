/**
 * Bulletproofs Range Proof Implementation
 * 
 * Implements Bulletproofs range proofs for ballot validity verification:
 * - Inner product argument
 * - Range proofs for vote validity (choice ∈ {0,1,2,...,n-1})
 * - Optimized for small ranges (2-10 candidates)
 * - Target <50ms proof generation on M4 Max
 * 
 * Based on the Bulletproofs paper by Bünz et al.
 * https://eprint.iacr.org/2017/1066.pdf
 * 
 * Requirements: 19.1, 19.6, 19.8, 19.9, 20
 */

#pragma once

#include "zk_field_arithmetic.h"
#include "zk_elliptic_curve.h"
#include "zk_hash.h"
#include <vector>
#include <memory>
#include <optional>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Bulletproofs Configuration
// ============================================================================

/**
 * Bulletproofs configuration parameters
 */
struct BulletproofsConfig {
    size_t max_range_bits;      // Maximum bits for range proof (e.g., 64)
    size_t max_aggregation;     // Maximum number of proofs to aggregate
    bool use_gpu;               // Use Metal GPU for MSM acceleration
    
    BulletproofsConfig(size_t bits = 64, size_t agg = 16, bool gpu = true)
        : max_range_bits(bits), max_aggregation(agg), use_gpu(gpu) {}
};

// ============================================================================
// Generator Points
// ============================================================================

/**
 * Bulletproofs generator points
 * 
 * Contains the public parameters:
 * - G, H: Base points for Pedersen commitments
 * - g_vec, h_vec: Vector generators for inner product argument
 */
struct BulletproofsGenerators {
    AffinePoint256 G;                      // Base point G
    AffinePoint256 H;                      // Base point H (for blinding)
    std::vector<AffinePoint256> g_vec;     // Vector of G generators
    std::vector<AffinePoint256> h_vec;     // Vector of H generators
    AffinePoint256 U;                      // Random point for inner product
    
    size_t size() const { return g_vec.size(); }
};

// ============================================================================
// Pedersen Commitment
// ============================================================================

/**
 * Pedersen commitment: C = v*G + r*H
 * 
 * Commits to value v with blinding factor r
 */
struct PedersenCommitment {
    AffinePoint256 point;
    
    PedersenCommitment() = default;
    explicit PedersenCommitment(const AffinePoint256& p) : point(p) {}
    
    bool operator==(const PedersenCommitment& other) const {
        return point == other.point;
    }
};

// ============================================================================
// Inner Product Proof
// ============================================================================

/**
 * Inner product argument proof
 * 
 * Proves that <a, b> = c for committed vectors a, b
 * Uses logarithmic communication via recursive halving
 */
struct InnerProductProof {
    std::vector<AffinePoint256> L;    // Left fold points
    std::vector<AffinePoint256> R;    // Right fold points
    FieldElement256 a;                 // Final scalar a
    FieldElement256 b;                 // Final scalar b
    
    /**
     * Get proof size in bytes
     */
    size_t size_bytes() const {
        // Each point is 64 bytes (compressed would be 33)
        // Each scalar is 32 bytes
        return L.size() * 64 + R.size() * 64 + 64;
    }
    
    /**
     * Serialize proof to bytes
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * Deserialize proof from bytes
     */
    static std::optional<InnerProductProof> deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// Range Proof
// ============================================================================

/**
 * Bulletproofs range proof
 * 
 * Proves that committed value v is in range [0, 2^n)
 * Proof size is O(log n) group elements
 */
struct RangeProof {
    AffinePoint256 A;                  // Vector commitment to a_L, a_R
    AffinePoint256 S;                  // Vector commitment to s_L, s_R
    AffinePoint256 T1;                 // Commitment to t_1
    AffinePoint256 T2;                 // Commitment to t_2
    FieldElement256 tau_x;             // Blinding factor for t(x)
    FieldElement256 mu;                // Blinding factor for P
    FieldElement256 t_hat;             // Evaluation t(x)
    InnerProductProof inner_proof;     // Inner product argument
    
    /**
     * Get proof size in bytes
     */
    size_t size_bytes() const {
        // 4 points (A, S, T1, T2) + 3 scalars (tau_x, mu, t_hat) + inner proof
        return 4 * 64 + 3 * 32 + inner_proof.size_bytes();
    }
    
    /**
     * Serialize proof to bytes
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * Deserialize proof from bytes
     */
    static std::optional<RangeProof> deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// Aggregated Range Proof
// ============================================================================

/**
 * Aggregated range proof for multiple values
 * 
 * Proves that multiple committed values are all in range [0, 2^n)
 * More efficient than individual proofs when proving multiple values
 */
struct AggregatedRangeProof {
    AffinePoint256 A;
    AffinePoint256 S;
    AffinePoint256 T1;
    AffinePoint256 T2;
    FieldElement256 tau_x;
    FieldElement256 mu;
    FieldElement256 t_hat;
    InnerProductProof inner_proof;
    size_t num_values;                 // Number of aggregated values
    
    size_t size_bytes() const {
        return 4 * 64 + 3 * 32 + inner_proof.size_bytes();
    }
    
    std::vector<uint8_t> serialize() const;
    static std::optional<AggregatedRangeProof> deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// Ballot Validity Proof
// ============================================================================

/**
 * Ballot validity proof for voting systems
 * 
 * Proves that a vote is valid (choice ∈ {0, 1, ..., num_candidates-1})
 * without revealing the actual choice.
 */
struct BallotValidityProof {
    PedersenCommitment commitment;     // Commitment to the vote
    RangeProof range_proof;            // Proof that vote is in valid range
    size_t num_candidates;             // Number of valid choices
    
    size_t size_bytes() const {
        return 64 + range_proof.size_bytes() + sizeof(size_t);
    }
    
    std::vector<uint8_t> serialize() const;
    static std::optional<BallotValidityProof> deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// Bulletproofs Prover
// ============================================================================

/**
 * Bulletproofs prover for range proofs
 * 
 * Generates range proofs for ballot validity verification.
 * Optimized for M4 Max with Metal GPU acceleration.
 */
class BulletproofsProver {
public:
    /**
     * Construct prover with given configuration
     */
    explicit BulletproofsProver(const BulletproofsConfig& config = BulletproofsConfig());
    
    /**
     * Generate public parameters (generators)
     * 
     * @param n Number of bits for range proof
     * @return Generator points
     */
    BulletproofsGenerators generate_generators(size_t n) const;
    
    /**
     * Create Pedersen commitment to value
     * 
     * @param value Value to commit
     * @param blinding Blinding factor (random if not provided)
     * @param gens Generator points
     * @return Commitment and blinding factor used
     */
    std::pair<PedersenCommitment, FieldElement256> commit(
        uint64_t value,
        const std::optional<FieldElement256>& blinding,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Generate range proof for single value
     * 
     * Proves that value is in range [0, 2^n)
     * 
     * @param value Value to prove
     * @param blinding Blinding factor used in commitment
     * @param n Number of bits (range is [0, 2^n))
     * @param gens Generator points
     * @return Range proof
     */
    RangeProof prove_range(
        uint64_t value,
        const FieldElement256& blinding,
        size_t n,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Generate aggregated range proof for multiple values
     * 
     * @param values Values to prove
     * @param blindings Blinding factors
     * @param n Number of bits per value
     * @param gens Generator points
     * @return Aggregated range proof
     */
    AggregatedRangeProof prove_range_aggregated(
        const std::vector<uint64_t>& values,
        const std::vector<FieldElement256>& blindings,
        size_t n,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Generate ballot validity proof
     * 
     * Proves that vote is in range [0, num_candidates)
     * 
     * @param vote Vote value (0 to num_candidates-1)
     * @param num_candidates Number of valid choices
     * @param gens Generator points
     * @return Ballot validity proof
     */
    BallotValidityProof prove_ballot_validity(
        uint64_t vote,
        size_t num_candidates,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Batch generate ballot validity proofs
     * 
     * @param votes Vector of votes
     * @param num_candidates Number of valid choices
     * @param gens Generator points
     * @return Vector of ballot validity proofs
     */
    std::vector<BallotValidityProof> prove_ballot_validity_batch(
        const std::vector<uint64_t>& votes,
        size_t num_candidates,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Get configuration
     */
    const BulletproofsConfig& config() const { return config_; }
    
private:
    BulletproofsConfig config_;
    const EllipticCurve256& curve_;
    const Field256& field_;
    
    /**
     * Generate inner product proof
     */
    InnerProductProof prove_inner_product(
        const std::vector<FieldElement256>& a,
        const std::vector<FieldElement256>& b,
        const std::vector<AffinePoint256>& g_vec,
        const std::vector<AffinePoint256>& h_vec,
        const AffinePoint256& U,
        Transcript& transcript) const;
    
    /**
     * Compute vector Pedersen commitment
     * P = sum(a_i * g_i) + sum(b_i * h_i)
     */
    JacobianPoint256 vector_commit(
        const std::vector<FieldElement256>& a,
        const std::vector<FieldElement256>& b,
        const std::vector<AffinePoint256>& g_vec,
        const std::vector<AffinePoint256>& h_vec) const;
    
    /**
     * Compute delta(y, z) for range proof
     */
    FieldElement256 compute_delta(
        const FieldElement256& y,
        const FieldElement256& z,
        size_t n) const;
    
    /**
     * Generate random blinding factor
     */
    FieldElement256 random_scalar() const;
    
    /**
     * Compute powers of a scalar: [1, x, x^2, ..., x^(n-1)]
     */
    std::vector<FieldElement256> powers_of(const FieldElement256& x, size_t n) const;
    
    /**
     * Compute bit decomposition of value
     */
    std::vector<FieldElement256> bit_decompose(uint64_t value, size_t n) const;
};

// ============================================================================
// Bulletproofs Verifier
// ============================================================================

/**
 * Bulletproofs verifier for range proofs
 * 
 * Verifies range proofs for ballot validity.
 * Supports batch verification for efficiency.
 */
class BulletproofsVerifier {
public:
    /**
     * Construct verifier with given configuration
     */
    explicit BulletproofsVerifier(const BulletproofsConfig& config = BulletproofsConfig());
    
    /**
     * Verify range proof for single value
     * 
     * @param commitment Pedersen commitment to value
     * @param proof Range proof
     * @param n Number of bits (range is [0, 2^n))
     * @param gens Generator points
     * @return true if proof is valid
     */
    bool verify_range(
        const PedersenCommitment& commitment,
        const RangeProof& proof,
        size_t n,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Verify aggregated range proof
     * 
     * @param commitments Pedersen commitments to values
     * @param proof Aggregated range proof
     * @param n Number of bits per value
     * @param gens Generator points
     * @return true if proof is valid
     */
    bool verify_range_aggregated(
        const std::vector<PedersenCommitment>& commitments,
        const AggregatedRangeProof& proof,
        size_t n,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Verify ballot validity proof
     * 
     * @param proof Ballot validity proof
     * @param gens Generator points
     * @return true if proof is valid
     */
    bool verify_ballot_validity(
        const BallotValidityProof& proof,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Batch verify multiple range proofs
     * 
     * More efficient than verifying individually due to
     * batched multi-scalar multiplication.
     * 
     * @param commitments Vector of commitments
     * @param proofs Vector of range proofs
     * @param n Number of bits
     * @param gens Generator points
     * @return true if all proofs are valid
     */
    bool batch_verify_range(
        const std::vector<PedersenCommitment>& commitments,
        const std::vector<RangeProof>& proofs,
        size_t n,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Batch verify multiple ballot validity proofs
     * 
     * @param proofs Vector of ballot validity proofs
     * @param gens Generator points
     * @return true if all proofs are valid
     */
    bool batch_verify_ballot_validity(
        const std::vector<BallotValidityProof>& proofs,
        const BulletproofsGenerators& gens) const;
    
    /**
     * Get configuration
     */
    const BulletproofsConfig& config() const { return config_; }
    
private:
    BulletproofsConfig config_;
    const EllipticCurve256& curve_;
    const Field256& field_;
    
    /**
     * Verify inner product proof
     */
    bool verify_inner_product(
        const JacobianPoint256& P,
        const FieldElement256& c,
        const InnerProductProof& proof,
        const std::vector<AffinePoint256>& g_vec,
        const std::vector<AffinePoint256>& h_vec,
        const AffinePoint256& U,
        Transcript& transcript) const;
    
    /**
     * Compute delta(y, z) for verification
     */
    FieldElement256 compute_delta(
        const FieldElement256& y,
        const FieldElement256& z,
        size_t n) const;
    
    /**
     * Compute powers of a scalar
     */
    std::vector<FieldElement256> powers_of(const FieldElement256& x, size_t n) const;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compute minimum bits needed to represent a range
 * 
 * @param max_value Maximum value in range (exclusive)
 * @return Number of bits needed
 */
size_t bits_needed(uint64_t max_value);

/**
 * Generate default generators for given bit size
 */
BulletproofsGenerators default_generators(size_t n);

/**
 * Verify that a value is in the valid range for given bit size
 */
bool value_in_range(uint64_t value, size_t n);

} // namespace zk
} // namespace fhe_accelerate
