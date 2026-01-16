/**
 * Zero-Knowledge Polynomial Commitment Schemes
 * 
 * Implements polynomial commitment schemes for ZK proof systems:
 * - KZG (Kate-Zaverucha-Goldberg) commitments for PLONK
 * - FRI (Fast Reed-Solomon IOP) commitments for STARKs
 * 
 * Reuses NTT infrastructure for FFT operations.
 * Uses Metal GPU for parallel commitment computation.
 * 
 * Requirements: 19, 20.3
 */

#pragma once

#include "zk_field_arithmetic.h"
#include "zk_elliptic_curve.h"
#include <vector>
#include <memory>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Polynomial Representation
// ============================================================================

/**
 * Polynomial over a finite field
 * Coefficients stored in coefficient form: p(x) = sum(coeffs[i] * x^i)
 */
template<typename FieldElement>
struct Polynomial {
    std::vector<FieldElement> coeffs;
    
    Polynomial() = default;
    explicit Polynomial(size_t degree) : coeffs(degree + 1) {}
    explicit Polynomial(const std::vector<FieldElement>& c) : coeffs(c) {}
    
    size_t degree() const { return coeffs.empty() ? 0 : coeffs.size() - 1; }
    bool is_zero() const;
    
    // Evaluate polynomial at point x
    template<typename Field>
    FieldElement evaluate(const FieldElement& x, const Field& field) const;
};

using Polynomial256 = Polynomial<FieldElement256>;
using Polynomial384 = Polynomial<FieldElement384>;

// ============================================================================
// KZG Commitment Scheme
// ============================================================================

/**
 * KZG Structured Reference String (SRS)
 * 
 * Contains powers of tau: [tau^0]_1, [tau^1]_1, ..., [tau^n]_1
 * and optionally [tau^0]_2, [tau^1]_2 for verification
 */
template<typename AffinePoint>
struct KZGSetup {
    std::vector<AffinePoint> powers_of_tau_g1;  // [tau^i]_1
    std::vector<AffinePoint> powers_of_tau_g2;  // [tau^i]_2 (for verification)
    size_t max_degree;
    
    KZGSetup() : max_degree(0) {}
};

using KZGSetup256 = KZGSetup<AffinePoint256>;
using KZGSetup384 = KZGSetup<AffinePoint384>;

/**
 * KZG Commitment (a single elliptic curve point)
 */
template<typename AffinePoint>
struct KZGCommitment {
    AffinePoint point;
    
    KZGCommitment() = default;
    explicit KZGCommitment(const AffinePoint& p) : point(p) {}
    
    bool operator==(const KZGCommitment& other) const { return point == other.point; }
};

using KZGCommitment256 = KZGCommitment<AffinePoint256>;
using KZGCommitment384 = KZGCommitment<AffinePoint384>;

/**
 * KZG Opening Proof
 * Proves that p(z) = y for committed polynomial p
 */
template<typename AffinePoint, typename FieldElement>
struct KZGProof {
    AffinePoint quotient;  // [q(tau)]_1 where q(x) = (p(x) - y) / (x - z)
    FieldElement point;    // z - the evaluation point
    FieldElement value;    // y = p(z)
    
    KZGProof() = default;
};

using KZGProof256 = KZGProof<AffinePoint256, FieldElement256>;
using KZGProof384 = KZGProof<AffinePoint384, FieldElement256>;

/**
 * KZG Polynomial Commitment Scheme for BN254
 */
class KZGScheme256 {
public:
    /**
     * Construct KZG scheme with given setup
     */
    explicit KZGScheme256(const KZGSetup256& setup);
    
    /**
     * Generate trusted setup (for testing only - real setup needs MPC)
     * 
     * @param max_degree Maximum polynomial degree to support
     * @param tau Secret value (should be destroyed after setup)
     */
    static KZGSetup256 generate_setup(size_t max_degree, const FieldElement256& tau);
    
    /**
     * Commit to a polynomial
     * 
     * @param poly Polynomial to commit (coefficients in Montgomery form)
     * @return Commitment point
     */
    KZGCommitment256 commit(const Polynomial256& poly) const;
    
    /**
     * GPU-accelerated commitment using Metal MSM
     */
    KZGCommitment256 commit_gpu(const Polynomial256& poly) const;
    
    /**
     * Create opening proof for polynomial at point z
     * 
     * @param poly The committed polynomial
     * @param z Evaluation point
     * @return Opening proof
     */
    KZGProof256 open(const Polynomial256& poly, const FieldElement256& z) const;
    
    /**
     * Verify opening proof
     * 
     * @param commitment The polynomial commitment
     * @param proof The opening proof
     * @return true if proof is valid
     */
    bool verify(const KZGCommitment256& commitment, const KZGProof256& proof) const;
    
    /**
     * Batch verify multiple opening proofs
     */
    bool batch_verify(const std::vector<KZGCommitment256>& commitments,
                      const std::vector<KZGProof256>& proofs) const;
    
    /**
     * Get maximum supported polynomial degree
     */
    size_t max_degree() const { return setup_.max_degree; }
    
private:
    KZGSetup256 setup_;
    const EllipticCurve256& curve_;
    const Field256& scalar_field_;
    const Field256& base_field_;
    
    // Compute quotient polynomial q(x) = (p(x) - y) / (x - z)
    Polynomial256 compute_quotient(const Polynomial256& poly,
                                    const FieldElement256& z,
                                    const FieldElement256& y) const;
};

/**
 * KZG Polynomial Commitment Scheme for BLS12-381
 */
class KZGScheme384 {
public:
    explicit KZGScheme384(const KZGSetup384& setup);
    
    static KZGSetup384 generate_setup(size_t max_degree, const FieldElement256& tau);
    
    KZGCommitment384 commit(const Polynomial256& poly) const;
    KZGCommitment384 commit_gpu(const Polynomial256& poly) const;
    
    KZGProof384 open(const Polynomial256& poly, const FieldElement256& z) const;
    bool verify(const KZGCommitment384& commitment, const KZGProof384& proof) const;
    bool batch_verify(const std::vector<KZGCommitment384>& commitments,
                      const std::vector<KZGProof384>& proofs) const;
    
    size_t max_degree() const { return setup_.max_degree; }
    
private:
    KZGSetup384 setup_;
    const EllipticCurve384& curve_;
    const Field256& scalar_field_;
    const Field384& base_field_;
    
    Polynomial256 compute_quotient(const Polynomial256& poly,
                                    const FieldElement256& z,
                                    const FieldElement256& y) const;
};

// ============================================================================
// FRI Commitment Scheme (for STARKs)
// ============================================================================

/**
 * FRI (Fast Reed-Solomon IOP) Configuration
 */
struct FRIConfig {
    size_t domain_size;           // Size of evaluation domain (power of 2)
    size_t num_queries;           // Number of query rounds
    size_t folding_factor;        // Folding factor per round (typically 2 or 4)
    size_t final_poly_degree;     // Degree of final polynomial
    
    FRIConfig(size_t domain = 1024, size_t queries = 30, 
              size_t folding = 2, size_t final_deg = 1)
        : domain_size(domain), num_queries(queries),
          folding_factor(folding), final_poly_degree(final_deg) {}
};

/**
 * Merkle tree node for FRI commitments
 */
struct MerkleNode {
    std::array<uint8_t, 32> hash;
    
    MerkleNode() : hash{} {}
    explicit MerkleNode(const std::array<uint8_t, 32>& h) : hash(h) {}
    
    bool operator==(const MerkleNode& other) const { return hash == other.hash; }
};

/**
 * FRI Commitment (Merkle root of polynomial evaluations)
 */
struct FRICommitment {
    MerkleNode root;
    size_t domain_size;
    
    FRICommitment() : domain_size(0) {}
};

/**
 * FRI Query Response
 */
struct FRIQueryResponse {
    std::vector<FieldElement256> values;           // Queried values
    std::vector<std::vector<MerkleNode>> paths;    // Merkle authentication paths
};

/**
 * FRI Proof
 */
struct FRIProof {
    std::vector<FRICommitment> layer_commitments;  // Commitments for each folding layer
    std::vector<FRIQueryResponse> query_responses; // Responses for each query
    Polynomial256 final_polynomial;                // Final low-degree polynomial
};

/**
 * FRI Polynomial Commitment Scheme
 * 
 * Uses FFT for polynomial evaluation and Merkle trees for commitments.
 */
class FRIScheme {
public:
    /**
     * Construct FRI scheme with given configuration
     */
    explicit FRIScheme(const FRIConfig& config);
    
    /**
     * Commit to a polynomial
     * 
     * @param poly Polynomial to commit
     * @return FRI commitment (Merkle root)
     */
    FRICommitment commit(const Polynomial256& poly) const;
    
    /**
     * GPU-accelerated commitment using Metal for FFT
     */
    FRICommitment commit_gpu(const Polynomial256& poly) const;
    
    /**
     * Create FRI proof for polynomial
     * 
     * @param poly The committed polynomial
     * @param commitment The commitment
     * @return FRI proof
     */
    FRIProof prove(const Polynomial256& poly, const FRICommitment& commitment) const;
    
    /**
     * Verify FRI proof
     * 
     * @param commitment The polynomial commitment
     * @param proof The FRI proof
     * @return true if proof is valid
     */
    bool verify(const FRICommitment& commitment, const FRIProof& proof) const;
    
    /**
     * Get configuration
     */
    const FRIConfig& config() const { return config_; }
    
private:
    FRIConfig config_;
    const Field256& field_;
    
    // Domain generator (primitive root of unity)
    FieldElement256 domain_generator_;
    
    // Precomputed domain elements
    std::vector<FieldElement256> domain_;
    
    // FFT operations (reusing NTT infrastructure)
    std::vector<FieldElement256> fft(const std::vector<FieldElement256>& coeffs) const;
    std::vector<FieldElement256> ifft(const std::vector<FieldElement256>& evals) const;
    
    // Merkle tree operations
    MerkleNode build_merkle_tree(const std::vector<FieldElement256>& leaves,
                                  std::vector<std::vector<MerkleNode>>& tree) const;
    std::vector<MerkleNode> get_merkle_path(const std::vector<std::vector<MerkleNode>>& tree,
                                             size_t index) const;
    bool verify_merkle_path(const MerkleNode& root, const FieldElement256& leaf,
                            size_t index, const std::vector<MerkleNode>& path) const;
    
    // FRI folding
    std::vector<FieldElement256> fold_evaluations(const std::vector<FieldElement256>& evals,
                                                   const FieldElement256& alpha) const;
    
    // Hash function for Merkle tree
    MerkleNode hash_field_elements(const FieldElement256* elems, size_t count) const;
    MerkleNode hash_nodes(const MerkleNode& left, const MerkleNode& right) const;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Polynomial arithmetic operations
 */
template<typename FieldElement, typename Field>
Polynomial<FieldElement> poly_add(const Polynomial<FieldElement>& a,
                                   const Polynomial<FieldElement>& b,
                                   const Field& field);

template<typename FieldElement, typename Field>
Polynomial<FieldElement> poly_sub(const Polynomial<FieldElement>& a,
                                   const Polynomial<FieldElement>& b,
                                   const Field& field);

template<typename FieldElement, typename Field>
Polynomial<FieldElement> poly_mul(const Polynomial<FieldElement>& a,
                                   const Polynomial<FieldElement>& b,
                                   const Field& field);

/**
 * Polynomial division: returns (quotient, remainder)
 */
template<typename FieldElement, typename Field>
std::pair<Polynomial<FieldElement>, Polynomial<FieldElement>>
poly_div(const Polynomial<FieldElement>& dividend,
         const Polynomial<FieldElement>& divisor,
         const Field& field);

/**
 * Generate random polynomial of given degree
 */
Polynomial256 random_polynomial_256(size_t degree, const Field256& field);

} // namespace zk
} // namespace fhe_accelerate
