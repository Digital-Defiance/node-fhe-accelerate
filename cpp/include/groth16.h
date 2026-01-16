/**
 * Groth16 Zero-Knowledge Proof System
 * 
 * Implements Groth16 zkSNARK for eligibility proofs:
 * - R1CS constraint system
 * - Trusted setup ceremony
 * - Proof generation with Metal GPU acceleration
 * - Pairing-based verification
 * 
 * Based on "On the Size of Pairing-based Non-interactive Arguments"
 * by Jens Groth (EUROCRYPT 2016)
 * 
 * Requirements: 19.2, 19.6, 19.8, 19.9, 20.2, 20.6
 */

#pragma once

#include "zk_field_arithmetic.h"
#include "zk_elliptic_curve.h"
#include "zk_hash.h"
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include <string>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// R1CS Constraint System
// ============================================================================

/**
 * Sparse vector representation for R1CS matrices
 * Stores (index, value) pairs for non-zero entries
 */
struct SparseVector {
    std::vector<std::pair<size_t, FieldElement256>> entries;
    
    SparseVector() = default;
    
    void add_term(size_t index, const FieldElement256& coeff);
    void add_term(size_t index, int64_t coeff, const Field256& field);
    
    FieldElement256 evaluate(const std::vector<FieldElement256>& witness,
                             const Field256& field) const;
    
    bool is_empty() const { return entries.empty(); }
    size_t num_terms() const { return entries.size(); }
};

/**
 * R1CS constraint: A * B = C
 * where A, B, C are linear combinations of witness variables
 */
struct R1CSConstraint {
    SparseVector a;  // Left input
    SparseVector b;  // Right input
    SparseVector c;  // Output
    
    R1CSConstraint() = default;
    R1CSConstraint(SparseVector a_, SparseVector b_, SparseVector c_)
        : a(std::move(a_)), b(std::move(b_)), c(std::move(c_)) {}
    
    /**
     * Check if constraint is satisfied by witness
     */
    bool is_satisfied(const std::vector<FieldElement256>& witness,
                      const Field256& field) const;
};

/**
 * R1CS constraint system
 * 
 * Represents a computation as a set of quadratic constraints:
 * For each constraint i: <A_i, w> * <B_i, w> = <C_i, w>
 * where w is the witness vector [1, public_inputs..., private_inputs...]
 */
class R1CS {
public:
    R1CS();
    
    // ========================================================================
    // Variable Management
    // ========================================================================
    
    /**
     * Allocate a new variable
     * @return Variable index
     */
    size_t allocate_variable();
    
    /**
     * Allocate multiple variables
     * @return Starting index
     */
    size_t allocate_variables(size_t count);
    
    /**
     * Mark variable as public input
     */
    void set_public_input(size_t var_index);
    
    /**
     * Get the constant "one" variable (always index 0)
     */
    size_t one() const { return 0; }
    
    /**
     * Get number of variables (including constant one)
     */
    size_t num_variables() const { return num_variables_; }
    
    /**
     * Get number of public inputs
     */
    size_t num_public_inputs() const { return public_inputs_.size(); }
    
    /**
     * Get number of constraints
     */
    size_t num_constraints() const { return constraints_.size(); }
    
    // ========================================================================
    // Constraint Building
    // ========================================================================
    
    /**
     * Add a constraint: A * B = C
     */
    void add_constraint(R1CSConstraint constraint);
    
    /**
     * Add constraint: a * b = c (single variable multiplication)
     */
    void add_multiplication_constraint(size_t a, size_t b, size_t c);
    
    /**
     * Add constraint: a + b = c (addition)
     */
    void add_addition_constraint(size_t a, size_t b, size_t c);
    
    /**
     * Add constraint: a = constant
     */
    void add_constant_constraint(size_t a, const FieldElement256& constant);
    
    /**
     * Add constraint: a = 0 or a = 1 (boolean)
     */
    void add_boolean_constraint(size_t a);
    
    /**
     * Add constraint: if selector = 1 then a = b
     */
    void add_conditional_equality(size_t selector, size_t a, size_t b);
    
    // ========================================================================
    // Witness Generation
    // ========================================================================
    
    /**
     * Create empty witness vector
     */
    std::vector<FieldElement256> create_witness() const;
    
    /**
     * Set witness value
     */
    void set_witness_value(std::vector<FieldElement256>& witness,
                           size_t var_index,
                           const FieldElement256& value) const;
    
    /**
     * Set witness value from integer
     */
    void set_witness_value(std::vector<FieldElement256>& witness,
                           size_t var_index,
                           uint64_t value) const;
    
    /**
     * Get witness value
     */
    const FieldElement256& get_witness_value(
        const std::vector<FieldElement256>& witness,
        size_t var_index) const;
    
    // ========================================================================
    // Verification
    // ========================================================================
    
    /**
     * Check if witness satisfies all constraints
     */
    bool is_satisfied(const std::vector<FieldElement256>& witness) const;
    
    /**
     * Get first unsatisfied constraint (for debugging)
     */
    std::optional<size_t> first_unsatisfied_constraint(
        const std::vector<FieldElement256>& witness) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const std::vector<R1CSConstraint>& constraints() const { return constraints_; }
    const std::vector<size_t>& public_inputs() const { return public_inputs_; }
    const Field256& field() const { return field_; }
    
private:
    std::vector<R1CSConstraint> constraints_;
    std::vector<size_t> public_inputs_;
    size_t num_variables_;
    const Field256& field_;
};

// ============================================================================
// Circuit Builder for Common Patterns
// ============================================================================

/**
 * High-level circuit builder for common ZK patterns
 */
class CircuitBuilder {
public:
    explicit CircuitBuilder(R1CS& r1cs);
    
    // ========================================================================
    // Basic Operations
    // ========================================================================
    
    /**
     * Add two variables: result = a + b
     */
    size_t add(size_t a, size_t b);
    
    /**
     * Subtract: result = a - b
     */
    size_t sub(size_t a, size_t b);
    
    /**
     * Multiply: result = a * b
     */
    size_t mul(size_t a, size_t b);
    
    /**
     * Constant: result = constant
     */
    size_t constant(const FieldElement256& value);
    size_t constant(uint64_t value);
    
    /**
     * Assert equality: a == b
     */
    void assert_equal(size_t a, size_t b);
    
    /**
     * Assert boolean: a ∈ {0, 1}
     */
    void assert_boolean(size_t a);
    
    // ========================================================================
    // Comparison Operations
    // ========================================================================
    
    /**
     * Less than: result = 1 if a < b, else 0
     * Requires bit decomposition
     */
    size_t less_than(size_t a, size_t b, size_t num_bits);
    
    /**
     * Range check: assert 0 <= a < 2^num_bits
     */
    void assert_in_range(size_t a, size_t num_bits);
    
    // ========================================================================
    // Merkle Tree Operations
    // ========================================================================
    
    /**
     * Poseidon hash of two elements
     */
    size_t poseidon_hash2(size_t left, size_t right);
    
    /**
     * Verify Merkle path
     * @param leaf Leaf value
     * @param path_elements Path elements (siblings)
     * @param path_indices Path indices (0 = left, 1 = right)
     * @param root Expected root
     */
    void verify_merkle_path(size_t leaf,
                            const std::vector<size_t>& path_elements,
                            const std::vector<size_t>& path_indices,
                            size_t root);
    
    // ========================================================================
    // Witness Computation Helpers
    // ========================================================================
    
    /**
     * Compute witness for addition
     */
    void compute_add_witness(std::vector<FieldElement256>& witness,
                             size_t result, size_t a, size_t b) const;
    
    /**
     * Compute witness for multiplication
     */
    void compute_mul_witness(std::vector<FieldElement256>& witness,
                             size_t result, size_t a, size_t b) const;
    
    /**
     * Compute witness for Poseidon hash
     */
    void compute_poseidon_witness(std::vector<FieldElement256>& witness,
                                  size_t result, size_t left, size_t right) const;
    
    R1CS& r1cs() { return r1cs_; }
    const R1CS& r1cs() const { return r1cs_; }
    
private:
    R1CS& r1cs_;
    PoseidonHash poseidon_;
};

// ============================================================================
// Eligibility Circuit
// ============================================================================

/**
 * Eligibility proof circuit
 * 
 * Proves that a voter is eligible without revealing identity:
 * - Voter ID is in the eligible voters Merkle tree
 * - Voter has not voted before (nullifier check)
 */
class EligibilityCircuit {
public:
    /**
     * Create eligibility circuit
     * @param tree_depth Depth of the Merkle tree
     */
    explicit EligibilityCircuit(size_t tree_depth);
    
    /**
     * Build the circuit constraints
     */
    void build();
    
    /**
     * Generate witness for a valid eligibility proof
     * 
     * @param voter_id Secret voter identifier
     * @param voter_secret Secret randomness for nullifier
     * @param merkle_path Merkle authentication path
     * @param path_indices Path direction indices
     * @param merkle_root Public Merkle root
     * @return Witness vector
     */
    std::vector<FieldElement256> generate_witness(
        const FieldElement256& voter_id,
        const FieldElement256& voter_secret,
        const std::vector<FieldElement256>& merkle_path,
        const std::vector<bool>& path_indices,
        const FieldElement256& merkle_root) const;
    
    /**
     * Get public inputs from witness
     */
    std::vector<FieldElement256> get_public_inputs(
        const std::vector<FieldElement256>& witness) const;
    
    /**
     * Get the R1CS constraint system
     */
    const R1CS& r1cs() const { return r1cs_; }
    R1CS& r1cs() { return r1cs_; }
    
    /**
     * Get tree depth
     */
    size_t tree_depth() const { return tree_depth_; }
    
    // Variable indices for public inputs
    size_t merkle_root_var() const { return merkle_root_var_; }
    size_t nullifier_var() const { return nullifier_var_; }
    
private:
    R1CS r1cs_;
    size_t tree_depth_;
    
    // Variable indices
    size_t voter_id_var_;
    size_t voter_secret_var_;
    size_t merkle_root_var_;
    size_t nullifier_var_;
    std::vector<size_t> merkle_path_vars_;
    std::vector<size_t> path_index_vars_;
    
    bool built_;
};



// ============================================================================
// Groth16 Proving Key
// ============================================================================

/**
 * Groth16 proving key
 * 
 * Contains the structured reference string (SRS) for proof generation.
 * Generated during trusted setup.
 */
struct Groth16ProvingKey {
    // G1 elements
    AffinePoint256 alpha_g1;                    // [α]_1
    AffinePoint256 beta_g1;                     // [β]_1
    AffinePoint256 delta_g1;                    // [δ]_1
    std::vector<AffinePoint256> a_query;        // [A_i(τ)]_1
    std::vector<AffinePoint256> b_g1_query;     // [B_i(τ)]_1
    std::vector<AffinePoint256> h_query;        // [H_i(τ)]_1 for QAP
    std::vector<AffinePoint256> l_query;        // [L_i(τ)]_1 for private inputs
    
    // G2 elements
    AffinePoint256 beta_g2;                     // [β]_2 (using 256-bit for BN254)
    AffinePoint256 delta_g2;                    // [δ]_2
    std::vector<AffinePoint256> b_g2_query;     // [B_i(τ)]_2
    
    // Metadata
    size_t num_variables;
    size_t num_public_inputs;
    size_t num_constraints;
    
    /**
     * Serialize proving key
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * Deserialize proving key
     */
    static std::optional<Groth16ProvingKey> deserialize(const uint8_t* data, size_t len);
    
    /**
     * Get approximate size in bytes
     */
    size_t size_bytes() const;
};

// ============================================================================
// Groth16 Verification Key
// ============================================================================

/**
 * Groth16 verification key
 * 
 * Compact key for proof verification.
 * Much smaller than proving key.
 */
struct Groth16VerificationKey {
    // G1 elements
    AffinePoint256 alpha_g1;                    // [α]_1
    std::vector<AffinePoint256> ic;             // [IC_i]_1 for public inputs
    
    // G2 elements
    AffinePoint256 beta_g2;                     // [β]_2
    AffinePoint256 gamma_g2;                    // [γ]_2
    AffinePoint256 delta_g2;                    // [δ]_2
    
    // Precomputed pairing values (optional, for faster verification)
    bool has_precomputed;
    // e(α, β) precomputed
    
    /**
     * Serialize verification key
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * Deserialize verification key
     */
    static std::optional<Groth16VerificationKey> deserialize(const uint8_t* data, size_t len);
    
    /**
     * Get size in bytes
     */
    size_t size_bytes() const;
};

// ============================================================================
// Groth16 Proof
// ============================================================================

/**
 * Groth16 proof
 * 
 * Compact proof consisting of 3 group elements.
 * ~200 bytes for BN254.
 */
struct Groth16Proof {
    AffinePoint256 a;    // [A]_1
    AffinePoint256 b;    // [B]_2 (stored as G1 point for BN254 G2)
    AffinePoint256 c;    // [C]_1
    
    /**
     * Serialize proof
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * Deserialize proof
     */
    static std::optional<Groth16Proof> deserialize(const uint8_t* data, size_t len);
    
    /**
     * Get size in bytes (~192-200 bytes)
     */
    size_t size_bytes() const { return 3 * 64; }  // 3 points, 64 bytes each
};

// ============================================================================
// Groth16 Setup
// ============================================================================

/**
 * Groth16 trusted setup
 * 
 * Generates proving and verification keys from R1CS.
 * In production, this should be done via MPC ceremony.
 */
class Groth16Setup {
public:
    /**
     * Generate keys from R1CS (for testing only - uses random toxic waste)
     * 
     * @param r1cs The constraint system
     * @return Pair of (proving_key, verification_key)
     */
    static std::pair<Groth16ProvingKey, Groth16VerificationKey>
    generate_keys(const R1CS& r1cs);
    
    /**
     * Generate keys with specific toxic waste (for deterministic testing)
     */
    static std::pair<Groth16ProvingKey, Groth16VerificationKey>
    generate_keys_deterministic(const R1CS& r1cs,
                                const FieldElement256& tau,
                                const FieldElement256& alpha,
                                const FieldElement256& beta,
                                const FieldElement256& gamma,
                                const FieldElement256& delta);
    
private:
    /**
     * Compute QAP polynomials from R1CS
     */
    static void compute_qap(const R1CS& r1cs,
                            std::vector<std::vector<FieldElement256>>& a_polys,
                            std::vector<std::vector<FieldElement256>>& b_polys,
                            std::vector<std::vector<FieldElement256>>& c_polys,
                            std::vector<FieldElement256>& t_poly);
    
    /**
     * Evaluate polynomial at point
     */
    static FieldElement256 evaluate_poly(const std::vector<FieldElement256>& poly,
                                          const FieldElement256& point,
                                          const Field256& field);
};

// ============================================================================
// Groth16 Prover
// ============================================================================

/**
 * Groth16 prover
 * 
 * Generates proofs using the proving key.
 * Optimized with Metal GPU for MSM operations.
 */
class Groth16Prover {
public:
    /**
     * Construct prover with proving key
     */
    explicit Groth16Prover(const Groth16ProvingKey& pk, bool use_gpu = true);
    
    /**
     * Generate proof
     * 
     * @param witness Full witness vector [1, public_inputs..., private_inputs...]
     * @return Groth16 proof
     */
    Groth16Proof prove(const std::vector<FieldElement256>& witness) const;
    
    /**
     * Generate proof with explicit randomness (for testing)
     */
    Groth16Proof prove_with_randomness(const std::vector<FieldElement256>& witness,
                                        const FieldElement256& r,
                                        const FieldElement256& s) const;
    
    /**
     * Batch prove multiple witnesses
     */
    std::vector<Groth16Proof> prove_batch(
        const std::vector<std::vector<FieldElement256>>& witnesses) const;
    
    /**
     * Get proving key
     */
    const Groth16ProvingKey& proving_key() const { return pk_; }
    
private:
    const Groth16ProvingKey& pk_;
    const EllipticCurve256& curve_;
    const Field256& field_;
    bool use_gpu_;
    
    /**
     * Compute h(x) coefficients for QAP
     */
    std::vector<FieldElement256> compute_h_coefficients(
        const std::vector<FieldElement256>& witness) const;
};

// ============================================================================
// Groth16 Verifier
// ============================================================================

/**
 * Groth16 verifier
 * 
 * Verifies proofs using pairing checks.
 * Target: <1ms verification time.
 */
class Groth16Verifier {
public:
    /**
     * Construct verifier with verification key
     */
    explicit Groth16Verifier(const Groth16VerificationKey& vk);
    
    /**
     * Verify proof
     * 
     * @param proof The Groth16 proof
     * @param public_inputs Public input values
     * @return true if proof is valid
     */
    bool verify(const Groth16Proof& proof,
                const std::vector<FieldElement256>& public_inputs) const;
    
    /**
     * Batch verify multiple proofs
     * More efficient than individual verification.
     */
    bool batch_verify(const std::vector<Groth16Proof>& proofs,
                      const std::vector<std::vector<FieldElement256>>& public_inputs) const;
    
    /**
     * Get verification key
     */
    const Groth16VerificationKey& verification_key() const { return vk_; }
    
private:
    const Groth16VerificationKey& vk_;
    const EllipticCurve256& curve_;
    const Field256& field_;
    
    /**
     * Compute pairing check
     * e(A, B) = e(α, β) * e(IC, γ) * e(C, δ)
     */
    bool pairing_check(const Groth16Proof& proof,
                       const JacobianPoint256& ic_sum) const;
    
    /**
     * Simplified pairing for BN254 (placeholder)
     * Real implementation would use optimal ate pairing
     */
    bool compute_pairing(const AffinePoint256& p1, const AffinePoint256& p2,
                         const AffinePoint256& q1, const AffinePoint256& q2) const;
};

// ============================================================================
// Eligibility Proof (High-Level API)
// ============================================================================

/**
 * Complete eligibility proof system
 * 
 * Combines circuit, setup, proving, and verification.
 */
class EligibilityProofSystem {
public:
    /**
     * Create eligibility proof system
     * @param tree_depth Merkle tree depth
     */
    explicit EligibilityProofSystem(size_t tree_depth);
    
    /**
     * Run trusted setup
     */
    void setup();
    
    /**
     * Generate eligibility proof
     * 
     * @param voter_id Secret voter identifier
     * @param voter_secret Secret randomness
     * @param merkle_path Authentication path
     * @param path_indices Path directions
     * @param merkle_root Public root
     * @return Proof and public inputs
     */
    std::pair<Groth16Proof, std::vector<FieldElement256>> prove(
        const FieldElement256& voter_id,
        const FieldElement256& voter_secret,
        const std::vector<FieldElement256>& merkle_path,
        const std::vector<bool>& path_indices,
        const FieldElement256& merkle_root) const;
    
    /**
     * Verify eligibility proof
     * 
     * @param proof The proof
     * @param merkle_root Expected Merkle root
     * @param nullifier Expected nullifier
     * @return true if valid
     */
    bool verify(const Groth16Proof& proof,
                const FieldElement256& merkle_root,
                const FieldElement256& nullifier) const;
    
    /**
     * Compute nullifier from voter ID and secret
     */
    FieldElement256 compute_nullifier(const FieldElement256& voter_id,
                                       const FieldElement256& voter_secret) const;
    
    /**
     * Compute leaf commitment from voter ID
     */
    FieldElement256 compute_leaf(const FieldElement256& voter_id) const;
    
    /**
     * Get verification key for external verification
     */
    const Groth16VerificationKey& verification_key() const;
    
    /**
     * Check if setup has been run
     */
    bool is_setup() const { return setup_complete_; }
    
private:
    EligibilityCircuit circuit_;
    std::unique_ptr<Groth16ProvingKey> pk_;
    std::unique_ptr<Groth16VerificationKey> vk_;
    std::unique_ptr<Groth16Prover> prover_;
    std::unique_ptr<Groth16Verifier> verifier_;
    PoseidonHash poseidon_;
    bool setup_complete_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compute Merkle root from leaves
 */
FieldElement256 compute_merkle_root(const std::vector<FieldElement256>& leaves);

/**
 * Compute Merkle path for a leaf
 */
std::pair<std::vector<FieldElement256>, std::vector<bool>>
compute_merkle_path(const std::vector<FieldElement256>& leaves, size_t leaf_index);

} // namespace zk
} // namespace fhe_accelerate
