/**
 * PLONK Zero-Knowledge Proof System
 * 
 * Implements PLONK zkSNARK for tally correctness proofs:
 * - Constraint system with custom gates for FHE operations
 * - Universal trusted setup with KZG commitments
 * - Proof generation with Metal GPU acceleration
 * - KZG opening verification
 * 
 * Based on "PLONK: Permutations over Lagrange-bases for Oecumenical
 * Noninteractive arguments of Knowledge" by Gabizon, Williamson, Ciobotaru
 * 
 * Requirements: 19.4, 19.6, 19.8, 19.9, 20, 20.3
 */

#pragma once

#include "zk_field_arithmetic.h"
#include "zk_elliptic_curve.h"
#include "zk_polynomial_commitment.h"
#include "zk_hash.h"
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include <string>
#include <functional>

namespace fhe_accelerate {
namespace zk {

// Forward declarations
class PLONKConstraintSystem;
class PLONKProver;
class PLONKVerifier;

// ============================================================================
// PLONK Gate Types
// ============================================================================

/**
 * Gate type enumeration for PLONK custom gates
 */
enum class PLONKGateType {
    ARITHMETIC,      // Standard arithmetic gate: qL*a + qR*b + qO*c + qM*a*b + qC = 0
    MULTIPLICATION,  // Multiplication gate: a * b = c
    ADDITION,        // Addition gate: a + b = c
    CONSTANT,        // Constant gate: a = constant
    BOOLEAN,         // Boolean constraint: a * (1 - a) = 0
    RANGE,           // Range check gate
    POSEIDON,        // Poseidon hash round gate
    FHE_ADD,         // FHE homomorphic addition gate
    FHE_MUL,         // FHE homomorphic multiplication gate
    FHE_NOISE,       // FHE noise budget check gate
    TALLY_SUM,       // Tally summation gate
    CUSTOM           // User-defined custom gate
};

/**
 * Wire indices for a PLONK gate
 */
struct PLONKWires {
    size_t a;  // Left input wire
    size_t b;  // Right input wire
    size_t c;  // Output wire
    
    PLONKWires() : a(0), b(0), c(0) {}
    PLONKWires(size_t a_, size_t b_, size_t c_) : a(a_), b(b_), c(c_) {}
};

/**
 * Selector coefficients for arithmetic gate
 * Gate equation: qL*a + qR*b + qO*c + qM*a*b + qC = 0
 */
struct PLONKSelectors {
    FieldElement256 qL;  // Left wire selector
    FieldElement256 qR;  // Right wire selector
    FieldElement256 qO;  // Output wire selector
    FieldElement256 qM;  // Multiplication selector
    FieldElement256 qC;  // Constant selector
    
    PLONKSelectors();
    
    // Factory methods for common gate types
    static PLONKSelectors addition(const Field256& field);
    static PLONKSelectors multiplication(const Field256& field);
    static PLONKSelectors constant(const FieldElement256& value, const Field256& field);
    static PLONKSelectors boolean_constraint(const Field256& field);
};

// ============================================================================
// PLONK Gate
// ============================================================================

/**
 * PLONK gate representation
 */
struct PLONKGate {
    PLONKGateType type;
    PLONKWires wires;
    PLONKSelectors selectors;
    
    // Optional custom gate data
    std::vector<FieldElement256> custom_data;
    
    PLONKGate() : type(PLONKGateType::ARITHMETIC) {}
    PLONKGate(PLONKGateType t, const PLONKWires& w, const PLONKSelectors& s)
        : type(t), wires(w), selectors(s) {}
    
    /**
     * Check if gate is satisfied by given wire values
     */
    bool is_satisfied(const FieldElement256& a_val,
                      const FieldElement256& b_val,
                      const FieldElement256& c_val,
                      const Field256& field) const;
};

// ============================================================================
// Copy Constraint (Permutation)
// ============================================================================

/**
 * Copy constraint representing wire equality
 * Used for permutation argument in PLONK
 */
struct CopyConstraint {
    size_t gate1;
    size_t wire1;  // 0=a, 1=b, 2=c
    size_t gate2;
    size_t wire2;
    
    CopyConstraint(size_t g1, size_t w1, size_t g2, size_t w2)
        : gate1(g1), wire1(w1), gate2(g2), wire2(w2) {}
};

// ============================================================================
// PLONK Constraint System
// ============================================================================

/**
 * PLONK constraint system
 * 
 * Represents a computation as a sequence of gates with copy constraints.
 * Supports variable-size circuits and custom gates for FHE operations.
 */
class PLONKConstraintSystem {
public:
    PLONKConstraintSystem();
    
    // ========================================================================
    // Variable Management
    // ========================================================================
    
    /**
     * Allocate a new variable (wire)
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
     * Get number of variables
     */
    size_t num_variables() const { return num_variables_; }
    
    /**
     * Get number of public inputs
     */
    size_t num_public_inputs() const { return public_inputs_.size(); }
    
    /**
     * Get number of gates
     */
    size_t num_gates() const { return gates_.size(); }
    
    // ========================================================================
    // Gate Building
    // ========================================================================
    
    /**
     * Add a generic arithmetic gate
     */
    void add_gate(const PLONKGate& gate);
    
    /**
     * Add multiplication gate: a * b = c
     */
    void add_multiplication_gate(size_t a, size_t b, size_t c);
    
    /**
     * Add addition gate: a + b = c
     */
    void add_addition_gate(size_t a, size_t b, size_t c);
    
    /**
     * Add constant gate: a = constant
     */
    void add_constant_gate(size_t a, const FieldElement256& constant);
    
    /**
     * Add boolean constraint: a * (1 - a) = 0
     */
    void add_boolean_gate(size_t a);
    
    /**
     * Add copy constraint (wire equality)
     */
    void add_copy_constraint(size_t var1, size_t var2);

    // ========================================================================
    // Custom Gates for FHE Operations
    // ========================================================================
    
    /**
     * Add FHE addition gate
     * Verifies homomorphic addition was performed correctly
     */
    void add_fhe_addition_gate(size_t ct1, size_t ct2, size_t result);
    
    /**
     * Add FHE multiplication gate
     * Verifies homomorphic multiplication was performed correctly
     */
    void add_fhe_multiplication_gate(size_t ct1, size_t ct2, size_t result);
    
    /**
     * Add tally summation gate
     * Verifies encrypted vote was added to tally correctly
     */
    void add_tally_sum_gate(size_t vote, size_t prev_tally, size_t new_tally);
    
    /**
     * Add Poseidon hash gate
     * Verifies Poseidon hash computation
     */
    void add_poseidon_gate(size_t input1, size_t input2, size_t output);
    
    /**
     * Add range check gate
     * Verifies value is in range [0, 2^bits)
     */
    void add_range_gate(size_t value, size_t bits);
    
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
     * Check if witness satisfies all gates
     */
    bool is_satisfied(const std::vector<FieldElement256>& witness) const;
    
    /**
     * Get first unsatisfied gate (for debugging)
     */
    std::optional<size_t> first_unsatisfied_gate(
        const std::vector<FieldElement256>& witness) const;

    // ========================================================================
    // Polynomial Representation
    // ========================================================================
    
    /**
     * Compute selector polynomials
     * Returns polynomials qL, qR, qO, qM, qC in coefficient form
     */
    void compute_selector_polynomials(
        Polynomial256& qL, Polynomial256& qR,
        Polynomial256& qO, Polynomial256& qM,
        Polynomial256& qC) const;
    
    /**
     * Compute permutation polynomials
     * Returns sigma1, sigma2, sigma3 for copy constraints
     */
    void compute_permutation_polynomials(
        Polynomial256& sigma1, Polynomial256& sigma2,
        Polynomial256& sigma3) const;
    
    /**
     * Compute wire polynomials from witness
     */
    void compute_wire_polynomials(
        const std::vector<FieldElement256>& witness,
        Polynomial256& a_poly, Polynomial256& b_poly,
        Polynomial256& c_poly) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const std::vector<PLONKGate>& gates() const { return gates_; }
    const std::vector<size_t>& public_inputs() const { return public_inputs_; }
    const std::vector<CopyConstraint>& copy_constraints() const { return copy_constraints_; }
    const Field256& field() const { return field_; }
    
    /**
     * Get domain size (next power of 2 >= num_gates)
     */
    size_t domain_size() const;
    
    /**
     * Get domain generator (primitive root of unity)
     */
    FieldElement256 domain_generator() const;
    
private:
    std::vector<PLONKGate> gates_;
    std::vector<size_t> public_inputs_;
    std::vector<CopyConstraint> copy_constraints_;
    std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> wire_to_gates_;
    size_t num_variables_;
    const Field256& field_;
    
    /**
     * Build permutation from copy constraints
     */
    void build_permutation(std::vector<size_t>& perm1,
                           std::vector<size_t>& perm2,
                           std::vector<size_t>& perm3) const;
};

// ============================================================================
// PLONK Circuit Builder
// ============================================================================

/**
 * High-level circuit builder for PLONK
 */
class PLONKCircuitBuilder {
public:
    explicit PLONKCircuitBuilder(PLONKConstraintSystem& cs);
    
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
    // FHE Operations
    // ========================================================================
    
    /**
     * FHE addition: result = ct1 + ct2 (homomorphic)
     */
    size_t fhe_add(size_t ct1, size_t ct2);
    
    /**
     * FHE multiplication: result = ct1 * ct2 (homomorphic)
     */
    size_t fhe_mul(size_t ct1, size_t ct2);
    
    /**
     * Tally sum: new_tally = prev_tally + vote
     */
    size_t tally_sum(size_t vote, size_t prev_tally);
    
    // ========================================================================
    // Hash Operations
    // ========================================================================
    
    /**
     * Poseidon hash of two elements
     */
    size_t poseidon_hash2(size_t left, size_t right);
    
    PLONKConstraintSystem& cs() { return cs_; }
    const PLONKConstraintSystem& cs() const { return cs_; }
    
private:
    PLONKConstraintSystem& cs_;
    PoseidonHash poseidon_;
};

// ============================================================================
// Tally Correctness Circuit
// ============================================================================

/**
 * Tally correctness proof circuit
 * 
 * Proves that the encrypted tally was computed correctly:
 * - Each vote was added to the running tally
 * - Final tally equals sum of all votes
 * - No votes were modified or omitted
 */
class TallyCorrectnessCircuit {
public:
    /**
     * Create tally correctness circuit
     * @param num_votes Number of votes to tally
     */
    explicit TallyCorrectnessCircuit(size_t num_votes);
    
    /**
     * Build the circuit constraints
     */
    void build();
    
    /**
     * Generate witness for a valid tally proof
     * 
     * @param votes Individual vote values (encrypted)
     * @param intermediate_tallies Running tallies after each vote
     * @param final_tally Final tally result
     * @return Witness vector
     */
    std::vector<FieldElement256> generate_witness(
        const std::vector<FieldElement256>& votes,
        const std::vector<FieldElement256>& intermediate_tallies,
        const FieldElement256& final_tally) const;
    
    /**
     * Get public inputs from witness
     */
    std::vector<FieldElement256> get_public_inputs(
        const std::vector<FieldElement256>& witness) const;
    
    /**
     * Get the constraint system
     */
    const PLONKConstraintSystem& constraint_system() const { return cs_; }
    PLONKConstraintSystem& constraint_system() { return cs_; }
    
    /**
     * Get number of votes
     */
    size_t num_votes() const { return num_votes_; }
    
    // Variable indices for public inputs
    size_t final_tally_var() const { return final_tally_var_; }
    size_t initial_tally_var() const { return initial_tally_var_; }
    
private:
    PLONKConstraintSystem cs_;
    size_t num_votes_;
    
    // Variable indices
    std::vector<size_t> vote_vars_;
    std::vector<size_t> tally_vars_;
    size_t initial_tally_var_;
    size_t final_tally_var_;
    
    bool built_;
};

// ============================================================================
// PLONK Setup (Universal Trusted Setup)
// ============================================================================

/**
 * PLONK universal setup parameters
 * 
 * Contains the structured reference string (SRS) for PLONK.
 * Universal: same setup works for any circuit up to max_degree.
 */
struct PLONKSetup {
    KZGSetup256 kzg_setup;           // KZG commitment setup
    size_t max_degree;                // Maximum supported circuit size
    
    /**
     * Serialize setup
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * Deserialize setup
     */
    static std::optional<PLONKSetup> deserialize(const uint8_t* data, size_t len);
    
    /**
     * Get size in bytes
     */
    size_t size_bytes() const;
};

/**
 * PLONK proving key (circuit-specific)
 */
struct PLONKProvingKey {
    // Selector polynomial commitments
    KZGCommitment256 qL_commit;
    KZGCommitment256 qR_commit;
    KZGCommitment256 qO_commit;
    KZGCommitment256 qM_commit;
    KZGCommitment256 qC_commit;
    
    // Permutation polynomial commitments
    KZGCommitment256 sigma1_commit;
    KZGCommitment256 sigma2_commit;
    KZGCommitment256 sigma3_commit;
    
    // Selector polynomials (for prover)
    Polynomial256 qL;
    Polynomial256 qR;
    Polynomial256 qO;
    Polynomial256 qM;
    Polynomial256 qC;
    
    // Permutation polynomials
    Polynomial256 sigma1;
    Polynomial256 sigma2;
    Polynomial256 sigma3;
    
    // Domain info
    size_t domain_size;
    FieldElement256 domain_generator;
    
    // Metadata
    size_t num_public_inputs;
    
    std::vector<uint8_t> serialize() const;
    static std::optional<PLONKProvingKey> deserialize(const uint8_t* data, size_t len);
    size_t size_bytes() const;
};

/**
 * PLONK verification key (circuit-specific)
 */
struct PLONKVerificationKey {
    // Selector polynomial commitments
    KZGCommitment256 qL_commit;
    KZGCommitment256 qR_commit;
    KZGCommitment256 qO_commit;
    KZGCommitment256 qM_commit;
    KZGCommitment256 qC_commit;
    
    // Permutation polynomial commitments
    KZGCommitment256 sigma1_commit;
    KZGCommitment256 sigma2_commit;
    KZGCommitment256 sigma3_commit;
    
    // Domain info
    size_t domain_size;
    FieldElement256 domain_generator;
    
    // Metadata
    size_t num_public_inputs;
    
    std::vector<uint8_t> serialize() const;
    static std::optional<PLONKVerificationKey> deserialize(const uint8_t* data, size_t len);
    size_t size_bytes() const;
};

// ============================================================================
// PLONK Proof
// ============================================================================

/**
 * PLONK proof
 * 
 * Contains commitments and evaluations for verification.
 * Target size: ~400 bytes
 */
struct PLONKProof {
    // Round 1: Wire commitments
    KZGCommitment256 a_commit;
    KZGCommitment256 b_commit;
    KZGCommitment256 c_commit;
    
    // Round 2: Permutation commitment
    KZGCommitment256 z_commit;
    
    // Round 3: Quotient commitments
    KZGCommitment256 t_lo_commit;
    KZGCommitment256 t_mid_commit;
    KZGCommitment256 t_hi_commit;
    
    // Round 4: Opening evaluations at zeta
    FieldElement256 a_eval;
    FieldElement256 b_eval;
    FieldElement256 c_eval;
    FieldElement256 sigma1_eval;
    FieldElement256 sigma2_eval;
    FieldElement256 z_omega_eval;
    
    // Round 5: Opening proofs
    KZGCommitment256 W_zeta;
    KZGCommitment256 W_zeta_omega;
    
    /**
     * Serialize proof
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * Deserialize proof
     */
    static std::optional<PLONKProof> deserialize(const uint8_t* data, size_t len);
    
    /**
     * Get size in bytes (~400 bytes)
     */
    size_t size_bytes() const;
};

// ============================================================================
// PLONK Setup Generator
// ============================================================================

/**
 * PLONK setup generator
 * 
 * Generates universal and circuit-specific keys.
 */
class PLONKSetupGenerator {
public:
    /**
     * Generate universal setup (for testing - uses random toxic waste)
     * 
     * @param max_degree Maximum polynomial degree to support
     * @return Universal setup parameters
     */
    static PLONKSetup generate_universal_setup(size_t max_degree);
    
    /**
     * Generate universal setup with specific tau (for deterministic testing)
     */
    static PLONKSetup generate_universal_setup_deterministic(
        size_t max_degree, const FieldElement256& tau);
    
    /**
     * Generate circuit-specific keys from universal setup
     * 
     * @param setup Universal setup
     * @param cs Constraint system
     * @return Pair of (proving_key, verification_key)
     */
    static std::pair<PLONKProvingKey, PLONKVerificationKey>
    generate_circuit_keys(const PLONKSetup& setup, const PLONKConstraintSystem& cs);
};

// ============================================================================
// PLONK Prover
// ============================================================================

/**
 * PLONK prover
 * 
 * Generates proofs using the proving key.
 * Optimized with Metal GPU for FFT and MSM operations.
 */
class PLONKProver {
public:
    /**
     * Construct prover with setup and proving key
     */
    PLONKProver(const PLONKSetup& setup, const PLONKProvingKey& pk, bool use_gpu = true);
    
    /**
     * Generate proof
     * 
     * @param witness Full witness vector
     * @param public_inputs Public input values
     * @return PLONK proof
     */
    PLONKProof prove(const std::vector<FieldElement256>& witness,
                     const std::vector<FieldElement256>& public_inputs) const;
    
    /**
     * Generate proof with explicit randomness (for testing)
     */
    PLONKProof prove_with_randomness(
        const std::vector<FieldElement256>& witness,
        const std::vector<FieldElement256>& public_inputs,
        const FieldElement256& b1, const FieldElement256& b2,
        const FieldElement256& b3, const FieldElement256& b4,
        const FieldElement256& b5, const FieldElement256& b6) const;
    
    /**
     * Get proving key
     */
    const PLONKProvingKey& proving_key() const { return pk_; }
    
private:
    const PLONKSetup& setup_;
    const PLONKProvingKey& pk_;
    std::unique_ptr<KZGScheme256> kzg_;
    const Field256& field_;
    bool use_gpu_;
    
    // Round computations
    void round1_wire_commitments(
        const std::vector<FieldElement256>& witness,
        const FieldElement256& b1, const FieldElement256& b2,
        const FieldElement256& b3, const FieldElement256& b4,
        const FieldElement256& b5, const FieldElement256& b6,
        Polynomial256& a_poly, Polynomial256& b_poly, Polynomial256& c_poly,
        KZGCommitment256& a_commit, KZGCommitment256& b_commit,
        KZGCommitment256& c_commit) const;
    
    void round2_permutation(
        const Polynomial256& a_poly, const Polynomial256& b_poly,
        const Polynomial256& c_poly,
        const FieldElement256& beta, const FieldElement256& gamma,
        Polynomial256& z_poly, KZGCommitment256& z_commit) const;
    
    void round3_quotient(
        const Polynomial256& a_poly, const Polynomial256& b_poly,
        const Polynomial256& c_poly, const Polynomial256& z_poly,
        const FieldElement256& alpha, const FieldElement256& beta,
        const FieldElement256& gamma,
        const std::vector<FieldElement256>& public_inputs,
        Polynomial256& t_poly,
        KZGCommitment256& t_lo, KZGCommitment256& t_mid,
        KZGCommitment256& t_hi) const;
    
    void round4_evaluations(
        const Polynomial256& a_poly, const Polynomial256& b_poly,
        const Polynomial256& c_poly, const Polynomial256& z_poly,
        const FieldElement256& zeta,
        FieldElement256& a_eval, FieldElement256& b_eval,
        FieldElement256& c_eval, FieldElement256& sigma1_eval,
        FieldElement256& sigma2_eval, FieldElement256& z_omega_eval) const;
    
    void round5_openings(
        const Polynomial256& a_poly, const Polynomial256& b_poly,
        const Polynomial256& c_poly, const Polynomial256& z_poly,
        const Polynomial256& t_poly,
        const FieldElement256& zeta, const FieldElement256& v,
        KZGCommitment256& W_zeta, KZGCommitment256& W_zeta_omega) const;
};

// ============================================================================
// PLONK Verifier
// ============================================================================

/**
 * PLONK verifier
 * 
 * Verifies proofs using KZG opening verification.
 * Target: <5ms verification time.
 */
class PLONKVerifier {
public:
    /**
     * Construct verifier with setup and verification key
     */
    PLONKVerifier(const PLONKSetup& setup, const PLONKVerificationKey& vk);
    
    /**
     * Verify proof
     * 
     * @param proof The PLONK proof
     * @param public_inputs Public input values
     * @return true if proof is valid
     */
    bool verify(const PLONKProof& proof,
                const std::vector<FieldElement256>& public_inputs) const;
    
    /**
     * Batch verify multiple proofs
     */
    bool batch_verify(const std::vector<PLONKProof>& proofs,
                      const std::vector<std::vector<FieldElement256>>& public_inputs) const;
    
    /**
     * Get verification key
     */
    const PLONKVerificationKey& verification_key() const { return vk_; }
    
private:
    const PLONKSetup& setup_;
    const PLONKVerificationKey& vk_;
    std::unique_ptr<KZGScheme256> kzg_;
    const Field256& field_;
    
    /**
     * Compute public input polynomial evaluation
     */
    FieldElement256 compute_public_input_eval(
        const std::vector<FieldElement256>& public_inputs,
        const FieldElement256& zeta) const;
    
    /**
     * Verify KZG opening
     */
    bool verify_opening(const KZGCommitment256& commitment,
                        const FieldElement256& point,
                        const FieldElement256& value,
                        const KZGCommitment256& proof) const;
};

// ============================================================================
// Tally Proof System (High-Level API)
// ============================================================================

/**
 * Complete tally correctness proof system
 * 
 * Combines circuit, setup, proving, and verification.
 */
class TallyProofSystem {
public:
    /**
     * Create tally proof system
     * @param num_votes Number of votes to tally
     */
    explicit TallyProofSystem(size_t num_votes);
    
    /**
     * Run trusted setup
     */
    void setup();
    
    /**
     * Generate tally correctness proof
     * 
     * @param votes Individual vote values
     * @param intermediate_tallies Running tallies
     * @param final_tally Final tally result
     * @return Proof and public inputs
     */
    std::pair<PLONKProof, std::vector<FieldElement256>> prove(
        const std::vector<FieldElement256>& votes,
        const std::vector<FieldElement256>& intermediate_tallies,
        const FieldElement256& final_tally) const;
    
    /**
     * Verify tally correctness proof
     * 
     * @param proof The proof
     * @param initial_tally Initial tally value (usually 0)
     * @param final_tally Expected final tally
     * @return true if valid
     */
    bool verify(const PLONKProof& proof,
                const FieldElement256& initial_tally,
                const FieldElement256& final_tally) const;
    
    /**
     * Get verification key for external verification
     */
    const PLONKVerificationKey& verification_key() const;
    
    /**
     * Check if setup has been run
     */
    bool is_setup() const { return setup_complete_; }
    
private:
    TallyCorrectnessCircuit circuit_;
    std::unique_ptr<PLONKSetup> universal_setup_;
    std::unique_ptr<PLONKProvingKey> pk_;
    std::unique_ptr<PLONKVerificationKey> vk_;
    std::unique_ptr<PLONKProver> prover_;
    std::unique_ptr<PLONKVerifier> verifier_;
    bool setup_complete_;
};

} // namespace zk
} // namespace fhe_accelerate
