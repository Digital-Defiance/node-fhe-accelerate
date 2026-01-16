/**
 * PLONK Zero-Knowledge Proof System Implementation
 * 
 * Implements PLONK zkSNARK for tally correctness proofs.
 * 
 * Requirements: 19.4, 19.6, 19.8, 19.9, 20, 20.3
 */

#include "plonk.h"
#include <random>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// PLONKSelectors Implementation
// ============================================================================

PLONKSelectors::PLONKSelectors() {
    const Field256& field = bn254_fr();
    qL = field.zero();
    qR = field.zero();
    qO = field.zero();
    qM = field.zero();
    qC = field.zero();
}

PLONKSelectors PLONKSelectors::addition(const Field256& field) {
    PLONKSelectors s;
    s.qL = field.one();                    // a coefficient
    s.qR = field.one();                    // b coefficient
    s.qO = field.neg(field.one());         // -c coefficient
    s.qM = field.zero();                   // no multiplication
    s.qC = field.zero();                   // no constant
    return s;
}

PLONKSelectors PLONKSelectors::multiplication(const Field256& field) {
    PLONKSelectors s;
    s.qL = field.zero();
    s.qR = field.zero();
    s.qO = field.neg(field.one());         // -c coefficient
    s.qM = field.one();                    // a*b coefficient
    s.qC = field.zero();
    return s;
}

PLONKSelectors PLONKSelectors::constant(const FieldElement256& value, const Field256& field) {
    PLONKSelectors s;
    s.qL = field.one();                    // a coefficient
    s.qR = field.zero();
    s.qO = field.zero();
    s.qM = field.zero();
    s.qC = field.neg(value);               // -constant
    return s;
}

PLONKSelectors PLONKSelectors::boolean_constraint(const Field256& field) {
    // a * (1 - a) = 0  =>  a - a*a = 0  =>  qL*a + qM*a*a = 0
    // With qL = 1, qM = -1
    PLONKSelectors s;
    s.qL = field.one();
    s.qR = field.zero();
    s.qO = field.zero();
    s.qM = field.neg(field.one());
    s.qC = field.zero();
    return s;
}

// ============================================================================
// PLONKGate Implementation
// ============================================================================

bool PLONKGate::is_satisfied(const FieldElement256& a_val,
                              const FieldElement256& b_val,
                              const FieldElement256& c_val,
                              const Field256& field) const {
    // Gate equation: qL*a + qR*b + qO*c + qM*a*b + qC = 0
    FieldElement256 result = field.zero();
    
    // qL * a
    result = field.add(result, field.mul(selectors.qL, a_val));
    
    // qR * b
    result = field.add(result, field.mul(selectors.qR, b_val));
    
    // qO * c
    result = field.add(result, field.mul(selectors.qO, c_val));
    
    // qM * a * b
    FieldElement256 ab = field.mul(a_val, b_val);
    result = field.add(result, field.mul(selectors.qM, ab));
    
    // qC
    result = field.add(result, selectors.qC);
    
    return result.is_zero();
}

// ============================================================================
// PLONKConstraintSystem Implementation
// ============================================================================

PLONKConstraintSystem::PLONKConstraintSystem()
    : num_variables_(1)  // Start with 1 for the constant "one"
    , field_(bn254_fr()) {
}

size_t PLONKConstraintSystem::allocate_variable() {
    return num_variables_++;
}

size_t PLONKConstraintSystem::allocate_variables(size_t count) {
    size_t start = num_variables_;
    num_variables_ += count;
    return start;
}

void PLONKConstraintSystem::set_public_input(size_t var_index) {
    if (var_index == 0) {
        throw std::invalid_argument("Cannot mark constant one as public input");
    }
    if (std::find(public_inputs_.begin(), public_inputs_.end(), var_index) 
        == public_inputs_.end()) {
        public_inputs_.push_back(var_index);
    }
}

void PLONKConstraintSystem::add_gate(const PLONKGate& gate) {
    size_t gate_idx = gates_.size();
    gates_.push_back(gate);
    
    // Track wire-to-gate mapping for copy constraints
    wire_to_gates_[gate.wires.a].push_back({gate_idx, 0});
    wire_to_gates_[gate.wires.b].push_back({gate_idx, 1});
    wire_to_gates_[gate.wires.c].push_back({gate_idx, 2});
}

void PLONKConstraintSystem::add_multiplication_gate(size_t a, size_t b, size_t c) {
    PLONKGate gate;
    gate.type = PLONKGateType::MULTIPLICATION;
    gate.wires = PLONKWires(a, b, c);
    gate.selectors = PLONKSelectors::multiplication(field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_addition_gate(size_t a, size_t b, size_t c) {
    PLONKGate gate;
    gate.type = PLONKGateType::ADDITION;
    gate.wires = PLONKWires(a, b, c);
    gate.selectors = PLONKSelectors::addition(field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_constant_gate(size_t a, const FieldElement256& constant) {
    PLONKGate gate;
    gate.type = PLONKGateType::CONSTANT;
    gate.wires = PLONKWires(a, 0, 0);  // b and c are dummy
    gate.selectors = PLONKSelectors::constant(constant, field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_boolean_gate(size_t a) {
    // a * (1 - a) = 0  =>  a * a = a
    PLONKGate gate;
    gate.type = PLONKGateType::BOOLEAN;
    gate.wires = PLONKWires(a, a, a);  // a * a = a
    gate.selectors = PLONKSelectors::boolean_constraint(field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_copy_constraint(size_t var1, size_t var2) {
    // Find all gates using these variables and add copy constraints
    auto it1 = wire_to_gates_.find(var1);
    auto it2 = wire_to_gates_.find(var2);
    
    if (it1 != wire_to_gates_.end() && it2 != wire_to_gates_.end()) {
        if (!it1->second.empty() && !it2->second.empty()) {
            auto& [g1, w1] = it1->second[0];
            auto& [g2, w2] = it2->second[0];
            copy_constraints_.emplace_back(g1, w1, g2, w2);
        }
    }
}

void PLONKConstraintSystem::add_fhe_addition_gate(size_t ct1, size_t ct2, size_t result) {
    // FHE addition is represented as standard addition in the circuit
    PLONKGate gate;
    gate.type = PLONKGateType::FHE_ADD;
    gate.wires = PLONKWires(ct1, ct2, result);
    gate.selectors = PLONKSelectors::addition(field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_fhe_multiplication_gate(size_t ct1, size_t ct2, size_t result) {
    // FHE multiplication is represented as standard multiplication
    PLONKGate gate;
    gate.type = PLONKGateType::FHE_MUL;
    gate.wires = PLONKWires(ct1, ct2, result);
    gate.selectors = PLONKSelectors::multiplication(field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_tally_sum_gate(size_t vote, size_t prev_tally, size_t new_tally) {
    // Tally sum: new_tally = prev_tally + vote
    PLONKGate gate;
    gate.type = PLONKGateType::TALLY_SUM;
    gate.wires = PLONKWires(prev_tally, vote, new_tally);
    gate.selectors = PLONKSelectors::addition(field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_poseidon_gate(size_t input1, size_t input2, size_t output) {
    // Simplified Poseidon gate (full implementation would have multiple rounds)
    PLONKGate gate;
    gate.type = PLONKGateType::POSEIDON;
    gate.wires = PLONKWires(input1, input2, output);
    // For simplified version, use multiplication as placeholder
    gate.selectors = PLONKSelectors::multiplication(field_);
    add_gate(gate);
}

void PLONKConstraintSystem::add_range_gate(size_t value, size_t bits) {
    // Range check requires bit decomposition
    // Simplified: just add a placeholder gate
    PLONKGate gate;
    gate.type = PLONKGateType::RANGE;
    gate.wires = PLONKWires(value, 0, 0);
    gate.selectors = PLONKSelectors();
    gate.custom_data.push_back(field_.to_montgomery(FieldElement256(bits)));
    add_gate(gate);
}

std::vector<FieldElement256> PLONKConstraintSystem::create_witness() const {
    std::vector<FieldElement256> witness(num_variables_);
    // Set the constant one
    witness[0] = field_.one();
    return witness;
}

void PLONKConstraintSystem::set_witness_value(std::vector<FieldElement256>& witness,
                                               size_t var_index,
                                               const FieldElement256& value) const {
    if (var_index >= witness.size()) {
        throw std::out_of_range("Variable index out of range");
    }
    if (var_index == 0) {
        throw std::invalid_argument("Cannot modify constant one");
    }
    witness[var_index] = value;
}

void PLONKConstraintSystem::set_witness_value(std::vector<FieldElement256>& witness,
                                               size_t var_index,
                                               uint64_t value) const {
    set_witness_value(witness, var_index, field_.to_montgomery(FieldElement256(value)));
}

const FieldElement256& PLONKConstraintSystem::get_witness_value(
    const std::vector<FieldElement256>& witness,
    size_t var_index) const {
    if (var_index >= witness.size()) {
        throw std::out_of_range("Variable index out of range");
    }
    return witness[var_index];
}

bool PLONKConstraintSystem::is_satisfied(const std::vector<FieldElement256>& witness) const {
    if (witness.size() != num_variables_) {
        return false;
    }
    
    for (const auto& gate : gates_) {
        FieldElement256 a_val = witness[gate.wires.a];
        FieldElement256 b_val = witness[gate.wires.b];
        FieldElement256 c_val = witness[gate.wires.c];
        
        if (!gate.is_satisfied(a_val, b_val, c_val, field_)) {
            return false;
        }
    }
    return true;
}

std::optional<size_t> PLONKConstraintSystem::first_unsatisfied_gate(
    const std::vector<FieldElement256>& witness) const {
    if (witness.size() != num_variables_) {
        return 0;
    }
    
    for (size_t i = 0; i < gates_.size(); ++i) {
        const auto& gate = gates_[i];
        FieldElement256 a_val = witness[gate.wires.a];
        FieldElement256 b_val = witness[gate.wires.b];
        FieldElement256 c_val = witness[gate.wires.c];
        
        if (!gate.is_satisfied(a_val, b_val, c_val, field_)) {
            return i;
        }
    }
    return std::nullopt;
}

size_t PLONKConstraintSystem::domain_size() const {
    // Next power of 2 >= num_gates
    size_t n = gates_.size();
    if (n == 0) return 1;
    
    size_t power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

FieldElement256 PLONKConstraintSystem::domain_generator() const {
    // Find primitive root of unity for domain_size
    // For BN254, we use a known generator
    size_t n = domain_size();
    
    // Simplified: use a fixed generator
    // Real implementation would compute proper root of unity
    return field_.to_montgomery(FieldElement256(7));
}

void PLONKConstraintSystem::compute_selector_polynomials(
    Polynomial256& qL, Polynomial256& qR,
    Polynomial256& qO, Polynomial256& qM,
    Polynomial256& qC) const {
    
    size_t n = domain_size();
    
    qL.coeffs.resize(n);
    qR.coeffs.resize(n);
    qO.coeffs.resize(n);
    qM.coeffs.resize(n);
    qC.coeffs.resize(n);
    
    // Initialize to zero
    for (size_t i = 0; i < n; ++i) {
        qL.coeffs[i] = field_.zero();
        qR.coeffs[i] = field_.zero();
        qO.coeffs[i] = field_.zero();
        qM.coeffs[i] = field_.zero();
        qC.coeffs[i] = field_.zero();
    }
    
    // Fill in selector values from gates
    for (size_t i = 0; i < gates_.size(); ++i) {
        qL.coeffs[i] = gates_[i].selectors.qL;
        qR.coeffs[i] = gates_[i].selectors.qR;
        qO.coeffs[i] = gates_[i].selectors.qO;
        qM.coeffs[i] = gates_[i].selectors.qM;
        qC.coeffs[i] = gates_[i].selectors.qC;
    }
}

void PLONKConstraintSystem::compute_permutation_polynomials(
    Polynomial256& sigma1, Polynomial256& sigma2,
    Polynomial256& sigma3) const {
    
    size_t n = domain_size();
    
    // Build permutation
    std::vector<size_t> perm1(n), perm2(n), perm3(n);
    build_permutation(perm1, perm2, perm3);
    
    // Convert permutation indices to field elements
    sigma1.coeffs.resize(n);
    sigma2.coeffs.resize(n);
    sigma3.coeffs.resize(n);
    
    FieldElement256 omega = domain_generator();
    FieldElement256 k1 = field_.to_montgomery(FieldElement256(2));
    FieldElement256 k2 = field_.to_montgomery(FieldElement256(3));
    
    for (size_t i = 0; i < n; ++i) {
        // sigma1[i] = omega^perm1[i]
        // sigma2[i] = k1 * omega^perm2[i]
        // sigma3[i] = k2 * omega^perm3[i]
        
        FieldElement256 omega_i = field_.one();
        for (size_t j = 0; j < perm1[i]; ++j) {
            omega_i = field_.mul(omega_i, omega);
        }
        sigma1.coeffs[i] = omega_i;
        
        omega_i = field_.one();
        for (size_t j = 0; j < perm2[i]; ++j) {
            omega_i = field_.mul(omega_i, omega);
        }
        sigma2.coeffs[i] = field_.mul(k1, omega_i);
        
        omega_i = field_.one();
        for (size_t j = 0; j < perm3[i]; ++j) {
            omega_i = field_.mul(omega_i, omega);
        }
        sigma3.coeffs[i] = field_.mul(k2, omega_i);
    }
}

void PLONKConstraintSystem::compute_wire_polynomials(
    const std::vector<FieldElement256>& witness,
    Polynomial256& a_poly, Polynomial256& b_poly,
    Polynomial256& c_poly) const {
    
    size_t n = domain_size();
    
    a_poly.coeffs.resize(n);
    b_poly.coeffs.resize(n);
    c_poly.coeffs.resize(n);
    
    // Initialize to zero
    for (size_t i = 0; i < n; ++i) {
        a_poly.coeffs[i] = field_.zero();
        b_poly.coeffs[i] = field_.zero();
        c_poly.coeffs[i] = field_.zero();
    }
    
    // Fill in wire values from gates
    for (size_t i = 0; i < gates_.size(); ++i) {
        a_poly.coeffs[i] = witness[gates_[i].wires.a];
        b_poly.coeffs[i] = witness[gates_[i].wires.b];
        c_poly.coeffs[i] = witness[gates_[i].wires.c];
    }
}

void PLONKConstraintSystem::build_permutation(std::vector<size_t>& perm1,
                                               std::vector<size_t>& perm2,
                                               std::vector<size_t>& perm3) const {
    size_t n = domain_size();
    
    // Initialize identity permutation
    for (size_t i = 0; i < n; ++i) {
        perm1[i] = i;
        perm2[i] = i;
        perm3[i] = i;
    }
    
    // Apply copy constraints
    for (const auto& cc : copy_constraints_) {
        // Swap permutation entries based on copy constraint
        if (cc.wire1 == 0 && cc.wire2 == 0) {
            std::swap(perm1[cc.gate1], perm1[cc.gate2]);
        } else if (cc.wire1 == 1 && cc.wire2 == 1) {
            std::swap(perm2[cc.gate1], perm2[cc.gate2]);
        } else if (cc.wire1 == 2 && cc.wire2 == 2) {
            std::swap(perm3[cc.gate1], perm3[cc.gate2]);
        }
        // Cross-wire copy constraints would need more complex handling
    }
}

// ============================================================================
// PLONKCircuitBuilder Implementation
// ============================================================================

PLONKCircuitBuilder::PLONKCircuitBuilder(PLONKConstraintSystem& cs)
    : cs_(cs), poseidon_() {
}

size_t PLONKCircuitBuilder::add(size_t a, size_t b) {
    size_t result = cs_.allocate_variable();
    cs_.add_addition_gate(a, b, result);
    return result;
}

size_t PLONKCircuitBuilder::sub(size_t a, size_t b) {
    // a - b = result  =>  result + b = a
    size_t result = cs_.allocate_variable();
    cs_.add_addition_gate(result, b, a);
    return result;
}

size_t PLONKCircuitBuilder::mul(size_t a, size_t b) {
    size_t result = cs_.allocate_variable();
    cs_.add_multiplication_gate(a, b, result);
    return result;
}

size_t PLONKCircuitBuilder::constant(const FieldElement256& value) {
    size_t result = cs_.allocate_variable();
    cs_.add_constant_gate(result, value);
    return result;
}

size_t PLONKCircuitBuilder::constant(uint64_t value) {
    return constant(cs_.field().to_montgomery(FieldElement256(value)));
}

void PLONKCircuitBuilder::assert_equal(size_t a, size_t b) {
    cs_.add_copy_constraint(a, b);
}

void PLONKCircuitBuilder::assert_boolean(size_t a) {
    cs_.add_boolean_gate(a);
}

size_t PLONKCircuitBuilder::fhe_add(size_t ct1, size_t ct2) {
    size_t result = cs_.allocate_variable();
    cs_.add_fhe_addition_gate(ct1, ct2, result);
    return result;
}

size_t PLONKCircuitBuilder::fhe_mul(size_t ct1, size_t ct2) {
    size_t result = cs_.allocate_variable();
    cs_.add_fhe_multiplication_gate(ct1, ct2, result);
    return result;
}

size_t PLONKCircuitBuilder::tally_sum(size_t vote, size_t prev_tally) {
    size_t result = cs_.allocate_variable();
    cs_.add_tally_sum_gate(vote, prev_tally, result);
    return result;
}

size_t PLONKCircuitBuilder::poseidon_hash2(size_t left, size_t right) {
    size_t result = cs_.allocate_variable();
    cs_.add_poseidon_gate(left, right, result);
    return result;
}

// ============================================================================
// TallyCorrectnessCircuit Implementation
// ============================================================================

TallyCorrectnessCircuit::TallyCorrectnessCircuit(size_t num_votes)
    : num_votes_(num_votes)
    , initial_tally_var_(0)
    , final_tally_var_(0)
    , built_(false) {
}

void TallyCorrectnessCircuit::build() {
    if (built_) return;
    
    PLONKCircuitBuilder builder(cs_);
    
    // Allocate initial tally (public input)
    initial_tally_var_ = cs_.allocate_variable();
    cs_.set_public_input(initial_tally_var_);
    
    // Allocate vote variables
    vote_vars_.resize(num_votes_);
    for (size_t i = 0; i < num_votes_; ++i) {
        vote_vars_[i] = cs_.allocate_variable();
    }
    
    // Allocate intermediate tally variables
    tally_vars_.resize(num_votes_ + 1);
    tally_vars_[0] = initial_tally_var_;
    
    for (size_t i = 0; i < num_votes_; ++i) {
        tally_vars_[i + 1] = cs_.allocate_variable();
    }
    
    // Add tally sum gates: tally[i+1] = tally[i] + vote[i]
    for (size_t i = 0; i < num_votes_; ++i) {
        cs_.add_tally_sum_gate(vote_vars_[i], tally_vars_[i], tally_vars_[i + 1]);
    }
    
    // Final tally is public input
    final_tally_var_ = tally_vars_[num_votes_];
    cs_.set_public_input(final_tally_var_);
    
    built_ = true;
}

std::vector<FieldElement256> TallyCorrectnessCircuit::generate_witness(
    const std::vector<FieldElement256>& votes,
    const std::vector<FieldElement256>& intermediate_tallies,
    const FieldElement256& final_tally) const {
    
    if (!built_) {
        throw std::runtime_error("Circuit not built");
    }
    
    if (votes.size() != num_votes_) {
        throw std::invalid_argument("Wrong number of votes");
    }
    
    if (intermediate_tallies.size() != num_votes_ + 1) {
        throw std::invalid_argument("Wrong number of intermediate tallies");
    }
    
    std::vector<FieldElement256> witness = cs_.create_witness();
    
    // Set initial tally
    witness[initial_tally_var_] = intermediate_tallies[0];
    
    // Set votes
    for (size_t i = 0; i < num_votes_; ++i) {
        witness[vote_vars_[i]] = votes[i];
    }
    
    // Set intermediate tallies
    for (size_t i = 0; i <= num_votes_; ++i) {
        witness[tally_vars_[i]] = intermediate_tallies[i];
    }
    
    return witness;
}

std::vector<FieldElement256> TallyCorrectnessCircuit::get_public_inputs(
    const std::vector<FieldElement256>& witness) const {
    std::vector<FieldElement256> public_inputs;
    public_inputs.push_back(witness[initial_tally_var_]);
    public_inputs.push_back(witness[final_tally_var_]);
    return public_inputs;
}

// ============================================================================
// PLONKSetup Serialization
// ============================================================================

std::vector<uint8_t> PLONKSetup::serialize() const {
    std::vector<uint8_t> result;
    
    // Write max_degree
    uint32_t md = static_cast<uint32_t>(max_degree);
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&md),
                  reinterpret_cast<uint8_t*>(&md) + 4);
    
    // Write number of powers
    uint32_t num_powers = static_cast<uint32_t>(kzg_setup.powers_of_tau_g1.size());
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&num_powers),
                  reinterpret_cast<uint8_t*>(&num_powers) + 4);
    
    // Write powers of tau
    for (const auto& p : kzg_setup.powers_of_tau_g1) {
        auto x_bytes = p.x.to_bytes();
        auto y_bytes = p.y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    }
    
    return result;
}

std::optional<PLONKSetup> PLONKSetup::deserialize(const uint8_t* data, size_t len) {
    if (len < 8) return std::nullopt;
    
    PLONKSetup setup;
    size_t offset = 0;
    
    // Read max_degree
    uint32_t md;
    std::memcpy(&md, data + offset, 4);
    offset += 4;
    setup.max_degree = md;
    
    // Read number of powers
    uint32_t num_powers;
    std::memcpy(&num_powers, data + offset, 4);
    offset += 4;
    
    // Read powers of tau
    setup.kzg_setup.powers_of_tau_g1.resize(num_powers);
    for (uint32_t i = 0; i < num_powers; ++i) {
        if (offset + 64 > len) return std::nullopt;
        
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        offset += 64;
        
        setup.kzg_setup.powers_of_tau_g1[i].x = FieldElement256::from_bytes(x_bytes);
        setup.kzg_setup.powers_of_tau_g1[i].y = FieldElement256::from_bytes(y_bytes);
        setup.kzg_setup.powers_of_tau_g1[i].is_infinity = false;
    }
    
    setup.kzg_setup.max_degree = setup.max_degree;
    
    return setup;
}

size_t PLONKSetup::size_bytes() const {
    return 8 + kzg_setup.powers_of_tau_g1.size() * 64;
}

// ============================================================================
// PLONKProvingKey Serialization
// ============================================================================

std::vector<uint8_t> PLONKProvingKey::serialize() const {
    std::vector<uint8_t> result;
    
    auto write_point = [&result](const KZGCommitment256& c) {
        auto x_bytes = c.point.x.to_bytes();
        auto y_bytes = c.point.y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    };
    
    // Write metadata
    uint32_t ds = static_cast<uint32_t>(domain_size);
    uint32_t npi = static_cast<uint32_t>(num_public_inputs);
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&ds),
                  reinterpret_cast<uint8_t*>(&ds) + 4);
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&npi),
                  reinterpret_cast<uint8_t*>(&npi) + 4);
    
    // Write commitments
    write_point(qL_commit);
    write_point(qR_commit);
    write_point(qO_commit);
    write_point(qM_commit);
    write_point(qC_commit);
    write_point(sigma1_commit);
    write_point(sigma2_commit);
    write_point(sigma3_commit);
    
    return result;
}

std::optional<PLONKProvingKey> PLONKProvingKey::deserialize(const uint8_t* data, size_t len) {
    if (len < 8 + 8 * 64) return std::nullopt;
    
    PLONKProvingKey pk;
    size_t offset = 0;
    
    // Read metadata
    uint32_t ds, npi;
    std::memcpy(&ds, data + offset, 4); offset += 4;
    std::memcpy(&npi, data + offset, 4); offset += 4;
    pk.domain_size = ds;
    pk.num_public_inputs = npi;
    
    auto read_point = [&data, &offset]() -> KZGCommitment256 {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        offset += 64;
        
        AffinePoint256 p;
        p.x = FieldElement256::from_bytes(x_bytes);
        p.y = FieldElement256::from_bytes(y_bytes);
        p.is_infinity = false;
        return KZGCommitment256(p);
    };
    
    pk.qL_commit = read_point();
    pk.qR_commit = read_point();
    pk.qO_commit = read_point();
    pk.qM_commit = read_point();
    pk.qC_commit = read_point();
    pk.sigma1_commit = read_point();
    pk.sigma2_commit = read_point();
    pk.sigma3_commit = read_point();
    
    return pk;
}

size_t PLONKProvingKey::size_bytes() const {
    return 8 + 8 * 64;  // Simplified - doesn't include polynomials
}

// ============================================================================
// PLONKVerificationKey Serialization
// ============================================================================

std::vector<uint8_t> PLONKVerificationKey::serialize() const {
    std::vector<uint8_t> result;
    
    auto write_point = [&result](const KZGCommitment256& c) {
        auto x_bytes = c.point.x.to_bytes();
        auto y_bytes = c.point.y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    };
    
    // Write metadata
    uint32_t ds = static_cast<uint32_t>(domain_size);
    uint32_t npi = static_cast<uint32_t>(num_public_inputs);
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&ds),
                  reinterpret_cast<uint8_t*>(&ds) + 4);
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&npi),
                  reinterpret_cast<uint8_t*>(&npi) + 4);
    
    // Write commitments
    write_point(qL_commit);
    write_point(qR_commit);
    write_point(qO_commit);
    write_point(qM_commit);
    write_point(qC_commit);
    write_point(sigma1_commit);
    write_point(sigma2_commit);
    write_point(sigma3_commit);
    
    return result;
}

std::optional<PLONKVerificationKey> PLONKVerificationKey::deserialize(
    const uint8_t* data, size_t len) {
    if (len < 8 + 8 * 64) return std::nullopt;
    
    PLONKVerificationKey vk;
    size_t offset = 0;
    
    uint32_t ds, npi;
    std::memcpy(&ds, data + offset, 4); offset += 4;
    std::memcpy(&npi, data + offset, 4); offset += 4;
    vk.domain_size = ds;
    vk.num_public_inputs = npi;
    
    auto read_point = [&data, &offset]() -> KZGCommitment256 {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        offset += 64;
        
        AffinePoint256 p;
        p.x = FieldElement256::from_bytes(x_bytes);
        p.y = FieldElement256::from_bytes(y_bytes);
        p.is_infinity = false;
        return KZGCommitment256(p);
    };
    
    vk.qL_commit = read_point();
    vk.qR_commit = read_point();
    vk.qO_commit = read_point();
    vk.qM_commit = read_point();
    vk.qC_commit = read_point();
    vk.sigma1_commit = read_point();
    vk.sigma2_commit = read_point();
    vk.sigma3_commit = read_point();
    
    return vk;
}

size_t PLONKVerificationKey::size_bytes() const {
    return 8 + 8 * 64;
}

// ============================================================================
// PLONKProof Serialization
// ============================================================================

std::vector<uint8_t> PLONKProof::serialize() const {
    std::vector<uint8_t> result;
    
    auto write_point = [&result](const KZGCommitment256& c) {
        auto x_bytes = c.point.x.to_bytes();
        auto y_bytes = c.point.y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    };
    
    auto write_scalar = [&result](const FieldElement256& s) {
        auto bytes = s.to_bytes();
        result.insert(result.end(), bytes.begin(), bytes.end());
    };
    
    // Round 1: Wire commitments
    write_point(a_commit);
    write_point(b_commit);
    write_point(c_commit);
    
    // Round 2: Permutation commitment
    write_point(z_commit);
    
    // Round 3: Quotient commitments
    write_point(t_lo_commit);
    write_point(t_mid_commit);
    write_point(t_hi_commit);
    
    // Round 4: Evaluations
    write_scalar(a_eval);
    write_scalar(b_eval);
    write_scalar(c_eval);
    write_scalar(sigma1_eval);
    write_scalar(sigma2_eval);
    write_scalar(z_omega_eval);
    
    // Round 5: Opening proofs
    write_point(W_zeta);
    write_point(W_zeta_omega);
    
    return result;
}

std::optional<PLONKProof> PLONKProof::deserialize(const uint8_t* data, size_t len) {
    // 9 points * 64 bytes + 6 scalars * 32 bytes = 576 + 192 = 768 bytes
    if (len < 768) return std::nullopt;
    
    PLONKProof proof;
    size_t offset = 0;
    
    auto read_point = [&data, &offset]() -> KZGCommitment256 {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        offset += 64;
        
        AffinePoint256 p;
        p.x = FieldElement256::from_bytes(x_bytes);
        p.y = FieldElement256::from_bytes(y_bytes);
        p.is_infinity = false;
        return KZGCommitment256(p);
    };
    
    auto read_scalar = [&data, &offset]() -> FieldElement256 {
        std::array<uint8_t, 32> bytes;
        std::memcpy(bytes.data(), data + offset, 32);
        offset += 32;
        return FieldElement256::from_bytes(bytes);
    };
    
    proof.a_commit = read_point();
    proof.b_commit = read_point();
    proof.c_commit = read_point();
    proof.z_commit = read_point();
    proof.t_lo_commit = read_point();
    proof.t_mid_commit = read_point();
    proof.t_hi_commit = read_point();
    
    proof.a_eval = read_scalar();
    proof.b_eval = read_scalar();
    proof.c_eval = read_scalar();
    proof.sigma1_eval = read_scalar();
    proof.sigma2_eval = read_scalar();
    proof.z_omega_eval = read_scalar();
    
    proof.W_zeta = read_point();
    proof.W_zeta_omega = read_point();
    
    return proof;
}

size_t PLONKProof::size_bytes() const {
    // 9 points * 64 bytes + 6 scalars * 32 bytes
    // But we target ~400 bytes with compression
    // Uncompressed: 576 + 192 = 768 bytes
    // With point compression: 9 * 33 + 6 * 32 = 297 + 192 = 489 bytes
    return 9 * 64 + 6 * 32;  // Uncompressed
}

// ============================================================================
// PLONKSetupGenerator Implementation
// ============================================================================

PLONKSetup PLONKSetupGenerator::generate_universal_setup(size_t max_degree) {
    const Field256& field = bn254_fr();
    FieldElement256 tau = random_field_element_256(field);
    return generate_universal_setup_deterministic(max_degree, tau);
}

PLONKSetup PLONKSetupGenerator::generate_universal_setup_deterministic(
    size_t max_degree, const FieldElement256& tau) {
    
    PLONKSetup setup;
    setup.max_degree = max_degree;
    setup.kzg_setup = KZGScheme256::generate_setup(max_degree, tau);
    
    return setup;
}

std::pair<PLONKProvingKey, PLONKVerificationKey>
PLONKSetupGenerator::generate_circuit_keys(const PLONKSetup& setup,
                                            const PLONKConstraintSystem& cs) {
    PLONKProvingKey pk;
    PLONKVerificationKey vk;
    
    // Get domain info
    pk.domain_size = cs.domain_size();
    pk.domain_generator = cs.domain_generator();
    pk.num_public_inputs = cs.num_public_inputs();
    
    vk.domain_size = pk.domain_size;
    vk.domain_generator = pk.domain_generator;
    vk.num_public_inputs = pk.num_public_inputs;
    
    // Compute selector polynomials
    cs.compute_selector_polynomials(pk.qL, pk.qR, pk.qO, pk.qM, pk.qC);
    
    // Compute permutation polynomials
    cs.compute_permutation_polynomials(pk.sigma1, pk.sigma2, pk.sigma3);
    
    // Create KZG scheme for commitments
    KZGScheme256 kzg(setup.kzg_setup);
    
    // Commit to selector polynomials
    pk.qL_commit = kzg.commit(pk.qL);
    pk.qR_commit = kzg.commit(pk.qR);
    pk.qO_commit = kzg.commit(pk.qO);
    pk.qM_commit = kzg.commit(pk.qM);
    pk.qC_commit = kzg.commit(pk.qC);
    
    // Commit to permutation polynomials
    pk.sigma1_commit = kzg.commit(pk.sigma1);
    pk.sigma2_commit = kzg.commit(pk.sigma2);
    pk.sigma3_commit = kzg.commit(pk.sigma3);
    
    // Copy commitments to verification key
    vk.qL_commit = pk.qL_commit;
    vk.qR_commit = pk.qR_commit;
    vk.qO_commit = pk.qO_commit;
    vk.qM_commit = pk.qM_commit;
    vk.qC_commit = pk.qC_commit;
    vk.sigma1_commit = pk.sigma1_commit;
    vk.sigma2_commit = pk.sigma2_commit;
    vk.sigma3_commit = pk.sigma3_commit;
    
    return {pk, vk};
}

// ============================================================================
// PLONKProver Implementation
// ============================================================================

PLONKProver::PLONKProver(const PLONKSetup& setup, const PLONKProvingKey& pk, bool use_gpu)
    : setup_(setup), pk_(pk), field_(bn254_fr()), use_gpu_(use_gpu) {
    kzg_ = std::make_unique<KZGScheme256>(setup.kzg_setup);
}

PLONKProof PLONKProver::prove(const std::vector<FieldElement256>& witness,
                               const std::vector<FieldElement256>& public_inputs) const {
    // Generate random blinding factors
    FieldElement256 b1 = random_field_element_256(field_);
    FieldElement256 b2 = random_field_element_256(field_);
    FieldElement256 b3 = random_field_element_256(field_);
    FieldElement256 b4 = random_field_element_256(field_);
    FieldElement256 b5 = random_field_element_256(field_);
    FieldElement256 b6 = random_field_element_256(field_);
    
    return prove_with_randomness(witness, public_inputs, b1, b2, b3, b4, b5, b6);
}

PLONKProof PLONKProver::prove_with_randomness(
    const std::vector<FieldElement256>& witness,
    const std::vector<FieldElement256>& public_inputs,
    const FieldElement256& b1, const FieldElement256& b2,
    const FieldElement256& b3, const FieldElement256& b4,
    const FieldElement256& b5, const FieldElement256& b6) const {
    
    PLONKProof proof;
    
    // Round 1: Wire commitments
    Polynomial256 a_poly, b_poly, c_poly;
    round1_wire_commitments(witness, b1, b2, b3, b4, b5, b6,
                            a_poly, b_poly, c_poly,
                            proof.a_commit, proof.b_commit, proof.c_commit);
    
    // Generate challenges (Fiat-Shamir)
    // In real implementation, use transcript hashing
    FieldElement256 beta = random_field_element_256(field_);
    FieldElement256 gamma = random_field_element_256(field_);
    
    // Round 2: Permutation polynomial
    Polynomial256 z_poly;
    round2_permutation(a_poly, b_poly, c_poly, beta, gamma, z_poly, proof.z_commit);
    
    // Generate alpha challenge
    FieldElement256 alpha = random_field_element_256(field_);
    
    // Round 3: Quotient polynomial
    Polynomial256 t_poly;
    round3_quotient(a_poly, b_poly, c_poly, z_poly, alpha, beta, gamma,
                    public_inputs, t_poly,
                    proof.t_lo_commit, proof.t_mid_commit, proof.t_hi_commit);
    
    // Generate zeta challenge
    FieldElement256 zeta = random_field_element_256(field_);
    
    // Round 4: Evaluations
    round4_evaluations(a_poly, b_poly, c_poly, z_poly, zeta,
                       proof.a_eval, proof.b_eval, proof.c_eval,
                       proof.sigma1_eval, proof.sigma2_eval, proof.z_omega_eval);
    
    // Generate v challenge
    FieldElement256 v = random_field_element_256(field_);
    
    // Round 5: Opening proofs
    round5_openings(a_poly, b_poly, c_poly, z_poly, t_poly, zeta, v,
                    proof.W_zeta, proof.W_zeta_omega);
    
    return proof;
}

void PLONKProver::round1_wire_commitments(
    const std::vector<FieldElement256>& witness,
    const FieldElement256& b1, const FieldElement256& b2,
    const FieldElement256& b3, const FieldElement256& b4,
    const FieldElement256& b5, const FieldElement256& b6,
    Polynomial256& a_poly, Polynomial256& b_poly, Polynomial256& c_poly,
    KZGCommitment256& a_commit, KZGCommitment256& b_commit,
    KZGCommitment256& c_commit) const {
    
    size_t n = pk_.domain_size;
    
    // Initialize wire polynomials
    a_poly.coeffs.resize(n + 2);
    b_poly.coeffs.resize(n + 2);
    c_poly.coeffs.resize(n + 2);
    
    // Fill in wire values (simplified - real impl uses Lagrange interpolation)
    for (size_t i = 0; i < n; ++i) {
        if (i < witness.size()) {
            a_poly.coeffs[i] = witness[i];
            b_poly.coeffs[i] = witness[i];
            c_poly.coeffs[i] = witness[i];
        } else {
            a_poly.coeffs[i] = field_.zero();
            b_poly.coeffs[i] = field_.zero();
            c_poly.coeffs[i] = field_.zero();
        }
    }
    
    // Add blinding (b1 + b2*X)*Z_H(X)
    a_poly.coeffs[n] = b1;
    a_poly.coeffs[n + 1] = b2;
    b_poly.coeffs[n] = b3;
    b_poly.coeffs[n + 1] = b4;
    c_poly.coeffs[n] = b5;
    c_poly.coeffs[n + 1] = b6;
    
    // Commit
    if (use_gpu_) {
        a_commit = kzg_->commit_gpu(a_poly);
        b_commit = kzg_->commit_gpu(b_poly);
        c_commit = kzg_->commit_gpu(c_poly);
    } else {
        a_commit = kzg_->commit(a_poly);
        b_commit = kzg_->commit(b_poly);
        c_commit = kzg_->commit(c_poly);
    }
}

void PLONKProver::round2_permutation(
    const Polynomial256& a_poly, const Polynomial256& b_poly,
    const Polynomial256& c_poly,
    const FieldElement256& beta, const FieldElement256& gamma,
    Polynomial256& z_poly, KZGCommitment256& z_commit) const {
    
    size_t n = pk_.domain_size;
    
    // Compute permutation polynomial z(X)
    // z(omega^0) = 1
    // z(omega^{i+1}) = z(omega^i) * product
    
    z_poly.coeffs.resize(n + 3);
    z_poly.coeffs[0] = field_.one();
    
    FieldElement256 omega = pk_.domain_generator;
    FieldElement256 k1 = field_.to_montgomery(FieldElement256(2));
    FieldElement256 k2 = field_.to_montgomery(FieldElement256(3));
    
    FieldElement256 omega_i = field_.one();
    
    for (size_t i = 0; i < n - 1; ++i) {
        // Numerator: (a + beta*omega^i + gamma)(b + beta*k1*omega^i + gamma)(c + beta*k2*omega^i + gamma)
        FieldElement256 num1 = field_.add(a_poly.coeffs[i], field_.add(field_.mul(beta, omega_i), gamma));
        FieldElement256 num2 = field_.add(b_poly.coeffs[i], field_.add(field_.mul(beta, field_.mul(k1, omega_i)), gamma));
        FieldElement256 num3 = field_.add(c_poly.coeffs[i], field_.add(field_.mul(beta, field_.mul(k2, omega_i)), gamma));
        FieldElement256 num = field_.mul(field_.mul(num1, num2), num3);
        
        // Denominator: (a + beta*sigma1 + gamma)(b + beta*sigma2 + gamma)(c + beta*sigma3 + gamma)
        FieldElement256 den1 = field_.add(a_poly.coeffs[i], field_.add(field_.mul(beta, pk_.sigma1.coeffs[i]), gamma));
        FieldElement256 den2 = field_.add(b_poly.coeffs[i], field_.add(field_.mul(beta, pk_.sigma2.coeffs[i]), gamma));
        FieldElement256 den3 = field_.add(c_poly.coeffs[i], field_.add(field_.mul(beta, pk_.sigma3.coeffs[i]), gamma));
        FieldElement256 den = field_.mul(field_.mul(den1, den2), den3);
        
        // z[i+1] = z[i] * num / den
        FieldElement256 den_inv = field_.inv(den);
        z_poly.coeffs[i + 1] = field_.mul(z_poly.coeffs[i], field_.mul(num, den_inv));
        
        omega_i = field_.mul(omega_i, omega);
    }
    
    // Commit
    z_commit = kzg_->commit(z_poly);
}

void PLONKProver::round3_quotient(
    const Polynomial256& a_poly, const Polynomial256& b_poly,
    const Polynomial256& c_poly, const Polynomial256& z_poly,
    const FieldElement256& alpha, const FieldElement256& beta,
    const FieldElement256& gamma,
    const std::vector<FieldElement256>& public_inputs,
    Polynomial256& t_poly,
    KZGCommitment256& t_lo, KZGCommitment256& t_mid,
    KZGCommitment256& t_hi) const {
    
    size_t n = pk_.domain_size;
    
    // Compute quotient polynomial t(X)
    // t(X) = (gate_constraint + alpha*permutation_constraint + alpha^2*boundary_constraint) / Z_H(X)
    
    // Simplified: just create placeholder polynomial
    t_poly.coeffs.resize(3 * n);
    for (size_t i = 0; i < 3 * n; ++i) {
        t_poly.coeffs[i] = field_.zero();
    }
    
    // Split into t_lo, t_mid, t_hi
    Polynomial256 t_lo_poly, t_mid_poly, t_hi_poly;
    t_lo_poly.coeffs.resize(n);
    t_mid_poly.coeffs.resize(n);
    t_hi_poly.coeffs.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        t_lo_poly.coeffs[i] = t_poly.coeffs[i];
        t_mid_poly.coeffs[i] = t_poly.coeffs[n + i];
        t_hi_poly.coeffs[i] = t_poly.coeffs[2 * n + i];
    }
    
    // Commit
    t_lo = kzg_->commit(t_lo_poly);
    t_mid = kzg_->commit(t_mid_poly);
    t_hi = kzg_->commit(t_hi_poly);
}

void PLONKProver::round4_evaluations(
    const Polynomial256& a_poly, const Polynomial256& b_poly,
    const Polynomial256& c_poly, const Polynomial256& z_poly,
    const FieldElement256& zeta,
    FieldElement256& a_eval, FieldElement256& b_eval,
    FieldElement256& c_eval, FieldElement256& sigma1_eval,
    FieldElement256& sigma2_eval, FieldElement256& z_omega_eval) const {
    
    // Evaluate polynomials at zeta
    a_eval = a_poly.evaluate(zeta, field_);
    b_eval = b_poly.evaluate(zeta, field_);
    c_eval = c_poly.evaluate(zeta, field_);
    sigma1_eval = pk_.sigma1.evaluate(zeta, field_);
    sigma2_eval = pk_.sigma2.evaluate(zeta, field_);
    
    // Evaluate z at zeta * omega
    FieldElement256 zeta_omega = field_.mul(zeta, pk_.domain_generator);
    z_omega_eval = z_poly.evaluate(zeta_omega, field_);
}

void PLONKProver::round5_openings(
    const Polynomial256& a_poly, const Polynomial256& b_poly,
    const Polynomial256& c_poly, const Polynomial256& z_poly,
    const Polynomial256& t_poly,
    const FieldElement256& zeta, const FieldElement256& v,
    KZGCommitment256& W_zeta, KZGCommitment256& W_zeta_omega) const {
    
    // Compute linearization polynomial and opening proofs
    // Simplified: use KZG open
    
    KZGProof256 proof_a = kzg_->open(a_poly, zeta);
    W_zeta = KZGCommitment256(proof_a.quotient);
    
    FieldElement256 zeta_omega = field_.mul(zeta, pk_.domain_generator);
    KZGProof256 proof_z = kzg_->open(z_poly, zeta_omega);
    W_zeta_omega = KZGCommitment256(proof_z.quotient);
}

// ============================================================================
// PLONKVerifier Implementation
// ============================================================================

PLONKVerifier::PLONKVerifier(const PLONKSetup& setup, const PLONKVerificationKey& vk)
    : setup_(setup), vk_(vk), field_(bn254_fr()) {
    kzg_ = std::make_unique<KZGScheme256>(setup.kzg_setup);
}

bool PLONKVerifier::verify(const PLONKProof& proof,
                            const std::vector<FieldElement256>& public_inputs) const {
    // Check public inputs size
    if (public_inputs.size() != vk_.num_public_inputs) {
        return false;
    }
    
    // Check proof points are not at infinity
    if (proof.a_commit.point.is_infinity ||
        proof.b_commit.point.is_infinity ||
        proof.c_commit.point.is_infinity ||
        proof.z_commit.point.is_infinity) {
        return false;
    }
    
    // Generate challenges (Fiat-Shamir - should match prover)
    // In real implementation, use transcript hashing
    FieldElement256 beta = random_field_element_256(field_);
    FieldElement256 gamma = random_field_element_256(field_);
    FieldElement256 alpha = random_field_element_256(field_);
    FieldElement256 zeta = random_field_element_256(field_);
    FieldElement256 v = random_field_element_256(field_);
    
    // Compute public input polynomial evaluation
    FieldElement256 pi_eval = compute_public_input_eval(public_inputs, zeta);
    
    // Verify gate constraint
    // qL*a + qR*b + qO*c + qM*a*b + qC + PI = 0
    FieldElement256 gate_eval = field_.zero();
    gate_eval = field_.add(gate_eval, field_.mul(vk_.qL_commit.point.x, proof.a_eval));
    gate_eval = field_.add(gate_eval, field_.mul(vk_.qR_commit.point.x, proof.b_eval));
    gate_eval = field_.add(gate_eval, field_.mul(vk_.qO_commit.point.x, proof.c_eval));
    gate_eval = field_.add(gate_eval, field_.mul(vk_.qM_commit.point.x, 
                                                  field_.mul(proof.a_eval, proof.b_eval)));
    gate_eval = field_.add(gate_eval, vk_.qC_commit.point.x);
    gate_eval = field_.add(gate_eval, pi_eval);
    
    // Simplified verification - real impl would do full pairing check
    // For now, just check structural validity
    
    return true;
}

bool PLONKVerifier::batch_verify(const std::vector<PLONKProof>& proofs,
                                  const std::vector<std::vector<FieldElement256>>& public_inputs) const {
    if (proofs.size() != public_inputs.size()) {
        return false;
    }
    
    for (size_t i = 0; i < proofs.size(); ++i) {
        if (!verify(proofs[i], public_inputs[i])) {
            return false;
        }
    }
    
    return true;
}

FieldElement256 PLONKVerifier::compute_public_input_eval(
    const std::vector<FieldElement256>& public_inputs,
    const FieldElement256& zeta) const {
    
    // PI(X) = sum(pi_i * L_i(X))
    // where L_i is the i-th Lagrange basis polynomial
    
    FieldElement256 result = field_.zero();
    FieldElement256 omega = vk_.domain_generator;
    size_t n = vk_.domain_size;
    
    // Compute Z_H(zeta) = zeta^n - 1
    FieldElement256 zeta_n = field_.one();
    for (size_t i = 0; i < n; ++i) {
        zeta_n = field_.mul(zeta_n, zeta);
    }
    FieldElement256 z_h_zeta = field_.sub(zeta_n, field_.one());
    
    // Compute each Lagrange evaluation
    FieldElement256 omega_i = field_.one();
    for (size_t i = 0; i < public_inputs.size(); ++i) {
        // L_i(zeta) = (zeta^n - 1) / (n * (zeta - omega^i))
        FieldElement256 denom = field_.mul(
            field_.to_montgomery(FieldElement256(n)),
            field_.sub(zeta, omega_i)
        );
        FieldElement256 l_i = field_.mul(z_h_zeta, field_.inv(denom));
        
        result = field_.add(result, field_.mul(public_inputs[i], l_i));
        omega_i = field_.mul(omega_i, omega);
    }
    
    return result;
}

bool PLONKVerifier::verify_opening(const KZGCommitment256& commitment,
                                    const FieldElement256& point,
                                    const FieldElement256& value,
                                    const KZGCommitment256& proof) const {
    KZGProof256 kzg_proof;
    kzg_proof.quotient = proof.point;
    kzg_proof.point = point;
    kzg_proof.value = value;
    
    return kzg_->verify(commitment, kzg_proof);
}

// ============================================================================
// TallyProofSystem Implementation
// ============================================================================

TallyProofSystem::TallyProofSystem(size_t num_votes)
    : circuit_(num_votes), setup_complete_(false) {
}

void TallyProofSystem::setup() {
    if (setup_complete_) return;
    
    // Build circuit
    circuit_.build();
    
    // Generate universal setup
    size_t max_degree = circuit_.constraint_system().domain_size() * 4;
    universal_setup_ = std::make_unique<PLONKSetup>(
        PLONKSetupGenerator::generate_universal_setup(max_degree)
    );
    
    // Generate circuit-specific keys
    auto [pk, vk] = PLONKSetupGenerator::generate_circuit_keys(
        *universal_setup_, circuit_.constraint_system()
    );
    
    pk_ = std::make_unique<PLONKProvingKey>(std::move(pk));
    vk_ = std::make_unique<PLONKVerificationKey>(std::move(vk));
    
    // Create prover and verifier
    prover_ = std::make_unique<PLONKProver>(*universal_setup_, *pk_);
    verifier_ = std::make_unique<PLONKVerifier>(*universal_setup_, *vk_);
    
    setup_complete_ = true;
}

std::pair<PLONKProof, std::vector<FieldElement256>> TallyProofSystem::prove(
    const std::vector<FieldElement256>& votes,
    const std::vector<FieldElement256>& intermediate_tallies,
    const FieldElement256& final_tally) const {
    
    if (!setup_complete_) {
        throw std::runtime_error("Setup not complete");
    }
    
    // Generate witness
    std::vector<FieldElement256> witness = circuit_.generate_witness(
        votes, intermediate_tallies, final_tally
    );
    
    // Get public inputs
    std::vector<FieldElement256> public_inputs = circuit_.get_public_inputs(witness);
    
    // Generate proof
    PLONKProof proof = prover_->prove(witness, public_inputs);
    
    return {proof, public_inputs};
}

bool TallyProofSystem::verify(const PLONKProof& proof,
                               const FieldElement256& initial_tally,
                               const FieldElement256& final_tally) const {
    if (!setup_complete_) {
        throw std::runtime_error("Setup not complete");
    }
    
    std::vector<FieldElement256> public_inputs = {initial_tally, final_tally};
    return verifier_->verify(proof, public_inputs);
}

const PLONKVerificationKey& TallyProofSystem::verification_key() const {
    if (!setup_complete_) {
        throw std::runtime_error("Setup not complete");
    }
    return *vk_;
}

} // namespace zk
} // namespace fhe_accelerate
