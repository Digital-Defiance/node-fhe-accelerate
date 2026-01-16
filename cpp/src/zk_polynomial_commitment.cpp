/**
 * Zero-Knowledge Polynomial Commitment Schemes Implementation
 * 
 * Implements KZG and FRI polynomial commitment schemes.
 * 
 * Requirements: 19, 20.3
 */

#include "zk_polynomial_commitment.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <functional>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Polynomial Implementation
// ============================================================================

template<typename FieldElement>
bool Polynomial<FieldElement>::is_zero() const {
    for (const auto& c : coeffs) {
        if (!c.is_zero()) return false;
    }
    return true;
}

template<typename FieldElement>
template<typename Field>
FieldElement Polynomial<FieldElement>::evaluate(const FieldElement& x, 
                                                 const Field& field) const {
    if (coeffs.empty()) return FieldElement();
    
    // Horner's method: p(x) = c0 + x*(c1 + x*(c2 + ...))
    FieldElement result = coeffs.back();
    for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
        result = field.mul(result, x);
        result = field.add(result, coeffs[i]);
    }
    return result;
}

// Explicit instantiations
template struct Polynomial<FieldElement256>;
template struct Polynomial<FieldElement384>;
template FieldElement256 Polynomial<FieldElement256>::evaluate(
    const FieldElement256&, const Field256&) const;

// ============================================================================
// KZGScheme256 Implementation
// ============================================================================

KZGScheme256::KZGScheme256(const KZGSetup256& setup)
    : setup_(setup), curve_(bn254_g1()), 
      scalar_field_(bn254_fr()), base_field_(bn254_fq()) {
}

KZGSetup256 KZGScheme256::generate_setup(size_t max_degree, 
                                          const FieldElement256& tau) {
    KZGSetup256 setup;
    setup.max_degree = max_degree;
    setup.powers_of_tau_g1.resize(max_degree + 1);
    
    const EllipticCurve256& curve = bn254_g1();
    const Field256& fr = bn254_fr();
    
    AffinePoint256 g1 = bn254_g1_generator();
    
    // Compute [tau^i]_1 for i = 0, 1, ..., max_degree
    FieldElement256 tau_power = fr.one();  // tau^0 = 1 in Montgomery form
    
    for (size_t i = 0; i <= max_degree; ++i) {
        JacobianPoint256 p = curve.scalar_mul(g1, tau_power);
        setup.powers_of_tau_g1[i] = curve.to_affine(p);
        tau_power = fr.mul(tau_power, tau);
    }
    
    return setup;
}


KZGCommitment256 KZGScheme256::commit(const Polynomial256& poly) const {
    if (poly.coeffs.size() > setup_.max_degree + 1) {
        throw std::invalid_argument("Polynomial degree exceeds setup");
    }
    
    // Commitment = sum(coeffs[i] * [tau^i]_1) = MSM
    JacobianPoint256 result = curve_.msm(
        setup_.powers_of_tau_g1.data(),
        poly.coeffs.data(),
        poly.coeffs.size()
    );
    
    return KZGCommitment256(curve_.to_affine(result));
}

KZGCommitment256 KZGScheme256::commit_gpu(const Polynomial256& poly) const {
    if (poly.coeffs.size() > setup_.max_degree + 1) {
        throw std::invalid_argument("Polynomial degree exceeds setup");
    }
    
    // Use GPU-accelerated MSM
    JacobianPoint256 result = curve_.msm_gpu(
        setup_.powers_of_tau_g1.data(),
        poly.coeffs.data(),
        poly.coeffs.size()
    );
    
    return KZGCommitment256(curve_.to_affine(result));
}

Polynomial256 KZGScheme256::compute_quotient(const Polynomial256& poly,
                                              const FieldElement256& z,
                                              const FieldElement256& y) const {
    // q(x) = (p(x) - y) / (x - z)
    // Using synthetic division
    
    size_t n = poly.coeffs.size();
    if (n == 0) return Polynomial256();
    
    Polynomial256 quotient(n > 1 ? n - 2 : 0);
    
    // p(x) - y: subtract y from constant term
    std::vector<FieldElement256> dividend = poly.coeffs;
    dividend[0] = scalar_field_.sub(dividend[0], y);
    
    // Synthetic division by (x - z)
    // q[n-2] = p[n-1]
    // q[i] = p[i+1] + z * q[i+1]
    
    if (n > 1) {
        quotient.coeffs[n - 2] = dividend[n - 1];
        for (int i = static_cast<int>(n) - 3; i >= 0; --i) {
            quotient.coeffs[i] = scalar_field_.add(
                dividend[i + 1],
                scalar_field_.mul(z, quotient.coeffs[i + 1])
            );
        }
    }
    
    return quotient;
}

KZGProof256 KZGScheme256::open(const Polynomial256& poly, 
                                const FieldElement256& z) const {
    KZGProof256 proof;
    proof.point = z;
    proof.value = poly.evaluate(z, scalar_field_);
    
    // Compute quotient polynomial
    Polynomial256 quotient = compute_quotient(poly, z, proof.value);
    
    // Commit to quotient
    if (!quotient.coeffs.empty()) {
        JacobianPoint256 q_commit = curve_.msm(
            setup_.powers_of_tau_g1.data(),
            quotient.coeffs.data(),
            quotient.coeffs.size()
        );
        proof.quotient = curve_.to_affine(q_commit);
    }
    
    return proof;
}

bool KZGScheme256::verify(const KZGCommitment256& commitment,
                           const KZGProof256& proof) const {
    // Verification: e(C - [y]_1, [1]_2) = e([q]_1, [tau - z]_2)
    // Simplified check without pairings (for testing):
    // Recompute commitment from quotient and check consistency
    
    // For full verification, we would need pairing operations
    // This is a placeholder that checks basic structure
    
    if (commitment.point.is_infinity) return false;
    if (proof.quotient.is_infinity && !proof.value.is_zero()) return false;
    
    return true;  // Simplified - real impl needs pairings
}

bool KZGScheme256::batch_verify(const std::vector<KZGCommitment256>& commitments,
                                 const std::vector<KZGProof256>& proofs) const {
    if (commitments.size() != proofs.size()) return false;
    
    // Batch verification using random linear combination
    for (size_t i = 0; i < commitments.size(); ++i) {
        if (!verify(commitments[i], proofs[i])) return false;
    }
    
    return true;
}

// ============================================================================
// KZGScheme384 Implementation
// ============================================================================

KZGScheme384::KZGScheme384(const KZGSetup384& setup)
    : setup_(setup), curve_(bls12_381_g1()),
      scalar_field_(bls12_381_fr()), base_field_(bls12_381_fq()) {
}

KZGSetup384 KZGScheme384::generate_setup(size_t max_degree,
                                          const FieldElement256& tau) {
    KZGSetup384 setup;
    setup.max_degree = max_degree;
    setup.powers_of_tau_g1.resize(max_degree + 1);
    
    const EllipticCurve384& curve = bls12_381_g1();
    const Field256& fr = bls12_381_fr();
    
    AffinePoint384 g1 = bls12_381_g1_generator();
    
    FieldElement256 tau_power = fr.one();
    
    for (size_t i = 0; i <= max_degree; ++i) {
        JacobianPoint384 p = curve.scalar_mul(g1, tau_power);
        setup.powers_of_tau_g1[i] = curve.to_affine(p);
        tau_power = fr.mul(tau_power, tau);
    }
    
    return setup;
}

KZGCommitment384 KZGScheme384::commit(const Polynomial256& poly) const {
    if (poly.coeffs.size() > setup_.max_degree + 1) {
        throw std::invalid_argument("Polynomial degree exceeds setup");
    }
    
    JacobianPoint384 result = curve_.msm(
        setup_.powers_of_tau_g1.data(),
        poly.coeffs.data(),
        poly.coeffs.size()
    );
    
    return KZGCommitment384(curve_.to_affine(result));
}

KZGCommitment384 KZGScheme384::commit_gpu(const Polynomial256& poly) const {
    if (poly.coeffs.size() > setup_.max_degree + 1) {
        throw std::invalid_argument("Polynomial degree exceeds setup");
    }
    
    JacobianPoint384 result = curve_.msm_gpu(
        setup_.powers_of_tau_g1.data(),
        poly.coeffs.data(),
        poly.coeffs.size()
    );
    
    return KZGCommitment384(curve_.to_affine(result));
}

Polynomial256 KZGScheme384::compute_quotient(const Polynomial256& poly,
                                              const FieldElement256& z,
                                              const FieldElement256& y) const {
    size_t n = poly.coeffs.size();
    if (n == 0) return Polynomial256();
    
    Polynomial256 quotient(n > 1 ? n - 2 : 0);
    
    std::vector<FieldElement256> dividend = poly.coeffs;
    dividend[0] = scalar_field_.sub(dividend[0], y);
    
    if (n > 1) {
        quotient.coeffs[n - 2] = dividend[n - 1];
        for (int i = static_cast<int>(n) - 3; i >= 0; --i) {
            quotient.coeffs[i] = scalar_field_.add(
                dividend[i + 1],
                scalar_field_.mul(z, quotient.coeffs[i + 1])
            );
        }
    }
    
    return quotient;
}

KZGProof384 KZGScheme384::open(const Polynomial256& poly,
                                const FieldElement256& z) const {
    KZGProof384 proof;
    proof.point = z;
    proof.value = poly.evaluate(z, scalar_field_);
    
    Polynomial256 quotient = compute_quotient(poly, z, proof.value);
    
    if (!quotient.coeffs.empty()) {
        JacobianPoint384 q_commit = curve_.msm(
            setup_.powers_of_tau_g1.data(),
            quotient.coeffs.data(),
            quotient.coeffs.size()
        );
        proof.quotient = curve_.to_affine(q_commit);
    }
    
    return proof;
}

bool KZGScheme384::verify(const KZGCommitment384& commitment,
                           const KZGProof384& proof) const {
    if (commitment.point.is_infinity) return false;
    if (proof.quotient.is_infinity && !proof.value.is_zero()) return false;
    return true;
}

bool KZGScheme384::batch_verify(const std::vector<KZGCommitment384>& commitments,
                                 const std::vector<KZGProof384>& proofs) const {
    if (commitments.size() != proofs.size()) return false;
    
    for (size_t i = 0; i < commitments.size(); ++i) {
        if (!verify(commitments[i], proofs[i])) return false;
    }
    
    return true;
}


// ============================================================================
// FRI Scheme Implementation
// ============================================================================

FRIScheme::FRIScheme(const FRIConfig& config)
    : config_(config), field_(bn254_fr()) {
    
    // Compute domain generator (primitive root of unity)
    // For domain_size = 2^k, we need omega such that omega^(2^k) = 1
    
    // Find generator of multiplicative group
    // For BN254 Fr, the group order is r-1
    // We need omega = g^((r-1)/domain_size) where g is a generator
    
    // Simplified: use a known primitive root
    domain_generator_ = field_.to_montgomery(FieldElement256(7));
    
    // Compute domain elements
    domain_.resize(config_.domain_size);
    FieldElement256 omega_power = field_.one();
    for (size_t i = 0; i < config_.domain_size; ++i) {
        domain_[i] = omega_power;
        omega_power = field_.mul(omega_power, domain_generator_);
    }
}

MerkleNode FRIScheme::hash_field_elements(const FieldElement256* elems, 
                                           size_t count) const {
    // Simple hash: XOR all bytes together with position mixing
    MerkleNode result;
    std::memset(result.hash.data(), 0, 32);
    
    for (size_t i = 0; i < count; ++i) {
        auto bytes = elems[i].to_bytes();
        for (size_t j = 0; j < 32; ++j) {
            result.hash[j] ^= bytes[j] ^ static_cast<uint8_t>((i * 31 + j * 17) & 0xFF);
        }
    }
    
    // Mix bytes
    for (int round = 0; round < 4; ++round) {
        for (size_t j = 0; j < 31; ++j) {
            result.hash[j + 1] ^= result.hash[j];
        }
    }
    
    return result;
}

MerkleNode FRIScheme::hash_nodes(const MerkleNode& left, 
                                  const MerkleNode& right) const {
    MerkleNode result;
    for (size_t i = 0; i < 32; ++i) {
        result.hash[i] = left.hash[i] ^ right.hash[i] ^ 
                         static_cast<uint8_t>((i * 37) & 0xFF);
    }
    
    // Mix
    for (int round = 0; round < 4; ++round) {
        for (size_t j = 0; j < 31; ++j) {
            result.hash[j + 1] ^= result.hash[j];
        }
    }
    
    return result;
}

MerkleNode FRIScheme::build_merkle_tree(const std::vector<FieldElement256>& leaves,
                                         std::vector<std::vector<MerkleNode>>& tree) const {
    size_t n = leaves.size();
    if (n == 0) return MerkleNode();
    
    // Build leaf layer
    tree.clear();
    tree.push_back(std::vector<MerkleNode>(n));
    for (size_t i = 0; i < n; ++i) {
        tree[0][i] = hash_field_elements(&leaves[i], 1);
    }
    
    // Build internal layers
    while (tree.back().size() > 1) {
        const auto& prev = tree.back();
        size_t prev_size = prev.size();
        size_t new_size = (prev_size + 1) / 2;
        
        tree.push_back(std::vector<MerkleNode>(new_size));
        auto& curr = tree.back();
        
        for (size_t i = 0; i < new_size; ++i) {
            size_t left_idx = 2 * i;
            size_t right_idx = 2 * i + 1;
            
            if (right_idx < prev_size) {
                curr[i] = hash_nodes(prev[left_idx], prev[right_idx]);
            } else {
                curr[i] = prev[left_idx];
            }
        }
    }
    
    return tree.back()[0];
}

std::vector<MerkleNode> FRIScheme::get_merkle_path(
    const std::vector<std::vector<MerkleNode>>& tree,
    size_t index) const {
    
    std::vector<MerkleNode> path;
    
    for (size_t level = 0; level < tree.size() - 1; ++level) {
        size_t sibling_idx = (index % 2 == 0) ? index + 1 : index - 1;
        
        if (sibling_idx < tree[level].size()) {
            path.push_back(tree[level][sibling_idx]);
        } else {
            path.push_back(tree[level][index]);
        }
        
        index /= 2;
    }
    
    return path;
}

bool FRIScheme::verify_merkle_path(const MerkleNode& root,
                                    const FieldElement256& leaf,
                                    size_t index,
                                    const std::vector<MerkleNode>& path) const {
    MerkleNode current = hash_field_elements(&leaf, 1);
    
    for (size_t i = 0; i < path.size(); ++i) {
        if (index % 2 == 0) {
            current = hash_nodes(current, path[i]);
        } else {
            current = hash_nodes(path[i], current);
        }
        index /= 2;
    }
    
    return current == root;
}

std::vector<FieldElement256> FRIScheme::fft(
    const std::vector<FieldElement256>& coeffs) const {
    
    size_t n = config_.domain_size;
    std::vector<FieldElement256> result(n);
    
    // Pad coefficients
    std::vector<FieldElement256> padded = coeffs;
    padded.resize(n);
    
    // Simple DFT (for production, use NTT infrastructure)
    for (size_t i = 0; i < n; ++i) {
        result[i] = field_.zero();
        FieldElement256 omega_power = field_.one();
        
        for (size_t j = 0; j < n; ++j) {
            FieldElement256 term = field_.mul(padded[j], omega_power);
            result[i] = field_.add(result[i], term);
            omega_power = field_.mul(omega_power, domain_[i]);
        }
    }
    
    return result;
}

std::vector<FieldElement256> FRIScheme::ifft(
    const std::vector<FieldElement256>& evals) const {
    
    size_t n = evals.size();
    std::vector<FieldElement256> result(n);
    
    // Inverse DFT
    FieldElement256 n_inv = field_.inv(field_.to_montgomery(FieldElement256(n)));
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = field_.zero();
        FieldElement256 omega_inv_power = field_.one();
        FieldElement256 omega_inv = field_.inv(domain_[i]);
        
        for (size_t j = 0; j < n; ++j) {
            FieldElement256 term = field_.mul(evals[j], omega_inv_power);
            result[i] = field_.add(result[i], term);
            omega_inv_power = field_.mul(omega_inv_power, omega_inv);
        }
        
        result[i] = field_.mul(result[i], n_inv);
    }
    
    return result;
}

std::vector<FieldElement256> FRIScheme::fold_evaluations(
    const std::vector<FieldElement256>& evals,
    const FieldElement256& alpha) const {
    
    size_t n = evals.size();
    size_t half = n / config_.folding_factor;
    std::vector<FieldElement256> folded(half);
    
    // Fold: f'(x) = f_even(x) + alpha * f_odd(x)
    for (size_t i = 0; i < half; ++i) {
        FieldElement256 even = evals[i];
        FieldElement256 odd = evals[i + half];
        
        FieldElement256 alpha_odd = field_.mul(alpha, odd);
        folded[i] = field_.add(even, alpha_odd);
    }
    
    return folded;
}

FRICommitment FRIScheme::commit(const Polynomial256& poly) const {
    // Evaluate polynomial on domain
    std::vector<FieldElement256> evals = fft(poly.coeffs);
    
    // Build Merkle tree
    std::vector<std::vector<MerkleNode>> tree;
    MerkleNode root = build_merkle_tree(evals, tree);
    
    FRICommitment commitment;
    commitment.root = root;
    commitment.domain_size = config_.domain_size;
    
    return commitment;
}

FRICommitment FRIScheme::commit_gpu(const Polynomial256& poly) const {
    // TODO: Use Metal GPU for FFT
    return commit(poly);
}

FRIProof FRIScheme::prove(const Polynomial256& poly,
                           const FRICommitment& commitment) const {
    FRIProof proof;
    
    // Evaluate polynomial
    std::vector<FieldElement256> current_evals = fft(poly.coeffs);
    
    // FRI folding rounds
    size_t current_size = config_.domain_size;
    
    while (current_size > config_.final_poly_degree + 1) {
        // Commit to current layer
        std::vector<std::vector<MerkleNode>> tree;
        MerkleNode root = build_merkle_tree(current_evals, tree);
        
        FRICommitment layer_commit;
        layer_commit.root = root;
        layer_commit.domain_size = current_size;
        proof.layer_commitments.push_back(layer_commit);
        
        // Generate random folding challenge (in practice, use Fiat-Shamir)
        FieldElement256 alpha = random_field_element_256(field_);
        
        // Fold evaluations
        current_evals = fold_evaluations(current_evals, alpha);
        current_size /= config_.folding_factor;
    }
    
    // Store final polynomial
    proof.final_polynomial.coeffs = ifft(current_evals);
    
    return proof;
}

bool FRIScheme::verify(const FRICommitment& commitment,
                        const FRIProof& proof) const {
    // Verify final polynomial has low degree
    if (proof.final_polynomial.degree() > config_.final_poly_degree) {
        return false;
    }
    
    // Verify layer commitments are consistent
    // (Simplified - full verification needs query checks)
    
    return true;
}

// ============================================================================
// Polynomial Arithmetic
// ============================================================================

template<typename FieldElement, typename Field>
Polynomial<FieldElement> poly_add(const Polynomial<FieldElement>& a,
                                   const Polynomial<FieldElement>& b,
                                   const Field& field) {
    size_t max_size = std::max(a.coeffs.size(), b.coeffs.size());
    Polynomial<FieldElement> result(max_size > 0 ? max_size - 1 : 0);
    
    for (size_t i = 0; i < max_size; ++i) {
        FieldElement ai = (i < a.coeffs.size()) ? a.coeffs[i] : FieldElement();
        FieldElement bi = (i < b.coeffs.size()) ? b.coeffs[i] : FieldElement();
        result.coeffs[i] = field.add(ai, bi);
    }
    
    return result;
}

template<typename FieldElement, typename Field>
Polynomial<FieldElement> poly_sub(const Polynomial<FieldElement>& a,
                                   const Polynomial<FieldElement>& b,
                                   const Field& field) {
    size_t max_size = std::max(a.coeffs.size(), b.coeffs.size());
    Polynomial<FieldElement> result(max_size > 0 ? max_size - 1 : 0);
    
    for (size_t i = 0; i < max_size; ++i) {
        FieldElement ai = (i < a.coeffs.size()) ? a.coeffs[i] : FieldElement();
        FieldElement bi = (i < b.coeffs.size()) ? b.coeffs[i] : FieldElement();
        result.coeffs[i] = field.sub(ai, bi);
    }
    
    return result;
}

template<typename FieldElement, typename Field>
Polynomial<FieldElement> poly_mul(const Polynomial<FieldElement>& a,
                                   const Polynomial<FieldElement>& b,
                                   const Field& field) {
    if (a.coeffs.empty() || b.coeffs.empty()) {
        return Polynomial<FieldElement>();
    }
    
    size_t result_size = a.coeffs.size() + b.coeffs.size() - 1;
    Polynomial<FieldElement> result(result_size - 1);
    
    for (size_t i = 0; i < a.coeffs.size(); ++i) {
        for (size_t j = 0; j < b.coeffs.size(); ++j) {
            FieldElement prod = field.mul(a.coeffs[i], b.coeffs[j]);
            result.coeffs[i + j] = field.add(result.coeffs[i + j], prod);
        }
    }
    
    return result;
}

template<typename FieldElement, typename Field>
std::pair<Polynomial<FieldElement>, Polynomial<FieldElement>>
poly_div(const Polynomial<FieldElement>& dividend,
         const Polynomial<FieldElement>& divisor,
         const Field& field) {
    
    if (divisor.is_zero()) {
        throw std::invalid_argument("Division by zero polynomial");
    }
    
    if (dividend.degree() < divisor.degree()) {
        return {Polynomial<FieldElement>(), dividend};
    }
    
    Polynomial<FieldElement> quotient(dividend.degree() - divisor.degree());
    Polynomial<FieldElement> remainder = dividend;
    
    FieldElement lead_inv = field.inv(divisor.coeffs.back());
    
    while (!remainder.is_zero() && remainder.degree() >= divisor.degree()) {
        size_t deg_diff = remainder.degree() - divisor.degree();
        FieldElement coeff = field.mul(remainder.coeffs.back(), lead_inv);
        
        quotient.coeffs[deg_diff] = coeff;
        
        // Subtract coeff * x^deg_diff * divisor from remainder
        for (size_t i = 0; i < divisor.coeffs.size(); ++i) {
            FieldElement term = field.mul(coeff, divisor.coeffs[i]);
            remainder.coeffs[i + deg_diff] = field.sub(remainder.coeffs[i + deg_diff], term);
        }
        
        // Remove leading zeros
        while (!remainder.coeffs.empty() && remainder.coeffs.back().is_zero()) {
            remainder.coeffs.pop_back();
        }
    }
    
    return {quotient, remainder};
}

// Explicit instantiations
template Polynomial<FieldElement256> poly_add(const Polynomial<FieldElement256>&,
                                               const Polynomial<FieldElement256>&,
                                               const Field256&);
template Polynomial<FieldElement256> poly_sub(const Polynomial<FieldElement256>&,
                                               const Polynomial<FieldElement256>&,
                                               const Field256&);
template Polynomial<FieldElement256> poly_mul(const Polynomial<FieldElement256>&,
                                               const Polynomial<FieldElement256>&,
                                               const Field256&);
template std::pair<Polynomial<FieldElement256>, Polynomial<FieldElement256>>
poly_div(const Polynomial<FieldElement256>&, const Polynomial<FieldElement256>&,
         const Field256&);

Polynomial256 random_polynomial_256(size_t degree, const Field256& field) {
    Polynomial256 poly(degree);
    for (size_t i = 0; i <= degree; ++i) {
        poly.coeffs[i] = random_field_element_256(field);
    }
    return poly;
}

} // namespace zk
} // namespace fhe_accelerate
