/**
 * Bulletproofs Range Proof Implementation
 * 
 * Implements Bulletproofs range proofs for ballot validity verification.
 * Based on the Bulletproofs paper by Bünz et al.
 * 
 * Requirements: 19.1, 19.6, 19.8, 19.9, 20
 */

#include "bulletproofs.h"
#include <random>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Utility Functions
// ============================================================================

size_t bits_needed(uint64_t max_value) {
    if (max_value == 0) return 1;
    size_t bits = 0;
    uint64_t v = max_value - 1;  // We need to represent [0, max_value)
    while (v > 0) {
        bits++;
        v >>= 1;
    }
    return bits == 0 ? 1 : bits;
}

bool value_in_range(uint64_t value, size_t n) {
    if (n >= 64) return true;
    return value < (1ULL << n);
}

// ============================================================================
// InnerProductProof Implementation
// ============================================================================

std::vector<uint8_t> InnerProductProof::serialize() const {
    std::vector<uint8_t> result;
    
    // Write number of rounds
    uint32_t rounds = static_cast<uint32_t>(L.size());
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&rounds),
                  reinterpret_cast<uint8_t*>(&rounds) + 4);

    // Write L points
    for (const auto& point : L) {
        auto x_bytes = point.x.to_bytes();
        auto y_bytes = point.y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    }
    
    // Write R points
    for (const auto& point : R) {
        auto x_bytes = point.x.to_bytes();
        auto y_bytes = point.y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    }
    
    // Write final scalars
    auto a_bytes = a.to_bytes();
    auto b_bytes = b.to_bytes();
    result.insert(result.end(), a_bytes.begin(), a_bytes.end());
    result.insert(result.end(), b_bytes.begin(), b_bytes.end());
    
    return result;
}

std::optional<InnerProductProof> InnerProductProof::deserialize(const uint8_t* data, size_t len) {
    if (len < 4) return std::nullopt;
    
    uint32_t rounds;
    std::memcpy(&rounds, data, 4);
    size_t offset = 4;
    
    // Check minimum size
    size_t expected_size = 4 + rounds * 64 * 2 + 64;  // rounds * (L + R) + (a + b)
    if (len < expected_size) return std::nullopt;
    
    InnerProductProof proof;
    
    // Read L points
    proof.L.resize(rounds);
    for (uint32_t i = 0; i < rounds; ++i) {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        proof.L[i].x = FieldElement256::from_bytes(x_bytes);
        proof.L[i].y = FieldElement256::from_bytes(y_bytes);
        proof.L[i].is_infinity = false;
        offset += 64;
    }

    // Read R points
    proof.R.resize(rounds);
    for (uint32_t i = 0; i < rounds; ++i) {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        proof.R[i].x = FieldElement256::from_bytes(x_bytes);
        proof.R[i].y = FieldElement256::from_bytes(y_bytes);
        proof.R[i].is_infinity = false;
        offset += 64;
    }
    
    // Read final scalars
    std::array<uint8_t, 32> a_bytes, b_bytes;
    std::memcpy(a_bytes.data(), data + offset, 32);
    std::memcpy(b_bytes.data(), data + offset + 32, 32);
    proof.a = FieldElement256::from_bytes(a_bytes);
    proof.b = FieldElement256::from_bytes(b_bytes);
    
    return proof;
}

// ============================================================================
// RangeProof Implementation
// ============================================================================

std::vector<uint8_t> RangeProof::serialize() const {
    std::vector<uint8_t> result;
    
    // Write A, S, T1, T2 points
    for (const auto* point : {&A, &S, &T1, &T2}) {
        auto x_bytes = point->x.to_bytes();
        auto y_bytes = point->y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    }
    
    // Write tau_x, mu, t_hat scalars
    for (const auto* scalar : {&tau_x, &mu, &t_hat}) {
        auto bytes = scalar->to_bytes();
        result.insert(result.end(), bytes.begin(), bytes.end());
    }
    
    // Write inner product proof
    auto inner_bytes = inner_proof.serialize();
    result.insert(result.end(), inner_bytes.begin(), inner_bytes.end());
    
    return result;
}

std::optional<RangeProof> RangeProof::deserialize(const uint8_t* data, size_t len) {
    // Minimum size: 4 points + 3 scalars + inner proof header
    if (len < 4 * 64 + 3 * 32 + 4) return std::nullopt;
    
    RangeProof proof;
    size_t offset = 0;
    
    // Read A, S, T1, T2 points
    for (auto* point : {&proof.A, &proof.S, &proof.T1, &proof.T2}) {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        point->x = FieldElement256::from_bytes(x_bytes);
        point->y = FieldElement256::from_bytes(y_bytes);
        point->is_infinity = false;
        offset += 64;
    }
    
    // Read tau_x, mu, t_hat scalars
    for (auto* scalar : {&proof.tau_x, &proof.mu, &proof.t_hat}) {
        std::array<uint8_t, 32> bytes;
        std::memcpy(bytes.data(), data + offset, 32);
        *scalar = FieldElement256::from_bytes(bytes);
        offset += 32;
    }
    
    // Read inner product proof
    auto inner = InnerProductProof::deserialize(data + offset, len - offset);
    if (!inner) return std::nullopt;
    proof.inner_proof = std::move(*inner);
    
    return proof;
}

// ============================================================================
// AggregatedRangeProof Implementation
// ============================================================================

std::vector<uint8_t> AggregatedRangeProof::serialize() const {
    std::vector<uint8_t> result;
    
    // Write num_values
    uint32_t nv = static_cast<uint32_t>(num_values);
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&nv),
                  reinterpret_cast<uint8_t*>(&nv) + 4);
    
    // Write points and scalars (same as RangeProof)
    for (const auto* point : {&A, &S, &T1, &T2}) {
        auto x_bytes = point->x.to_bytes();
        auto y_bytes = point->y.to_bytes();
        result.insert(result.end(), x_bytes.begin(), x_bytes.end());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    }
    
    for (const auto* scalar : {&tau_x, &mu, &t_hat}) {
        auto bytes = scalar->to_bytes();
        result.insert(result.end(), bytes.begin(), bytes.end());
    }
    
    auto inner_bytes = inner_proof.serialize();
    result.insert(result.end(), inner_bytes.begin(), inner_bytes.end());
    
    return result;
}

std::optional<AggregatedRangeProof> AggregatedRangeProof::deserialize(const uint8_t* data, size_t len) {
    if (len < 4 + 4 * 64 + 3 * 32 + 4) return std::nullopt;
    
    AggregatedRangeProof proof;
    size_t offset = 0;
    
    // Read num_values
    uint32_t nv;
    std::memcpy(&nv, data, 4);
    proof.num_values = nv;
    offset += 4;
    
    // Read points
    for (auto* point : {&proof.A, &proof.S, &proof.T1, &proof.T2}) {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), data + offset, 32);
        std::memcpy(y_bytes.data(), data + offset + 32, 32);
        point->x = FieldElement256::from_bytes(x_bytes);
        point->y = FieldElement256::from_bytes(y_bytes);
        point->is_infinity = false;
        offset += 64;
    }
    
    // Read scalars
    for (auto* scalar : {&proof.tau_x, &proof.mu, &proof.t_hat}) {
        std::array<uint8_t, 32> bytes;
        std::memcpy(bytes.data(), data + offset, 32);
        *scalar = FieldElement256::from_bytes(bytes);
        offset += 32;
    }
    
    auto inner = InnerProductProof::deserialize(data + offset, len - offset);
    if (!inner) return std::nullopt;
    proof.inner_proof = std::move(*inner);
    
    return proof;
}

// ============================================================================
// BallotValidityProof Implementation
// ============================================================================

std::vector<uint8_t> BallotValidityProof::serialize() const {
    std::vector<uint8_t> result;
    
    // Write num_candidates
    uint32_t nc = static_cast<uint32_t>(num_candidates);
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&nc),
                  reinterpret_cast<uint8_t*>(&nc) + 4);
    
    // Write commitment
    auto x_bytes = commitment.point.x.to_bytes();
    auto y_bytes = commitment.point.y.to_bytes();
    result.insert(result.end(), x_bytes.begin(), x_bytes.end());
    result.insert(result.end(), y_bytes.begin(), y_bytes.end());
    
    // Write range proof
    auto proof_bytes = range_proof.serialize();
    result.insert(result.end(), proof_bytes.begin(), proof_bytes.end());
    
    return result;
}

std::optional<BallotValidityProof> BallotValidityProof::deserialize(const uint8_t* data, size_t len) {
    if (len < 4 + 64) return std::nullopt;
    
    BallotValidityProof proof;
    size_t offset = 0;
    
    // Read num_candidates
    uint32_t nc;
    std::memcpy(&nc, data, 4);
    proof.num_candidates = nc;
    offset += 4;
    
    // Read commitment
    std::array<uint8_t, 32> x_bytes, y_bytes;
    std::memcpy(x_bytes.data(), data + offset, 32);
    std::memcpy(y_bytes.data(), data + offset + 32, 32);
    proof.commitment.point.x = FieldElement256::from_bytes(x_bytes);
    proof.commitment.point.y = FieldElement256::from_bytes(y_bytes);
    proof.commitment.point.is_infinity = false;
    offset += 64;
    
    // Read range proof
    auto range = RangeProof::deserialize(data + offset, len - offset);
    if (!range) return std::nullopt;
    proof.range_proof = std::move(*range);
    
    return proof;
}

// ============================================================================
// BulletproofsProver Implementation
// ============================================================================

BulletproofsProver::BulletproofsProver(const BulletproofsConfig& config)
    : config_(config)
    , curve_(bn254_g1())
    , field_(bn254_fr()) {
}

BulletproofsGenerators BulletproofsProver::generate_generators(size_t n) const {
    BulletproofsGenerators gens;
    
    // Use hash-to-curve for deterministic generator generation
    // G is the standard generator
    gens.G = bn254_g1_generator();
    
    // H is derived from hashing "H"
    gens.H = hash_to_curve_256(reinterpret_cast<const uint8_t*>("Bulletproofs_H"), 14, curve_);
    
    // U is derived from hashing "U"
    gens.U = hash_to_curve_256(reinterpret_cast<const uint8_t*>("Bulletproofs_U"), 14, curve_);
    
    // Generate g_vec and h_vec
    gens.g_vec.resize(n);
    gens.h_vec.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        // g_i = hash_to_curve("Bulletproofs_g" || i)
        std::string g_label = "Bulletproofs_g_" + std::to_string(i);
        gens.g_vec[i] = hash_to_curve_256(
            reinterpret_cast<const uint8_t*>(g_label.c_str()),
            g_label.size(), curve_);
        
        // h_i = hash_to_curve("Bulletproofs_h" || i)
        std::string h_label = "Bulletproofs_h_" + std::to_string(i);
        gens.h_vec[i] = hash_to_curve_256(
            reinterpret_cast<const uint8_t*>(h_label.c_str()),
            h_label.size(), curve_);
    }
    
    return gens;
}

FieldElement256 BulletproofsProver::random_scalar() const {
    return random_field_element_256(field_);
}

std::vector<FieldElement256> BulletproofsProver::powers_of(const FieldElement256& x, size_t n) const {
    std::vector<FieldElement256> result(n);
    result[0] = field_.one();  // x^0 = 1
    for (size_t i = 1; i < n; ++i) {
        result[i] = field_.mul(result[i - 1], x);
    }
    return result;
}

std::vector<FieldElement256> BulletproofsProver::bit_decompose(uint64_t value, size_t n) const {
    std::vector<FieldElement256> bits(n);
    for (size_t i = 0; i < n; ++i) {
        if ((value >> i) & 1) {
            bits[i] = field_.one();
        } else {
            bits[i] = field_.zero();
        }
    }
    return bits;
}

std::pair<PedersenCommitment, FieldElement256> BulletproofsProver::commit(
    uint64_t value,
    const std::optional<FieldElement256>& blinding,
    const BulletproofsGenerators& gens) const {
    
    // Generate or use provided blinding factor
    FieldElement256 r = blinding.value_or(random_scalar());
    
    // Convert value to field element
    FieldElement256 v = field_.to_montgomery(FieldElement256(value));
    
    // C = v*G + r*H
    auto vG = curve_.scalar_mul(gens.G, v);
    auto rH = curve_.scalar_mul(gens.H, r);
    auto C = curve_.add(vG, rH);
    
    PedersenCommitment commitment(curve_.to_affine(C));
    return {commitment, r};
}

JacobianPoint256 BulletproofsProver::vector_commit(
    const std::vector<FieldElement256>& a,
    const std::vector<FieldElement256>& b,
    const std::vector<AffinePoint256>& g_vec,
    const std::vector<AffinePoint256>& h_vec) const {
    
    size_t n = a.size();
    
    // Use MSM for efficiency
    std::vector<AffinePoint256> points;
    std::vector<FieldElement256> scalars;
    
    points.reserve(2 * n);
    scalars.reserve(2 * n);
    
    for (size_t i = 0; i < n; ++i) {
        points.push_back(g_vec[i]);
        scalars.push_back(a[i]);
        points.push_back(h_vec[i]);
        scalars.push_back(b[i]);
    }
    
    if (config_.use_gpu) {
        return curve_.msm_gpu(points.data(), scalars.data(), points.size());
    } else {
        return curve_.msm(points.data(), scalars.data(), points.size());
    }
}

FieldElement256 BulletproofsProver::compute_delta(
    const FieldElement256& y,
    const FieldElement256& z,
    size_t n) const {
    
    // delta(y, z) = (z - z^2) * <1^n, y^n> - z^3 * <1^n, 2^n>
    // where <1^n, y^n> = sum(y^i) for i in [0, n)
    // and <1^n, 2^n> = sum(2^i) = 2^n - 1
    
    auto z2 = field_.square(z);
    auto z3 = field_.mul(z2, z);
    
    // Compute <1^n, y^n> = (y^n - 1) / (y - 1) if y != 1
    auto y_powers = powers_of(y, n + 1);
    FieldElement256 sum_y_powers = field_.zero();
    for (size_t i = 0; i < n; ++i) {
        sum_y_powers = field_.add(sum_y_powers, y_powers[i]);
    }
    
    // Compute 2^n - 1
    FieldElement256 two = field_.to_montgomery(FieldElement256(2));
    auto two_powers = powers_of(two, n);
    FieldElement256 sum_two_powers = field_.zero();
    for (size_t i = 0; i < n; ++i) {
        sum_two_powers = field_.add(sum_two_powers, two_powers[i]);
    }
    
    // (z - z^2) * sum_y_powers
    auto z_minus_z2 = field_.sub(z, z2);
    auto term1 = field_.mul(z_minus_z2, sum_y_powers);
    
    // z^3 * sum_two_powers
    auto term2 = field_.mul(z3, sum_two_powers);
    
    return field_.sub(term1, term2);
}

InnerProductProof BulletproofsProver::prove_inner_product(
    const std::vector<FieldElement256>& a,
    const std::vector<FieldElement256>& b,
    const std::vector<AffinePoint256>& g_vec,
    const std::vector<AffinePoint256>& h_vec,
    const AffinePoint256& U,
    Transcript& transcript) const {
    
    InnerProductProof proof;
    size_t n = a.size();
    
    // Make mutable copies
    std::vector<FieldElement256> a_vec = a;
    std::vector<FieldElement256> b_vec = b;
    std::vector<AffinePoint256> g = g_vec;
    std::vector<AffinePoint256> h = h_vec;
    
    // Recursive halving
    while (n > 1) {
        size_t n_half = n / 2;
        
        // Compute L and R
        // L = <a_lo, g_hi> + <b_hi, h_lo> + <a_lo, b_hi> * U
        // R = <a_hi, g_lo> + <b_lo, h_hi> + <a_hi, b_lo> * U
        
        // Split vectors
        std::vector<FieldElement256> a_lo(a_vec.begin(), a_vec.begin() + n_half);
        std::vector<FieldElement256> a_hi(a_vec.begin() + n_half, a_vec.end());
        std::vector<FieldElement256> b_lo(b_vec.begin(), b_vec.begin() + n_half);
        std::vector<FieldElement256> b_hi(b_vec.begin() + n_half, b_vec.end());
        std::vector<AffinePoint256> g_lo(g.begin(), g.begin() + n_half);
        std::vector<AffinePoint256> g_hi(g.begin() + n_half, g.end());
        std::vector<AffinePoint256> h_lo(h.begin(), h.begin() + n_half);
        std::vector<AffinePoint256> h_hi(h.begin() + n_half, h.end());

        // Compute inner products
        FieldElement256 c_L = field_.zero();
        FieldElement256 c_R = field_.zero();
        for (size_t i = 0; i < n_half; ++i) {
            c_L = field_.add(c_L, field_.mul(a_lo[i], b_hi[i]));
            c_R = field_.add(c_R, field_.mul(a_hi[i], b_lo[i]));
        }
        
        // Compute L point
        std::vector<AffinePoint256> L_points;
        std::vector<FieldElement256> L_scalars;
        for (size_t i = 0; i < n_half; ++i) {
            L_points.push_back(g_hi[i]);
            L_scalars.push_back(a_lo[i]);
            L_points.push_back(h_lo[i]);
            L_scalars.push_back(b_hi[i]);
        }
        L_points.push_back(U);
        L_scalars.push_back(c_L);
        
        auto L_jac = curve_.msm(L_points.data(), L_scalars.data(), L_points.size());
        auto L = curve_.to_affine(L_jac);
        
        // Compute R point
        std::vector<AffinePoint256> R_points;
        std::vector<FieldElement256> R_scalars;
        for (size_t i = 0; i < n_half; ++i) {
            R_points.push_back(g_lo[i]);
            R_scalars.push_back(a_hi[i]);
            R_points.push_back(h_hi[i]);
            R_scalars.push_back(b_lo[i]);
        }
        R_points.push_back(U);
        R_scalars.push_back(c_R);
        
        auto R_jac = curve_.msm(R_points.data(), R_scalars.data(), R_points.size());
        auto R = curve_.to_affine(R_jac);
        
        proof.L.push_back(L);
        proof.R.push_back(R);
        
        // Get challenge
        transcript.append_point("L", L);
        transcript.append_point("R", R);
        auto x = transcript.challenge_field_element("x");
        auto x_inv = field_.inv(x);
        
        // Update vectors
        a_vec.resize(n_half);
        b_vec.resize(n_half);
        g.resize(n_half);
        h.resize(n_half);
        
        for (size_t i = 0; i < n_half; ++i) {
            // a' = a_lo * x + a_hi * x^(-1)
            a_vec[i] = field_.add(field_.mul(a_lo[i], x), field_.mul(a_hi[i], x_inv));
            // b' = b_lo * x^(-1) + b_hi * x
            b_vec[i] = field_.add(field_.mul(b_lo[i], x_inv), field_.mul(b_hi[i], x));
            // g' = g_lo * x^(-1) + g_hi * x
            auto g_term1 = curve_.scalar_mul(g_lo[i], x_inv);
            auto g_term2 = curve_.scalar_mul(g_hi[i], x);
            g[i] = curve_.to_affine(curve_.add(g_term1, g_term2));
            // h' = h_lo * x + h_hi * x^(-1)
            auto h_term1 = curve_.scalar_mul(h_lo[i], x);
            auto h_term2 = curve_.scalar_mul(h_hi[i], x_inv);
            h[i] = curve_.to_affine(curve_.add(h_term1, h_term2));
        }
        
        n = n_half;
    }
    
    proof.a = a_vec[0];
    proof.b = b_vec[0];
    
    return proof;
}

RangeProof BulletproofsProver::prove_range(
    uint64_t value,
    const FieldElement256& blinding,
    size_t n,
    const BulletproofsGenerators& gens) const {
    
    if (!value_in_range(value, n)) {
        throw std::invalid_argument("Value out of range");
    }
    
    if (gens.size() < n) {
        throw std::invalid_argument("Not enough generators");
    }
    
    RangeProof proof;
    Transcript transcript("Bulletproofs_RangeProof");
    
    // Bit decomposition: a_L[i] = (value >> i) & 1
    auto a_L = bit_decompose(value, n);
    
    // a_R = a_L - 1^n (so a_R[i] = a_L[i] - 1)
    std::vector<FieldElement256> a_R(n);
    for (size_t i = 0; i < n; ++i) {
        a_R[i] = field_.sub(a_L[i], field_.one());
    }
    
    // Random blinding vectors
    std::vector<FieldElement256> s_L(n), s_R(n);
    for (size_t i = 0; i < n; ++i) {
        s_L[i] = random_scalar();
        s_R[i] = random_scalar();
    }
    
    // Blinding factors
    auto alpha = random_scalar();
    auto rho = random_scalar();
    
    // A = h^alpha * g^a_L * h^a_R
    auto A_jac = curve_.scalar_mul(gens.H, alpha);
    auto a_commit = vector_commit(a_L, a_R, gens.g_vec, gens.h_vec);
    A_jac = curve_.add(A_jac, a_commit);
    proof.A = curve_.to_affine(A_jac);
    
    // S = h^rho * g^s_L * h^s_R
    auto S_jac = curve_.scalar_mul(gens.H, rho);
    auto s_commit = vector_commit(s_L, s_R, gens.g_vec, gens.h_vec);
    S_jac = curve_.add(S_jac, s_commit);
    proof.S = curve_.to_affine(S_jac);
    
    // Get challenges y, z
    transcript.append_point("A", proof.A);
    transcript.append_point("S", proof.S);
    auto y = transcript.challenge_field_element("y");
    auto z = transcript.challenge_field_element("z");
    
    // Compute polynomial coefficients
    auto y_powers = powers_of(y, n);
    auto two = field_.to_montgomery(FieldElement256(2));
    auto two_powers = powers_of(two, n);
    
    auto z2 = field_.square(z);

    // l(X) = (a_L - z*1^n) + s_L*X
    // r(X) = y^n ∘ (a_R + z*1^n + s_R*X) + z^2*2^n
    // t(X) = <l(X), r(X)> = t_0 + t_1*X + t_2*X^2
    
    // Compute t_0, t_1, t_2
    // t_0 = <a_L - z*1^n, y^n ∘ (a_R + z*1^n) + z^2*2^n>
    // t_1 = <a_L - z*1^n, y^n ∘ s_R> + <s_L, y^n ∘ (a_R + z*1^n) + z^2*2^n>
    // t_2 = <s_L, y^n ∘ s_R>
    
    std::vector<FieldElement256> l_0(n), r_0(n);
    for (size_t i = 0; i < n; ++i) {
        l_0[i] = field_.sub(a_L[i], z);
        auto r_term = field_.add(a_R[i], z);
        r_term = field_.mul(y_powers[i], r_term);
        r_term = field_.add(r_term, field_.mul(z2, two_powers[i]));
        r_0[i] = r_term;
    }
    
    FieldElement256 t_0 = field_.zero();
    FieldElement256 t_1 = field_.zero();
    FieldElement256 t_2 = field_.zero();
    
    for (size_t i = 0; i < n; ++i) {
        t_0 = field_.add(t_0, field_.mul(l_0[i], r_0[i]));
        
        // t_1 terms
        auto y_s_R = field_.mul(y_powers[i], s_R[i]);
        t_1 = field_.add(t_1, field_.mul(l_0[i], y_s_R));
        t_1 = field_.add(t_1, field_.mul(s_L[i], r_0[i]));
        
        // t_2 = <s_L, y^n ∘ s_R>
        t_2 = field_.add(t_2, field_.mul(s_L[i], y_s_R));
    }
    
    // Commit to t_1, t_2
    auto tau_1 = random_scalar();
    auto tau_2 = random_scalar();
    
    // T_1 = t_1*G + tau_1*H
    auto T1_jac = curve_.add(
        curve_.scalar_mul(gens.G, t_1),
        curve_.scalar_mul(gens.H, tau_1)
    );
    proof.T1 = curve_.to_affine(T1_jac);
    
    // T_2 = t_2*G + tau_2*H
    auto T2_jac = curve_.add(
        curve_.scalar_mul(gens.G, t_2),
        curve_.scalar_mul(gens.H, tau_2)
    );
    proof.T2 = curve_.to_affine(T2_jac);
    
    // Get challenge x
    transcript.append_point("T1", proof.T1);
    transcript.append_point("T2", proof.T2);
    auto x = transcript.challenge_field_element("x");
    auto x2 = field_.square(x);

    // Compute l, r vectors at x
    std::vector<FieldElement256> l_vec(n), r_vec(n);
    for (size_t i = 0; i < n; ++i) {
        // l = l_0 + s_L * x
        l_vec[i] = field_.add(l_0[i], field_.mul(s_L[i], x));
        // r = r_0 + y^i * s_R * x
        r_vec[i] = field_.add(r_0[i], field_.mul(field_.mul(y_powers[i], s_R[i]), x));
    }
    
    // t_hat = t_0 + t_1*x + t_2*x^2
    proof.t_hat = field_.add(t_0, field_.add(field_.mul(t_1, x), field_.mul(t_2, x2)));
    
    // tau_x = tau_2*x^2 + tau_1*x + z^2*gamma
    // where gamma is the blinding factor for the value commitment
    proof.tau_x = field_.add(
        field_.mul(tau_2, x2),
        field_.add(field_.mul(tau_1, x), field_.mul(z2, blinding))
    );
    
    // mu = alpha + rho*x
    proof.mu = field_.add(alpha, field_.mul(rho, x));
    
    // Prepare for inner product argument
    // Need to compute h' = h^(y^(-i))
    std::vector<AffinePoint256> h_prime(n);
    auto y_inv = field_.inv(y);
    auto y_inv_powers = powers_of(y_inv, n);
    
    for (size_t i = 0; i < n; ++i) {
        auto h_scaled = curve_.scalar_mul(gens.h_vec[i], y_inv_powers[i]);
        h_prime[i] = curve_.to_affine(h_scaled);
    }
    
    // Inner product argument
    transcript.append_field_element("t_hat", proof.t_hat);
    transcript.append_field_element("tau_x", proof.tau_x);
    transcript.append_field_element("mu", proof.mu);
    
    auto w = transcript.challenge_field_element("w");
    
    // U' = w * U
    auto U_prime = curve_.to_affine(curve_.scalar_mul(gens.U, w));
    
    proof.inner_proof = prove_inner_product(
        l_vec, r_vec,
        std::vector<AffinePoint256>(gens.g_vec.begin(), gens.g_vec.begin() + n),
        h_prime,
        U_prime,
        transcript
    );
    
    return proof;
}

AggregatedRangeProof BulletproofsProver::prove_range_aggregated(
    const std::vector<uint64_t>& values,
    const std::vector<FieldElement256>& blindings,
    size_t n,
    const BulletproofsGenerators& gens) const {
    
    // For simplicity, implement as single proof for now
    // Full aggregation would interleave the bit vectors
    if (values.size() != blindings.size()) {
        throw std::invalid_argument("Values and blindings must have same size");
    }
    
    if (values.empty()) {
        throw std::invalid_argument("Must have at least one value");
    }
    
    // For single value, just wrap the regular proof
    auto proof = prove_range(values[0], blindings[0], n, gens);
    
    AggregatedRangeProof agg;
    agg.A = proof.A;
    agg.S = proof.S;
    agg.T1 = proof.T1;
    agg.T2 = proof.T2;
    agg.tau_x = proof.tau_x;
    agg.mu = proof.mu;
    agg.t_hat = proof.t_hat;
    agg.inner_proof = std::move(proof.inner_proof);
    agg.num_values = values.size();
    
    return agg;
}

BallotValidityProof BulletproofsProver::prove_ballot_validity(
    uint64_t vote,
    size_t num_candidates,
    const BulletproofsGenerators& gens) const {
    
    if (vote >= num_candidates) {
        throw std::invalid_argument("Vote out of range");
    }
    
    // Compute bits needed for range
    size_t n = bits_needed(num_candidates);
    
    // Round up to power of 2 for inner product argument
    size_t n_padded = 1;
    while (n_padded < n) n_padded *= 2;
    
    // Ensure we have enough generators
    if (gens.size() < n_padded) {
        throw std::invalid_argument("Not enough generators for ballot proof");
    }
    
    BallotValidityProof proof;
    proof.num_candidates = num_candidates;
    
    // Create commitment
    auto [commitment, blinding] = commit(vote, std::nullopt, gens);
    proof.commitment = commitment;
    
    // Create range proof
    proof.range_proof = prove_range(vote, blinding, n_padded, gens);
    
    return proof;
}

std::vector<BallotValidityProof> BulletproofsProver::prove_ballot_validity_batch(
    const std::vector<uint64_t>& votes,
    size_t num_candidates,
    const BulletproofsGenerators& gens) const {
    
    std::vector<BallotValidityProof> proofs;
    proofs.reserve(votes.size());
    
    for (uint64_t vote : votes) {
        proofs.push_back(prove_ballot_validity(vote, num_candidates, gens));
    }
    
    return proofs;
}

// ============================================================================
// BulletproofsVerifier Implementation
// ============================================================================

BulletproofsVerifier::BulletproofsVerifier(const BulletproofsConfig& config)
    : config_(config)
    , curve_(bn254_g1())
    , field_(bn254_fr()) {
}

std::vector<FieldElement256> BulletproofsVerifier::powers_of(const FieldElement256& x, size_t n) const {
    std::vector<FieldElement256> result(n);
    result[0] = field_.one();
    for (size_t i = 1; i < n; ++i) {
        result[i] = field_.mul(result[i - 1], x);
    }
    return result;
}

FieldElement256 BulletproofsVerifier::compute_delta(
    const FieldElement256& y,
    const FieldElement256& z,
    size_t n) const {
    
    auto z2 = field_.square(z);
    auto z3 = field_.mul(z2, z);
    
    auto y_powers = powers_of(y, n + 1);
    FieldElement256 sum_y_powers = field_.zero();
    for (size_t i = 0; i < n; ++i) {
        sum_y_powers = field_.add(sum_y_powers, y_powers[i]);
    }
    
    FieldElement256 two = field_.to_montgomery(FieldElement256(2));
    auto two_powers = powers_of(two, n);
    FieldElement256 sum_two_powers = field_.zero();
    for (size_t i = 0; i < n; ++i) {
        sum_two_powers = field_.add(sum_two_powers, two_powers[i]);
    }
    
    auto z_minus_z2 = field_.sub(z, z2);
    auto term1 = field_.mul(z_minus_z2, sum_y_powers);
    auto term2 = field_.mul(z3, sum_two_powers);
    
    return field_.sub(term1, term2);
}

bool BulletproofsVerifier::verify_inner_product(
    const JacobianPoint256& P,
    const FieldElement256& c,
    const InnerProductProof& proof,
    const std::vector<AffinePoint256>& g_vec,
    const std::vector<AffinePoint256>& h_vec,
    const AffinePoint256& U,
    Transcript& transcript) const {
    
    size_t n = g_vec.size();
    size_t rounds = proof.L.size();
    
    // Check that n = 2^rounds
    if ((1ULL << rounds) != n) {
        return false;
    }
    
    // Collect challenges
    std::vector<FieldElement256> challenges(rounds);
    for (size_t i = 0; i < rounds; ++i) {
        transcript.append_point("L", proof.L[i]);
        transcript.append_point("R", proof.R[i]);
        challenges[i] = transcript.challenge_field_element("x");
    }
    
    // Compute s values for final check
    // s[i] = product of x_j^(b_j) where b_j is the j-th bit of i
    std::vector<FieldElement256> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = field_.one();
        for (size_t j = 0; j < rounds; ++j) {
            size_t bit = (i >> (rounds - 1 - j)) & 1;
            if (bit == 1) {
                s[i] = field_.mul(s[i], challenges[j]);
            } else {
                s[i] = field_.mul(s[i], field_.inv(challenges[j]));
            }
        }
    }
    
    // Compute g_final = sum(s[i] * g[i])
    // Compute h_final = sum(s[i]^(-1) * h[i])
    std::vector<AffinePoint256> points;
    std::vector<FieldElement256> scalars;
    
    for (size_t i = 0; i < n; ++i) {
        points.push_back(g_vec[i]);
        scalars.push_back(field_.mul(s[i], proof.a));
        
        points.push_back(h_vec[i]);
        scalars.push_back(field_.mul(field_.inv(s[i]), proof.b));
    }
    
    // Add U * (a * b)
    points.push_back(U);
    scalars.push_back(field_.mul(proof.a, proof.b));
    
    // Add L and R contributions
    for (size_t i = 0; i < rounds; ++i) {
        auto x2 = field_.square(challenges[i]);
        auto x_inv2 = field_.inv(x2);
        
        points.push_back(proof.L[i]);
        scalars.push_back(x2);
        
        points.push_back(proof.R[i]);
        scalars.push_back(x_inv2);
    }
    
    // Compute expected P
    auto expected = curve_.msm(points.data(), scalars.data(), points.size());
    
    // Check P == expected
    auto P_affine = curve_.to_affine(P);
    auto expected_affine = curve_.to_affine(expected);
    
    return P_affine == expected_affine;
}

bool BulletproofsVerifier::verify_range(
    const PedersenCommitment& commitment,
    const RangeProof& proof,
    size_t n,
    const BulletproofsGenerators& gens) const {
    
    if (gens.size() < n) {
        return false;
    }
    
    Transcript transcript("Bulletproofs_RangeProof");
    
    // Reconstruct challenges
    transcript.append_point("A", proof.A);
    transcript.append_point("S", proof.S);
    auto y = transcript.challenge_field_element("y");
    auto z = transcript.challenge_field_element("z");
    
    transcript.append_point("T1", proof.T1);
    transcript.append_point("T2", proof.T2);
    auto x = transcript.challenge_field_element("x");
    
    transcript.append_field_element("t_hat", proof.t_hat);
    transcript.append_field_element("tau_x", proof.tau_x);
    transcript.append_field_element("mu", proof.mu);
    
    auto w = transcript.challenge_field_element("w");
    
    // Verify t_hat commitment
    // t_hat * G + tau_x * H == z^2 * V + delta * G + x * T1 + x^2 * T2
    auto z2 = field_.square(z);
    auto x2 = field_.square(x);
    auto delta = compute_delta(y, z, n);
    
    // LHS: t_hat * G + tau_x * H
    auto lhs = curve_.add(
        curve_.scalar_mul(gens.G, proof.t_hat),
        curve_.scalar_mul(gens.H, proof.tau_x)
    );
    
    // RHS: z^2 * V + delta * G + x * T1 + x^2 * T2
    auto rhs = curve_.scalar_mul(commitment.point, z2);
    rhs = curve_.add(rhs, curve_.scalar_mul(gens.G, delta));
    rhs = curve_.add(rhs, curve_.scalar_mul(proof.T1, x));
    rhs = curve_.add(rhs, curve_.scalar_mul(proof.T2, x2));
    
    auto lhs_affine = curve_.to_affine(lhs);
    auto rhs_affine = curve_.to_affine(rhs);
    
    if (lhs_affine != rhs_affine) {
        return false;
    }
    
    // Compute P for inner product verification
    // P = A + x*S - z*sum(g) + sum(z*y^i + z^2*2^i)*h_i - mu*H
    auto y_powers = powers_of(y, n);
    auto two = field_.to_montgomery(FieldElement256(2));
    auto two_powers = powers_of(two, n);
    
    std::vector<AffinePoint256> P_points;
    std::vector<FieldElement256> P_scalars;
    
    // A
    P_points.push_back(proof.A);
    P_scalars.push_back(field_.one());
    
    // x * S
    P_points.push_back(proof.S);
    P_scalars.push_back(x);
    
    // -z * sum(g_i)
    for (size_t i = 0; i < n; ++i) {
        P_points.push_back(gens.g_vec[i]);
        P_scalars.push_back(field_.neg(z));
    }

    // (z*y^i + z^2*2^i) * h_i
    auto y_inv = field_.inv(y);
    auto y_inv_powers = powers_of(y_inv, n);
    
    std::vector<AffinePoint256> h_prime(n);
    for (size_t i = 0; i < n; ++i) {
        auto h_scaled = curve_.scalar_mul(gens.h_vec[i], y_inv_powers[i]);
        h_prime[i] = curve_.to_affine(h_scaled);
        
        auto coeff = field_.add(field_.mul(z, y_powers[i]), field_.mul(z2, two_powers[i]));
        P_points.push_back(h_prime[i]);
        P_scalars.push_back(coeff);
    }
    
    // -mu * H
    P_points.push_back(gens.H);
    P_scalars.push_back(field_.neg(proof.mu));
    
    auto P = curve_.msm(P_points.data(), P_scalars.data(), P_points.size());
    
    // U' = w * U
    auto U_prime = curve_.to_affine(curve_.scalar_mul(gens.U, w));
    
    // Verify inner product
    return verify_inner_product(
        P, proof.t_hat, proof.inner_proof,
        std::vector<AffinePoint256>(gens.g_vec.begin(), gens.g_vec.begin() + n),
        h_prime,
        U_prime,
        transcript
    );
}

bool BulletproofsVerifier::verify_range_aggregated(
    const std::vector<PedersenCommitment>& commitments,
    const AggregatedRangeProof& proof,
    size_t n,
    const BulletproofsGenerators& gens) const {
    
    if (commitments.empty()) {
        return false;
    }
    
    // For single value, verify as regular range proof
    RangeProof range_proof;
    range_proof.A = proof.A;
    range_proof.S = proof.S;
    range_proof.T1 = proof.T1;
    range_proof.T2 = proof.T2;
    range_proof.tau_x = proof.tau_x;
    range_proof.mu = proof.mu;
    range_proof.t_hat = proof.t_hat;
    range_proof.inner_proof = proof.inner_proof;
    
    return verify_range(commitments[0], range_proof, n, gens);
}

bool BulletproofsVerifier::verify_ballot_validity(
    const BallotValidityProof& proof,
    const BulletproofsGenerators& gens) const {
    
    size_t n = bits_needed(proof.num_candidates);
    size_t n_padded = 1;
    while (n_padded < n) n_padded *= 2;
    
    return verify_range(proof.commitment, proof.range_proof, n_padded, gens);
}

bool BulletproofsVerifier::batch_verify_range(
    const std::vector<PedersenCommitment>& commitments,
    const std::vector<RangeProof>& proofs,
    size_t n,
    const BulletproofsGenerators& gens) const {
    
    if (commitments.size() != proofs.size()) {
        return false;
    }
    
    // For now, verify individually
    // Full batch verification would combine all checks into single MSM
    for (size_t i = 0; i < commitments.size(); ++i) {
        if (!verify_range(commitments[i], proofs[i], n, gens)) {
            return false;
        }
    }
    
    return true;
}

bool BulletproofsVerifier::batch_verify_ballot_validity(
    const std::vector<BallotValidityProof>& proofs,
    const BulletproofsGenerators& gens) const {
    
    for (const auto& proof : proofs) {
        if (!verify_ballot_validity(proof, gens)) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// Default Generators
// ============================================================================

BulletproofsGenerators default_generators(size_t n) {
    BulletproofsProver prover;
    return prover.generate_generators(n);
}

} // namespace zk
} // namespace fhe_accelerate
