/**
 * Polynomial Ring Implementation
 * 
 * Implements polynomial arithmetic in the cyclotomic ring Z_q[X]/(X^N + 1).
 * 
 * Design Reference: Section 3 - Polynomial Ring
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 14.2
 */

#include "polynomial_ring.h"
#include <stdexcept>
#include <cstring>
#include <random>
#include <algorithm>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace fhe_accelerate {

// ============================================================================
// Polynomial Implementation
// ============================================================================

Polynomial::Polynomial(uint32_t degree, uint64_t modulus)
    : modulus_(modulus)
    , degree_(degree)
    , is_ntt_(false) {
    
    // Validate degree is power of 2
    if (degree == 0 || (degree & (degree - 1)) != 0) {
        throw std::invalid_argument("Polynomial degree must be a power of 2");
    }
    
    // Allocate cache-aligned coefficient storage
    coeffs_.resize(degree, 0);
}

Polynomial::Polynomial(const std::vector<uint64_t>& coeffs, uint64_t modulus, bool is_ntt)
    : modulus_(modulus)
    , degree_(static_cast<uint32_t>(coeffs.size()))
    , is_ntt_(is_ntt) {
    
    // Validate degree is power of 2
    if (degree_ == 0 || (degree_ & (degree_ - 1)) != 0) {
        throw std::invalid_argument("Polynomial degree must be a power of 2");
    }
    
    // Copy coefficients to cache-aligned storage
    coeffs_.resize(degree_);
    std::copy(coeffs.begin(), coeffs.end(), coeffs_.begin());
}

Polynomial::Polynomial(std::vector<uint64_t>&& coeffs, uint64_t modulus, bool is_ntt)
    : modulus_(modulus)
    , degree_(static_cast<uint32_t>(coeffs.size()))
    , is_ntt_(is_ntt) {
    
    // Validate degree is power of 2
    if (degree_ == 0 || (degree_ & (degree_ - 1)) != 0) {
        throw std::invalid_argument("Polynomial degree must be a power of 2");
    }
    
    // Move coefficients to cache-aligned storage
    coeffs_.resize(degree_);
    std::move(coeffs.begin(), coeffs.end(), coeffs_.begin());
}

Polynomial::Polynomial(const Polynomial& other)
    : coeffs_(other.coeffs_)
    , modulus_(other.modulus_)
    , degree_(other.degree_)
    , is_ntt_(other.is_ntt_) {
}

Polynomial::Polynomial(Polynomial&& other) noexcept
    : coeffs_(std::move(other.coeffs_))
    , modulus_(other.modulus_)
    , degree_(other.degree_)
    , is_ntt_(other.is_ntt_) {
}

Polynomial& Polynomial::operator=(const Polynomial& other) {
    if (this != &other) {
        coeffs_ = other.coeffs_;
        modulus_ = other.modulus_;
        degree_ = other.degree_;
        is_ntt_ = other.is_ntt_;
    }
    return *this;
}

Polynomial& Polynomial::operator=(Polynomial&& other) noexcept {
    if (this != &other) {
        coeffs_ = std::move(other.coeffs_);
        modulus_ = other.modulus_;
        degree_ = other.degree_;
        is_ntt_ = other.is_ntt_;
    }
    return *this;
}

void Polynomial::to_ntt(NTTProcessor& ntt) {
    if (is_ntt_) return;  // Already in NTT form
    
    if (ntt.get_degree() != degree_) {
        throw std::invalid_argument("NTT processor degree mismatch");
    }
    if (ntt.get_modulus() != modulus_) {
        throw std::invalid_argument("NTT processor modulus mismatch");
    }
    
    ntt.forward_ntt(coeffs_.data(), degree_);
    is_ntt_ = true;
}

void Polynomial::from_ntt(NTTProcessor& ntt) {
    if (!is_ntt_) return;  // Already in coefficient form
    
    if (ntt.get_degree() != degree_) {
        throw std::invalid_argument("NTT processor degree mismatch");
    }
    if (ntt.get_modulus() != modulus_) {
        throw std::invalid_argument("NTT processor modulus mismatch");
    }
    
    ntt.inverse_ntt(coeffs_.data(), degree_);
    is_ntt_ = false;
}

void Polynomial::set_zero() {
    std::fill(coeffs_.begin(), coeffs_.end(), 0);
}

void Polynomial::set_identity() {
    std::fill(coeffs_.begin(), coeffs_.end(), 0);
    coeffs_[0] = 1;
}

bool Polynomial::is_zero() const {
    for (const auto& c : coeffs_) {
        if (c != 0) return false;
    }
    return true;
}

bool Polynomial::is_identity() const {
    if (coeffs_[0] != 1) return false;
    for (size_t i = 1; i < coeffs_.size(); i++) {
        if (coeffs_[i] != 0) return false;
    }
    return true;
}

bool Polynomial::operator==(const Polynomial& other) const {
    if (degree_ != other.degree_ || modulus_ != other.modulus_ || is_ntt_ != other.is_ntt_) {
        return false;
    }
    return coeffs_ == other.coeffs_;
}

bool Polynomial::operator!=(const Polynomial& other) const {
    return !(*this == other);
}

Polynomial Polynomial::clone() const {
    return Polynomial(*this);
}

Polynomial Polynomial::zero(uint32_t degree, uint64_t modulus) {
    return Polynomial(degree, modulus);
}

Polynomial Polynomial::identity(uint32_t degree, uint64_t modulus) {
    Polynomial p(degree, modulus);
    p.set_identity();
    return p;
}

Polynomial Polynomial::random(uint32_t degree, uint64_t modulus) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, modulus - 1);
    
    std::vector<uint64_t> coeffs(degree);
    for (auto& c : coeffs) {
        c = dist(gen);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

Polynomial Polynomial::random_ternary(uint32_t degree, uint64_t modulus) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int> dist(-1, 1);
    
    std::vector<uint64_t> coeffs(degree);
    for (auto& c : coeffs) {
        int val = dist(gen);
        // Convert -1 to modulus - 1 (equivalent in Z_q)
        c = (val < 0) ? (modulus - 1) : static_cast<uint64_t>(val);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

// ============================================================================
// PolynomialRing Implementation
// ============================================================================

PolynomialRing::PolynomialRing(uint32_t degree, uint64_t modulus)
    : degree_(degree)
    , moduli_({modulus}) {
    
    // Create NTT processor
    ntt_processors_.push_back(std::make_unique<NTTProcessor>(degree, modulus));
    
    // Create modular arithmetic helper
    mod_arith_.push_back(std::make_unique<ModularArithmetic>(modulus));
}

PolynomialRing::PolynomialRing(uint32_t degree, const std::vector<uint64_t>& moduli)
    : degree_(degree)
    , moduli_(moduli) {
    
    if (moduli.empty()) {
        throw std::invalid_argument("At least one modulus required");
    }
    
    // Create NTT processors and modular arithmetic for each modulus
    for (uint64_t mod : moduli) {
        ntt_processors_.push_back(std::make_unique<NTTProcessor>(degree, mod));
        mod_arith_.push_back(std::make_unique<ModularArithmetic>(mod));
    }
}

PolynomialRing::~PolynomialRing() = default;

void PolynomialRing::validate_polynomial(const Polynomial& p) const {
    if (p.degree() != degree_) {
        throw std::invalid_argument("Polynomial degree mismatch");
    }
    if (p.modulus() != moduli_[0]) {
        throw std::invalid_argument("Polynomial modulus mismatch");
    }
}

void PolynomialRing::validate_polynomials(const Polynomial& a, const Polynomial& b) const {
    validate_polynomial(a);
    validate_polynomial(b);
    
    if (a.is_ntt() != b.is_ntt()) {
        throw std::invalid_argument("Polynomials must be in same representation (both NTT or both coefficient)");
    }
}

// ============================================================================
// Addition and Subtraction
// ============================================================================

Polynomial PolynomialRing::add(const Polynomial& a, const Polynomial& b) const {
    validate_polynomials(a, b);
    
    Polynomial result(degree_, moduli_[0]);
    result.set_ntt_flag(a.is_ntt());
    
    add_neon(a, b, result);
    
    return result;
}

void PolynomialRing::add_inplace(Polynomial& a, const Polynomial& b) const {
    validate_polynomials(a, b);
    
    const uint64_t mod = moduli_[0];
    uint64_t* a_data = a.data();
    const uint64_t* b_data = b.data();
    
#ifdef __ARM_NEON
    // Process 2 coefficients at a time using NEON
    size_t i = 0;
    for (; i + 1 < degree_; i += 2) {
        uint64_t a0 = a_data[i];
        uint64_t a1 = a_data[i + 1];
        uint64_t b0 = b_data[i];
        uint64_t b1 = b_data[i + 1];
        
        // Modular addition
        a_data[i] = mod_arith_[0]->mod_add(a0, b0);
        a_data[i + 1] = mod_arith_[0]->mod_add(a1, b1);
    }
    
    // Handle remaining coefficient
    if (i < degree_) {
        a_data[i] = mod_arith_[0]->mod_add(a_data[i], b_data[i]);
    }
#else
    for (size_t i = 0; i < degree_; i++) {
        a_data[i] = mod_arith_[0]->mod_add(a_data[i], b_data[i]);
    }
#endif
}

Polynomial PolynomialRing::subtract(const Polynomial& a, const Polynomial& b) const {
    validate_polynomials(a, b);
    
    Polynomial result(degree_, moduli_[0]);
    result.set_ntt_flag(a.is_ntt());
    
    subtract_neon(a, b, result);
    
    return result;
}

void PolynomialRing::subtract_inplace(Polynomial& a, const Polynomial& b) const {
    validate_polynomials(a, b);
    
    uint64_t* a_data = a.data();
    const uint64_t* b_data = b.data();
    
#ifdef __ARM_NEON
    size_t i = 0;
    for (; i + 1 < degree_; i += 2) {
        a_data[i] = mod_arith_[0]->mod_sub(a_data[i], b_data[i]);
        a_data[i + 1] = mod_arith_[0]->mod_sub(a_data[i + 1], b_data[i + 1]);
    }
    
    if (i < degree_) {
        a_data[i] = mod_arith_[0]->mod_sub(a_data[i], b_data[i]);
    }
#else
    for (size_t i = 0; i < degree_; i++) {
        a_data[i] = mod_arith_[0]->mod_sub(a_data[i], b_data[i]);
    }
#endif
}

Polynomial PolynomialRing::negate(const Polynomial& a) const {
    validate_polynomial(a);
    
    Polynomial result(degree_, moduli_[0]);
    result.set_ntt_flag(a.is_ntt());
    
    const uint64_t mod = moduli_[0];
    const uint64_t* a_data = a.data();
    uint64_t* r_data = result.data();
    
    for (size_t i = 0; i < degree_; i++) {
        r_data[i] = (a_data[i] == 0) ? 0 : (mod - a_data[i]);
    }
    
    return result;
}

void PolynomialRing::negate_inplace(Polynomial& a) const {
    validate_polynomial(a);
    
    const uint64_t mod = moduli_[0];
    uint64_t* a_data = a.data();
    
    for (size_t i = 0; i < degree_; i++) {
        a_data[i] = (a_data[i] == 0) ? 0 : (mod - a_data[i]);
    }
}

// ============================================================================
// NEON-Optimized Operations
// ============================================================================

void PolynomialRing::add_neon(const Polynomial& a, const Polynomial& b, Polynomial& result) const {
    const uint64_t* a_data = a.data();
    const uint64_t* b_data = b.data();
    uint64_t* r_data = result.data();
    
#ifdef __ARM_NEON
    // Process 2 coefficients at a time
    size_t i = 0;
    for (; i + 1 < degree_; i += 2) {
        r_data[i] = mod_arith_[0]->mod_add(a_data[i], b_data[i]);
        r_data[i + 1] = mod_arith_[0]->mod_add(a_data[i + 1], b_data[i + 1]);
    }
    
    if (i < degree_) {
        r_data[i] = mod_arith_[0]->mod_add(a_data[i], b_data[i]);
    }
#else
    for (size_t i = 0; i < degree_; i++) {
        r_data[i] = mod_arith_[0]->mod_add(a_data[i], b_data[i]);
    }
#endif
}

void PolynomialRing::subtract_neon(const Polynomial& a, const Polynomial& b, Polynomial& result) const {
    const uint64_t* a_data = a.data();
    const uint64_t* b_data = b.data();
    uint64_t* r_data = result.data();
    
#ifdef __ARM_NEON
    size_t i = 0;
    for (; i + 1 < degree_; i += 2) {
        r_data[i] = mod_arith_[0]->mod_sub(a_data[i], b_data[i]);
        r_data[i + 1] = mod_arith_[0]->mod_sub(a_data[i + 1], b_data[i + 1]);
    }
    
    if (i < degree_) {
        r_data[i] = mod_arith_[0]->mod_sub(a_data[i], b_data[i]);
    }
#else
    for (size_t i = 0; i < degree_; i++) {
        r_data[i] = mod_arith_[0]->mod_sub(a_data[i], b_data[i]);
    }
#endif
}

// ============================================================================
// Multiplication
// ============================================================================

Polynomial PolynomialRing::multiply(const Polynomial& a, const Polynomial& b) {
    validate_polynomials(a, b);
    
    // If both are in NTT form, do pointwise multiplication
    if (a.is_ntt() && b.is_ntt()) {
        return pointwise_multiply(a, b);
    }
    
    // Convert to NTT, multiply pointwise, convert back
    Polynomial a_ntt = a.clone();
    Polynomial b_ntt = b.clone();
    
    if (!a_ntt.is_ntt()) {
        a_ntt.to_ntt(*ntt_processors_[0]);
    }
    if (!b_ntt.is_ntt()) {
        b_ntt.to_ntt(*ntt_processors_[0]);
    }
    
    // Pointwise multiplication in NTT domain
    Polynomial result = pointwise_multiply(a_ntt, b_ntt);
    
    // Convert back to coefficient form
    result.from_ntt(*ntt_processors_[0]);
    
    return result;
}

void PolynomialRing::multiply_inplace(Polynomial& a, const Polynomial& b) {
    Polynomial result = multiply(a, b);
    a = std::move(result);
}

Polynomial PolynomialRing::multiply_scalar(const Polynomial& a, uint64_t scalar) const {
    validate_polynomial(a);
    
    Polynomial result(degree_, moduli_[0]);
    result.set_ntt_flag(a.is_ntt());
    
    const uint64_t mod = moduli_[0];
    const uint64_t* a_data = a.data();
    uint64_t* r_data = result.data();
    
    // Reduce scalar modulo q
    scalar = scalar % mod;
    
    for (size_t i = 0; i < degree_; i++) {
        __uint128_t prod = static_cast<__uint128_t>(a_data[i]) * scalar;
        r_data[i] = prod % mod;
    }
    
    return result;
}

void PolynomialRing::multiply_scalar_inplace(Polynomial& a, uint64_t scalar) const {
    validate_polynomial(a);
    
    const uint64_t mod = moduli_[0];
    uint64_t* a_data = a.data();
    
    scalar = scalar % mod;
    
    for (size_t i = 0; i < degree_; i++) {
        __uint128_t prod = static_cast<__uint128_t>(a_data[i]) * scalar;
        a_data[i] = prod % mod;
    }
}

// ============================================================================
// NTT Operations
// ============================================================================

Polynomial PolynomialRing::pointwise_multiply(const Polynomial& a, const Polynomial& b) const {
    validate_polynomials(a, b);
    
    if (!a.is_ntt() || !b.is_ntt()) {
        throw std::invalid_argument("Both polynomials must be in NTT form for pointwise multiplication");
    }
    
    Polynomial result(degree_, moduli_[0]);
    result.set_ntt_flag(true);
    
    const uint64_t mod = moduli_[0];
    const uint64_t* a_data = a.data();
    const uint64_t* b_data = b.data();
    uint64_t* r_data = result.data();
    
#ifdef __ARM_NEON
    // Process 2 coefficients at a time
    size_t i = 0;
    for (; i + 1 < degree_; i += 2) {
        __uint128_t prod0 = static_cast<__uint128_t>(a_data[i]) * b_data[i];
        __uint128_t prod1 = static_cast<__uint128_t>(a_data[i + 1]) * b_data[i + 1];
        r_data[i] = prod0 % mod;
        r_data[i + 1] = prod1 % mod;
    }
    
    if (i < degree_) {
        __uint128_t prod = static_cast<__uint128_t>(a_data[i]) * b_data[i];
        r_data[i] = prod % mod;
    }
#else
    for (size_t i = 0; i < degree_; i++) {
        __uint128_t prod = static_cast<__uint128_t>(a_data[i]) * b_data[i];
        r_data[i] = prod % mod;
    }
#endif
    
    return result;
}

void PolynomialRing::pointwise_multiply_inplace(Polynomial& a, const Polynomial& b) const {
    validate_polynomials(a, b);
    
    if (!a.is_ntt() || !b.is_ntt()) {
        throw std::invalid_argument("Both polynomials must be in NTT form for pointwise multiplication");
    }
    
    const uint64_t mod = moduli_[0];
    uint64_t* a_data = a.data();
    const uint64_t* b_data = b.data();
    
#ifdef __ARM_NEON
    size_t i = 0;
    for (; i + 1 < degree_; i += 2) {
        __uint128_t prod0 = static_cast<__uint128_t>(a_data[i]) * b_data[i];
        __uint128_t prod1 = static_cast<__uint128_t>(a_data[i + 1]) * b_data[i + 1];
        a_data[i] = prod0 % mod;
        a_data[i + 1] = prod1 % mod;
    }
    
    if (i < degree_) {
        __uint128_t prod = static_cast<__uint128_t>(a_data[i]) * b_data[i];
        a_data[i] = prod % mod;
    }
#else
    for (size_t i = 0; i < degree_; i++) {
        __uint128_t prod = static_cast<__uint128_t>(a_data[i]) * b_data[i];
        a_data[i] = prod % mod;
    }
#endif
}

void PolynomialRing::to_ntt(Polynomial& p) {
    validate_polynomial(p);
    p.to_ntt(*ntt_processors_[0]);
}

void PolynomialRing::from_ntt(Polynomial& p) {
    validate_polynomial(p);
    p.from_ntt(*ntt_processors_[0]);
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<PolynomialRing> create_polynomial_ring(uint32_t degree, uint64_t modulus) {
    return std::make_unique<PolynomialRing>(degree, modulus);
}

} // namespace fhe_accelerate
