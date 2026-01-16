/**
 * Zero-Knowledge Proof Field Arithmetic Implementation
 * 
 * Implements Montgomery arithmetic for ZK proof system fields:
 * - BLS12-381 scalar field (Fr) and base field (Fq)
 * - BN254 scalar field (Fr) and base field (Fq)
 * 
 * Requirements: 19, 20
 */

#include "zk_field_arithmetic.h"
#include <cstring>
#include <random>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// FieldElement256 Implementation
// ============================================================================

bool FieldElement256::operator==(const FieldElement256& other) const {
    return limbs[0] == other.limbs[0] && limbs[1] == other.limbs[1] &&
           limbs[2] == other.limbs[2] && limbs[3] == other.limbs[3];
}

bool FieldElement256::operator<(const FieldElement256& other) const {
    // Compare from most significant limb
    for (int i = 3; i >= 0; --i) {
        if (limbs[i] < other.limbs[i]) return true;
        if (limbs[i] > other.limbs[i]) return false;
    }
    return false;  // Equal
}

bool FieldElement256::is_zero() const {
    return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
}

std::string FieldElement256::to_hex() const {
    std::ostringstream oss;
    oss << "0x";
    for (int i = 3; i >= 0; --i) {
        oss << std::hex << std::setfill('0') << std::setw(16) << limbs[i];
    }
    return oss.str();
}

FieldElement256 FieldElement256::from_hex(const std::string& hex) {
    FieldElement256 result;
    std::string h = hex;
    
    // Remove 0x prefix if present
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) {
        h = h.substr(2);
    }
    
    // Pad to 64 characters
    while (h.size() < 64) {
        h = "0" + h;
    }
    
    // Parse 16 hex chars per limb, from most significant
    for (int i = 0; i < 4; ++i) {
        std::string limb_hex = h.substr((3 - i) * 16, 16);
        result.limbs[i] = std::stoull(limb_hex, nullptr, 16);
    }
    
    return result;
}

std::array<uint8_t, 32> FieldElement256::to_bytes() const {
    std::array<uint8_t, 32> bytes;
    // Big-endian: most significant byte first
    for (int i = 3; i >= 0; --i) {
        for (int j = 7; j >= 0; --j) {
            bytes[(3 - i) * 8 + (7 - j)] = (limbs[i] >> (j * 8)) & 0xFF;
        }
    }
    return bytes;
}

FieldElement256 FieldElement256::from_bytes(const std::array<uint8_t, 32>& bytes) {
    FieldElement256 result;
    for (int i = 0; i < 4; ++i) {
        result.limbs[3 - i] = 0;
        for (int j = 0; j < 8; ++j) {
            result.limbs[3 - i] |= static_cast<uint64_t>(bytes[i * 8 + j]) << ((7 - j) * 8);
        }
    }
    return result;
}

// ============================================================================
// FieldElement384 Implementation
// ============================================================================

bool FieldElement384::operator==(const FieldElement384& other) const {
    for (int i = 0; i < 6; ++i) {
        if (limbs[i] != other.limbs[i]) return false;
    }
    return true;
}

bool FieldElement384::operator<(const FieldElement384& other) const {
    for (int i = 5; i >= 0; --i) {
        if (limbs[i] < other.limbs[i]) return true;
        if (limbs[i] > other.limbs[i]) return false;
    }
    return false;
}

bool FieldElement384::is_zero() const {
    for (int i = 0; i < 6; ++i) {
        if (limbs[i] != 0) return false;
    }
    return true;
}

std::string FieldElement384::to_hex() const {
    std::ostringstream oss;
    oss << "0x";
    for (int i = 5; i >= 0; --i) {
        oss << std::hex << std::setfill('0') << std::setw(16) << limbs[i];
    }
    return oss.str();
}

FieldElement384 FieldElement384::from_hex(const std::string& hex) {
    FieldElement384 result;
    std::string h = hex;
    
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) {
        h = h.substr(2);
    }
    
    while (h.size() < 96) {
        h = "0" + h;
    }
    
    for (int i = 0; i < 6; ++i) {
        std::string limb_hex = h.substr((5 - i) * 16, 16);
        result.limbs[i] = std::stoull(limb_hex, nullptr, 16);
    }
    
    return result;
}

std::array<uint8_t, 48> FieldElement384::to_bytes() const {
    std::array<uint8_t, 48> bytes;
    for (int i = 5; i >= 0; --i) {
        for (int j = 7; j >= 0; --j) {
            bytes[(5 - i) * 8 + (7 - j)] = (limbs[i] >> (j * 8)) & 0xFF;
        }
    }
    return bytes;
}

FieldElement384 FieldElement384::from_bytes(const std::array<uint8_t, 48>& bytes) {
    FieldElement384 result;
    for (int i = 0; i < 6; ++i) {
        result.limbs[5 - i] = 0;
        for (int j = 0; j < 8; ++j) {
            result.limbs[5 - i] |= static_cast<uint64_t>(bytes[i * 8 + j]) << ((7 - j) * 8);
        }
    }
    return result;
}

// ============================================================================
// Field256 Implementation
// ============================================================================

Field256::Field256(const std::array<uint64_t, 4>& modulus,
                   const std::array<uint64_t, 4>& r,
                   const std::array<uint64_t, 4>& r2,
                   uint64_t inv)
    : modulus_(modulus), r_(r), r2_(r2), zero_(), inv_(inv) {
}

void Field256::add_with_carry(const uint64_t* a, const uint64_t* b,
                              uint64_t* result, uint64_t& carry) {
    carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t sum = static_cast<__uint128_t>(a[i]) + b[i] + carry;
        result[i] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
}

void Field256::sub_with_borrow(const uint64_t* a, const uint64_t* b,
                               uint64_t* result, uint64_t& borrow) {
    borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t ai = a[i];
        uint64_t bi = b[i];
        uint64_t diff = ai - bi - borrow;
        borrow = (ai < bi + borrow) ? 1 : 0;
        result[i] = diff;
    }
}

void Field256::mul_wide(const uint64_t* a, const uint64_t* b, uint64_t* result) {
    // Schoolbook multiplication producing 8-limb result
    std::memset(result, 0, 8 * sizeof(uint64_t));
    
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            __uint128_t prod = static_cast<__uint128_t>(a[i]) * b[j];
            prod += result[i + j];
            prod += carry;
            result[i + j] = static_cast<uint64_t>(prod);
            carry = static_cast<uint64_t>(prod >> 64);
        }
        result[i + 4] = carry;
    }
}

FieldElement256 Field256::montgomery_reduce(const std::array<uint64_t, 8>& product) const {
    // Montgomery reduction for 256-bit field
    // Algorithm: REDC
    std::array<uint64_t, 8> t;
    std::copy(product.begin(), product.end(), t.begin());
    
    for (int i = 0; i < 4; ++i) {
        // m = t[i] * inv mod 2^64
        uint64_t m = t[i] * inv_;
        
        // t = t + m * modulus * 2^(64*i)
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            __uint128_t prod = static_cast<__uint128_t>(m) * modulus_.limbs[j];
            prod += t[i + j];
            prod += carry;
            t[i + j] = static_cast<uint64_t>(prod);
            carry = static_cast<uint64_t>(prod >> 64);
        }
        
        // Propagate carry
        for (int j = 4; j < 8 - i && carry; ++j) {
            __uint128_t sum = static_cast<__uint128_t>(t[i + j]) + carry;
            t[i + j] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
    }
    
    // Extract high 4 limbs
    FieldElement256 result;
    for (int i = 0; i < 4; ++i) {
        result.limbs[i] = t[4 + i];
    }
    
    // Final conditional subtraction
    if (result >= modulus_) {
        uint64_t borrow;
        sub_with_borrow(result.limbs.data(), modulus_.limbs.data(),
                        result.limbs.data(), borrow);
    }
    
    return result;
}

FieldElement256 Field256::add(const FieldElement256& a, const FieldElement256& b) const {
    FieldElement256 result;
    uint64_t carry;
    add_with_carry(a.limbs.data(), b.limbs.data(), result.limbs.data(), carry);
    
    // Conditional subtraction if result >= modulus or carry
    if (carry || result >= modulus_) {
        uint64_t borrow;
        sub_with_borrow(result.limbs.data(), modulus_.limbs.data(),
                        result.limbs.data(), borrow);
    }
    
    return result;
}

FieldElement256 Field256::sub(const FieldElement256& a, const FieldElement256& b) const {
    FieldElement256 result;
    uint64_t borrow;
    sub_with_borrow(a.limbs.data(), b.limbs.data(), result.limbs.data(), borrow);
    
    // If borrow, add modulus
    if (borrow) {
        uint64_t carry;
        add_with_carry(result.limbs.data(), modulus_.limbs.data(),
                       result.limbs.data(), carry);
    }
    
    return result;
}

FieldElement256 Field256::mul(const FieldElement256& a, const FieldElement256& b) const {
    std::array<uint64_t, 8> product;
    mul_wide(a.limbs.data(), b.limbs.data(), product.data());
    return montgomery_reduce(product);
}

FieldElement256 Field256::neg(const FieldElement256& a) const {
    if (a.is_zero()) {
        return a;
    }
    return sub(modulus_, a);
}

FieldElement256 Field256::square(const FieldElement256& a) const {
    return mul(a, a);
}

FieldElement256 Field256::inv(const FieldElement256& a) const {
    if (a.is_zero()) {
        throw std::invalid_argument("Cannot invert zero");
    }
    
    // Compute a^(p-2) mod p using Fermat's little theorem
    // p - 2 = modulus - 2
    FieldElement256 exp = modulus_;
    
    // Subtract 2 from exp
    if (exp.limbs[0] >= 2) {
        exp.limbs[0] -= 2;
    } else {
        exp.limbs[0] = exp.limbs[0] - 2;  // Will wrap, need borrow
        for (int i = 1; i < 4; ++i) {
            if (exp.limbs[i] > 0) {
                exp.limbs[i]--;
                break;
            }
            exp.limbs[i] = UINT64_MAX;
        }
    }
    
    return pow(a, exp);
}

FieldElement256 Field256::pow(const FieldElement256& base, const FieldElement256& exp) const {
    FieldElement256 result = r_;  // 1 in Montgomery form
    FieldElement256 b = base;
    
    // Binary exponentiation
    for (int i = 0; i < 4; ++i) {
        uint64_t e = exp.limbs[i];
        for (int j = 0; j < 64; ++j) {
            if (e & 1) {
                result = mul(result, b);
            }
            b = square(b);
            e >>= 1;
        }
    }
    
    return result;
}

FieldElement256 Field256::to_montgomery(const FieldElement256& a) const {
    return mul(a, r2_);
}

FieldElement256 Field256::from_montgomery(const FieldElement256& a) const {
    std::array<uint64_t, 8> product = {0};
    for (int i = 0; i < 4; ++i) {
        product[i] = a.limbs[i];
    }
    return montgomery_reduce(product);
}

void Field256::batch_add(const FieldElement256* a, const FieldElement256* b,
                         FieldElement256* result, size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        result[i] = add(a[i], b[i]);
    }
}

void Field256::batch_mul(const FieldElement256* a, const FieldElement256* b,
                         FieldElement256* result, size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        result[i] = mul(a[i], b[i]);
    }
}

// ============================================================================
// Field384 Implementation
// ============================================================================

Field384::Field384(const std::array<uint64_t, 6>& modulus,
                   const std::array<uint64_t, 6>& r,
                   const std::array<uint64_t, 6>& r2,
                   uint64_t inv)
    : modulus_(modulus), r_(r), r2_(r2), zero_(), inv_(inv) {
}

FieldElement384 Field384::montgomery_reduce(const std::array<uint64_t, 12>& product) const {
    std::array<uint64_t, 12> t;
    std::copy(product.begin(), product.end(), t.begin());
    
    for (int i = 0; i < 6; ++i) {
        uint64_t m = t[i] * inv_;
        
        uint64_t carry = 0;
        for (int j = 0; j < 6; ++j) {
            __uint128_t prod = static_cast<__uint128_t>(m) * modulus_.limbs[j];
            prod += t[i + j];
            prod += carry;
            t[i + j] = static_cast<uint64_t>(prod);
            carry = static_cast<uint64_t>(prod >> 64);
        }
        
        for (int j = 6; j < 12 - i && carry; ++j) {
            __uint128_t sum = static_cast<__uint128_t>(t[i + j]) + carry;
            t[i + j] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
    }
    
    FieldElement384 result;
    for (int i = 0; i < 6; ++i) {
        result.limbs[i] = t[6 + i];
    }
    
    if (result >= modulus_) {
        uint64_t borrow = 0;
        for (int i = 0; i < 6; ++i) {
            uint64_t ai = result.limbs[i];
            uint64_t bi = modulus_.limbs[i];
            uint64_t diff = ai - bi - borrow;
            borrow = (ai < bi + borrow) ? 1 : 0;
            result.limbs[i] = diff;
        }
    }
    
    return result;
}

FieldElement384 Field384::add(const FieldElement384& a, const FieldElement384& b) const {
    FieldElement384 result;
    uint64_t carry = 0;
    
    for (int i = 0; i < 6; ++i) {
        __uint128_t sum = static_cast<__uint128_t>(a.limbs[i]) + b.limbs[i] + carry;
        result.limbs[i] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
    
    if (carry || result >= modulus_) {
        uint64_t borrow = 0;
        for (int i = 0; i < 6; ++i) {
            uint64_t ri = result.limbs[i];
            uint64_t mi = modulus_.limbs[i];
            uint64_t diff = ri - mi - borrow;
            borrow = (ri < mi + borrow) ? 1 : 0;
            result.limbs[i] = diff;
        }
    }
    
    return result;
}

FieldElement384 Field384::sub(const FieldElement384& a, const FieldElement384& b) const {
    FieldElement384 result;
    uint64_t borrow = 0;
    
    for (int i = 0; i < 6; ++i) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t diff = ai - bi - borrow;
        borrow = (ai < bi + borrow) ? 1 : 0;
        result.limbs[i] = diff;
    }
    
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 6; ++i) {
            __uint128_t sum = static_cast<__uint128_t>(result.limbs[i]) + modulus_.limbs[i] + carry;
            result.limbs[i] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
    }
    
    return result;
}

FieldElement384 Field384::mul(const FieldElement384& a, const FieldElement384& b) const {
    std::array<uint64_t, 12> product = {0};
    
    for (int i = 0; i < 6; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 6; ++j) {
            __uint128_t prod = static_cast<__uint128_t>(a.limbs[i]) * b.limbs[j];
            prod += product[i + j];
            prod += carry;
            product[i + j] = static_cast<uint64_t>(prod);
            carry = static_cast<uint64_t>(prod >> 64);
        }
        product[i + 6] = carry;
    }
    
    return montgomery_reduce(product);
}

FieldElement384 Field384::neg(const FieldElement384& a) const {
    if (a.is_zero()) {
        return a;
    }
    return sub(modulus_, a);
}

FieldElement384 Field384::square(const FieldElement384& a) const {
    return mul(a, a);
}

FieldElement384 Field384::inv(const FieldElement384& a) const {
    if (a.is_zero()) {
        throw std::invalid_argument("Cannot invert zero");
    }
    
    FieldElement384 exp = modulus_;
    if (exp.limbs[0] >= 2) {
        exp.limbs[0] -= 2;
    } else {
        exp.limbs[0] = exp.limbs[0] - 2;
        for (int i = 1; i < 6; ++i) {
            if (exp.limbs[i] > 0) {
                exp.limbs[i]--;
                break;
            }
            exp.limbs[i] = UINT64_MAX;
        }
    }
    
    return pow(a, exp);
}

FieldElement384 Field384::pow(const FieldElement384& base, const FieldElement384& exp) const {
    FieldElement384 result = r_;
    FieldElement384 b = base;
    
    for (int i = 0; i < 6; ++i) {
        uint64_t e = exp.limbs[i];
        for (int j = 0; j < 64; ++j) {
            if (e & 1) {
                result = mul(result, b);
            }
            b = square(b);
            e >>= 1;
        }
    }
    
    return result;
}

FieldElement384 Field384::to_montgomery(const FieldElement384& a) const {
    return mul(a, r2_);
}

FieldElement384 Field384::from_montgomery(const FieldElement384& a) const {
    std::array<uint64_t, 12> product = {0};
    for (int i = 0; i < 6; ++i) {
        product[i] = a.limbs[i];
    }
    return montgomery_reduce(product);
}

// ============================================================================
// Pre-configured Field Instances
// ============================================================================

const Field256& bls12_381_fr() {
    static Field256 field(
        bls12_381::FR_MODULUS,
        bls12_381::FR_R,
        bls12_381::FR_R2,
        bls12_381::FR_INV
    );
    return field;
}

const Field384& bls12_381_fq() {
    static Field384 field(
        bls12_381::FQ_MODULUS,
        bls12_381::FQ_R,
        bls12_381::FQ_R2,
        bls12_381::FQ_INV
    );
    return field;
}

const Field256& bn254_fr() {
    static Field256 field(
        bn254::FR_MODULUS,
        bn254::FR_R,
        bn254::FR_R2,
        bn254::FR_INV
    );
    return field;
}

const Field256& bn254_fq() {
    static Field256 field(
        bn254::FQ_MODULUS,
        bn254::FQ_R,
        bn254::FQ_R2,
        bn254::FQ_INV
    );
    return field;
}

// ============================================================================
// Utility Functions
// ============================================================================

FieldElement256 random_field_element_256(const Field256& field) {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    
    FieldElement256 result;
    do {
        for (int i = 0; i < 4; ++i) {
            result.limbs[i] = gen();
        }
    } while (result >= field.modulus());
    
    return field.to_montgomery(result);
}

FieldElement384 random_field_element_384(const Field384& field) {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    
    FieldElement384 result;
    do {
        for (int i = 0; i < 6; ++i) {
            result.limbs[i] = gen();
        }
    } while (result >= field.modulus());
    
    return field.to_montgomery(result);
}

bool is_valid_field_element(const FieldElement256& elem, const Field256& field) {
    return elem < field.modulus();
}

bool is_valid_field_element(const FieldElement384& elem, const Field384& field) {
    return elem < field.modulus();
}

} // namespace zk
} // namespace fhe_accelerate
