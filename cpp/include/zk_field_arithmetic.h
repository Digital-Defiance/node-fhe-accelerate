/**
 * Zero-Knowledge Proof Field Arithmetic
 * 
 * Implements finite field arithmetic for ZK proof systems:
 * - BLS12-381 scalar field (Fr) and base field (Fq)
 * - BN254 scalar field (Fr) and base field (Fq)
 * 
 * Uses Montgomery form for efficient modular arithmetic.
 * Reuses FHE modular arithmetic infrastructure where possible.
 * 
 * Requirements: 19, 20
 */

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <memory>
#include "modular_arithmetic.h"

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Field Element Types (Multi-Limb for 256+ bit fields)
// ============================================================================

/**
 * 256-bit field element (4 x 64-bit limbs)
 * Used for BLS12-381 scalar field and BN254 fields
 * Limbs stored in little-endian order
 */
struct FieldElement256 {
    std::array<uint64_t, 4> limbs;
    
    FieldElement256() : limbs{0, 0, 0, 0} {}
    explicit FieldElement256(uint64_t value) : limbs{value, 0, 0, 0} {}
    FieldElement256(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) 
        : limbs{l0, l1, l2, l3} {}
    explicit FieldElement256(const std::array<uint64_t, 4>& l) : limbs(l) {}
    
    bool operator==(const FieldElement256& other) const;
    bool operator!=(const FieldElement256& other) const { return !(*this == other); }
    bool operator<(const FieldElement256& other) const;
    bool operator>=(const FieldElement256& other) const { return !(*this < other); }
    
    bool is_zero() const;
    
    // Convert to/from hex string
    std::string to_hex() const;
    static FieldElement256 from_hex(const std::string& hex);
    
    // Convert to/from bytes (big-endian)
    std::array<uint8_t, 32> to_bytes() const;
    static FieldElement256 from_bytes(const std::array<uint8_t, 32>& bytes);
};

/**
 * 384-bit field element (6 x 64-bit limbs)
 * Used for BLS12-381 base field Fq
 */
struct FieldElement384 {
    std::array<uint64_t, 6> limbs;
    
    FieldElement384() : limbs{0, 0, 0, 0, 0, 0} {}
    explicit FieldElement384(uint64_t value) : limbs{value, 0, 0, 0, 0, 0} {}
    explicit FieldElement384(const std::array<uint64_t, 6>& l) : limbs(l) {}
    
    bool operator==(const FieldElement384& other) const;
    bool operator!=(const FieldElement384& other) const { return !(*this == other); }
    bool operator<(const FieldElement384& other) const;
    bool operator>=(const FieldElement384& other) const { return !(*this < other); }
    
    bool is_zero() const;
    
    std::string to_hex() const;
    static FieldElement384 from_hex(const std::string& hex);
    
    std::array<uint8_t, 48> to_bytes() const;
    static FieldElement384 from_bytes(const std::array<uint8_t, 48>& bytes);
};

// ============================================================================
// Curve Parameters
// ============================================================================

/**
 * BLS12-381 curve parameters
 * 
 * Scalar field Fr: r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
 * Base field Fq: q = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
 */
namespace bls12_381 {
    // Scalar field modulus r (255 bits)
    constexpr std::array<uint64_t, 4> FR_MODULUS = {
        0xffffffff00000001ULL,
        0x53bda402fffe5bfeULL,
        0x3339d80809a1d805ULL,
        0x73eda753299d7d48ULL
    };
    
    // R = 2^256 mod r (Montgomery constant)
    constexpr std::array<uint64_t, 4> FR_R = {
        0x00000001fffffffeULL,
        0x5884b7fa00034802ULL,
        0x998c4fefecbc4ff5ULL,
        0x1824b159acc5056fULL
    };
    
    // R^2 mod r
    constexpr std::array<uint64_t, 4> FR_R2 = {
        0xc999e990f3f29c6dULL,
        0x2b6cedcb87925c23ULL,
        0x05d314967254398fULL,
        0x0748d9d99f59ff11ULL
    };
    
    // -r^(-1) mod 2^64
    constexpr uint64_t FR_INV = 0xfffffffeffffffffULL;
    
    // Base field modulus q (381 bits)
    constexpr std::array<uint64_t, 6> FQ_MODULUS = {
        0xb9feffffffffaaabULL,
        0x1eabfffeb153ffffULL,
        0x6730d2a0f6b0f624ULL,
        0x64774b84f38512bfULL,
        0x4b1ba7b6434bacd7ULL,
        0x1a0111ea397fe69aULL
    };
    
    // R = 2^384 mod q
    constexpr std::array<uint64_t, 6> FQ_R = {
        0x760900000002fffdULL,
        0xebf4000bc40c0002ULL,
        0x5f48985753c758baULL,
        0x77ce585370525745ULL,
        0x5c071a97a256ec6dULL,
        0x15f65ec3fa80e493ULL
    };
    
    // R^2 mod q
    constexpr std::array<uint64_t, 6> FQ_R2 = {
        0xf4df1f341c341746ULL,
        0x0a76e6a609d104f1ULL,
        0x8de5476c4c95b6d5ULL,
        0x67eb88a9939d83c0ULL,
        0x9a793e85b519952dULL,
        0x11988fe592cae3aaULL
    };
    
    // -q^(-1) mod 2^64
    constexpr uint64_t FQ_INV = 0x89f3fffcfffcfffdULL;
}

/**
 * BN254 curve parameters (also known as alt_bn128)
 * 
 * Scalar field Fr: r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
 * Base field Fq: q = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
 */
namespace bn254 {
    // Scalar field modulus r (254 bits)
    constexpr std::array<uint64_t, 4> FR_MODULUS = {
        0x43e1f593f0000001ULL,
        0x2833e84879b97091ULL,
        0xb85045b68181585dULL,
        0x30644e72e131a029ULL
    };
    
    // R = 2^256 mod r
    constexpr std::array<uint64_t, 4> FR_R = {
        0xac96341c4ffffffbULL,
        0x36fc76959f60cd29ULL,
        0x666ea36f7879462eULL,
        0x0e0a77c19a07df2fULL
    };
    
    // R^2 mod r
    constexpr std::array<uint64_t, 4> FR_R2 = {
        0x1bb8e645ae216da7ULL,
        0x53fe3ab1e35c59e3ULL,
        0x8c49833d53bb8085ULL,
        0x0216d0b17f4e44a5ULL
    };
    
    // -r^(-1) mod 2^64
    constexpr uint64_t FR_INV = 0xc2e1f593efffffffULL;
    
    // Base field modulus q (254 bits)
    constexpr std::array<uint64_t, 4> FQ_MODULUS = {
        0x3c208c16d87cfd47ULL,
        0x97816a916871ca8dULL,
        0xb85045b68181585dULL,
        0x30644e72e131a029ULL
    };
    
    // R = 2^256 mod q
    constexpr std::array<uint64_t, 4> FQ_R = {
        0xd35d438dc58f0d9dULL,
        0x0a78eb28f5c70b3dULL,
        0x666ea36f7879462cULL,
        0x0e0a77c19a07df2fULL
    };
    
    // R^2 mod q
    constexpr std::array<uint64_t, 4> FQ_R2 = {
        0xf32cfc5b538afa89ULL,
        0xb5e71911d44501fbULL,
        0x47ab1eff0a417ff6ULL,
        0x06d89f71cab8351fULL
    };
    
    // -q^(-1) mod 2^64
    constexpr uint64_t FQ_INV = 0x87d20782e4866389ULL;
}

// ============================================================================
// Field Arithmetic Classes
// ============================================================================

/**
 * Montgomery arithmetic for 256-bit fields
 * 
 * Provides efficient modular arithmetic using Montgomery representation.
 * Used for BLS12-381 Fr and BN254 Fr/Fq fields.
 */
class Field256 {
public:
    /**
     * Construct field arithmetic for given modulus
     * 
     * @param modulus Field modulus (4 limbs)
     * @param r Montgomery R constant
     * @param r2 Montgomery R^2 constant
     * @param inv Montgomery inverse constant (-p^(-1) mod 2^64)
     */
    Field256(const std::array<uint64_t, 4>& modulus,
             const std::array<uint64_t, 4>& r,
             const std::array<uint64_t, 4>& r2,
             uint64_t inv);
    
    // Field operations (inputs/outputs in Montgomery form)
    FieldElement256 add(const FieldElement256& a, const FieldElement256& b) const;
    FieldElement256 sub(const FieldElement256& a, const FieldElement256& b) const;
    FieldElement256 mul(const FieldElement256& a, const FieldElement256& b) const;
    FieldElement256 neg(const FieldElement256& a) const;
    FieldElement256 square(const FieldElement256& a) const;
    
    // Inversion using Fermat's little theorem: a^(-1) = a^(p-2) mod p
    FieldElement256 inv(const FieldElement256& a) const;
    
    // Exponentiation: a^exp mod p
    FieldElement256 pow(const FieldElement256& base, const FieldElement256& exp) const;
    
    // Convert to/from Montgomery form
    FieldElement256 to_montgomery(const FieldElement256& a) const;
    FieldElement256 from_montgomery(const FieldElement256& a) const;
    
    // Get field constants
    const FieldElement256& modulus() const { return modulus_; }
    const FieldElement256& one() const { return r_; }  // 1 in Montgomery form
    const FieldElement256& zero() const { return zero_; }
    
    // Batch operations (NEON optimized)
    void batch_add(const FieldElement256* a, const FieldElement256* b,
                   FieldElement256* result, size_t count) const;
    void batch_mul(const FieldElement256* a, const FieldElement256* b,
                   FieldElement256* result, size_t count) const;
    
private:
    FieldElement256 modulus_;
    FieldElement256 r_;      // R mod p (Montgomery 1)
    FieldElement256 r2_;     // R^2 mod p
    FieldElement256 zero_;   // Zero element
    uint64_t inv_;           // -p^(-1) mod 2^64
    
    // Montgomery reduction
    FieldElement256 montgomery_reduce(const std::array<uint64_t, 8>& product) const;
    
    // Multi-limb arithmetic helpers
    static void add_with_carry(const uint64_t* a, const uint64_t* b,
                               uint64_t* result, uint64_t& carry);
    static void sub_with_borrow(const uint64_t* a, const uint64_t* b,
                                uint64_t* result, uint64_t& borrow);
    static void mul_wide(const uint64_t* a, const uint64_t* b,
                         uint64_t* result);  // 8-limb result
};

/**
 * Montgomery arithmetic for 384-bit fields
 * 
 * Used for BLS12-381 base field Fq.
 */
class Field384 {
public:
    Field384(const std::array<uint64_t, 6>& modulus,
             const std::array<uint64_t, 6>& r,
             const std::array<uint64_t, 6>& r2,
             uint64_t inv);
    
    // Field operations
    FieldElement384 add(const FieldElement384& a, const FieldElement384& b) const;
    FieldElement384 sub(const FieldElement384& a, const FieldElement384& b) const;
    FieldElement384 mul(const FieldElement384& a, const FieldElement384& b) const;
    FieldElement384 neg(const FieldElement384& a) const;
    FieldElement384 square(const FieldElement384& a) const;
    FieldElement384 inv(const FieldElement384& a) const;
    FieldElement384 pow(const FieldElement384& base, const FieldElement384& exp) const;
    
    FieldElement384 to_montgomery(const FieldElement384& a) const;
    FieldElement384 from_montgomery(const FieldElement384& a) const;
    
    const FieldElement384& modulus() const { return modulus_; }
    const FieldElement384& one() const { return r_; }
    const FieldElement384& zero() const { return zero_; }
    
private:
    FieldElement384 modulus_;
    FieldElement384 r_;
    FieldElement384 r2_;
    FieldElement384 zero_;
    uint64_t inv_;
    
    FieldElement384 montgomery_reduce(const std::array<uint64_t, 12>& product) const;
};

// ============================================================================
// Pre-configured Field Instances
// ============================================================================

/**
 * Get BLS12-381 scalar field Fr arithmetic
 */
const Field256& bls12_381_fr();

/**
 * Get BLS12-381 base field Fq arithmetic
 */
const Field384& bls12_381_fq();

/**
 * Get BN254 scalar field Fr arithmetic
 */
const Field256& bn254_fr();

/**
 * Get BN254 base field Fq arithmetic
 */
const Field256& bn254_fq();

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate random field element
 */
FieldElement256 random_field_element_256(const Field256& field);
FieldElement384 random_field_element_384(const Field384& field);

/**
 * Check if element is in valid range [0, modulus)
 */
bool is_valid_field_element(const FieldElement256& elem, const Field256& field);
bool is_valid_field_element(const FieldElement384& elem, const Field384& field);

} // namespace zk
} // namespace fhe_accelerate
