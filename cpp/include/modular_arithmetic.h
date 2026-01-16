#pragma once

#include <cstdint>
#include <vector>
#include <arm_neon.h>

namespace fhe_accelerate {

/// Montgomery reduction constants for a given modulus
struct MontgomeryConstants {
    uint64_t modulus;           // The modulus q
    uint64_t r_mod_q;           // R mod q, where R = 2^64
    uint64_t r2_mod_q;          // R^2 mod q for conversion to Montgomery form
    uint64_t q_inv;             // -q^(-1) mod R for Montgomery reduction
    
    MontgomeryConstants(uint64_t q);
};

/// Modular arithmetic operations using Montgomery representation
class ModularArithmetic {
public:
    explicit ModularArithmetic(uint64_t modulus);
    
    // Montgomery multiplication: (a * b * R^-1) mod q
    uint64_t montgomery_mul(uint64_t a, uint64_t b) const;
    
    // Modular addition: (a + b) mod q
    uint64_t mod_add(uint64_t a, uint64_t b) const;
    
    // Modular subtraction: (a - b) mod q
    uint64_t mod_sub(uint64_t a, uint64_t b) const;
    
    // Convert to Montgomery form: a * R mod q
    uint64_t to_montgomery(uint64_t a) const;
    
    // Convert from Montgomery form: a * R^-1 mod q
    uint64_t from_montgomery(uint64_t a) const;
    
    // NEON SIMD vectorized operations (process 2 values at once)
    void montgomery_mul_neon(const uint64_t* a, const uint64_t* b, 
                             uint64_t* result, size_t count) const;
    
    void mod_add_neon(const uint64_t* a, const uint64_t* b,
                      uint64_t* result, size_t count) const;
    
    void mod_sub_neon(const uint64_t* a, const uint64_t* b,
                      uint64_t* result, size_t count) const;
    
    // Getters
    uint64_t get_modulus() const { return constants_.modulus; }
    const MontgomeryConstants& get_constants() const { return constants_; }
    
private:
    MontgomeryConstants constants_;
    
    // Helper: compute a * b as 128-bit result
    inline void mul_128(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) const;
    
    // Helper: Montgomery reduction of 128-bit value
    inline uint64_t montgomery_reduce(uint64_t hi, uint64_t lo) const;
};

/// Barrett reduction for modular multiplication (alternative to Montgomery)
class BarrettReducer {
public:
    explicit BarrettReducer(uint64_t modulus);
    
    // Barrett modular multiplication: (a * b) mod q
    uint64_t barrett_mul(uint64_t a, uint64_t b) const;
    
    // Barrett reduction: x mod q
    uint64_t barrett_reduce(uint64_t x) const;
    
private:
    uint64_t modulus_;
    uint64_t mu_;  // floor(2^128 / modulus) for Barrett reduction
};

// Factory function for creating ModularArithmetic instances (for cxx bridge)
std::unique_ptr<ModularArithmetic> create_modular_arithmetic(uint64_t modulus);

/// Multi-limb integer representation for coefficients > 64 bits
/// Limbs are stored in little-endian order (limbs[0] is least significant)
class MultiLimbInteger {
public:
    // Constructors
    MultiLimbInteger(); // Default: 2 limbs, all zero
    explicit MultiLimbInteger(size_t num_limbs);
    explicit MultiLimbInteger(const std::vector<uint64_t>& limbs);
    
    // Factory method for creating from a single uint64_t value
    static MultiLimbInteger from_u64(uint64_t value);
    
    // Accessors
    size_t num_limbs() const { return limbs_.size(); }
    uint64_t get_limb(size_t index) const { return limbs_[index]; }
    void set_limb(size_t index, uint64_t value) { limbs_[index] = value; }
    const std::vector<uint64_t>& limbs() const { return limbs_; }
    
    // Comparison
    bool operator==(const MultiLimbInteger& other) const;
    bool operator<(const MultiLimbInteger& other) const;
    bool operator>(const MultiLimbInteger& other) const;
    
    // Check if zero
    bool is_zero() const;
    
private:
    std::vector<uint64_t> limbs_;
};

/// Montgomery constants for multi-limb modular arithmetic
struct MultiLimbMontgomeryConstants {
    MultiLimbInteger modulus;           // The modulus q
    MultiLimbInteger r_mod_q;           // R mod q, where R = 2^(64*num_limbs)
    MultiLimbInteger r2_mod_q;          // R^2 mod q for conversion to Montgomery form
    uint64_t q_inv;                     // -q[0]^(-1) mod 2^64 for Montgomery reduction
    size_t num_limbs;
    
    MultiLimbMontgomeryConstants(const MultiLimbInteger& q);
};

/// Multi-limb modular arithmetic using Montgomery representation
class MultiLimbModularArithmetic {
public:
    explicit MultiLimbModularArithmetic(const MultiLimbInteger& modulus);
    
    // Montgomery multiplication: (a * b * R^-1) mod q
    MultiLimbInteger montgomery_mul(const MultiLimbInteger& a, 
                                    const MultiLimbInteger& b) const;
    
    // Modular addition: (a + b) mod q
    MultiLimbInteger mod_add(const MultiLimbInteger& a, 
                             const MultiLimbInteger& b) const;
    
    // Modular subtraction: (a - b) mod q
    MultiLimbInteger mod_sub(const MultiLimbInteger& a, 
                             const MultiLimbInteger& b) const;
    
    // Convert to Montgomery form: a * R mod q
    MultiLimbInteger to_montgomery(const MultiLimbInteger& a) const;
    
    // Convert from Montgomery form: a * R^-1 mod q
    MultiLimbInteger from_montgomery(const MultiLimbInteger& a) const;
    
    // NEON vectorized operations for limb-level operations
    void montgomery_mul_neon(const MultiLimbInteger* a, const MultiLimbInteger* b,
                             MultiLimbInteger* result, size_t count) const;
    
    void mod_add_neon(const MultiLimbInteger* a, const MultiLimbInteger* b,
                      MultiLimbInteger* result, size_t count) const;
    
    void mod_sub_neon(const MultiLimbInteger* a, const MultiLimbInteger* b,
                      MultiLimbInteger* result, size_t count) const;
    
    // Getters
    const MultiLimbInteger& get_modulus() const { return constants_.modulus; }
    const MultiLimbMontgomeryConstants& get_constants() const { return constants_; }
    
private:
    MultiLimbMontgomeryConstants constants_;
    
    // Helper: multi-limb addition with carry
    // Returns carry out
    uint64_t add_limbs(const uint64_t* a, const uint64_t* b, 
                       uint64_t* result, size_t num_limbs) const;
    
    // Helper: multi-limb subtraction with borrow
    // Returns borrow out
    uint64_t sub_limbs(const uint64_t* a, const uint64_t* b,
                       uint64_t* result, size_t num_limbs) const;
    
    // Helper: multi-limb multiplication (schoolbook)
    // result must have space for 2*num_limbs
    void mul_limbs(const uint64_t* a, const uint64_t* b,
                   uint64_t* result, size_t num_limbs) const;
    
    // Helper: Montgomery reduction for multi-limb
    MultiLimbInteger montgomery_reduce(const std::vector<uint64_t>& product) const;
    
    // Helper: compare multi-limb integers
    // Returns: -1 if a < b, 0 if a == b, 1 if a > b
    int compare_limbs(const uint64_t* a, const uint64_t* b, size_t num_limbs) const;
    
    // NEON optimized limb operations
    uint64_t add_limbs_neon(const uint64_t* a, const uint64_t* b,
                            uint64_t* result, size_t num_limbs) const;
    
    uint64_t sub_limbs_neon(const uint64_t* a, const uint64_t* b,
                            uint64_t* result, size_t num_limbs) const;
    
    void mul_limbs_neon(const uint64_t* a, const uint64_t* b,
                        uint64_t* result, size_t num_limbs) const;
};

} // namespace fhe_accelerate
