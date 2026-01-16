#include "modular_arithmetic.h"
#include <stdexcept>
#include <cstring>

namespace fhe_accelerate {

// Helper: Extended GCD to compute modular inverse
static uint64_t mod_inverse(uint64_t a, uint64_t m) {
    if (m == 0) return 0;
    
    int64_t m0 = m;
    int64_t x0 = 0, x1 = 1;
    
    if (m == 1) return 0;
    
    while (a > 1) {
        int64_t q = a / m;
        int64_t t = m;
        
        m = a % m;
        a = t;
        t = x0;
        
        x0 = x1 - q * x0;
        x1 = t;
    }
    
    if (x1 < 0) x1 += m0;
    
    return static_cast<uint64_t>(x1);
}

// Helper: Compute a^b mod m using binary exponentiation
static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    
    while (exp > 0) {
        if (exp & 1) {
            // Use 128-bit multiplication to avoid overflow
            __uint128_t tmp = static_cast<__uint128_t>(result) * base;
            result = tmp % mod;
        }
        __uint128_t tmp = static_cast<__uint128_t>(base) * base;
        base = tmp % mod;
        exp >>= 1;
    }
    
    return result;
}

MontgomeryConstants::MontgomeryConstants(uint64_t q) : modulus(q) {
    if (q == 0 || (q & 1) == 0) {
        throw std::invalid_argument("Modulus must be odd and non-zero for Montgomery arithmetic");
    }
    
    // Compute R mod q, where R = 2^64
    // R mod q = (2^64) mod q
    __uint128_t r = static_cast<__uint128_t>(1) << 64;
    r_mod_q = r % q;
    
    // Compute R^2 mod q
    __uint128_t r2 = static_cast<__uint128_t>(r_mod_q) * r_mod_q;
    r2_mod_q = r2 % q;
    
    // Compute q_inv = -q^(-1) mod 2^64
    // We need q * q_inv ≡ -1 (mod 2^64)
    // This is equivalent to q * q_inv ≡ 2^64 - 1 (mod 2^64)
    uint64_t q_inv_tmp = mod_inverse(q, UINT64_MAX);
    q_inv = (~q_inv_tmp) + 1;  // Negate: -x = ~x + 1 in two's complement
}

ModularArithmetic::ModularArithmetic(uint64_t modulus) 
    : constants_(modulus) {
}

inline void ModularArithmetic::mul_128(uint64_t a, uint64_t b, 
                                       uint64_t& hi, uint64_t& lo) const {
    __uint128_t product = static_cast<__uint128_t>(a) * b;
    lo = static_cast<uint64_t>(product);
    hi = static_cast<uint64_t>(product >> 64);
}

inline uint64_t ModularArithmetic::montgomery_reduce(uint64_t hi, uint64_t lo) const {
    // Montgomery reduction: given x = hi * 2^64 + lo, compute x * R^-1 mod q
    // Algorithm:
    //   m = (lo * q_inv) mod 2^64
    //   t = (x + m * q) / 2^64
    //   if t >= q then t = t - q
    //   return t
    
    uint64_t m = lo * constants_.q_inv;
    
    // Compute m * q as 128-bit
    uint64_t m_q_hi, m_q_lo;
    mul_128(m, constants_.modulus, m_q_hi, m_q_lo);
    
    // Add (hi, lo) + (m_q_hi, m_q_lo)
    __uint128_t sum = (static_cast<__uint128_t>(hi) << 64) + lo +
                      (static_cast<__uint128_t>(m_q_hi) << 64) + m_q_lo;
    
    // Divide by 2^64 (take high 64 bits)
    uint64_t t = static_cast<uint64_t>(sum >> 64);
    
    // Final conditional subtraction
    if (t >= constants_.modulus) {
        t -= constants_.modulus;
    }
    
    return t;
}

uint64_t ModularArithmetic::montgomery_mul(uint64_t a, uint64_t b) const {
    // Compute a * b as 128-bit
    uint64_t hi, lo;
    mul_128(a, b, hi, lo);
    
    // Montgomery reduction
    return montgomery_reduce(hi, lo);
}

uint64_t ModularArithmetic::mod_add(uint64_t a, uint64_t b) const {
    // Ensure inputs are reduced
    a %= constants_.modulus;
    b %= constants_.modulus;
    
    // Compute sum with overflow check
    uint64_t sum = a + b;
    
    // Conditional subtraction to avoid overflow
    if (sum < a || sum >= constants_.modulus) {
        sum -= constants_.modulus;
    }
    
    return sum;
}

uint64_t ModularArithmetic::mod_sub(uint64_t a, uint64_t b) const {
    // Ensure inputs are reduced
    a %= constants_.modulus;
    b %= constants_.modulus;
    
    // Compute difference
    uint64_t diff;
    if (a >= b) {
        diff = a - b;
    } else {
        // Handle underflow: a - b + q
        diff = constants_.modulus - (b - a);
    }
    
    return diff;
}

uint64_t ModularArithmetic::to_montgomery(uint64_t a) const {
    // Convert to Montgomery form: a * R mod q
    // We use: a * R mod q = a * R^2 * R^-1 mod q = montgomery_mul(a, R^2 mod q)
    return montgomery_mul(a, constants_.r2_mod_q);
}

uint64_t ModularArithmetic::from_montgomery(uint64_t a) const {
    // Convert from Montgomery form: a * R^-1 mod q
    // We use: montgomery_mul(a, 1) = a * 1 * R^-1 mod q
    return montgomery_reduce(0, a);
}

void ModularArithmetic::montgomery_mul_neon(const uint64_t* a, const uint64_t* b,
                                            uint64_t* result, size_t count) const {
    // Process 2 elements at a time using NEON
    size_t i = 0;
    
    // NEON can process 2x64-bit values per register
    for (; i + 1 < count; i += 2) {
        // For now, fall back to scalar for each pair
        // Full NEON optimization would require 128-bit arithmetic support
        result[i] = montgomery_mul(a[i], b[i]);
        result[i + 1] = montgomery_mul(a[i + 1], b[i + 1]);
    }
    
    // Handle remaining element
    if (i < count) {
        result[i] = montgomery_mul(a[i], b[i]);
    }
}

void ModularArithmetic::mod_add_neon(const uint64_t* a, const uint64_t* b,
                                     uint64_t* result, size_t count) const {
    size_t i = 0;
    uint64x2_t vmodulus = vdupq_n_u64(constants_.modulus);
    
    // Process 2 elements at a time
    for (; i + 1 < count; i += 2) {
        // Load values
        uint64x2_t va = vld1q_u64(&a[i]);
        uint64x2_t vb = vld1q_u64(&b[i]);
        
        // Reduce inputs first (scalar for now, as NEON doesn't have native mod)
        uint64_t a0 = a[i] % constants_.modulus;
        uint64_t a1 = a[i + 1] % constants_.modulus;
        uint64_t b0 = b[i] % constants_.modulus;
        uint64_t b1 = b[i + 1] % constants_.modulus;
        
        // Add
        uint64_t sum0 = a0 + b0;
        uint64_t sum1 = a1 + b1;
        
        // Conditional subtraction
        if (sum0 < a0 || sum0 >= constants_.modulus) sum0 -= constants_.modulus;
        if (sum1 < a1 || sum1 >= constants_.modulus) sum1 -= constants_.modulus;
        
        result[i] = sum0;
        result[i + 1] = sum1;
    }
    
    // Handle remaining element
    if (i < count) {
        result[i] = mod_add(a[i], b[i]);
    }
}

void ModularArithmetic::mod_sub_neon(const uint64_t* a, const uint64_t* b,
                                     uint64_t* result, size_t count) const {
    size_t i = 0;
    
    // Process 2 elements at a time
    for (; i + 1 < count; i += 2) {
        result[i] = mod_sub(a[i], b[i]);
        result[i + 1] = mod_sub(a[i + 1], b[i + 1]);
    }
    
    // Handle remaining element
    if (i < count) {
        result[i] = mod_sub(a[i], b[i]);
    }
}

// Barrett Reducer Implementation
BarrettReducer::BarrettReducer(uint64_t modulus) : modulus_(modulus) {
    if (modulus == 0) {
        throw std::invalid_argument("Modulus must be non-zero for Barrett reduction");
    }
    
    // Compute mu = floor(2^128 / modulus)
    // For 64-bit implementation, we approximate with 2^64
    __uint128_t numerator = static_cast<__uint128_t>(1) << 64;
    mu_ = numerator / modulus;
}

uint64_t BarrettReducer::barrett_reduce(uint64_t x) const {
    // Barrett reduction: x mod q
    // q1 = floor(x / 2^64) * mu
    // q2 = floor(q1 / 2^64)
    // r = x - q2 * modulus
    // if r >= modulus then r = r - modulus
    
    __uint128_t tmp = static_cast<__uint128_t>(x) * mu_;
    uint64_t q2 = static_cast<uint64_t>(tmp >> 64);
    
    uint64_t r = x - q2 * modulus_;
    
    if (r >= modulus_) {
        r -= modulus_;
    }
    
    return r;
}

uint64_t BarrettReducer::barrett_mul(uint64_t a, uint64_t b) const {
    // Compute a * b
    __uint128_t product = static_cast<__uint128_t>(a) * b;
    
    // Reduce using Barrett
    // For values that fit in 64 bits after multiplication
    if (product < (static_cast<__uint128_t>(1) << 64)) {
        return barrett_reduce(static_cast<uint64_t>(product));
    }
    
    // For larger products, use direct modulo
    return product % modulus_;
}

// ============================================================================
// Multi-Limb Integer Implementation
// ============================================================================

fhe_accelerate::MultiLimbInteger::MultiLimbInteger()
    : limbs_(2, 0) {
}

fhe_accelerate::MultiLimbInteger::MultiLimbInteger(size_t num_limbs) 
    : limbs_(num_limbs, 0) {
}

fhe_accelerate::MultiLimbInteger::MultiLimbInteger(const std::vector<uint64_t>& limbs)
    : limbs_(limbs) {
    if (limbs_.empty()) {
        limbs_.resize(2, 0);
    }
}

fhe_accelerate::MultiLimbInteger fhe_accelerate::MultiLimbInteger::from_u64(uint64_t value) {
    std::vector<uint64_t> limbs(2, 0);
    limbs[0] = value;
    return MultiLimbInteger(limbs);
}

bool fhe_accelerate::MultiLimbInteger::operator==(const MultiLimbInteger& other) const {
    if (limbs_.size() != other.limbs_.size()) {
        return false;
    }
    return std::memcmp(limbs_.data(), other.limbs_.data(), 
                       limbs_.size() * sizeof(uint64_t)) == 0;
}

bool fhe_accelerate::MultiLimbInteger::operator<(const MultiLimbInteger& other) const {
    size_t max_limbs = std::max(limbs_.size(), other.limbs_.size());
    
    // Compare from most significant to least significant
    for (size_t i = max_limbs; i > 0; --i) {
        size_t idx = i - 1;
        uint64_t a_limb = (idx < limbs_.size()) ? limbs_[idx] : 0;
        uint64_t b_limb = (idx < other.limbs_.size()) ? other.limbs_[idx] : 0;
        
        if (a_limb < b_limb) return true;
        if (a_limb > b_limb) return false;
    }
    
    return false;  // Equal
}

bool fhe_accelerate::MultiLimbInteger::operator>(const MultiLimbInteger& other) const {
    return other < *this;
}

bool fhe_accelerate::MultiLimbInteger::is_zero() const {
    for (uint64_t limb : limbs_) {
        if (limb != 0) return false;
    }
    return true;
}

// ============================================================================
// Multi-Limb Montgomery Constants
// ============================================================================

// Helper: compute modular inverse for multi-limb
static uint64_t compute_q_inv_limb(uint64_t q0) {
    // Compute -q0^(-1) mod 2^64
    // Using Newton's method: x_{n+1} = x_n * (2 - q0 * x_n)
    uint64_t x = q0;  // Initial approximation
    
    // Newton iterations (5 iterations sufficient for 64-bit)
    for (int i = 0; i < 5; ++i) {
        x = x * (2 - q0 * x);
    }
    
    return (~x) + 1;  // Negate
}

// Helper: proper multi-limb modular reduction using Barrett-like approach
static fhe_accelerate::MultiLimbInteger multi_limb_mod_proper(
    const std::vector<uint64_t>& a_limbs,
    const fhe_accelerate::MultiLimbInteger& modulus) {
    
    size_t num_limbs = modulus.num_limbs();
    size_t a_size = a_limbs.size();
    
    // If a is already smaller than modulus, return it
    fhe_accelerate::MultiLimbInteger a(a_limbs);
    if (a_size <= num_limbs && a < modulus) {
        return a;
    }
    
    // Use long division approach
    // This is O(n^2) but correct for initialization
    std::vector<uint64_t> remainder = a_limbs;
    if (remainder.size() < num_limbs) {
        remainder.resize(num_limbs, 0);
    }
    
    // Repeatedly subtract modulus * 2^k where k is chosen to keep result positive
    // Start from the most significant position
    for (int bit_pos = (remainder.size() * 64) - 1; bit_pos >= 0; --bit_pos) {
        // Check if we can subtract modulus << bit_pos
        size_t limb_shift = bit_pos / 64;
        size_t bit_shift = bit_pos % 64;
        
        // Create shifted modulus
        std::vector<uint64_t> shifted_mod(remainder.size(), 0);
        for (size_t i = 0; i < num_limbs && (i + limb_shift) < shifted_mod.size(); ++i) {
            uint64_t mod_limb = modulus.get_limb(i);
            if (bit_shift == 0) {
                shifted_mod[i + limb_shift] = mod_limb;
            } else {
                shifted_mod[i + limb_shift] |= (mod_limb << bit_shift);
                if (i + limb_shift + 1 < shifted_mod.size()) {
                    shifted_mod[i + limb_shift + 1] = (mod_limb >> (64 - bit_shift));
                }
            }
        }
        
        // Compare remainder with shifted_mod
        bool can_subtract = false;
        for (int i = remainder.size() - 1; i >= 0; --i) {
            if (remainder[i] > shifted_mod[i]) {
                can_subtract = true;
                break;
            } else if (remainder[i] < shifted_mod[i]) {
                break;
            }
        }
        
        // If we can subtract, do it
        if (can_subtract) {
            uint64_t borrow = 0;
            for (size_t i = 0; i < remainder.size(); ++i) {
                uint64_t r = remainder[i];
                uint64_t s = shifted_mod[i];
                uint64_t diff = r - s - borrow;
                borrow = (r < s + borrow) ? 1 : 0;
                remainder[i] = diff;
            }
        }
    }
    
    // Trim to num_limbs
    remainder.resize(num_limbs);
    return fhe_accelerate::MultiLimbInteger(remainder);
}

// Helper: compute R mod q where R = 2^(64*num_limbs)
static fhe_accelerate::MultiLimbInteger compute_r_mod_q(const fhe_accelerate::MultiLimbInteger& modulus) {
    size_t num_limbs = modulus.num_limbs();
    
    // R = 2^(64*num_limbs) = 1 followed by num_limbs zero limbs
    std::vector<uint64_t> r_limbs(num_limbs + 1, 0);
    r_limbs[num_limbs] = 1;
    
    return multi_limb_mod_proper(r_limbs, modulus);
}

// Helper: compute R^2 mod q
static fhe_accelerate::MultiLimbInteger compute_r2_mod_q(
    const fhe_accelerate::MultiLimbInteger& modulus,
    const fhe_accelerate::MultiLimbInteger& r_mod_q) {
    
    size_t num_limbs = modulus.num_limbs();
    
    // Compute (R mod q)^2 using schoolbook multiplication
    std::vector<uint64_t> product(2 * num_limbs, 0);
    
    for (size_t i = 0; i < num_limbs; ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < num_limbs; ++j) {
            __uint128_t prod = static_cast<__uint128_t>(r_mod_q.get_limb(i)) * 
                              r_mod_q.get_limb(j);
            prod += product[i + j];
            prod += carry;
            
            product[i + j] = static_cast<uint64_t>(prod);
            carry = static_cast<uint64_t>(prod >> 64);
        }
        product[i + num_limbs] = carry;
    }
    
    // Reduce modulo q
    return multi_limb_mod_proper(product, modulus);
}

// Helper: multi-limb modular exponentiation for computing R^2 mod q
fhe_accelerate::MultiLimbMontgomeryConstants::MultiLimbMontgomeryConstants(const MultiLimbInteger& q)
    : modulus(q), num_limbs(q.num_limbs()) {
    
    if (q.is_zero() || (q.get_limb(0) & 1) == 0) {
        throw std::invalid_argument("Modulus must be odd and non-zero for Montgomery arithmetic");
    }
    
    // Compute R mod q where R = 2^(64*num_limbs)
    r_mod_q = compute_r_mod_q(modulus);
    
    // Compute R^2 mod q
    r2_mod_q = compute_r2_mod_q(modulus, r_mod_q);
    
    // Compute q_inv = -q[0]^(-1) mod 2^64
    q_inv = compute_q_inv_limb(modulus.get_limb(0));
}

// ============================================================================
// Multi-Limb Modular Arithmetic Implementation
// ============================================================================

fhe_accelerate::MultiLimbModularArithmetic::MultiLimbModularArithmetic(const MultiLimbInteger& modulus)
    : constants_(modulus) {
}

uint64_t fhe_accelerate::MultiLimbModularArithmetic::add_limbs(const uint64_t* a, const uint64_t* b,
                                                uint64_t* result, size_t num_limbs) const {
    uint64_t carry = 0;
    
    for (size_t i = 0; i < num_limbs; ++i) {
        __uint128_t sum = static_cast<__uint128_t>(a[i]) + b[i] + carry;
        result[i] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
    
    return carry;
}

uint64_t fhe_accelerate::MultiLimbModularArithmetic::sub_limbs(const uint64_t* a, const uint64_t* b,
                                                uint64_t* result, size_t num_limbs) const {
    uint64_t borrow = 0;
    
    for (size_t i = 0; i < num_limbs; ++i) {
        uint64_t a_limb = a[i];
        uint64_t b_limb = b[i];
        
        uint64_t diff = a_limb - b_limb - borrow;
        borrow = (a_limb < b_limb + borrow) ? 1 : 0;
        result[i] = diff;
    }
    
    return borrow;
}

void fhe_accelerate::MultiLimbModularArithmetic::mul_limbs(const uint64_t* a, const uint64_t* b,
                                           uint64_t* result, size_t num_limbs) const {
    // Schoolbook multiplication
    // result must have space for 2*num_limbs
    std::memset(result, 0, 2 * num_limbs * sizeof(uint64_t));
    
    for (size_t i = 0; i < num_limbs; ++i) {
        uint64_t carry = 0;
        
        for (size_t j = 0; j < num_limbs; ++j) {
            __uint128_t product = static_cast<__uint128_t>(a[i]) * b[j];
            product += result[i + j];
            product += carry;
            
            result[i + j] = static_cast<uint64_t>(product);
            carry = static_cast<uint64_t>(product >> 64);
        }
        
        result[i + num_limbs] = carry;
    }
}

int fhe_accelerate::MultiLimbModularArithmetic::compare_limbs(const uint64_t* a, const uint64_t* b,
                                              size_t num_limbs) const {
    // Compare from most significant to least significant
    for (size_t i = num_limbs; i > 0; --i) {
        size_t idx = i - 1;
        if (a[idx] < b[idx]) return -1;
        if (a[idx] > b[idx]) return 1;
    }
    return 0;
}

fhe_accelerate::MultiLimbInteger fhe_accelerate::MultiLimbModularArithmetic::montgomery_reduce(
    const std::vector<uint64_t>& product) const {
    
    size_t num_limbs = constants_.num_limbs;
    std::vector<uint64_t> t = product;
    
    // Ensure t has enough space
    if (t.size() < 2 * num_limbs) {
        t.resize(2 * num_limbs, 0);
    }
    
    // Montgomery reduction algorithm for multi-limb
    // For each limb position:
    for (size_t i = 0; i < num_limbs; ++i) {
        // m = t[i] * q_inv mod 2^64
        uint64_t m = t[i] * constants_.q_inv;
        
        // t = t + m * q
        uint64_t carry = 0;
        for (size_t j = 0; j < num_limbs; ++j) {
            __uint128_t prod = static_cast<__uint128_t>(m) * 
                              constants_.modulus.get_limb(j);
            prod += t[i + j];
            prod += carry;
            
            t[i + j] = static_cast<uint64_t>(prod);
            carry = static_cast<uint64_t>(prod >> 64);
        }
        
        // Propagate carry
        for (size_t j = num_limbs; j < 2 * num_limbs - i && carry; ++j) {
            __uint128_t sum = static_cast<__uint128_t>(t[i + j]) + carry;
            t[i + j] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
    }
    
    // Extract high limbs (divide by R)
    std::vector<uint64_t> result_limbs(num_limbs);
    for (size_t i = 0; i < num_limbs; ++i) {
        result_limbs[i] = t[num_limbs + i];
    }
    
    MultiLimbInteger result(result_limbs);
    
    // Final conditional subtraction
    if (!(result < constants_.modulus)) {
        std::vector<uint64_t> temp(num_limbs);
        sub_limbs(result_limbs.data(), constants_.modulus.limbs().data(),
                  temp.data(), num_limbs);
        result = MultiLimbInteger(temp);
    }
    
    return result;
}

fhe_accelerate::MultiLimbInteger fhe_accelerate::MultiLimbModularArithmetic::montgomery_mul(
    const MultiLimbInteger& a, const MultiLimbInteger& b) const {
    
    size_t num_limbs = constants_.num_limbs;
    
    // Multiply a * b
    std::vector<uint64_t> product(2 * num_limbs);
    mul_limbs(a.limbs().data(), b.limbs().data(), product.data(), num_limbs);
    
    // Montgomery reduction
    return montgomery_reduce(product);
}

fhe_accelerate::MultiLimbInteger fhe_accelerate::MultiLimbModularArithmetic::mod_add(
    const MultiLimbInteger& a, const MultiLimbInteger& b) const {
    
    size_t num_limbs = constants_.num_limbs;
    std::vector<uint64_t> result(num_limbs);
    
    // Add limbs
    uint64_t carry = add_limbs(a.limbs().data(), b.limbs().data(),
                               result.data(), num_limbs);
    
    MultiLimbInteger sum(result);
    
    // Conditional subtraction if sum >= modulus or carry
    if (carry || !(sum < constants_.modulus)) {
        std::vector<uint64_t> temp(num_limbs);
        sub_limbs(result.data(), constants_.modulus.limbs().data(),
                  temp.data(), num_limbs);
        sum = MultiLimbInteger(temp);
    }
    
    return sum;
}

fhe_accelerate::MultiLimbInteger fhe_accelerate::MultiLimbModularArithmetic::mod_sub(
    const MultiLimbInteger& a, const MultiLimbInteger& b) const {
    
    size_t num_limbs = constants_.num_limbs;
    std::vector<uint64_t> result(num_limbs);
    
    // Subtract limbs
    uint64_t borrow = sub_limbs(a.limbs().data(), b.limbs().data(),
                                result.data(), num_limbs);
    
    MultiLimbInteger diff(result);
    
    // If borrow, add modulus
    if (borrow) {
        std::vector<uint64_t> temp(num_limbs);
        add_limbs(result.data(), constants_.modulus.limbs().data(),
                  temp.data(), num_limbs);
        diff = MultiLimbInteger(temp);
    }
    
    return diff;
}

fhe_accelerate::MultiLimbInteger fhe_accelerate::MultiLimbModularArithmetic::to_montgomery(
    const MultiLimbInteger& a) const {
    
    // Convert to Montgomery form: a * R mod q
    return montgomery_mul(a, constants_.r2_mod_q);
}

fhe_accelerate::MultiLimbInteger fhe_accelerate::MultiLimbModularArithmetic::from_montgomery(
    const MultiLimbInteger& a) const {
    
    // Convert from Montgomery form: a * R^-1 mod q
    size_t num_limbs = constants_.num_limbs;
    std::vector<uint64_t> product(2 * num_limbs, 0);
    
    // Copy a to low limbs
    for (size_t i = 0; i < num_limbs; ++i) {
        product[i] = a.get_limb(i);
    }
    
    return montgomery_reduce(product);
}

// ============================================================================
// NEON Optimized Multi-Limb Operations
// ============================================================================

uint64_t fhe_accelerate::MultiLimbModularArithmetic::add_limbs_neon(const uint64_t* a, const uint64_t* b,
                                                     uint64_t* result, size_t num_limbs) const {
    uint64_t carry = 0;
    size_t i = 0;
    
    // Process 2 limbs at a time with NEON
    for (; i + 1 < num_limbs; i += 2) {
        // Load pairs of limbs
        uint64x2_t va = vld1q_u64(&a[i]);
        uint64x2_t vb = vld1q_u64(&b[i]);
        
        // Add with carry handling (scalar for now due to carry complexity)
        __uint128_t sum0 = static_cast<__uint128_t>(a[i]) + b[i] + carry;
        result[i] = static_cast<uint64_t>(sum0);
        carry = static_cast<uint64_t>(sum0 >> 64);
        
        __uint128_t sum1 = static_cast<__uint128_t>(a[i+1]) + b[i+1] + carry;
        result[i+1] = static_cast<uint64_t>(sum1);
        carry = static_cast<uint64_t>(sum1 >> 64);
    }
    
    // Handle remaining limbs
    for (; i < num_limbs; ++i) {
        __uint128_t sum = static_cast<__uint128_t>(a[i]) + b[i] + carry;
        result[i] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
    
    return carry;
}

uint64_t fhe_accelerate::MultiLimbModularArithmetic::sub_limbs_neon(const uint64_t* a, const uint64_t* b,
                                                     uint64_t* result, size_t num_limbs) const {
    uint64_t borrow = 0;
    size_t i = 0;
    
    // Process 2 limbs at a time with NEON
    for (; i + 1 < num_limbs; i += 2) {
        // Scalar subtraction with borrow (NEON doesn't have native borrow propagation)
        uint64_t diff0 = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) ? 1 : 0;
        result[i] = diff0;
        
        uint64_t diff1 = a[i+1] - b[i+1] - borrow;
        borrow = (a[i+1] < b[i+1] + borrow) ? 1 : 0;
        result[i+1] = diff1;
    }
    
    // Handle remaining limbs
    for (; i < num_limbs; ++i) {
        uint64_t diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) ? 1 : 0;
        result[i] = diff;
    }
    
    return borrow;
}

void fhe_accelerate::MultiLimbModularArithmetic::mul_limbs_neon(const uint64_t* a, const uint64_t* b,
                                                uint64_t* result, size_t num_limbs) const {
    // NEON-optimized schoolbook multiplication
    std::memset(result, 0, 2 * num_limbs * sizeof(uint64_t));
    
    for (size_t i = 0; i < num_limbs; ++i) {
        uint64_t carry = 0;
        size_t j = 0;
        
        // Process 2 multiplications at a time
        for (; j + 1 < num_limbs; j += 2) {
            // First multiplication
            __uint128_t product0 = static_cast<__uint128_t>(a[i]) * b[j];
            product0 += result[i + j];
            product0 += carry;
            result[i + j] = static_cast<uint64_t>(product0);
            carry = static_cast<uint64_t>(product0 >> 64);
            
            // Second multiplication
            __uint128_t product1 = static_cast<__uint128_t>(a[i]) * b[j + 1];
            product1 += result[i + j + 1];
            product1 += carry;
            result[i + j + 1] = static_cast<uint64_t>(product1);
            carry = static_cast<uint64_t>(product1 >> 64);
        }
        
        // Handle remaining multiplication
        for (; j < num_limbs; ++j) {
            __uint128_t product = static_cast<__uint128_t>(a[i]) * b[j];
            product += result[i + j];
            product += carry;
            result[i + j] = static_cast<uint64_t>(product);
            carry = static_cast<uint64_t>(product >> 64);
        }
        
        result[i + num_limbs] = carry;
    }
}

void fhe_accelerate::MultiLimbModularArithmetic::montgomery_mul_neon(const MultiLimbInteger* a,
                                                     const MultiLimbInteger* b,
                                                     MultiLimbInteger* result,
                                                     size_t count) const {
    // Process multiple multi-limb multiplications
    for (size_t i = 0; i < count; ++i) {
        result[i] = montgomery_mul(a[i], b[i]);
    }
}

void fhe_accelerate::MultiLimbModularArithmetic::mod_add_neon(const MultiLimbInteger* a,
                                              const MultiLimbInteger* b,
                                              MultiLimbInteger* result,
                                              size_t count) const {
    // Process multiple multi-limb additions
    for (size_t i = 0; i < count; ++i) {
        result[i] = mod_add(a[i], b[i]);
    }
}

void fhe_accelerate::MultiLimbModularArithmetic::mod_sub_neon(const MultiLimbInteger* a,
                                              const MultiLimbInteger* b,
                                              MultiLimbInteger* result,
                                              size_t count) const {
    // Process multiple multi-limb subtractions
    for (size_t i = 0; i < count; ++i) {
        result[i] = mod_sub(a[i], b[i]);
    }
}

} // namespace fhe_accelerate
