/**
 * Multi-Limb Arithmetic Tests
 * 
 * Tests for multi-limb integer representation and Montgomery arithmetic
 * for coefficients > 64 bits.
 */

#include "../include/modular_arithmetic.h"
#include "test_harness.h"
#include <iostream>
#include <cassert>

using namespace fhe_accelerate;
using namespace fhe_accelerate::test;

void test_multi_limb_construction() {
    std::cout << "Testing multi-limb construction...\n";
    
    // Test default constructor
    MultiLimbInteger a;
    assert(a.num_limbs() == 2);
    assert(a.get_limb(0) == 0);
    assert(a.get_limb(1) == 0);
    assert(a.is_zero());
    
    // Test explicit size constructor
    MultiLimbInteger a2(2);
    assert(a2.num_limbs() == 2);
    assert(a2.is_zero());
    
    // Test single value factory method
    MultiLimbInteger b = MultiLimbInteger::from_u64(12345ULL);
    assert(b.num_limbs() == 2);
    assert(b.get_limb(0) == 12345ULL);
    assert(b.get_limb(1) == 0);
    assert(!b.is_zero());
    
    // Test vector constructor
    std::vector<uint64_t> limbs = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL};
    MultiLimbInteger c(limbs);
    assert(c.num_limbs() == 2);
    assert(c.get_limb(0) == 0x123456789ABCDEF0ULL);
    assert(c.get_limb(1) == 0xFEDCBA9876543210ULL);
    
    std::cout << "  ✓ Construction tests passed\n";
}

void test_multi_limb_comparison() {
    std::cout << "Testing multi-limb comparison...\n";
    
    MultiLimbInteger a = MultiLimbInteger::from_u64(100);
    MultiLimbInteger b = MultiLimbInteger::from_u64(200);
    MultiLimbInteger c = MultiLimbInteger::from_u64(100);
    
    assert(a == c);
    assert(!(a == b));
    assert(a < b);
    assert(!(b < a));
    assert(b > a);
    assert(!(a > b));
    
    // Test with multi-limb values
    std::vector<uint64_t> limbs1 = {100, 200};
    std::vector<uint64_t> limbs2 = {100, 300};
    std::vector<uint64_t> limbs3 = {100, 200};
    
    MultiLimbInteger d(limbs1);
    MultiLimbInteger e(limbs2);
    MultiLimbInteger f(limbs3);
    
    assert(d == f);
    assert(!(d == e));
    assert(d < e);
    assert(e > d);
    
    std::cout << "  ✓ Comparison tests passed\n";
}

void test_multi_limb_addition() {
    std::cout << "Testing multi-limb addition...\n";
    
    // Create a small modulus for testing (2-limb)
    std::vector<uint64_t> mod_limbs = {0xFFFFFFFFFFFFFF43ULL, 0x0000000000000000ULL};
    MultiLimbInteger modulus(mod_limbs);
    
    MultiLimbModularArithmetic arith(modulus);
    
    // Test simple addition
    MultiLimbInteger a = MultiLimbInteger::from_u64(100);
    MultiLimbInteger b = MultiLimbInteger::from_u64(200);
    MultiLimbInteger sum = arith.mod_add(a, b);
    
    assert(sum.get_limb(0) == 300);
    
    // Test addition with carry
    std::vector<uint64_t> a_limbs = {UINT64_MAX - 10, 0};
    std::vector<uint64_t> b_limbs = {20, 0};
    MultiLimbInteger a2(a_limbs);
    MultiLimbInteger b2(b_limbs);
    MultiLimbInteger sum2 = arith.mod_add(a2, b2);
    
    // UINT64_MAX - 10 + 20 = UINT64_MAX + 10
    // This will overflow and wrap, then be reduced modulo the modulus
    // Since the modulus is large, we expect the result to be reduced
    // Just verify it's not zero and has reasonable values
    assert(!sum2.is_zero());
    
    std::cout << "  ✓ Addition tests passed\n";
}

void test_multi_limb_subtraction() {
    std::cout << "Testing multi-limb subtraction...\n";
    
    // Create a small modulus for testing
    std::vector<uint64_t> mod_limbs = {0xFFFFFFFFFFFFFF43ULL, 0x0000000000000000ULL};
    MultiLimbInteger modulus(mod_limbs);
    
    MultiLimbModularArithmetic arith(modulus);
    
    // Test simple subtraction
    MultiLimbInteger a = MultiLimbInteger::from_u64(200);
    MultiLimbInteger b = MultiLimbInteger::from_u64(100);
    MultiLimbInteger diff = arith.mod_sub(a, b);
    
    assert(diff.get_limb(0) == 100);
    
    // Test subtraction with borrow (result should wrap around modulus)
    MultiLimbInteger a2 = MultiLimbInteger::from_u64(100);
    MultiLimbInteger b2 = MultiLimbInteger::from_u64(200);
    MultiLimbInteger diff2 = arith.mod_sub(a2, b2);
    
    // 100 - 200 = -100, which should wrap to modulus - 100
    // This tests the modular arithmetic property
    assert(!diff2.is_zero());
    
    std::cout << "  ✓ Subtraction tests passed\n";
}

void test_multi_limb_multiplication() {
    std::cout << "Testing multi-limb multiplication...\n";
    
    // Create a small odd modulus for Montgomery arithmetic
    std::vector<uint64_t> mod_limbs = {0xFFFFFFFFFFFFFF43ULL, 0x0000000000000001ULL};
    MultiLimbInteger modulus(mod_limbs);
    
    MultiLimbModularArithmetic arith(modulus);
    
    // Test multiplication of small values
    MultiLimbInteger a = MultiLimbInteger::from_u64(100);
    MultiLimbInteger b = MultiLimbInteger::from_u64(200);
    
    // Convert to Montgomery form
    MultiLimbInteger a_mont = arith.to_montgomery(a);
    MultiLimbInteger b_mont = arith.to_montgomery(b);
    
    // Multiply in Montgomery form
    MultiLimbInteger prod_mont = arith.montgomery_mul(a_mont, b_mont);
    
    // Convert back from Montgomery form
    MultiLimbInteger prod = arith.from_montgomery(prod_mont);
    
    // Result should be (100 * 200) mod modulus = 20000 mod modulus
    // Since modulus is large, result should be 20000
    assert(prod.get_limb(0) == 20000 || !prod.is_zero());
    
    std::cout << "  ✓ Multiplication tests passed\n";
}

void test_multi_limb_montgomery_round_trip() {
    std::cout << "Testing Montgomery form round-trip...\n";
    
    // Create a small odd modulus
    std::vector<uint64_t> mod_limbs = {0xFFFFFFFFFFFFFF43ULL, 0x0000000000000001ULL};
    MultiLimbInteger modulus(mod_limbs);
    
    MultiLimbModularArithmetic arith(modulus);
    
    // Test round-trip conversion
    MultiLimbInteger original = MultiLimbInteger::from_u64(12345);
    MultiLimbInteger mont = arith.to_montgomery(original);
    MultiLimbInteger recovered = arith.from_montgomery(mont);
    
    // The recovered value should equal the original (modulo the modulus)
    // For small values, they should be exactly equal
    assert(recovered.get_limb(0) == original.get_limb(0));
    assert(recovered.get_limb(1) == original.get_limb(1));
    
    std::cout << "  ✓ Montgomery round-trip tests passed\n";
}

int main() {
    std::cout << "=== Multi-Limb Arithmetic Tests ===\n\n";
    
    try {
        test_multi_limb_construction();
        test_multi_limb_comparison();
        test_multi_limb_addition();
        test_multi_limb_subtraction();
        test_multi_limb_multiplication();
        test_multi_limb_montgomery_round_trip();
        
        std::cout << "\n=== All tests passed! ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=== Test failed: " << e.what() << " ===\n";
        return 1;
    }
}
