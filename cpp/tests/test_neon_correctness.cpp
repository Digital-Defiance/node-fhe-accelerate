/**
 * NEON Optimization Correctness Tests
 * 
 * This test verifies that NEON-optimized operations produce the same results
 * as the scalar implementations, ensuring correctness of the vectorized code.
 * 
 * **Validates: Requirements 2.4, 2.6**
 * - NEON SIMD vectorized variants produce correct results
 * - Vectorized operations match scalar implementations
 */

#include "../include/modular_arithmetic.h"
#include "test_harness.h"
#include <iostream>
#include <cassert>
#include <cstring>

using namespace fhe_accelerate;
using namespace fhe_accelerate::test;

// Test modulus - NTT-friendly prime
const uint64_t TEST_MODULUS = 132120577ULL; // 2^27 - 2^11 + 1

void test_montgomery_mul_neon_correctness() {
    std::cout << "Testing Montgomery multiplication NEON correctness...\n";
    
    ModularArithmetic arith(TEST_MODULUS);
    TestRandom rng(12345);
    
    const size_t count = 100;
    std::vector<uint64_t> a(count), b(count);
    std::vector<uint64_t> result_scalar(count), result_neon(count);
    
    // Generate random test data
    for (size_t i = 0; i < count; ++i) {
        a[i] = rng.next_coefficient(TEST_MODULUS);
        b[i] = rng.next_coefficient(TEST_MODULUS);
    }
    
    // Convert to Montgomery form
    std::vector<uint64_t> a_mont(count), b_mont(count);
    for (size_t i = 0; i < count; ++i) {
        a_mont[i] = arith.to_montgomery(a[i]);
        b_mont[i] = arith.to_montgomery(b[i]);
    }
    
    // Compute using scalar implementation
    for (size_t i = 0; i < count; ++i) {
        result_scalar[i] = arith.montgomery_mul(a_mont[i], b_mont[i]);
    }
    
    // Compute using NEON implementation
    arith.montgomery_mul_neon(a_mont.data(), b_mont.data(), result_neon.data(), count);
    
    // Compare results
    for (size_t i = 0; i < count; ++i) {
        if (result_scalar[i] != result_neon[i]) {
            std::cerr << "  MISMATCH at index " << i << ": scalar=" << result_scalar[i] 
                      << ", neon=" << result_neon[i] << "\n";
            assert(false && "NEON Montgomery multiplication mismatch");
        }
    }
    
    std::cout << "  ✓ Montgomery multiplication NEON correctness verified (" << count << " values)\n";
}

void test_mod_add_neon_correctness() {
    std::cout << "Testing modular addition NEON correctness...\n";
    
    ModularArithmetic arith(TEST_MODULUS);
    TestRandom rng(23456);
    
    const size_t count = 100;
    std::vector<uint64_t> a(count), b(count);
    std::vector<uint64_t> result_scalar(count), result_neon(count);
    
    // Generate random test data
    for (size_t i = 0; i < count; ++i) {
        a[i] = rng.next_coefficient(TEST_MODULUS);
        b[i] = rng.next_coefficient(TEST_MODULUS);
    }
    
    // Compute using scalar implementation
    for (size_t i = 0; i < count; ++i) {
        result_scalar[i] = arith.mod_add(a[i], b[i]);
    }
    
    // Compute using NEON implementation
    arith.mod_add_neon(a.data(), b.data(), result_neon.data(), count);
    
    // Compare results
    for (size_t i = 0; i < count; ++i) {
        if (result_scalar[i] != result_neon[i]) {
            std::cerr << "  MISMATCH at index " << i << ": scalar=" << result_scalar[i] 
                      << ", neon=" << result_neon[i] << "\n";
            assert(false && "NEON modular addition mismatch");
        }
    }
    
    std::cout << "  ✓ Modular addition NEON correctness verified (" << count << " values)\n";
}

void test_mod_sub_neon_correctness() {
    std::cout << "Testing modular subtraction NEON correctness...\n";
    
    ModularArithmetic arith(TEST_MODULUS);
    TestRandom rng(34567);
    
    const size_t count = 100;
    std::vector<uint64_t> a(count), b(count);
    std::vector<uint64_t> result_scalar(count), result_neon(count);
    
    // Generate random test data
    for (size_t i = 0; i < count; ++i) {
        a[i] = rng.next_coefficient(TEST_MODULUS);
        b[i] = rng.next_coefficient(TEST_MODULUS);
    }
    
    // Compute using scalar implementation
    for (size_t i = 0; i < count; ++i) {
        result_scalar[i] = arith.mod_sub(a[i], b[i]);
    }
    
    // Compute using NEON implementation
    arith.mod_sub_neon(a.data(), b.data(), result_neon.data(), count);
    
    // Compare results
    for (size_t i = 0; i < count; ++i) {
        if (result_scalar[i] != result_neon[i]) {
            std::cerr << "  MISMATCH at index " << i << ": scalar=" << result_scalar[i] 
                      << ", neon=" << result_neon[i] << "\n";
            assert(false && "NEON modular subtraction mismatch");
        }
    }
    
    std::cout << "  ✓ Modular subtraction NEON correctness verified (" << count << " values)\n";
}

void test_multi_limb_neon_correctness() {
    std::cout << "Testing multi-limb NEON correctness...\n";
    
    // Create a 2-limb odd modulus
    std::vector<uint64_t> mod_limbs = {0xFFFFFFFFFFFFFF43ULL, 0x0000000000000001ULL};
    MultiLimbInteger modulus(mod_limbs);
    
    MultiLimbModularArithmetic arith(modulus);
    TestRandom rng(45678);
    
    const size_t count = 50;
    std::vector<MultiLimbInteger> a(count), b(count);
    std::vector<MultiLimbInteger> result_scalar(count), result_neon(count);
    
    // Generate random test data
    for (size_t i = 0; i < count; ++i) {
        std::vector<uint64_t> a_limbs = {rng.next_u64(), rng.next_u64_bounded(2)};
        std::vector<uint64_t> b_limbs = {rng.next_u64(), rng.next_u64_bounded(2)};
        a[i] = MultiLimbInteger(a_limbs);
        b[i] = MultiLimbInteger(b_limbs);
    }
    
    // Test mod_add
    for (size_t i = 0; i < count; ++i) {
        result_scalar[i] = arith.mod_add(a[i], b[i]);
    }
    arith.mod_add_neon(a.data(), b.data(), result_neon.data(), count);
    
    for (size_t i = 0; i < count; ++i) {
        if (!(result_scalar[i] == result_neon[i])) {
            std::cerr << "  MISMATCH in mod_add at index " << i << "\n";
            assert(false && "NEON multi-limb mod_add mismatch");
        }
    }
    std::cout << "  ✓ Multi-limb mod_add NEON correctness verified\n";
    
    // Test mod_sub
    for (size_t i = 0; i < count; ++i) {
        result_scalar[i] = arith.mod_sub(a[i], b[i]);
    }
    arith.mod_sub_neon(a.data(), b.data(), result_neon.data(), count);
    
    for (size_t i = 0; i < count; ++i) {
        if (!(result_scalar[i] == result_neon[i])) {
            std::cerr << "  MISMATCH in mod_sub at index " << i << "\n";
            assert(false && "NEON multi-limb mod_sub mismatch");
        }
    }
    std::cout << "  ✓ Multi-limb mod_sub NEON correctness verified\n";
    
    // Test montgomery_mul
    std::vector<MultiLimbInteger> a_mont(count), b_mont(count);
    for (size_t i = 0; i < count; ++i) {
        a_mont[i] = arith.to_montgomery(a[i]);
        b_mont[i] = arith.to_montgomery(b[i]);
    }
    
    for (size_t i = 0; i < count; ++i) {
        result_scalar[i] = arith.montgomery_mul(a_mont[i], b_mont[i]);
    }
    arith.montgomery_mul_neon(a_mont.data(), b_mont.data(), result_neon.data(), count);
    
    for (size_t i = 0; i < count; ++i) {
        if (!(result_scalar[i] == result_neon[i])) {
            std::cerr << "  MISMATCH in montgomery_mul at index " << i << "\n";
            assert(false && "NEON multi-limb montgomery_mul mismatch");
        }
    }
    std::cout << "  ✓ Multi-limb montgomery_mul NEON correctness verified\n";
}

void test_neon_edge_cases() {
    std::cout << "Testing NEON edge cases...\n";
    
    ModularArithmetic arith(TEST_MODULUS);
    
    // Test with odd count (to verify handling of remaining elements)
    const size_t odd_count = 7;
    std::vector<uint64_t> a(odd_count), b(odd_count);
    std::vector<uint64_t> result_scalar(odd_count), result_neon(odd_count);
    
    for (size_t i = 0; i < odd_count; ++i) {
        a[i] = i * 1000;
        b[i] = i * 500;
    }
    
    // Test mod_add with odd count
    for (size_t i = 0; i < odd_count; ++i) {
        result_scalar[i] = arith.mod_add(a[i], b[i]);
    }
    arith.mod_add_neon(a.data(), b.data(), result_neon.data(), odd_count);
    
    for (size_t i = 0; i < odd_count; ++i) {
        assert(result_scalar[i] == result_neon[i] && "NEON odd count mismatch");
    }
    std::cout << "  ✓ NEON handles odd element counts correctly\n";
    
    // Test with single element
    const size_t single_count = 1;
    std::vector<uint64_t> single_a = {12345};
    std::vector<uint64_t> single_b = {67890};
    std::vector<uint64_t> single_result_scalar(1), single_result_neon(1);
    
    single_result_scalar[0] = arith.mod_add(single_a[0], single_b[0]);
    arith.mod_add_neon(single_a.data(), single_b.data(), single_result_neon.data(), single_count);
    
    assert(single_result_scalar[0] == single_result_neon[0] && "NEON single element mismatch");
    std::cout << "  ✓ NEON handles single element correctly\n";
    
    // Test with zero values
    std::vector<uint64_t> zeros(10, 0);
    std::vector<uint64_t> values(10);
    std::vector<uint64_t> zero_result(10);
    
    for (size_t i = 0; i < 10; ++i) {
        values[i] = i * 1000;
    }
    
    arith.mod_add_neon(zeros.data(), values.data(), zero_result.data(), 10);
    for (size_t i = 0; i < 10; ++i) {
        assert(zero_result[i] == values[i] && "NEON zero addition mismatch");
    }
    std::cout << "  ✓ NEON handles zero values correctly\n";
}

int main() {
    std::cout << "=== NEON Optimization Correctness Tests ===\n\n";
    
    try {
        test_montgomery_mul_neon_correctness();
        test_mod_add_neon_correctness();
        test_mod_sub_neon_correctness();
        test_multi_limb_neon_correctness();
        test_neon_edge_cases();
        
        std::cout << "\n=== All NEON correctness tests passed! ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=== Test failed: " << e.what() << " ===\n";
        return 1;
    }
}
