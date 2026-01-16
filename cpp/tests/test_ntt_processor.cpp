/**
 * NTT Processor Tests
 * 
 * Tests for the Number Theoretic Transform implementation.
 * Includes property-based tests for round-trip consistency.
 * 
 * Feature: fhe-accelerate, Property 1: NTT Round-Trip Consistency
 * Validates: Requirements 1.6, 1.2
 */

#include "test_harness.h"
#include "../include/ntt_processor.h"
#include <iostream>
#include <cstring>
#include <chrono>

using namespace fhe_accelerate;
using namespace fhe_accelerate::test;

// ============================================================================
// Test Configuration
// ============================================================================

// NTT-friendly primes for different polynomial degrees
// These are primes q such that q ≡ 1 (mod 2N)
static const uint64_t PRIME_1024 = 132120577ULL;        // 2^27 - 2^11 + 1
static const uint64_t PRIME_2048 = 1099511627777ULL;    // 2^40 - 2^13 + 1
static const uint64_t PRIME_4096 = 4611686018326724609ULL; // 2^62 - 2^15 + 1

// ============================================================================
// Unit Tests
// ============================================================================

void test_is_power_of_two() {
    std::cout << "Testing is_power_of_two...\n";
    
    TestAssert::assert_true(NTTProcessor::is_power_of_two(1), "1 is power of 2");
    TestAssert::assert_true(NTTProcessor::is_power_of_two(2), "2 is power of 2");
    TestAssert::assert_true(NTTProcessor::is_power_of_two(4), "4 is power of 2");
    TestAssert::assert_true(NTTProcessor::is_power_of_two(1024), "1024 is power of 2");
    TestAssert::assert_true(NTTProcessor::is_power_of_two(32768), "32768 is power of 2");
    
    TestAssert::assert_false(NTTProcessor::is_power_of_two(0), "0 is not power of 2");
    TestAssert::assert_false(NTTProcessor::is_power_of_two(3), "3 is not power of 2");
    TestAssert::assert_false(NTTProcessor::is_power_of_two(5), "5 is not power of 2");
    TestAssert::assert_false(NTTProcessor::is_power_of_two(1000), "1000 is not power of 2");
    
    std::cout << "  PASSED\n";
}

void test_log2_pow2() {
    std::cout << "Testing log2_pow2...\n";
    
    TestAssert::assert_equal(NTTProcessor::log2_pow2(1), 0, "log2(1) = 0");
    TestAssert::assert_equal(NTTProcessor::log2_pow2(2), 1, "log2(2) = 1");
    TestAssert::assert_equal(NTTProcessor::log2_pow2(4), 2, "log2(4) = 2");
    TestAssert::assert_equal(NTTProcessor::log2_pow2(1024), 10, "log2(1024) = 10");
    TestAssert::assert_equal(NTTProcessor::log2_pow2(32768), 15, "log2(32768) = 15");
    
    std::cout << "  PASSED\n";
}

void test_bit_reverse() {
    std::cout << "Testing bit_reverse...\n";
    
    // 3 bits: 0b000 -> 0b000, 0b001 -> 0b100, 0b010 -> 0b010, etc.
    TestAssert::assert_equal(NTTProcessor::bit_reverse(0, 3), 0, "bit_reverse(0, 3) = 0");
    TestAssert::assert_equal(NTTProcessor::bit_reverse(1, 3), 4, "bit_reverse(1, 3) = 4");
    TestAssert::assert_equal(NTTProcessor::bit_reverse(2, 3), 2, "bit_reverse(2, 3) = 2");
    TestAssert::assert_equal(NTTProcessor::bit_reverse(3, 3), 6, "bit_reverse(3, 3) = 6");
    TestAssert::assert_equal(NTTProcessor::bit_reverse(4, 3), 1, "bit_reverse(4, 3) = 1");
    TestAssert::assert_equal(NTTProcessor::bit_reverse(5, 3), 5, "bit_reverse(5, 3) = 5");
    TestAssert::assert_equal(NTTProcessor::bit_reverse(6, 3), 3, "bit_reverse(6, 3) = 3");
    TestAssert::assert_equal(NTTProcessor::bit_reverse(7, 3), 7, "bit_reverse(7, 3) = 7");
    
    std::cout << "  PASSED\n";
}

void test_mod_pow() {
    std::cout << "Testing mod_pow...\n";
    
    // 2^10 mod 1000 = 1024 mod 1000 = 24
    TestAssert::assert_equal(NTTProcessor::mod_pow(2, 10, 1000), 24, "2^10 mod 1000 = 24");
    
    // 3^5 mod 7 = 243 mod 7 = 5
    TestAssert::assert_equal(NTTProcessor::mod_pow(3, 5, 7), 5, "3^5 mod 7 = 5");
    
    // Any number to power 0 is 1
    TestAssert::assert_equal(NTTProcessor::mod_pow(123, 0, 1000), 1, "123^0 mod 1000 = 1");
    
    // Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
    uint64_t prime = 17;
    TestAssert::assert_equal(NTTProcessor::mod_pow(3, prime - 1, prime), 1, "Fermat's little theorem");
    
    std::cout << "  PASSED\n";
}

void test_mod_inverse() {
    std::cout << "Testing mod_inverse...\n";
    
    // 3 * 5 ≡ 1 (mod 7), so 3^(-1) ≡ 5 (mod 7)
    uint64_t inv = NTTProcessor::mod_inverse(3, 7);
    TestAssert::assert_equal((3 * inv) % 7, 1, "3 * 3^(-1) ≡ 1 (mod 7)");
    
    // Test with larger prime
    uint64_t prime = PRIME_1024;
    uint64_t a = 12345;
    inv = NTTProcessor::mod_inverse(a, prime);
    __uint128_t product = static_cast<__uint128_t>(a) * inv;
    TestAssert::assert_equal(product % prime, 1, "a * a^(-1) ≡ 1 (mod prime)");
    
    std::cout << "  PASSED\n";
}

void test_find_primitive_root() {
    std::cout << "Testing find_primitive_root...\n";
    
    // Test for degree 1024 with known NTT-friendly prime
    uint32_t degree = 1024;
    uint64_t modulus = PRIME_1024;
    
    uint64_t omega = NTTProcessor::find_primitive_root(degree, modulus);
    
    // Verify ω^(2N) ≡ 1 (mod q)
    uint64_t omega_2n = NTTProcessor::mod_pow(omega, 2 * degree, modulus);
    TestAssert::assert_equal(omega_2n, 1, "ω^(2N) ≡ 1 (mod q)");
    
    // Verify ω^N ≡ -1 (mod q)
    uint64_t omega_n = NTTProcessor::mod_pow(omega, degree, modulus);
    TestAssert::assert_equal(omega_n, modulus - 1, "ω^N ≡ -1 (mod q)");
    
    std::cout << "  PASSED\n";
}

void test_ntt_processor_construction() {
    std::cout << "Testing NTTProcessor construction...\n";
    
    // Valid construction
    NTTProcessor ntt(1024, PRIME_1024);
    TestAssert::assert_equal(ntt.get_degree(), 1024, "Degree should be 1024");
    TestAssert::assert_equal(ntt.get_modulus(), PRIME_1024, "Modulus should match");
    
    // Test twiddle factors are computed
    const TwiddleFactors& twiddles = ntt.get_twiddles();
    TestAssert::assert_equal(twiddles.degree, 1024, "Twiddle degree should be 1024");
    TestAssert::assert_equal(twiddles.forward.size(), 1024, "Forward twiddles size");
    TestAssert::assert_equal(twiddles.inverse.size(), 1024, "Inverse twiddles size");
    
    std::cout << "  PASSED\n";
}

void test_ntt_simple() {
    std::cout << "Testing simple NTT forward/inverse...\n";
    
    uint32_t degree = 8;  // Small degree for easy verification
    // Need a prime q ≡ 1 (mod 16) for degree 8
    // 17 works: 17 - 1 = 16 = 2 * 8
    uint64_t modulus = 17;
    
    NTTProcessor ntt(degree, modulus);
    
    // Simple polynomial: [1, 2, 3, 4, 5, 6, 7, 8]
    std::vector<uint64_t> coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> original = coeffs;
    
    // Forward NTT
    ntt.forward_ntt(coeffs.data(), degree);
    
    // Inverse NTT
    ntt.inverse_ntt(coeffs.data(), degree);
    
    // Verify round-trip
    for (uint32_t i = 0; i < degree; i++) {
        TestAssert::assert_equal(coeffs[i], original[i], 
            "Coefficient " + std::to_string(i) + " should match after round-trip");
    }
    
    std::cout << "  PASSED\n";
}

// ============================================================================
// Property-Based Tests
// ============================================================================

/**
 * Property 1: NTT Round-Trip Consistency
 * 
 * For any valid polynomial p with coefficients in Z_q, applying forward NTT
 * followed by inverse NTT SHALL produce a polynomial equal to the original p.
 * 
 * Validates: Requirements 1.6, 1.2
 */
void test_property_ntt_round_trip() {
    std::cout << "\n=== Property Test: NTT Round-Trip Consistency ===\n";
    std::cout << "Feature: fhe-accelerate, Property 1: NTT Round-Trip Consistency\n";
    std::cout << "Validates: Requirements 1.6, 1.2\n\n";
    
    TestRandom rng(42);  // Fixed seed for reproducibility
    
    // Test configurations: (degree, modulus, iterations)
    // Note: Larger degrees with 128-bit modular arithmetic are slow
    // We test smaller degrees thoroughly and larger degrees with fewer iterations
    std::vector<std::tuple<uint32_t, uint64_t, size_t>> configs = {
        {8, 17, 100},                    // Small test case
        {16, 97, 100},                   // 97 ≡ 1 (mod 32)
        {1024, PRIME_1024, 20},          // Standard FHE degree
    };
    
    size_t total_iterations = 0;
    size_t passed = 0;
    size_t failed = 0;
    
    for (const auto& [degree, modulus, num_iters] : configs) {
        std::cout << "Testing degree=" << degree << ", modulus=" << modulus << " (" << num_iters << " iterations)\n";
        
        NTTProcessor ntt(degree, modulus);
        
        // Run iterations per configuration
        for (size_t iter = 0; iter < num_iters; iter++) {
            total_iterations++;
            
            // Generate random polynomial
            std::vector<uint64_t> coeffs(degree);
            for (uint32_t i = 0; i < degree; i++) {
                coeffs[i] = rng.next_coefficient(modulus);
            }
            
            // Save original
            std::vector<uint64_t> original = coeffs;
            
            // Forward NTT
            ntt.forward_ntt(coeffs.data(), degree);
            
            // Inverse NTT
            ntt.inverse_ntt(coeffs.data(), degree);
            
            // Verify round-trip
            bool match = true;
            for (uint32_t i = 0; i < degree; i++) {
                if (coeffs[i] != original[i]) {
                    match = false;
                    std::cerr << "  Iteration " << iter << " failed at index " << i
                              << ": expected " << original[i] << ", got " << coeffs[i] << "\n";
                    break;
                }
            }
            
            if (match) {
                passed++;
            } else {
                failed++;
            }
        }
        
        std::cout << "  Completed " << num_iters << " iterations\n";
    }
    
    std::cout << "\nProperty Test Results:\n";
    std::cout << "  Total iterations: " << total_iterations << "\n";
    std::cout << "  Passed: " << passed << "\n";
    std::cout << "  Failed: " << failed << "\n";
    
    if (failed > 0) {
        throw std::runtime_error("Property test failed: NTT Round-Trip Consistency");
    }
    
    std::cout << "\n=== Property Test PASSED ===\n";
}

/**
 * Test NTT round-trip with NEON optimization
 */
void test_property_ntt_round_trip_neon() {
    std::cout << "\n=== Property Test: NTT Round-Trip (NEON) ===\n";
    
    TestRandom rng(123);
    
    std::vector<std::tuple<uint32_t, uint64_t, size_t>> configs = {
        {1024, PRIME_1024, 10},
    };
    
    size_t passed = 0;
    size_t failed = 0;
    
    for (const auto& [degree, modulus, num_iters] : configs) {
        std::cout << "Testing NEON NTT: degree=" << degree << " (" << num_iters << " iterations)\n";
        
        NTTProcessor ntt(degree, modulus);
        
        for (size_t iter = 0; iter < num_iters; iter++) {
            std::vector<uint64_t> coeffs(degree);
            for (uint32_t i = 0; i < degree; i++) {
                coeffs[i] = rng.next_coefficient(modulus);
            }
            
            std::vector<uint64_t> original = coeffs;
            
            // Use NEON-optimized NTT
            ntt.forward_ntt_neon(coeffs.data(), degree);
            ntt.inverse_ntt_neon(coeffs.data(), degree);
            
            bool match = true;
            for (uint32_t i = 0; i < degree; i++) {
                if (coeffs[i] != original[i]) {
                    match = false;
                    break;
                }
            }
            
            if (match) passed++;
            else failed++;
        }
    }
    
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    
    if (failed > 0) {
        throw std::runtime_error("Property test failed: NTT Round-Trip (NEON)");
    }
    
    std::cout << "=== Property Test PASSED ===\n";
}

/**
 * Test that forward NTT produces different output (not identity)
 */
void test_ntt_transforms_data() {
    std::cout << "Testing that NTT actually transforms data...\n";
    
    NTTProcessor ntt(1024, PRIME_1024);
    TestRandom rng(42);
    
    std::vector<uint64_t> coeffs(1024);
    for (uint32_t i = 0; i < 1024; i++) {
        coeffs[i] = rng.next_coefficient(PRIME_1024);
    }
    
    std::vector<uint64_t> original = coeffs;
    
    // Forward NTT
    ntt.forward_ntt(coeffs.data(), 1024);
    
    // Check that at least some coefficients changed
    size_t changed = 0;
    for (uint32_t i = 0; i < 1024; i++) {
        if (coeffs[i] != original[i]) {
            changed++;
        }
    }
    
    // Most coefficients should change (unless polynomial is very special)
    TestAssert::assert_true(changed > 100, "NTT should transform most coefficients");
    
    std::cout << "  " << changed << " coefficients changed after forward NTT\n";
    std::cout << "  PASSED\n";
}

/**
 * Performance benchmark
 */
void benchmark_ntt() {
    std::cout << "\n=== NTT Performance Benchmark ===\n";
    
    // Only benchmark degree 1024 for now (larger degrees are slow without Montgomery optimization)
    std::vector<std::pair<uint32_t, uint64_t>> configs = {
        {1024, PRIME_1024},
    };
    
    TestRandom rng(42);
    
    for (const auto& [degree, modulus] : configs) {
        NTTProcessor ntt(degree, modulus);
        
        // Generate random polynomial
        std::vector<uint64_t> coeffs(degree);
        for (uint32_t i = 0; i < degree; i++) {
            coeffs[i] = rng.next_coefficient(modulus);
        }
        
        // Warm up
        for (int i = 0; i < 10; i++) {
            std::vector<uint64_t> temp = coeffs;
            ntt.forward_ntt(temp.data(), degree);
        }
        
        // Benchmark forward NTT
        const int iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            std::vector<uint64_t> temp = coeffs;
            ntt.forward_ntt(temp.data(), degree);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_us = static_cast<double>(duration.count()) / iterations;
        double ops_per_sec = 1000000.0 / avg_us;
        
        std::cout << "Degree " << degree << ": " << avg_us << " µs/NTT, "
                  << static_cast<int>(ops_per_sec) << " NTT/sec\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "NTT Processor Tests\n";
    std::cout << "========================================\n\n";
    
    try {
        // Unit tests
        test_is_power_of_two();
        test_log2_pow2();
        test_bit_reverse();
        test_mod_pow();
        test_mod_inverse();
        test_find_primitive_root();
        test_ntt_processor_construction();
        test_ntt_simple();
        test_ntt_transforms_data();
        
        // Property-based tests
        test_property_ntt_round_trip();
        test_property_ntt_round_trip_neon();
        
        // Performance benchmark
        benchmark_ntt();
        
        std::cout << "\n========================================\n";
        std::cout << "All tests PASSED!\n";
        std::cout << "========================================\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n========================================\n";
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        std::cerr << "========================================\n";
        return 1;
    }
}
