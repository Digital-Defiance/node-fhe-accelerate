/**
 * Example C++ Property Tests
 * 
 * This file demonstrates how to use the C++ test harness for property-based testing.
 * These tests will be expanded as the native implementation progresses.
 * 
 * To compile and run (once native implementation is ready):
 *   clang++ -std=c++17 -I../include -o test_runner example_property_test.cpp
 *   ./test_runner
 */

#include "test_harness.h"
#include <iostream>

using namespace fhe_accelerate::test;

/**
 * Example Property 1: Polynomial Addition Commutativity
 * For any polynomials p1, p2: p1 + p2 = p2 + p1
 */
void test_polynomial_addition_commutativity() {
    TestRandom rng(42);
    PolynomialGenerator gen(rng);
    
    PropertyTestRunner runner("Polynomial Addition Commutativity", 100);
    
    runner.run([&]() {
        uint32_t degree = rng.next_poly_degree();
        uint64_t modulus = NTTPrimeGenerator::get_prime(degree);
        
        TestPolynomial p1 = gen.generate(degree, modulus);
        TestPolynomial p2 = gen.generate(degree, modulus);
        
        // Compute p1 + p2
        TestPolynomial sum1(degree, modulus);
        for (uint32_t i = 0; i < degree; i++) {
            sum1.coeffs[i] = (p1.coeffs[i] + p2.coeffs[i]) % modulus;
        }
        
        // Compute p2 + p1
        TestPolynomial sum2(degree, modulus);
        for (uint32_t i = 0; i < degree; i++) {
            sum2.coeffs[i] = (p2.coeffs[i] + p1.coeffs[i]) % modulus;
        }
        
        // Assert equality
        TestAssert::assert_polynomials_equal(sum1, sum2, "Addition should be commutative");
    });
}

/**
 * Example Property 2: Polynomial Addition Identity
 * For any polynomial p: p + 0 = p
 */
void test_polynomial_addition_identity() {
    TestRandom rng(43);
    PolynomialGenerator gen(rng);
    
    PropertyTestRunner runner("Polynomial Addition Identity", 100);
    
    runner.run([&]() {
        uint32_t degree = rng.next_poly_degree();
        uint64_t modulus = NTTPrimeGenerator::get_prime(degree);
        
        TestPolynomial p = gen.generate(degree, modulus);
        TestPolynomial zero = gen.generate_zero(degree, modulus);
        
        // Compute p + 0
        TestPolynomial sum(degree, modulus);
        for (uint32_t i = 0; i < degree; i++) {
            sum.coeffs[i] = (p.coeffs[i] + zero.coeffs[i]) % modulus;
        }
        
        // Assert p + 0 = p
        TestAssert::assert_polynomials_equal(sum, p, "Adding zero should preserve polynomial");
    });
}

/**
 * Example Property 3: Modular Reduction Correctness
 * For any coefficient a and modulus q: (a mod q) < q
 */
void test_modular_reduction_bounds() {
    TestRandom rng(44);
    
    PropertyTestRunner runner("Modular Reduction Bounds", 100);
    
    runner.run([&]() {
        uint32_t degree = rng.next_poly_degree();
        uint64_t modulus = NTTPrimeGenerator::get_prime(degree);
        
        uint64_t a = rng.next_u64();
        uint64_t reduced = a % modulus;
        
        TestAssert::assert_true(reduced < modulus, "Reduced value should be less than modulus");
    });
}

/**
 * Example Property 4: Small Coefficient Generation
 * Small coefficients should be in {-1, 0, 1}
 */
void test_small_coefficient_generation() {
    TestRandom rng(45);
    PolynomialGenerator gen(rng);
    
    PropertyTestRunner runner("Small Coefficient Generation", 100);
    
    runner.run([&]() {
        uint32_t degree = rng.next_poly_degree();
        uint64_t modulus = NTTPrimeGenerator::get_prime(degree);
        
        TestPolynomial small = gen.generate_small(degree, modulus);
        
        // Check each coefficient is in {0, 1, modulus-1} (representing {0, 1, -1})
        for (uint32_t i = 0; i < degree; i++) {
            uint64_t coeff = small.coeffs[i];
            bool valid = (coeff == 0) || (coeff == 1) || (coeff == modulus - 1);
            TestAssert::assert_true(valid, "Small coefficient should be in {-1, 0, 1}");
        }
    });
}

/**
 * Main test runner
 */
int main() {
    std::cout << "=== FHE Accelerate C++ Property Tests ===\n\n";
    
    try {
        test_polynomial_addition_commutativity();
        std::cout << "✓ Polynomial addition commutativity passed\n\n";
        
        test_polynomial_addition_identity();
        std::cout << "✓ Polynomial addition identity passed\n\n";
        
        test_modular_reduction_bounds();
        std::cout << "✓ Modular reduction bounds passed\n\n";
        
        test_small_coefficient_generation();
        std::cout << "✓ Small coefficient generation passed\n\n";
        
        std::cout << "=== All tests passed! ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=== Test suite failed ===\n";
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
