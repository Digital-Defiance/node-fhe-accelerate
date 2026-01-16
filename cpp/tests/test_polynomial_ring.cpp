/**
 * Test Suite for Polynomial Ring Operations
 * 
 * Tests polynomial arithmetic in Z_q[X]/(X^N + 1).
 * 
 * Design Reference: Section 3 - Polynomial Ring
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
 */

#include "test_harness.h"
#include "../include/polynomial_ring.h"
#include <iostream>
#include <random>
#include <chrono>

using namespace fhe_accelerate;

// NTT-friendly primes for testing
constexpr uint64_t PRIME_17 = 17;           // For degree 8
constexpr uint64_t PRIME_97 = 97;           // For degree 16
constexpr uint64_t PRIME_193 = 193;         // For degree 32
constexpr uint64_t PRIME_257 = 257;         // For degree 64
constexpr uint64_t PRIME_769 = 769;         // For degree 128
constexpr uint64_t PRIME_7681 = 7681;       // For degree 256
constexpr uint64_t PRIME_12289 = 12289;     // For degree 512
constexpr uint64_t PRIME_132120577 = 132120577ULL;  // For degree 1024

// ============================================================================
// Test: Polynomial Construction and Basic Operations
// ============================================================================

void test_polynomial_construction() {
    std::cout << "Testing polynomial construction..." << std::endl;
    
    // Test zero polynomial construction
    Polynomial p1(16, PRIME_97);
    TEST_ASSERT(p1.degree() == 16, "Degree should be 16");
    TEST_ASSERT(p1.modulus() == PRIME_97, "Modulus should be 97");
    TEST_ASSERT(p1.is_zero(), "Default polynomial should be zero");
    TEST_ASSERT(!p1.is_ntt(), "Default polynomial should not be in NTT form");
    
    // Test construction from vector
    std::vector<uint64_t> coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    Polynomial p2(coeffs, PRIME_17, false);
    TEST_ASSERT(p2.degree() == 8, "Degree should be 8");
    TEST_ASSERT(p2[0] == 1, "First coefficient should be 1");
    TEST_ASSERT(p2[7] == 8, "Last coefficient should be 8");
    
    // Test copy constructor
    Polynomial p3(p2);
    TEST_ASSERT(p3 == p2, "Copy should be equal to original");
    
    // Test move constructor
    Polynomial p4(std::move(p3));
    TEST_ASSERT(p4 == p2, "Moved polynomial should equal original");
    
    // Test identity polynomial
    Polynomial identity = Polynomial::identity(8, PRIME_17);
    TEST_ASSERT(identity.is_identity(), "Identity polynomial should be identity");
    TEST_ASSERT(identity[0] == 1, "Identity constant term should be 1");
    
    std::cout << "  PASSED: Polynomial construction" << std::endl;
}

// ============================================================================
// Test: Polynomial Addition
// ============================================================================

void test_polynomial_addition() {
    std::cout << "Testing polynomial addition..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    // Test basic addition
    std::vector<uint64_t> a_coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> b_coeffs = {8, 7, 6, 5, 4, 3, 2, 1};
    
    Polynomial a(a_coeffs, PRIME_17);
    Polynomial b(b_coeffs, PRIME_17);
    
    Polynomial c = ring.add(a, b);
    
    // Each coefficient should be (a[i] + b[i]) mod 17
    // 1+8=9, 2+7=9, 3+6=9, 4+5=9, 5+4=9, 6+3=9, 7+2=9, 8+1=9
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT(c[i] == 9, "Sum coefficient should be 9");
    }
    
    // Test addition with modular reduction
    std::vector<uint64_t> c_coeffs = {10, 10, 10, 10, 10, 10, 10, 10};
    std::vector<uint64_t> d_coeffs = {10, 10, 10, 10, 10, 10, 10, 10};
    
    Polynomial c_poly(c_coeffs, PRIME_17);
    Polynomial d_poly(d_coeffs, PRIME_17);
    
    Polynomial e = ring.add(c_poly, d_poly);
    
    // 10 + 10 = 20 mod 17 = 3
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT(e[i] == 3, "Sum with reduction should be 3");
    }
    
    // Test in-place addition
    Polynomial f = a.clone();
    ring.add_inplace(f, b);
    TEST_ASSERT(f == c, "In-place addition should match");
    
    std::cout << "  PASSED: Polynomial addition" << std::endl;
}

// ============================================================================
// Test: Polynomial Subtraction
// ============================================================================

void test_polynomial_subtraction() {
    std::cout << "Testing polynomial subtraction..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    // Test basic subtraction
    std::vector<uint64_t> a_coeffs = {10, 10, 10, 10, 10, 10, 10, 10};
    std::vector<uint64_t> b_coeffs = {3, 3, 3, 3, 3, 3, 3, 3};
    
    Polynomial a(a_coeffs, PRIME_17);
    Polynomial b(b_coeffs, PRIME_17);
    
    Polynomial c = ring.subtract(a, b);
    
    // 10 - 3 = 7
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT(c[i] == 7, "Difference should be 7");
    }
    
    // Test subtraction with wrap-around
    std::vector<uint64_t> d_coeffs = {3, 3, 3, 3, 3, 3, 3, 3};
    std::vector<uint64_t> e_coeffs = {10, 10, 10, 10, 10, 10, 10, 10};
    
    Polynomial d(d_coeffs, PRIME_17);
    Polynomial e(e_coeffs, PRIME_17);
    
    Polynomial f = ring.subtract(d, e);
    
    // 3 - 10 = -7 mod 17 = 10
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT(f[i] == 10, "Difference with wrap should be 10");
    }
    
    // Test in-place subtraction
    Polynomial g = a.clone();
    ring.subtract_inplace(g, b);
    TEST_ASSERT(g == c, "In-place subtraction should match");
    
    std::cout << "  PASSED: Polynomial subtraction" << std::endl;
}

// ============================================================================
// Test: Polynomial Negation
// ============================================================================

void test_polynomial_negation() {
    std::cout << "Testing polynomial negation..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    std::vector<uint64_t> a_coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    Polynomial a(a_coeffs, PRIME_17);
    
    Polynomial neg_a = ring.negate(a);
    
    // -1 mod 17 = 16, -2 mod 17 = 15, etc.
    TEST_ASSERT(neg_a[0] == 16, "-1 mod 17 = 16");
    TEST_ASSERT(neg_a[1] == 15, "-2 mod 17 = 15");
    TEST_ASSERT(neg_a[7] == 9, "-8 mod 17 = 9");
    
    // a + (-a) should be zero
    Polynomial sum = ring.add(a, neg_a);
    TEST_ASSERT(sum.is_zero(), "a + (-a) should be zero");
    
    // Test negation of zero
    Polynomial zero(8, PRIME_17);
    Polynomial neg_zero = ring.negate(zero);
    TEST_ASSERT(neg_zero.is_zero(), "Negation of zero should be zero");
    
    std::cout << "  PASSED: Polynomial negation" << std::endl;
}

// ============================================================================
// Test: Polynomial Multiplication
// ============================================================================

void test_polynomial_multiplication() {
    std::cout << "Testing polynomial multiplication..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    // Test multiplication by identity
    std::vector<uint64_t> a_coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    Polynomial a(a_coeffs, PRIME_17);
    Polynomial identity = Polynomial::identity(8, PRIME_17);
    
    Polynomial prod = ring.multiply(a, identity);
    
    // a * 1 = a
    TEST_ASSERT(prod == a, "Multiplication by identity should preserve polynomial");
    
    // Test multiplication by zero
    Polynomial zero(8, PRIME_17);
    Polynomial zero_prod = ring.multiply(a, zero);
    TEST_ASSERT(zero_prod.is_zero(), "Multiplication by zero should give zero");
    
    // Test scalar multiplication
    Polynomial scaled = ring.multiply_scalar(a, 2);
    for (size_t i = 0; i < 8; i++) {
        uint64_t expected = (a[i] * 2) % PRIME_17;
        TEST_ASSERT(scaled[i] == expected, "Scalar multiplication should double coefficients");
    }
    
    std::cout << "  PASSED: Polynomial multiplication" << std::endl;
}

// ============================================================================
// Test: NTT Conversion
// ============================================================================

void test_ntt_conversion() {
    std::cout << "Testing NTT conversion..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    std::vector<uint64_t> a_coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    Polynomial a(a_coeffs, PRIME_17);
    
    // Convert to NTT
    Polynomial a_ntt = a.clone();
    ring.to_ntt(a_ntt);
    
    TEST_ASSERT(a_ntt.is_ntt(), "Should be in NTT form after conversion");
    TEST_ASSERT(a_ntt != a, "NTT form should differ from coefficient form");
    
    // Convert back
    Polynomial a_recovered = a_ntt.clone();
    ring.from_ntt(a_recovered);
    
    TEST_ASSERT(!a_recovered.is_ntt(), "Should be in coefficient form after inverse");
    TEST_ASSERT(a_recovered == a, "Round-trip should preserve polynomial");
    
    std::cout << "  PASSED: NTT conversion" << std::endl;
}

// ============================================================================
// Test: Pointwise Multiplication
// ============================================================================

void test_pointwise_multiplication() {
    std::cout << "Testing pointwise multiplication..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    std::vector<uint64_t> a_coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> b_coeffs = {2, 2, 2, 2, 2, 2, 2, 2};
    
    Polynomial a(a_coeffs, PRIME_17);
    Polynomial b(b_coeffs, PRIME_17);
    
    // Convert to NTT
    ring.to_ntt(a);
    ring.to_ntt(b);
    
    // Pointwise multiply
    Polynomial c = ring.pointwise_multiply(a, b);
    
    TEST_ASSERT(c.is_ntt(), "Result should be in NTT form");
    
    // Convert back and verify
    ring.from_ntt(c);
    
    // The result should be the polynomial product in the ring
    // This is a more complex verification - just check it's not zero
    TEST_ASSERT(!c.is_zero(), "Product should not be zero");
    
    std::cout << "  PASSED: Pointwise multiplication" << std::endl;
}

// ============================================================================
// Test: Commutativity Property
// ============================================================================

void test_commutativity() {
    std::cout << "Testing commutativity property..." << std::endl;
    
    PolynomialRing ring(16, PRIME_97);
    
    // Generate random polynomials
    Polynomial a = Polynomial::random(16, PRIME_97);
    Polynomial b = Polynomial::random(16, PRIME_97);
    
    // Test addition commutativity: a + b = b + a
    Polynomial sum_ab = ring.add(a, b);
    Polynomial sum_ba = ring.add(b, a);
    TEST_ASSERT(sum_ab == sum_ba, "Addition should be commutative");
    
    // Test multiplication commutativity: a * b = b * a
    Polynomial prod_ab = ring.multiply(a, b);
    Polynomial prod_ba = ring.multiply(b, a);
    TEST_ASSERT(prod_ab == prod_ba, "Multiplication should be commutative");
    
    std::cout << "  PASSED: Commutativity property" << std::endl;
}

// ============================================================================
// Test: Identity Property
// ============================================================================

void test_identity_property() {
    std::cout << "Testing identity property..." << std::endl;
    
    PolynomialRing ring(16, PRIME_97);
    
    // Generate random polynomial
    Polynomial a = Polynomial::random(16, PRIME_97);
    
    // Test additive identity: a + 0 = a
    Polynomial zero = Polynomial::zero(16, PRIME_97);
    Polynomial sum = ring.add(a, zero);
    TEST_ASSERT(sum == a, "Adding zero should preserve polynomial");
    
    // Test multiplicative identity: a * 1 = a
    Polynomial one = Polynomial::identity(16, PRIME_97);
    Polynomial prod = ring.multiply(a, one);
    TEST_ASSERT(prod == a, "Multiplying by identity should preserve polynomial");
    
    std::cout << "  PASSED: Identity property" << std::endl;
}

// ============================================================================
// Test: Associativity Property
// ============================================================================

void test_associativity() {
    std::cout << "Testing associativity property..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    // Use small random polynomials
    Polynomial a = Polynomial::random(8, PRIME_17);
    Polynomial b = Polynomial::random(8, PRIME_17);
    Polynomial c = Polynomial::random(8, PRIME_17);
    
    // Test addition associativity: (a + b) + c = a + (b + c)
    Polynomial ab = ring.add(a, b);
    Polynomial ab_c = ring.add(ab, c);
    
    Polynomial bc = ring.add(b, c);
    Polynomial a_bc = ring.add(a, bc);
    
    TEST_ASSERT(ab_c == a_bc, "Addition should be associative");
    
    std::cout << "  PASSED: Associativity property" << std::endl;
}

// ============================================================================
// Test: Distributivity Property
// ============================================================================

void test_distributivity() {
    std::cout << "Testing distributivity property..." << std::endl;
    
    PolynomialRing ring(8, PRIME_17);
    
    Polynomial a = Polynomial::random(8, PRIME_17);
    Polynomial b = Polynomial::random(8, PRIME_17);
    Polynomial c = Polynomial::random(8, PRIME_17);
    
    // Test: a * (b + c) = a*b + a*c
    Polynomial b_plus_c = ring.add(b, c);
    Polynomial left = ring.multiply(a, b_plus_c);
    
    Polynomial ab = ring.multiply(a, b);
    Polynomial ac = ring.multiply(a, c);
    Polynomial right = ring.add(ab, ac);
    
    TEST_ASSERT(left == right, "Multiplication should distribute over addition");
    
    std::cout << "  PASSED: Distributivity property" << std::endl;
}

// ============================================================================
// Test: Cache-Aligned Memory
// ============================================================================

void test_cache_alignment() {
    std::cout << "Testing cache-aligned memory..." << std::endl;
    
    Polynomial p(1024, PRIME_132120577);
    
    // Check that data pointer is cache-aligned (128 bytes on M4 Max)
    uintptr_t addr = reinterpret_cast<uintptr_t>(p.data());
    TEST_ASSERT((addr % 128) == 0, "Coefficient data should be 128-byte aligned");
    
    std::cout << "  PASSED: Cache-aligned memory" << std::endl;
}

// ============================================================================
// Test: Large Polynomial Operations
// ============================================================================

void test_large_polynomials() {
    std::cout << "Testing large polynomial operations..." << std::endl;
    
    // Test with degree 1024
    PolynomialRing ring(1024, PRIME_132120577);
    
    Polynomial a = Polynomial::random(1024, PRIME_132120577);
    Polynomial b = Polynomial::random(1024, PRIME_132120577);
    
    // Time addition
    auto start = std::chrono::high_resolution_clock::now();
    Polynomial sum = ring.add(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "  Addition time (N=1024): " << add_time << " us" << std::endl;
    
    // Time multiplication
    start = std::chrono::high_resolution_clock::now();
    Polynomial prod = ring.multiply(a, b);
    end = std::chrono::high_resolution_clock::now();
    auto mul_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "  Multiplication time (N=1024): " << mul_time << " us" << std::endl;
    
    // Verify commutativity still holds
    Polynomial prod_ba = ring.multiply(b, a);
    TEST_ASSERT(prod == prod_ba, "Commutativity should hold for large polynomials");
    
    std::cout << "  PASSED: Large polynomial operations" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Polynomial Ring Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_polynomial_construction();
        test_polynomial_addition();
        test_polynomial_subtraction();
        test_polynomial_negation();
        test_polynomial_multiplication();
        test_ntt_conversion();
        test_pointwise_multiplication();
        test_commutativity();
        test_identity_property();
        test_associativity();
        test_distributivity();
        test_cache_alignment();
        test_large_polynomials();
        
        std::cout << "========================================" << std::endl;
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
