/**
 * C++ Test Harness for FHE Accelerate
 * 
 * This header provides a lightweight test harness for C++ property-based testing.
 * It includes random generators for FHE data structures and assertion utilities.
 * 
 * The harness is designed to work with the native C++ implementation and can be
 * compiled independently of the Node.js bindings for faster iteration.
 */

#ifndef FHE_ACCELERATE_TEST_HARNESS_H
#define FHE_ACCELERATE_TEST_HARNESS_H

#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <functional>
#include <iostream>
#include <sstream>

namespace fhe_accelerate {
namespace test {

/**
 * Random number generator for tests
 * Uses a seeded generator for reproducibility
 */
class TestRandom {
public:
    explicit TestRandom(uint64_t seed = 42) : gen_(seed), dist_(0, UINT64_MAX) {}
    
    // Generate random uint64_t
    uint64_t next_u64() {
        return dist_(gen_);
    }
    
    // Generate random uint64_t in range [0, max)
    uint64_t next_u64_bounded(uint64_t max) {
        if (max == 0) return 0;
        return dist_(gen_) % max;
    }
    
    // Generate random coefficient modulo q
    uint64_t next_coefficient(uint64_t modulus) {
        return next_u64_bounded(modulus);
    }
    
    // Generate random small coefficient in {-1, 0, 1}
    int32_t next_small_coefficient() {
        return static_cast<int32_t>(next_u64_bounded(3)) - 1;
    }
    
    // Generate random boolean
    bool next_bool() {
        return (next_u64() & 1) == 1;
    }
    
    // Generate random polynomial degree (power of 2)
    uint32_t next_poly_degree() {
        const uint32_t degrees[] = {1024, 2048, 4096, 8192, 16384, 32768};
        return degrees[next_u64_bounded(6)];
    }
    
    // Reseed the generator
    void reseed(uint64_t seed) {
        gen_.seed(seed);
    }
    
private:
    std::mt19937_64 gen_;
    std::uniform_int_distribution<uint64_t> dist_;
};

/**
 * Test polynomial structure
 */
struct TestPolynomial {
    std::vector<uint64_t> coeffs;
    uint64_t modulus;
    uint32_t degree;
    bool is_ntt;
    
    TestPolynomial(uint32_t deg, uint64_t mod)
        : coeffs(deg, 0), modulus(mod), degree(deg), is_ntt(false) {}
};

/**
 * Random polynomial generator
 */
class PolynomialGenerator {
public:
    explicit PolynomialGenerator(TestRandom& rng) : rng_(rng) {}
    
    // Generate random polynomial with given degree and modulus
    TestPolynomial generate(uint32_t degree, uint64_t modulus) {
        TestPolynomial poly(degree, modulus);
        for (uint32_t i = 0; i < degree; i++) {
            poly.coeffs[i] = rng_.next_coefficient(modulus);
        }
        return poly;
    }
    
    // Generate zero polynomial
    TestPolynomial generate_zero(uint32_t degree, uint64_t modulus) {
        return TestPolynomial(degree, modulus);
    }
    
    // Generate identity polynomial (constant term 1, rest 0)
    TestPolynomial generate_identity(uint32_t degree, uint64_t modulus) {
        TestPolynomial poly(degree, modulus);
        poly.coeffs[0] = 1;
        return poly;
    }
    
    // Generate polynomial with small coefficients (for secret keys)
    TestPolynomial generate_small(uint32_t degree, uint64_t modulus) {
        TestPolynomial poly(degree, modulus);
        for (uint32_t i = 0; i < degree; i++) {
            int32_t small = rng_.next_small_coefficient();
            poly.coeffs[i] = (small < 0) ? modulus + small : small;
        }
        return poly;
    }
    
private:
    TestRandom& rng_;
};

/**
 * NTT-friendly prime generator
 */
class NTTPrimeGenerator {
public:
    // Get NTT-friendly prime for given degree
    // These are primes of the form q = 1 (mod 2N)
    static uint64_t get_prime(uint32_t degree) {
        switch (degree) {
            case 1024:
                return 132120577ULL; // 2^27 - 2^11 + 1
            case 2048:
                return 1099511627777ULL; // 2^40 - 2^13 + 1
            case 4096:
            case 8192:
            case 16384:
            case 32768:
                return 4611686018326724609ULL; // 2^62 - 2^15 + 1
            default:
                return 1099511627777ULL;
        }
    }
    
    // Get list of NTT-friendly primes for RNS representation
    static std::vector<uint64_t> get_prime_chain(uint32_t degree, size_t count) {
        std::vector<uint64_t> primes;
        // For now, return the same prime multiple times
        // In production, we'd have a larger set of distinct primes
        uint64_t prime = get_prime(degree);
        for (size_t i = 0; i < count; i++) {
            primes.push_back(prime);
        }
        return primes;
    }
};

/**
 * Polynomial comparison utilities
 */
class PolynomialComparator {
public:
    // Check if two polynomials are equal
    static bool equal(const TestPolynomial& a, const TestPolynomial& b) {
        if (a.degree != b.degree) return false;
        if (a.modulus != b.modulus) return false;
        if (a.is_ntt != b.is_ntt) return false;
        
        for (uint32_t i = 0; i < a.degree; i++) {
            if (a.coeffs[i] != b.coeffs[i]) return false;
        }
        return true;
    }
    
    // Check if two polynomials are equal modulo the modulus
    static bool equal_mod(const TestPolynomial& a, const TestPolynomial& b) {
        if (a.degree != b.degree) return false;
        if (a.modulus != b.modulus) return false;
        
        for (uint32_t i = 0; i < a.degree; i++) {
            uint64_t diff = (a.coeffs[i] >= b.coeffs[i]) 
                ? (a.coeffs[i] - b.coeffs[i]) 
                : (b.coeffs[i] - a.coeffs[i]);
            if (diff % a.modulus != 0) return false;
        }
        return true;
    }
    
    // Check if two polynomials are approximately equal (within tolerance)
    static bool approx_equal(const TestPolynomial& a, const TestPolynomial& b, uint64_t tolerance = 1) {
        if (a.degree != b.degree) return false;
        if (a.modulus != b.modulus) return false;
        
        for (uint32_t i = 0; i < a.degree; i++) {
            uint64_t diff = (a.coeffs[i] >= b.coeffs[i]) 
                ? (a.coeffs[i] - b.coeffs[i]) 
                : (b.coeffs[i] - a.coeffs[i]);
            uint64_t normalized_diff = diff % a.modulus;
            if (normalized_diff > tolerance && normalized_diff < a.modulus - tolerance) {
                return false;
            }
        }
        return true;
    }
};

/**
 * Test assertion utilities
 */
class TestAssert {
public:
    static void assert_true(bool condition, const std::string& message = "") {
        if (!condition) {
            throw std::runtime_error("Assertion failed: " + message);
        }
    }
    
    static void assert_false(bool condition, const std::string& message = "") {
        assert_true(!condition, message);
    }
    
    static void assert_equal(uint64_t actual, uint64_t expected, const std::string& message = "") {
        if (actual != expected) {
            std::ostringstream oss;
            oss << "Assertion failed: " << message 
                << " (expected=" << expected << ", actual=" << actual << ")";
            throw std::runtime_error(oss.str());
        }
    }
    
    static void assert_polynomials_equal(
        const TestPolynomial& actual,
        const TestPolynomial& expected,
        const std::string& message = ""
    ) {
        if (!PolynomialComparator::equal(actual, expected)) {
            std::ostringstream oss;
            oss << "Assertion failed: " << message << "\n"
                << "  Polynomials not equal\n"
                << "  Degree: actual=" << actual.degree << ", expected=" << expected.degree << "\n"
                << "  Modulus: actual=" << actual.modulus << ", expected=" << expected.modulus << "\n"
                << "  IsNTT: actual=" << actual.is_ntt << ", expected=" << expected.is_ntt;
            
            // Find first differing coefficient
            for (uint32_t i = 0; i < actual.degree && i < expected.degree; i++) {
                if (actual.coeffs[i] != expected.coeffs[i]) {
                    oss << "\n  First difference at index " << i 
                        << ": actual=" << actual.coeffs[i] 
                        << ", expected=" << expected.coeffs[i];
                    break;
                }
            }
            
            throw std::runtime_error(oss.str());
        }
    }
};

/**
 * Property test runner
 * Runs a property test with multiple iterations
 */
class PropertyTestRunner {
public:
    PropertyTestRunner(const std::string& name, size_t iterations = 100)
        : name_(name), iterations_(iterations), passed_(0), failed_(0) {}
    
    // Run a property test
    template<typename Func>
    void run(Func property) {
        std::cout << "Running property test: " << name_ << " (" << iterations_ << " iterations)\n";
        
        for (size_t i = 0; i < iterations_; i++) {
            try {
                property();
                passed_++;
            } catch (const std::exception& e) {
                failed_++;
                std::cerr << "  Iteration " << i << " failed: " << e.what() << "\n";
            }
        }
        
        std::cout << "  Results: " << passed_ << " passed, " << failed_ << " failed\n";
        
        if (failed_ > 0) {
            throw std::runtime_error("Property test failed: " + name_);
        }
    }
    
private:
    std::string name_;
    size_t iterations_;
    size_t passed_;
    size_t failed_;
};

} // namespace test
} // namespace fhe_accelerate

// Convenience macro for test assertions
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": " << message; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#endif // FHE_ACCELERATE_TEST_HARNESS_H
