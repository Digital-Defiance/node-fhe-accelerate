# C++ Test Harness for FHE Accelerate

This directory contains the C++ test harness and property-based tests for the native FHE implementation.

## Overview

The test harness provides:

- **Random Generators**: Seeded random number generators for reproducible tests
- **Polynomial Generators**: Generate random, zero, identity, and small-coefficient polynomials
- **NTT Prime Generators**: Provide NTT-friendly primes for various polynomial degrees
- **Comparison Utilities**: Compare polynomials with exact or approximate equality
- **Assertion Utilities**: Rich assertion helpers with detailed error messages
- **Property Test Runner**: Execute property tests with configurable iteration counts

## Files

- `test_harness.h` - Main test harness header with all utilities
- `example_property_test.cpp` - Example property tests demonstrating usage
- `README.md` - This file

## Building and Running Tests

### Prerequisites

- Clang or GCC with C++17 support
- macOS with Apple Silicon (for Metal/SME features)

### Compile Example Tests

```bash
# From the cpp/tests directory
clang++ -std=c++17 -I../include -o test_runner example_property_test.cpp

# Run the tests
./test_runner
```

### Expected Output

```
=== FHE Accelerate C++ Property Tests ===

Running property test: Polynomial Addition Commutativity (100 iterations)
  Results: 100 passed, 0 failed
✓ Polynomial addition commutativity passed

Running property test: Polynomial Addition Identity (100 iterations)
  Results: 100 passed, 0 failed
✓ Polynomial addition identity passed

Running property test: Modular Reduction Bounds (100 iterations)
  Results: 100 passed, 0 failed
✓ Modular reduction bounds passed

Running property test: Small Coefficient Generation (100 iterations)
  Results: 100 passed, 0 failed
✓ Small coefficient generation passed

=== All tests passed! ===
```

## Writing Property Tests

### Basic Structure

```cpp
#include "test_harness.h"

using namespace fhe_accelerate::test;

void test_my_property() {
    TestRandom rng(seed);
    PolynomialGenerator gen(rng);
    
    PropertyTestRunner runner("My Property Name", 100);
    
    runner.run([&]() {
        // Generate random test data
        uint32_t degree = rng.next_poly_degree();
        uint64_t modulus = NTTPrimeGenerator::get_prime(degree);
        TestPolynomial p = gen.generate(degree, modulus);
        
        // Perform operation
        // ...
        
        // Assert property holds
        TestAssert::assert_true(condition, "Property should hold");
    });
}
```

### Available Generators

#### TestRandom

```cpp
TestRandom rng(42);  // Seeded for reproducibility

uint64_t val = rng.next_u64();                    // Random uint64_t
uint64_t bounded = rng.next_u64_bounded(100);     // Random in [0, 100)
uint64_t coeff = rng.next_coefficient(modulus);   // Random in [0, modulus)
int32_t small = rng.next_small_coefficient();     // Random in {-1, 0, 1}
bool flag = rng.next_bool();                      // Random boolean
uint32_t degree = rng.next_poly_degree();         // Random power of 2 degree
```

#### PolynomialGenerator

```cpp
PolynomialGenerator gen(rng);

TestPolynomial p = gen.generate(degree, modulus);        // Random polynomial
TestPolynomial zero = gen.generate_zero(degree, modulus); // Zero polynomial
TestPolynomial one = gen.generate_identity(degree, modulus); // Identity
TestPolynomial sk = gen.generate_small(degree, modulus);  // Small coefficients
```

#### NTTPrimeGenerator

```cpp
uint64_t prime = NTTPrimeGenerator::get_prime(4096);
std::vector<uint64_t> primes = NTTPrimeGenerator::get_prime_chain(4096, 3);
```

### Available Assertions

```cpp
TestAssert::assert_true(condition, "message");
TestAssert::assert_false(condition, "message");
TestAssert::assert_equal(actual, expected, "message");
TestAssert::assert_polynomials_equal(p1, p2, "message");
```

### Available Comparisons

```cpp
bool exact = PolynomialComparator::equal(p1, p2);
bool modular = PolynomialComparator::equal_mod(p1, p2);
bool approx = PolynomialComparator::approx_equal(p1, p2, tolerance);
```

## Integration with Native Implementation

As the native C++ FHE implementation progresses, these tests will be expanded to cover:

1. **NTT Operations** (Property 1: NTT Round-Trip Consistency)
   - Forward/inverse NTT correctness
   - Twiddle factor computation
   - Hardware backend equivalence

2. **Modular Arithmetic** (Property 2: Modular Multiplication Correctness)
   - Montgomery multiplication
   - Barrett reduction
   - Multi-limb arithmetic

3. **Polynomial Ring Operations** (Properties 3, 4)
   - Multiplication commutativity
   - Multiplicative identity
   - Reduction modulo X^N + 1

4. **Cryptographic Operations** (Properties 5-10)
   - Key generation and serialization
   - Encryption/decryption round-trip
   - Homomorphic operations
   - Bootstrapping

5. **System Properties** (Properties 11-12)
   - Parameter validation
   - Streaming equivalence

## Property Test Configuration

- **Default iterations**: 100 per property
- **Seeded RNG**: All tests use seeded generators for reproducibility
- **Shrinking**: Manual shrinking by reducing random input ranges on failure
- **Parallel execution**: Tests can be run in parallel (future enhancement)

## Best Practices

1. **Use descriptive test names** that clearly state the property being tested
2. **Include property numbers** from the design document in comments
3. **Use seeded RNG** for reproducibility (different seed per test)
4. **Test edge cases** explicitly in addition to random generation
5. **Provide clear error messages** in assertions
6. **Keep tests focused** on a single property
7. **Document expected behavior** in comments

## Future Enhancements

- [ ] Integration with CMake build system
- [ ] Parallel test execution
- [ ] Test result reporting (JSON output)
- [ ] Automatic shrinking on failure
- [ ] Coverage analysis
- [ ] Benchmark integration
- [ ] CI/CD integration
