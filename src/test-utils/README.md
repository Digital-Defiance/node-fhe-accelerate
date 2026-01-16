# Testing Infrastructure for @digitaldefiance/node-fhe-accelerate

This directory contains the testing infrastructure for property-based testing and unit testing of the FHE acceleration library.

## Overview

The testing infrastructure provides:

1. **Property-Based Testing Configuration** - Standard configurations for fast-check tests
2. **Arbitrary Generators** - Generators for FHE-specific data types (polynomials, coefficients, parameters)
3. **Ciphertext Comparison Utilities** - Helpers for comparing encrypted data and polynomials
4. **Mock Data Structures** - Test structures for use before native implementation is complete

## Files

- `property-test-config.ts` - Property test configuration and arbitrary generators
- `ciphertext-comparison.ts` - Comparison utilities for FHE data structures
- `example.prop.test.ts` - Example property tests demonstrating usage
- `index.ts` - Main export file
- `README.md` - This file

## Property-Based Testing

### Configuration

Three standard configurations are provided:

```typescript
import { PROPERTY_TEST_CONFIG, FAST_PROPERTY_TEST_CONFIG, EXHAUSTIVE_PROPERTY_TEST_CONFIG } from './test-utils';

// Standard: 100 iterations
fc.assert(fc.property(...), PROPERTY_TEST_CONFIG);

// Fast: 10 iterations (for development)
fc.assert(fc.property(...), FAST_PROPERTY_TEST_CONFIG);

// Exhaustive: 1000 iterations (for critical properties)
fc.assert(fc.property(...), EXHAUSTIVE_PROPERTY_TEST_CONFIG);
```

### Arbitrary Generators

The library provides generators for FHE-specific types:

```typescript
import {
  arbitrarySecurityLevel,
  arbitraryPolyDegree,
  arbitraryNTTPrime,
  arbitraryCoefficient,
  arbitraryPolynomialCoeffs,
  arbitrarySmallCoefficient,
  arbitraryPlaintext,
  arbitraryPlaintextVector,
  arbitraryParameterPreset,
  arbitraryCustomParameters,
} from './test-utils';

// Generate security levels (128, 192, or 256)
const secLevel = arbitrarySecurityLevel();

// Generate polynomial degrees (powers of 2 from 1024 to 32768)
const degree = arbitraryPolyDegree();

// Generate NTT-friendly primes for a given degree
const prime = arbitraryNTTPrime(4096);

// Generate polynomial coefficients
const coeffs = arbitraryPolynomialCoeffs(4096, BigInt('1099511627777'));

// Generate small coefficients for secret keys ({-1, 0, 1})
const smallCoeff = arbitrarySmallCoefficient();

// Generate plaintexts (default 4 bits)
const plaintext = arbitraryPlaintext(4);

// Generate plaintext vectors for SIMD
const vector = arbitraryPlaintextVector(1024, 4);

// Generate parameter presets
const preset = arbitraryParameterPreset();

// Generate custom parameters
const params = arbitraryCustomParameters();
```

### Writing Property Tests

Property tests should follow this pattern:

```typescript
import { describe, it } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG, arbitraryPolyDegree } from './test-utils';

describe('My Feature', () => {
  // Feature: fhe-accelerate, Property X: Property Name
  it('should satisfy property X', () => {
    fc.assert(
      fc.property(
        arbitraryPolyDegree(),
        (degree) => {
          // Test logic here
          return true; // or use expect() assertions
        }
      ),
      PROPERTY_TEST_CONFIG
    );
  });
});
```

**Important**: Always include a comment referencing the design document property number:

```typescript
// Feature: fhe-accelerate, Property 1: NTT Round-Trip Consistency
```

## Ciphertext Comparison

### Polynomial Comparison

```typescript
import {
  polynomialsEqual,
  polynomialsApproxEqual,
  assertPolynomialsEqual,
  assertPolynomialsApproxEqual,
  type TestPolynomial,
} from './test-utils';

const poly1: TestPolynomial = {
  coeffs: [1n, 2n, 3n],
  modulus: 17n,
  degree: 3,
  isNTT: false,
};

const poly2: TestPolynomial = {
  coeffs: [1n, 2n, 3n],
  modulus: 17n,
  degree: 3,
  isNTT: false,
};

// Exact equality
if (polynomialsEqual(poly1, poly2)) {
  // ...
}

// Approximate equality (with tolerance)
if (polynomialsApproxEqual(poly1, poly2, 1n)) {
  // ...
}

// Assertions with detailed error messages
assertPolynomialsEqual(poly1, poly2, 'Polynomials should be equal');
assertPolynomialsApproxEqual(poly1, poly2, 1n, 'Polynomials should be approximately equal');
```

### Helper Functions

```typescript
import {
  createZeroPolynomial,
  createIdentityPolynomial,
  createRandomPolynomial,
  addPolynomials,
  negatePolynomial,
} from './test-utils';

// Create special polynomials
const zero = createZeroPolynomial(4096, BigInt('1099511627777'));
const identity = createIdentityPolynomial(4096, BigInt('1099511627777'));
const random = createRandomPolynomial(4096, BigInt('1099511627777'));

// Polynomial operations
const sum = addPolynomials(poly1, poly2);
const negated = negatePolynomial(poly1);
```

### Array Comparison

```typescript
import {
  bigIntArraysEqual,
  bigIntArraysEqualMod,
  floatArraysApproxEqual,
} from './test-utils';

// Exact equality for bigint arrays
bigIntArraysEqual([1n, 2n, 3n], [1n, 2n, 3n]); // true

// Modular equality
bigIntArraysEqualMod([1n, 2n, 3n], [18n, 19n, 20n], 17n); // true

// Approximate equality for floats (CKKS)
floatArraysApproxEqual([1.0, 2.0], [1.000001, 2.000001], 1e-5); // true
```

## Mock Data Structures

Before the native implementation is complete, use mock structures for testing:

```typescript
import {
  type MockCiphertext,
  type MockPlaintext,
  mockCiphertextsEqual,
  mockPlaintextsEqual,
} from './test-utils';

const mockCt: MockCiphertext = {
  components: [poly1, poly2],
  level: 0,
  scale: 1.0,
};

const mockPt: MockPlaintext = {
  value: 42n,
  encoding: 'integer',
};

// Compare mock structures
if (mockCiphertextsEqual(mockCt1, mockCt2)) {
  // ...
}

if (mockPlaintextsEqual(mockPt1, mockPt2)) {
  // ...
}
```

## Running Tests

```bash
# Run all tests
yarn test

# Run tests in watch mode
yarn test:watch

# Run tests with coverage
yarn test:coverage

# Run only property tests
yarn test example.prop.test.ts
```

## C++ Test Harness

See `cpp/tests/README.md` for information about the C++ test harness for native code testing.

## Best Practices

1. **Use Standard Configurations**: Always use `PROPERTY_TEST_CONFIG` for consistency
2. **Reference Design Properties**: Include comments linking tests to design document properties
3. **Use Appropriate Generators**: Choose generators that match your test's domain
4. **Test Edge Cases**: Use unit tests for specific edge cases, property tests for general properties
5. **Keep Tests Focused**: Each test should verify a single property or behavior
6. **Use Descriptive Names**: Test names should clearly state what property is being tested
7. **Handle Failures Gracefully**: Property test failures should provide clear counterexamples

## Integration with Design Document

This testing infrastructure implements the testing strategy defined in the design document:

- **Property Tests**: Verify universal properties across randomly generated inputs (Properties 1-12)
- **Unit Tests**: Verify specific examples, edge cases, and error conditions
- **Dual Approach**: Both test types are complementary and required for comprehensive validation

### Property Test Tagging

Each property test must include a comment referencing its design property:

```typescript
// Feature: fhe-accelerate, Property 1: NTT Round-Trip Consistency
// Feature: fhe-accelerate, Property 2: Modular Multiplication Correctness
// etc.
```

This ensures traceability between requirements, design, and tests.

## Future Enhancements

- [ ] Integration with native C++ tests
- [ ] Automated property shrinking improvements
- [ ] Performance benchmarking integration
- [ ] Coverage analysis for property tests
- [ ] CI/CD integration with property test reporting
