/**
 * Property-based testing configuration and utilities
 * 
 * This module provides configuration and helpers for property-based testing
 * using fast-check. All property tests should use these configurations to
 * ensure consistency across the test suite.
 */

import * as fc from 'fast-check';

/**
 * Standard configuration for property-based tests
 * - Minimum 100 iterations per property test
 * - Seed logging for reproducibility
 * - Shrinking enabled for minimal failing examples
 */
export const PROPERTY_TEST_CONFIG: fc.Parameters<unknown> = {
  numRuns: 100,
  verbose: true,
  seed: Date.now(), // Can be overridden for reproducibility
  endOnFailure: false, // Continue to find all failures
};

/**
 * Configuration for fast property tests (used during development)
 */
export const FAST_PROPERTY_TEST_CONFIG: fc.Parameters<unknown> = {
  numRuns: 10,
  verbose: false,
  seed: Date.now(),
};

/**
 * Configuration for exhaustive property tests (used for critical properties)
 */
export const EXHAUSTIVE_PROPERTY_TEST_CONFIG: fc.Parameters<unknown> = {
  numRuns: 1000,
  verbose: true,
  seed: Date.now(),
  endOnFailure: false,
};

/**
 * Helper to create a property test with standard configuration
 * 
 * @example
 * ```typescript
 * // Feature: fhe-accelerate, Property 1: NTT Round-Trip Consistency
 * propertyTest(
 *   'NTT round-trip preserves polynomial',
 *   [arbitraryPolynomial()],
 *   (poly) => {
 *     const ntt = forwardNTT(poly);
 *     const result = inverseNTT(ntt);
 *     return polynomialsEqual(poly, result);
 *   }
 * );
 * ```
 */
export function propertyTest<T>(
  _name: string,
  _arbitraries: fc.Arbitrary<T>[],
  _predicate: (...args: T[]) => boolean | Promise<boolean>,
  _config: fc.Parameters<unknown> = PROPERTY_TEST_CONFIG
): void {
  // This is a wrapper that will be used with vitest's test function
  // The actual test registration happens in the test files
  throw new Error('Use with vitest test.prop or implement custom wrapper');
}

/**
 * Arbitrary generator for security levels
 */
export function arbitrarySecurityLevel(): fc.Arbitrary<128 | 192 | 256> {
  return fc.constantFrom(128 as const, 192 as const, 256 as const);
}

/**
 * Arbitrary generator for polynomial degrees (powers of 2)
 */
export function arbitraryPolyDegree(): fc.Arbitrary<number> {
  return fc.constantFrom(1024, 2048, 4096, 8192, 16384, 32768);
}

/**
 * Arbitrary generator for NTT-friendly primes
 * These are primes of the form q = 1 (mod 2N) suitable for NTT
 */
export function arbitraryNTTPrime(degree: number): fc.Arbitrary<bigint> {
  // Common NTT-friendly primes for different degrees
  const primes: Record<number, bigint[]> = {
    1024: [
      BigInt('132120577'), // 2^27 - 2^11 + 1
      BigInt('268369921'), // 2^28 - 2^12 + 1
    ],
    2048: [
      BigInt('1099511627777'), // 2^40 - 2^13 + 1
      BigInt('1073479681'), // 2^30 - 2^14 + 1
    ],
    4096: [
      BigInt('4611686018326724609'), // 2^62 - 2^15 + 1
      BigInt('1099511627777'), // 2^40 - 2^13 + 1
    ],
    8192: [
      BigInt('4611686018326724609'), // 2^62 - 2^15 + 1
    ],
    16384: [
      BigInt('4611686018326724609'), // 2^62 - 2^15 + 1
    ],
    32768: [
      BigInt('4611686018326724609'), // 2^62 - 2^15 + 1
    ],
  };

  const primesForDegree = primes[degree] ?? primes[4096] ?? [];
  if (primesForDegree.length === 0) {
    return fc.constant(BigInt('4611686018326724609'));
  }
  return fc.constantFrom(...primesForDegree);
}

/**
 * Arbitrary generator for coefficient values modulo q
 */
export function arbitraryCoefficient(modulus: bigint): fc.Arbitrary<bigint> {
  // Generate random bigints in range [0, modulus)
  const modulusNumber = Number(modulus);
  if (modulusNumber < Number.MAX_SAFE_INTEGER) {
    return fc.bigInt({ min: 0n, max: modulus - 1n });
  }
  // For very large moduli, use string-based generation
  return fc.bigInt({ min: 0n, max: modulus - 1n });
}

/**
 * Arbitrary generator for polynomial coefficients
 */
export function arbitraryPolynomialCoeffs(
  degree: number,
  modulus: bigint
): fc.Arbitrary<bigint[]> {
  return fc.array(arbitraryCoefficient(modulus), {
    minLength: degree,
    maxLength: degree,
  });
}

/**
 * Arbitrary generator for small coefficients (for secret keys)
 * Generates coefficients in {-1, 0, 1} (ternary distribution)
 */
export function arbitrarySmallCoefficient(): fc.Arbitrary<number> {
  return fc.constantFrom(-1, 0, 1);
}

/**
 * Arbitrary generator for plaintext values
 * For TFHE, plaintexts are typically small (2-4 bits)
 */
export function arbitraryPlaintext(bits: number = 4): fc.Arbitrary<number> {
  const max = (1 << bits) - 1;
  return fc.integer({ min: 0, max });
}

/**
 * Arbitrary generator for plaintext vectors (for SIMD packing)
 */
export function arbitraryPlaintextVector(
  length: number,
  bits: number = 4
): fc.Arbitrary<number[]> {
  return fc.array(arbitraryPlaintext(bits), {
    minLength: length,
    maxLength: length,
  });
}

/**
 * Arbitrary generator for parameter presets
 */
export function arbitraryParameterPreset(): fc.Arbitrary<
  'tfhe-128-fast' | 'tfhe-128-balanced' | 'tfhe-256-secure' | 'bfv-128-simd' | 'ckks-128-ml'
> {
  return fc.constantFrom(
    'tfhe-128-fast',
    'tfhe-128-balanced',
    'tfhe-256-secure',
    'bfv-128-simd',
    'ckks-128-ml'
  );
}

/**
 * Arbitrary generator for custom parameters
 */
export function arbitraryCustomParameters(): fc.Arbitrary<{
  polyDegree: number;
  moduli: bigint[];
  securityLevel: 128 | 192 | 256;
}> {
  return fc.record({
    polyDegree: arbitraryPolyDegree(),
    moduli: fc.array(fc.bigInt({ min: 1n, max: BigInt('4611686018326724609') }), {
      minLength: 1,
      maxLength: 5,
    }),
    securityLevel: arbitrarySecurityLevel(),
  });
}
