/**
 * Test utilities for ciphertext comparison and validation
 * 
 * This module provides utilities for comparing ciphertexts, polynomials,
 * and other FHE data structures in tests. These utilities handle the
 * complexity of comparing encrypted data and numerical precision issues.
 */

/**
 * Tolerance for floating-point comparisons
 * Used for CKKS scheme where results are approximate
 */
export const FLOAT_TOLERANCE = 1e-6;

/**
 * Compare two bigint arrays for equality
 */
export function bigIntArraysEqual(a: bigint[], b: bigint[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((val, idx) => val === b[idx]);
}

/**
 * Compare two bigint arrays with modular reduction
 * Useful when comparing coefficients that may differ by multiples of the modulus
 */
export function bigIntArraysEqualMod(a: bigint[], b: bigint[], modulus: bigint): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((val, idx) => {
    const bVal = b[idx];
    if (bVal === undefined) return false;
    const diff = (val - bVal) % modulus;
    return diff === 0n || diff === modulus;
  });
}

/**
 * Compare two number arrays for approximate equality (for CKKS)
 */
export function floatArraysApproxEqual(
  a: number[],
  b: number[],
  tolerance: number = FLOAT_TOLERANCE
): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((val, idx) => {
    const bVal = b[idx];
    if (bVal === undefined) return false;
    return Math.abs(val - bVal) <= tolerance;
  });
}

/**
 * Polynomial representation for testing
 */
export interface TestPolynomial {
  coeffs: bigint[];
  modulus: bigint;
  degree: number;
  isNTT?: boolean;
}

/**
 * Compare two polynomials for equality
 */
export function polynomialsEqual(a: TestPolynomial, b: TestPolynomial): boolean {
  if (a.degree !== b.degree) {
    return false;
  }
  if (a.modulus !== b.modulus) {
    return false;
  }
  if (a.isNTT !== b.isNTT) {
    return false;
  }
  return bigIntArraysEqualMod(a.coeffs, b.coeffs, a.modulus);
}

/**
 * Compare two polynomials with a tolerance for coefficient differences
 * Useful when small numerical errors are expected
 */
export function polynomialsApproxEqual(
  a: TestPolynomial,
  b: TestPolynomial,
  maxCoeffDiff: bigint = 1n
): boolean {
  if (a.degree !== b.degree) {
    return false;
  }
  if (a.modulus !== b.modulus) {
    return false;
  }

  return a.coeffs.every((val, idx) => {
    const bCoeff = b.coeffs[idx];
    if (bCoeff === undefined) return false;
    const diff = val > bCoeff ? val - bCoeff : bCoeff - val;
    const normalizedDiff = diff % a.modulus;
    return normalizedDiff <= maxCoeffDiff || normalizedDiff >= a.modulus - maxCoeffDiff;
  });
}

/**
 * Mock ciphertext structure for testing (before native implementation)
 */
export interface MockCiphertext {
  components: TestPolynomial[];
  level: number;
  scale?: number;
}

/**
 * Mock plaintext structure for testing (before native implementation)
 */
export interface MockPlaintext {
  value: bigint | bigint[] | number[];
  encoding: 'integer' | 'simd' | 'ckks';
}

/**
 * Compare two mock ciphertexts for equality
 */
export function mockCiphertextsEqual(a: MockCiphertext, b: MockCiphertext): boolean {
  if (a.components.length !== b.components.length) {
    return false;
  }
  if (a.level !== b.level) {
    return false;
  }
  if (a.scale !== b.scale) {
    return false;
  }
  return a.components.every((poly, idx) => {
    const bPoly = b.components[idx];
    if (!bPoly) return false;
    return polynomialsEqual(poly, bPoly);
  });
}

/**
 * Compare two mock plaintexts for equality
 */
export function mockPlaintextsEqual(a: MockPlaintext, b: MockPlaintext): boolean {
  if (a.encoding !== b.encoding) {
    return false;
  }

  if (typeof a.value === 'bigint' && typeof b.value === 'bigint') {
    return a.value === b.value;
  }

  if (Array.isArray(a.value) && Array.isArray(b.value)) {
    if (a.encoding === 'ckks') {
      return floatArraysApproxEqual(a.value as number[], b.value as number[]);
    }
    return bigIntArraysEqual(a.value as bigint[], b.value as bigint[]);
  }

  return false;
}

/**
 * Helper to create a zero polynomial
 */
export function createZeroPolynomial(degree: number, modulus: bigint): TestPolynomial {
  return {
    coeffs: new Array(degree).fill(0n),
    modulus,
    degree,
    isNTT: false,
  };
}

/**
 * Helper to create an identity polynomial (constant term 1, rest 0)
 */
export function createIdentityPolynomial(degree: number, modulus: bigint): TestPolynomial {
  const coeffs = new Array(degree).fill(0n);
  coeffs[0] = 1n;
  return {
    coeffs,
    modulus,
    degree,
    isNTT: false,
  };
}

/**
 * Helper to create a random polynomial
 */
export function createRandomPolynomial(degree: number, modulus: bigint): TestPolynomial {
  const coeffs = new Array(degree).fill(0n).map(() => {
    // Generate random bigint in range [0, modulus)
    const bytes = new Uint8Array(8);
    crypto.getRandomValues(bytes);
    let value = 0n;
    for (let i = 0; i < 8; i++) {
      const byte = bytes[i];
      if (byte !== undefined) {
        value = (value << 8n) | BigInt(byte);
      }
    }
    return value % modulus;
  });

  return {
    coeffs,
    modulus,
    degree,
    isNTT: false,
  };
}

/**
 * Helper to add two polynomials (coefficient-wise modular addition)
 */
export function addPolynomials(a: TestPolynomial, b: TestPolynomial): TestPolynomial {
  if (a.degree !== b.degree || a.modulus !== b.modulus) {
    throw new Error('Polynomials must have same degree and modulus');
  }

  const coeffs = a.coeffs.map((val, idx) => {
    const bCoeff = b.coeffs[idx];
    if (bCoeff === undefined) return val;
    return (val + bCoeff) % a.modulus;
  });

  return {
    coeffs,
    modulus: a.modulus,
    degree: a.degree,
    isNTT: a.isNTT ?? false,
  };
}

/**
 * Helper to negate a polynomial (coefficient-wise modular negation)
 */
export function negatePolynomial(p: TestPolynomial): TestPolynomial {
  const coeffs = p.coeffs.map((val) => (val === 0n ? 0n : p.modulus - val));

  return {
    coeffs,
    modulus: p.modulus,
    degree: p.degree,
    isNTT: p.isNTT ?? false,
  };
}

/**
 * Assertion helper for polynomial equality with detailed error message
 */
export function assertPolynomialsEqual(
  actual: TestPolynomial,
  expected: TestPolynomial,
  message?: string
): void {
  if (!polynomialsEqual(actual, expected)) {
    const prefix = message ? `${message}: ` : '';
    const details = `
${prefix}Polynomials not equal
  Degree: actual=${actual.degree}, expected=${expected.degree}
  Modulus: actual=${actual.modulus}, expected=${expected.modulus}
  IsNTT: actual=${actual.isNTT}, expected=${expected.isNTT}
  First differing coefficient at index: ${actual.coeffs.findIndex((val, idx) => val !== expected.coeffs[idx])}
    `;
    throw new Error(details.trim());
  }
}

/**
 * Assertion helper for approximate polynomial equality
 */
export function assertPolynomialsApproxEqual(
  actual: TestPolynomial,
  expected: TestPolynomial,
  maxCoeffDiff: bigint = 1n,
  message?: string
): void {
  if (!polynomialsApproxEqual(actual, expected, maxCoeffDiff)) {
    const prefix = message ? `${message}: ` : '';
    const firstDiff = actual.coeffs.findIndex((val, idx) => {
      const expCoeff = expected.coeffs[idx];
      if (expCoeff === undefined) return true;
      const diff = val > expCoeff ? val - expCoeff : expCoeff - val;
      const normalizedDiff = diff % actual.modulus;
      return normalizedDiff > maxCoeffDiff && normalizedDiff < actual.modulus - maxCoeffDiff;
    });
    const actualCoeff = actual.coeffs[firstDiff];
    const expectedCoeff = expected.coeffs[firstDiff];
    const diffVal = actualCoeff !== undefined && expectedCoeff !== undefined
      ? (actualCoeff > expectedCoeff ? actualCoeff - expectedCoeff : expectedCoeff - actualCoeff)
      : 0n;
    const details = `
${prefix}Polynomials not approximately equal (tolerance=${maxCoeffDiff})
  Degree: actual=${actual.degree}, expected=${expected.degree}
  Modulus: actual=${actual.modulus}, expected=${expected.modulus}
  First differing coefficient at index ${firstDiff}:
    actual=${actualCoeff}
    expected=${expectedCoeff}
    diff=${diffVal}
    `;
    throw new Error(details.trim());
  }
}
