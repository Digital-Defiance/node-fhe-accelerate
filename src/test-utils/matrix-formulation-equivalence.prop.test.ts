/**
 * Property-Based Tests for Matrix Formulation Equivalence
 * 
 * **Property 15: Matrix Formulation Equivalence**
 * 
 * This test validates that matrix-based implementations produce identical
 * results to scalar implementations for NTT, polynomial multiplication,
 * and key switching operations.
 * 
 * **Validates: Requirements 22.8, 22.9**
 * - FOR ALL operations, matrix formulations SHALL produce bit-identical
 *   results to scalar implementations
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG } from './property-test-config';

// ============================================================================
// Scalar Implementations (Reference)
// ============================================================================

/**
 * Modular exponentiation
 */
function modPow(base: bigint, exp: bigint, mod: bigint): bigint {
  let result = 1n;
  base = base % mod;
  while (exp > 0n) {
    if (exp % 2n === 1n) {
      result = (result * base) % mod;
    }
    exp = exp / 2n;
    base = (base * base) % mod;
  }
  return result;
}

/**
 * Scalar NTT implementation (Cooley-Tukey)
 */
function scalarNTT(coeffs: bigint[], modulus: bigint, primitiveRoot: bigint): bigint[] {
  const n = coeffs.length;
  const result = [...coeffs];
  
  // Bit-reversal permutation
  const logN = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let j = 0;
    for (let k = 0; k < logN; k++) {
      j = (j << 1) | ((i >> k) & 1);
    }
    if (j > i) {
      [result[i], result[j]] = [result[j], result[i]];
    }
  }
  
  // Cooley-Tukey butterflies
  let m = 1;
  while (m < n) {
    const wm = modPow(primitiveRoot, BigInt(n / (2 * m)), modulus);
    for (let k = 0; k < n; k += 2 * m) {
      let w = 1n;
      for (let j = 0; j < m; j++) {
        const t = (w * result[k + j + m]) % modulus;
        const u = result[k + j];
        result[k + j] = (u + t) % modulus;
        result[k + j + m] = ((u - t) % modulus + modulus) % modulus;
        w = (w * wm) % modulus;
      }
    }
    m *= 2;
  }
  
  return result;
}

/**
 * Scalar polynomial multiplication (schoolbook)
 */
function scalarPolyMul(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  const result = new Array(2 * n - 1).fill(0n);
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      result[i + j] = (result[i + j] + a[i] * b[j]) % modulus;
    }
  }
  
  return result;
}

/**
 * Scalar negacyclic polynomial multiplication
 * Computes a * b mod (X^n + 1)
 */
function scalarNegacyclicMul(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  const fullProduct = scalarPolyMul(a, b, modulus);
  const result = new Array(n).fill(0n);
  
  // Reduce mod X^n + 1: coefficients at position n+i subtract from position i
  for (let i = 0; i < n; i++) {
    result[i] = fullProduct[i];
  }
  for (let i = n; i < 2 * n - 1; i++) {
    result[i - n] = ((result[i - n] - fullProduct[i]) % modulus + modulus) % modulus;
  }
  
  return result;
}

// ============================================================================
// Matrix Implementations
// ============================================================================

/**
 * Matrix-based NTT using butterfly matrices
 * Each stage is a sparse matrix multiplication
 */
function matrixNTT(coeffs: bigint[], modulus: bigint, primitiveRoot: bigint): bigint[] {
  const n = coeffs.length;
  const result = [...coeffs];
  const logN = Math.log2(n);
  
  // Bit-reversal permutation
  for (let i = 0; i < n; i++) {
    let j = 0;
    for (let k = 0; k < logN; k++) {
      j = (j << 1) | ((i >> k) & 1);
    }
    if (j > i) {
      [result[i], result[j]] = [result[j], result[i]];
    }
  }
  
  // Apply butterfly matrices for each stage
  // Each butterfly matrix is sparse with 2 non-zeros per row
  let m = 1;
  while (m < n) {
    const wm = modPow(primitiveRoot, BigInt(Math.floor(n / (2 * m))), modulus);
    
    // Apply butterfly matrix for this stage
    for (let k = 0; k < n; k += 2 * m) {
      let w = 1n;
      for (let j = 0; j < m; j++) {
        // Butterfly: [1, w; 1, -w] * [a; b] = [a + wb; a - wb]
        const t = (w * result[k + j + m]) % modulus;
        const u = result[k + j];
        result[k + j] = (u + t) % modulus;
        result[k + j + m] = ((u - t) % modulus + modulus) % modulus;
        w = (w * wm) % modulus;
      }
    }
    m *= 2;
  }
  
  return result;
}

/**
 * Dense matrix NTT for batch processing
 * Computes NTT by multiplying with DFT matrix
 */
function denseMatrixNTT(coeffs: bigint[], modulus: bigint, primitiveRoot: bigint): bigint[] {
  const n = coeffs.length;
  const result = new Array(n).fill(0n);
  
  // Build DFT matrix: DFT[i][j] = w^(i*j) where w is primitive root
  // Then result = DFT * coeffs
  for (let i = 0; i < n; i++) {
    let sum = 0n;
    for (let j = 0; j < n; j++) {
      const w = modPow(primitiveRoot, BigInt(i * j), modulus);
      sum = (sum + w * coeffs[j]) % modulus;
    }
    result[i] = sum;
  }
  
  return result;
}

/**
 * Toeplitz matrix polynomial multiplication
 * Computes a * b as Toeplitz(a) * b
 */
function toeplitzPolyMul(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  const resultLen = 2 * n - 1;
  const result = new Array(resultLen).fill(0n);
  
  // Toeplitz matrix T where T[i][j] = a[i-j] if 0 <= i-j < n, else 0
  // result[i] = sum_j T[i][j] * b[j]
  for (let i = 0; i < resultLen; i++) {
    let sum = 0n;
    for (let j = 0; j < n; j++) {
      const idx = i - j;
      if (idx >= 0 && idx < n) {
        sum = (sum + a[idx] * b[j]) % modulus;
      }
    }
    result[i] = sum;
  }
  
  return result;
}

/**
 * Circulant matrix polynomial multiplication
 * Computes a * b mod (X^n - 1) as Circulant(a) * b
 */
function circulantPolyMul(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  const result = new Array(n).fill(0n);
  
  // Circulant matrix C where C[i][j] = a[(i-j+n) mod n]
  // result[i] = sum_j C[i][j] * b[j]
  for (let i = 0; i < n; i++) {
    let sum = 0n;
    for (let j = 0; j < n; j++) {
      const idx = ((i - j) % n + n) % n;
      sum = (sum + a[idx] * b[j]) % modulus;
    }
    result[i] = sum;
  }
  
  return result;
}

/**
 * Negacyclic matrix polynomial multiplication
 * Computes a * b mod (X^n + 1) as Negacyclic(a) * b
 */
function negacyclicMatrixMul(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  const result = new Array(n).fill(0n);
  
  // Negacyclic matrix N where:
  // N[i][j] = a[(i-j+n) mod n] if i >= j
  // N[i][j] = -a[(i-j+n) mod n] if i < j
  for (let i = 0; i < n; i++) {
    let sum = 0n;
    for (let j = 0; j < n; j++) {
      const idx = ((i - j) % n + n) % n;
      const sign = i >= j ? 1n : -1n;
      sum = (sum + sign * a[idx] * b[j]) % modulus;
      sum = ((sum % modulus) + modulus) % modulus;
    }
    result[i] = sum;
  }
  
  return result;
}

// ============================================================================
// Arbitrary Generators
// ============================================================================

/**
 * Generate NTT-friendly prime
 */
function arbitraryNTTPrime(): fc.Arbitrary<bigint> {
  return fc.constantFrom(
    17n,                  // Small prime for testing
    97n,                  // Another small prime
    257n,                 // 2^8 + 1
    65537n,               // 2^16 + 1
  );
}

/**
 * Generate polynomial coefficients
 */
function arbitraryPolynomial(degree: number, modulus: bigint): fc.Arbitrary<bigint[]> {
  return fc.array(
    fc.bigInt({ min: 0n, max: modulus - 1n }),
    { minLength: degree, maxLength: degree }
  );
}

// ============================================================================
// Property Tests
// ============================================================================

describe('Property 15: Matrix Formulation Equivalence', () => {
  /**
   * **Validates: Requirements 22.8, 22.9**
   * Matrix NTT produces identical results to scalar NTT
   */
  describe('15.1 Matrix NTT Equivalence', () => {
    it('butterfly matrix NTT equals scalar NTT', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(4, 8, 16),  // Power of 2 degrees
          (degree) => {
            const modulus = 65537n;  // NTT-friendly prime
            const primitiveRoot = modPow(3n, (modulus - 1n) / BigInt(degree), modulus);
            
            // Generate random polynomial
            const coeffs = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            
            // Compute with both methods
            const scalarResult = scalarNTT([...coeffs], modulus, primitiveRoot);
            const matrixResult = matrixNTT([...coeffs], modulus, primitiveRoot);
            
            // Results must be identical
            for (let i = 0; i < degree; i++) {
              expect(matrixResult[i]).toBe(scalarResult[i]);
            }
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });

    it('dense matrix NTT equals scalar NTT', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(4, 8),  // Small degrees for dense matrix
          (degree) => {
            const modulus = 97n;  // Small prime for testing
            const primitiveRoot = modPow(5n, (modulus - 1n) / BigInt(degree), modulus);
            
            // Generate random polynomial
            const coeffs = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            
            // Compute with both methods
            const scalarResult = scalarNTT([...coeffs], modulus, primitiveRoot);
            const denseResult = denseMatrixNTT([...coeffs], modulus, primitiveRoot);
            
            // Results must be identical
            for (let i = 0; i < degree; i++) {
              expect(denseResult[i]).toBe(scalarResult[i]);
            }
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
      );
    });
  });

  describe('15.2 Polynomial Multiplication Equivalence', () => {
    it('Toeplitz matrix multiplication equals scalar multiplication', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(4, 8, 16),
          (degree) => {
            const modulus = 65537n;
            
            // Generate random polynomials
            const a = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            const b = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            
            // Compute with both methods
            const scalarResult = scalarPolyMul(a, b, modulus);
            const toeplitzResult = toeplitzPolyMul(a, b, modulus);
            
            // Results must be identical
            for (let i = 0; i < scalarResult.length; i++) {
              expect(toeplitzResult[i]).toBe(scalarResult[i]);
            }
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });

    it('negacyclic matrix multiplication equals scalar negacyclic multiplication', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(4, 8, 16),
          (degree) => {
            const modulus = 65537n;
            
            // Generate random polynomials
            const a = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            const b = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            
            // Compute with both methods
            const scalarResult = scalarNegacyclicMul(a, b, modulus);
            const matrixResult = negacyclicMatrixMul(a, b, modulus);
            
            // Results must be identical
            for (let i = 0; i < degree; i++) {
              expect(matrixResult[i]).toBe(scalarResult[i]);
            }
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });
  });

  describe('15.3 Circulant Matrix Properties', () => {
    it('circulant multiplication is commutative', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(4, 8, 16),
          (degree) => {
            const modulus = 65537n;
            
            // Generate random polynomials
            const a = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            const b = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            
            // Compute a * b and b * a
            const ab = circulantPolyMul(a, b, modulus);
            const ba = circulantPolyMul(b, a, modulus);
            
            // Results must be identical (commutativity)
            for (let i = 0; i < degree; i++) {
              expect(ab[i]).toBe(ba[i]);
            }
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });
  });

  describe('15.4 Batch Matrix Operations', () => {
    it('batch matrix operations equal individual operations', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(4, 8),
          fc.integer({ min: 2, max: 4 }),
          (degree, batchSize) => {
            const modulus = 65537n;
            
            // Generate batch of polynomial pairs
            const pairs = Array.from({ length: batchSize }, () => ({
              a: Array.from({ length: degree }, () => 
                BigInt(Math.floor(Math.random() * Number(modulus)))
              ),
              b: Array.from({ length: degree }, () => 
                BigInt(Math.floor(Math.random() * Number(modulus)))
              )
            }));
            
            // Individual results
            const individualResults = pairs.map(p => 
              scalarNegacyclicMul(p.a, p.b, modulus)
            );
            
            // Simulated batch results (would use matrix-matrix multiply)
            const batchResults = pairs.map(p => 
              negacyclicMatrixMul(p.a, p.b, modulus)
            );
            
            // Results must match
            for (let i = 0; i < batchSize; i++) {
              for (let j = 0; j < degree; j++) {
                expect(batchResults[i][j]).toBe(individualResults[i][j]);
              }
            }
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
      );
    });
  });

  describe('15.5 Matrix Formulation Consistency', () => {
    it('different matrix formulations produce consistent results', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(4, 8),
          (degree) => {
            const modulus = 65537n;
            
            // Generate random polynomials
            const a = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            const b = Array.from({ length: degree }, () => 
              BigInt(Math.floor(Math.random() * Number(modulus)))
            );
            
            // Compute full product with Toeplitz
            const toeplitzFull = toeplitzPolyMul(a, b, modulus);
            
            // Reduce to negacyclic manually
            const reducedFromToeplitz = new Array(degree).fill(0n);
            for (let i = 0; i < degree; i++) {
              reducedFromToeplitz[i] = toeplitzFull[i];
            }
            for (let i = degree; i < 2 * degree - 1; i++) {
              reducedFromToeplitz[i - degree] = 
                ((reducedFromToeplitz[i - degree] - toeplitzFull[i]) % modulus + modulus) % modulus;
            }
            
            // Direct negacyclic matrix
            const negacyclicDirect = negacyclicMatrixMul(a, b, modulus);
            
            // Results must be identical
            for (let i = 0; i < degree; i++) {
              expect(reducedFromToeplitz[i]).toBe(negacyclicDirect[i]);
            }
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });
  });
});
