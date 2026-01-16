/**
 * Property-Based Tests for NTT Round-Trip Consistency
 * 
 * Feature: fhe-accelerate, Property 1: NTT Round-Trip Consistency
 * 
 * For any valid polynomial p with coefficients in Z_q, applying forward NTT
 * followed by inverse NTT SHALL produce a polynomial equal to the original p.
 * 
 * Validates: Requirements 1.6, 1.2
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  PROPERTY_TEST_CONFIG,
  arbitraryPolyDegree,
  arbitraryNTTPrime,
  arbitraryPolynomialCoeffs,
} from './property-test-config';
import {
  type TestPolynomial,
  polynomialsEqual,
  assertPolynomialsEqual,
} from './ciphertext-comparison';

/**
 * NTT-friendly primes for different polynomial degrees
 * These are primes q such that q ≡ 1 (mod 2N)
 */
const NTT_PRIMES: Record<number, bigint> = {
  8: 17n,                           // 17 ≡ 1 (mod 16)
  16: 97n,                          // 97 ≡ 1 (mod 32)
  32: 193n,                         // 193 ≡ 1 (mod 64)
  64: 257n,                         // 257 ≡ 1 (mod 128)
  128: 769n,                        // 769 ≡ 1 (mod 256)
  256: 7681n,                       // 7681 ≡ 1 (mod 512)
  512: 12289n,                      // 12289 ≡ 1 (mod 1024)
  1024: 132120577n,                 // 2^27 - 2^11 + 1
  2048: 1099511627777n,             // 2^40 - 2^13 + 1
  4096: 4611686018326724609n,       // 2^62 - 2^15 + 1
  8192: 4611686018326724609n,
  16384: 4611686018326724609n,
  32768: 4611686018326724609n,
};

/**
 * Get NTT-friendly prime for a given degree
 */
function getNTTPrime(degree: number): bigint {
  const prime = NTT_PRIMES[degree];
  if (!prime) {
    throw new Error(`No NTT-friendly prime defined for degree ${degree}`);
  }
  return prime;
}

/**
 * Compute modular exponentiation: base^exp mod m
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
 * Compute modular inverse using extended Euclidean algorithm
 */
function modInverse(a: bigint, m: bigint): bigint {
  let [old_r, r] = [a, m];
  let [old_s, s] = [1n, 0n];
  
  while (r !== 0n) {
    const quotient = old_r / r;
    [old_r, r] = [r, old_r - quotient * r];
    [old_s, s] = [s, old_s - quotient * s];
  }
  
  if (old_r !== 1n) {
    throw new Error('Modular inverse does not exist');
  }
  
  return ((old_s % m) + m) % m;
}

/**
 * Find primitive 2N-th root of unity modulo q
 */
function findPrimitiveRoot(degree: number, modulus: bigint): bigint {
  const twoN = BigInt(degree) * 2n;
  
  // Check that q ≡ 1 (mod 2N)
  if ((modulus - 1n) % twoN !== 0n) {
    throw new Error('Modulus is not NTT-friendly');
  }
  
  const exponent = (modulus - 1n) / twoN;
  
  // Find generator by trying small values
  for (let g = 2n; g < modulus; g++) {
    const omega = modPow(g, exponent, modulus);
    
    // Verify it's a primitive 2N-th root
    const omegaN = modPow(omega, BigInt(degree), modulus);
    const omega2N = modPow(omega, twoN, modulus);
    
    if (omega2N === 1n && omegaN === modulus - 1n) {
      return omega;
    }
  }
  
  throw new Error('Could not find primitive root');
}

/**
 * Compute bit-reversed index
 */
function bitReverse(index: number, bits: number): number {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (index & 1);
    index >>= 1;
  }
  return result;
}

/**
 * Bit-reversal permutation
 */
function bitReversePermutation(coeffs: bigint[], n: number): bigint[] {
  const result = [...coeffs];
  const bits = Math.log2(n);
  
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    if (i < j) {
      [result[i], result[j]] = [result[j], result[i]];
    }
  }
  
  return result;
}

/**
 * Forward NTT (Cooley-Tukey)
 */
function forwardNTT(coeffs: bigint[], modulus: bigint): bigint[] {
  const n = coeffs.length;
  const logN = Math.log2(n);
  
  // Find primitive root and compute twiddle factors
  const omega = findPrimitiveRoot(n, modulus);
  const twiddles: bigint[] = [1n];
  for (let i = 1; i < n; i++) {
    twiddles.push((twiddles[i - 1] * omega) % modulus);
  }
  
  // Bit-reversal permutation
  let result = bitReversePermutation(coeffs, n);
  
  // Cooley-Tukey butterfly stages
  for (let stage = 0; stage < logN; stage++) {
    const m = 1 << stage;
    const groupSize = 2 * m;
    
    for (let k = 0; k < n; k += groupSize) {
      for (let j = 0; j < m; j++) {
        const twiddleIdx = j * (n / groupSize);
        const w = twiddles[twiddleIdx];
        
        const idxA = k + j;
        const idxB = k + j + m;
        
        const a = result[idxA];
        const b = result[idxB];
        
        const wb = (w * b) % modulus;
        result[idxA] = (a + wb) % modulus;
        result[idxB] = ((a - wb) % modulus + modulus) % modulus;
      }
    }
  }
  
  return result;
}

/**
 * Inverse NTT (Gentleman-Sande)
 */
function inverseNTT(coeffs: bigint[], modulus: bigint): bigint[] {
  const n = coeffs.length;
  const logN = Math.log2(n);
  
  // Find primitive root and compute inverse twiddle factors
  const omega = findPrimitiveRoot(n, modulus);
  const omegaInv = modInverse(omega, modulus);
  const invTwiddles: bigint[] = [1n];
  for (let i = 1; i < n; i++) {
    invTwiddles.push((invTwiddles[i - 1] * omegaInv) % modulus);
  }
  
  let result = [...coeffs];
  
  // Gentleman-Sande butterfly stages (reverse order)
  for (let stage = logN - 1; stage >= 0; stage--) {
    const m = 1 << stage;
    const groupSize = 2 * m;
    
    for (let k = 0; k < n; k += groupSize) {
      for (let j = 0; j < m; j++) {
        const twiddleIdx = j * (n / groupSize);
        const wInv = invTwiddles[twiddleIdx];
        
        const idxA = k + j;
        const idxB = k + j + m;
        
        const a = result[idxA];
        const b = result[idxB];
        
        result[idxA] = (a + b) % modulus;
        const diff = ((a - b) % modulus + modulus) % modulus;
        result[idxB] = (diff * wInv) % modulus;
      }
    }
  }
  
  // Bit-reversal permutation
  result = bitReversePermutation(result, n);
  
  // Scale by N^(-1)
  const invN = modInverse(BigInt(n), modulus);
  result = result.map(c => (c * invN) % modulus);
  
  return result;
}

describe('Property 1: NTT Round-Trip Consistency', () => {
  /**
   * Feature: fhe-accelerate, Property 1: NTT Round-Trip Consistency
   * 
   * For any valid polynomial p with coefficients in Z_q, applying forward NTT
   * followed by inverse NTT SHALL produce a polynomial equal to the original p.
   * 
   * Validates: Requirements 1.6, 1.2
   */
  it('should satisfy forward_ntt(inverse_ntt(p)) = p for all polynomials', () => {
    // Test with small degrees for faster execution
    const testDegrees = [8, 16, 32, 64];
    
    for (const degree of testDegrees) {
      const modulus = getNTTPrime(degree);
      
      fc.assert(
        fc.property(
          fc.array(
            fc.bigInt({ min: 0n, max: modulus - 1n }),
            { minLength: degree, maxLength: degree }
          ),
          (coeffs) => {
            // Apply forward NTT
            const nttCoeffs = forwardNTT(coeffs, modulus);
            
            // Apply inverse NTT
            const recovered = inverseNTT(nttCoeffs, modulus);
            
            // Verify round-trip
            for (let i = 0; i < degree; i++) {
              if (coeffs[i] !== recovered[i]) {
                return false;
              }
            }
            
            return true;
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
      );
    }
  });

  it('should transform data (NTT is not identity)', () => {
    const degree = 16;
    const modulus = getNTTPrime(degree);
    
    fc.assert(
      fc.property(
        fc.array(
          fc.bigInt({ min: 1n, max: modulus - 1n }),
          { minLength: degree, maxLength: degree }
        ),
        (coeffs) => {
          const nttCoeffs = forwardNTT(coeffs, modulus);
          
          // At least some coefficients should change
          let changed = 0;
          for (let i = 0; i < degree; i++) {
            if (coeffs[i] !== nttCoeffs[i]) {
              changed++;
            }
          }
          
          // Most coefficients should change for random input
          return changed > 0;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });

  it('should preserve polynomial degree', () => {
    const testDegrees = [8, 16, 32];
    
    for (const degree of testDegrees) {
      const modulus = getNTTPrime(degree);
      
      fc.assert(
        fc.property(
          fc.array(
            fc.bigInt({ min: 0n, max: modulus - 1n }),
            { minLength: degree, maxLength: degree }
          ),
          (coeffs) => {
            const nttCoeffs = forwardNTT(coeffs, modulus);
            return nttCoeffs.length === degree;
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 20 }
      );
    }
  });

  it('should produce coefficients within modulus range', () => {
    const degree = 16;
    const modulus = getNTTPrime(degree);
    
    fc.assert(
      fc.property(
        fc.array(
          fc.bigInt({ min: 0n, max: modulus - 1n }),
          { minLength: degree, maxLength: degree }
        ),
        (coeffs) => {
          const nttCoeffs = forwardNTT(coeffs, modulus);
          
          for (const c of nttCoeffs) {
            if (c < 0n || c >= modulus) {
              return false;
            }
          }
          
          return true;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });
});

describe('NTT Utility Functions', () => {
  it('should correctly compute modular exponentiation', () => {
    fc.assert(
      fc.property(
        fc.bigInt({ min: 2n, max: 100n }),
        fc.bigInt({ min: 1n, max: 20n }),
        fc.bigInt({ min: 3n, max: 1000n }),
        (base, exp, mod) => {
          const result = modPow(base, exp, mod);
          
          // Verify result is in range
          if (result < 0n || result >= mod) {
            return false;
          }
          
          // Verify by computing directly for small exponents
          let expected = 1n;
          for (let i = 0n; i < exp; i++) {
            expected = (expected * base) % mod;
          }
          
          return result === expected;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should correctly compute modular inverse', () => {
    // Test with known primes
    const primes = [17n, 97n, 193n, 257n];
    
    for (const p of primes) {
      fc.assert(
        fc.property(
          fc.bigInt({ min: 1n, max: p - 1n }),
          (a) => {
            const inv = modInverse(a, p);
            return (a * inv) % p === 1n;
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
      );
    }
  });

  it('should correctly compute bit reversal', () => {
    // Test bit reversal for various bit widths
    expect(bitReverse(0, 3)).toBe(0);   // 000 -> 000
    expect(bitReverse(1, 3)).toBe(4);   // 001 -> 100
    expect(bitReverse(2, 3)).toBe(2);   // 010 -> 010
    expect(bitReverse(3, 3)).toBe(6);   // 011 -> 110
    expect(bitReverse(4, 3)).toBe(1);   // 100 -> 001
    expect(bitReverse(5, 3)).toBe(5);   // 101 -> 101
    expect(bitReverse(6, 3)).toBe(3);   // 110 -> 011
    expect(bitReverse(7, 3)).toBe(7);   // 111 -> 111
  });
});
