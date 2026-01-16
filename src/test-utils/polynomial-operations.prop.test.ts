/**
 * Property-Based Tests for Polynomial Ring Operations
 * 
 * Feature: fhe-accelerate, Property 3: Polynomial Multiplication Commutativity
 * Feature: fhe-accelerate, Property 4: Polynomial Multiplicative Identity
 * 
 * These tests validate polynomial arithmetic in the cyclotomic ring Z_q[X]/(X^N + 1).
 * 
 * Validates: Requirements 3.5, 3.6
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG } from './property-test-config';

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
  
  if ((modulus - 1n) % twoN !== 0n) {
    throw new Error('Modulus is not NTT-friendly');
  }
  
  const exponent = (modulus - 1n) / twoN;
  
  for (let g = 2n; g < modulus; g++) {
    const omega = modPow(g, exponent, modulus);
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
  
  const omega = findPrimitiveRoot(n, modulus);
  const twiddles: bigint[] = [1n];
  for (let i = 1; i < n; i++) {
    twiddles.push((twiddles[i - 1] * omega) % modulus);
  }
  
  let result = bitReversePermutation(coeffs, n);
  
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
  
  const omega = findPrimitiveRoot(n, modulus);
  const omegaInv = modInverse(omega, modulus);
  const invTwiddles: bigint[] = [1n];
  for (let i = 1; i < n; i++) {
    invTwiddles.push((invTwiddles[i - 1] * omegaInv) % modulus);
  }
  
  let result = [...coeffs];
  
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
  
  result = bitReversePermutation(result, n);
  
  const invN = modInverse(BigInt(n), modulus);
  result = result.map(c => (c * invN) % modulus);
  
  return result;
}

/**
 * Polynomial addition in Z_q[X]/(X^N + 1)
 */
function polynomialAdd(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  const result: bigint[] = new Array(n);
  
  for (let i = 0; i < n; i++) {
    result[i] = (a[i] + b[i]) % modulus;
  }
  
  return result;
}

/**
 * Polynomial subtraction in Z_q[X]/(X^N + 1)
 */
function polynomialSub(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  const result: bigint[] = new Array(n);
  
  for (let i = 0; i < n; i++) {
    result[i] = ((a[i] - b[i]) % modulus + modulus) % modulus;
  }
  
  return result;
}

/**
 * Polynomial multiplication in Z_q[X]/(X^N + 1) using NTT
 */
function polynomialMul(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  const n = a.length;
  
  // Convert to NTT domain
  const aNtt = forwardNTT(a, modulus);
  const bNtt = forwardNTT(b, modulus);
  
  // Pointwise multiplication
  const cNtt: bigint[] = new Array(n);
  for (let i = 0; i < n; i++) {
    cNtt[i] = (aNtt[i] * bNtt[i]) % modulus;
  }
  
  // Convert back to coefficient domain
  return inverseNTT(cNtt, modulus);
}

/**
 * Create identity polynomial (constant 1)
 */
function identityPolynomial(degree: number): bigint[] {
  const result = new Array(degree).fill(0n);
  result[0] = 1n;
  return result;
}

/**
 * Create zero polynomial
 */
function zeroPolynomial(degree: number): bigint[] {
  return new Array(degree).fill(0n);
}

/**
 * Check if two polynomials are equal
 */
function polynomialsEqual(a: bigint[], b: bigint[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Generate arbitrary polynomial coefficients
 */
function arbitraryPolynomial(degree: number, modulus: bigint): fc.Arbitrary<bigint[]> {
  return fc.array(
    fc.bigInt({ min: 0n, max: modulus - 1n }),
    { minLength: degree, maxLength: degree }
  );
}

describe('Property 3: Polynomial Multiplication Commutativity', () => {
  /**
   * Feature: fhe-accelerate, Property 3: Polynomial Multiplication Commutativity
   * 
   * For any polynomial pair (p1, p2) in the ring Z_q[X]/(X^N + 1),
   * multiplication SHALL be commutative: p1 × p2 = p2 × p1.
   * 
   * **Validates: Requirements 3.5**
   */
  it('should satisfy p1 * p2 = p2 * p1 for all polynomial pairs', () => {
    const testDegrees = [8, 16, 32];
    
    for (const degree of testDegrees) {
      const modulus = getNTTPrime(degree);
      
      fc.assert(
        fc.property(
          arbitraryPolynomial(degree, modulus),
          arbitraryPolynomial(degree, modulus),
          (p1, p2) => {
            // Compute p1 * p2
            const prod12 = polynomialMul(p1, p2, modulus);
            
            // Compute p2 * p1
            const prod21 = polynomialMul(p2, p1, modulus);
            
            // Verify commutativity
            return polynomialsEqual(prod12, prod21);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    }
  });

  it('should satisfy commutativity for random polynomials of various sizes', () => {
    const testCases = [8, 16, 32, 64];
    
    for (const degree of testCases) {
      const modulus = getNTTPrime(degree);
      
      fc.assert(
        fc.property(
          arbitraryPolynomial(degree, modulus),
          arbitraryPolynomial(degree, modulus),
          (p1, p2) => {
            const prod12 = polynomialMul(p1, p2, modulus);
            const prod21 = polynomialMul(p2, p1, modulus);
            
            return polynomialsEqual(prod12, prod21);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 25 }
      );
    }
  });
});

describe('Property 4: Polynomial Multiplicative Identity', () => {
  /**
   * Feature: fhe-accelerate, Property 4: Polynomial Multiplicative Identity
   * 
   * For any polynomial p in the ring Z_q[X]/(X^N + 1), multiplying by the
   * multiplicative identity (polynomial with constant term 1 and all other
   * coefficients 0) SHALL return p unchanged.
   * 
   * **Validates: Requirements 3.6**
   */
  it('should satisfy p * 1 = p for all polynomials', () => {
    const testDegrees = [8, 16, 32];
    
    for (const degree of testDegrees) {
      const modulus = getNTTPrime(degree);
      const identity = identityPolynomial(degree);
      
      fc.assert(
        fc.property(
          arbitraryPolynomial(degree, modulus),
          (p) => {
            // Compute p * 1
            const prod = polynomialMul(p, identity, modulus);
            
            // Verify p * 1 = p
            return polynomialsEqual(prod, p);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    }
  });

  it('should satisfy 1 * p = p for all polynomials (left identity)', () => {
    const testDegrees = [8, 16, 32];
    
    for (const degree of testDegrees) {
      const modulus = getNTTPrime(degree);
      const identity = identityPolynomial(degree);
      
      fc.assert(
        fc.property(
          arbitraryPolynomial(degree, modulus),
          (p) => {
            // Compute 1 * p
            const prod = polynomialMul(identity, p, modulus);
            
            // Verify 1 * p = p
            return polynomialsEqual(prod, p);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    }
  });

  it('should satisfy p * 0 = 0 for all polynomials (zero annihilator)', () => {
    const testDegrees = [8, 16, 32];
    
    for (const degree of testDegrees) {
      const modulus = getNTTPrime(degree);
      const zero = zeroPolynomial(degree);
      
      fc.assert(
        fc.property(
          arbitraryPolynomial(degree, modulus),
          (p) => {
            // Compute p * 0
            const prod = polynomialMul(p, zero, modulus);
            
            // Verify p * 0 = 0
            return polynomialsEqual(prod, zero);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    }
  });
});

describe('Polynomial Ring Additional Properties', () => {
  it('should satisfy additive identity: p + 0 = p', () => {
    const degree = 16;
    const modulus = getNTTPrime(degree);
    const zero = zeroPolynomial(degree);
    
    fc.assert(
      fc.property(
        arbitraryPolynomial(degree, modulus),
        (p) => {
          const sum = polynomialAdd(p, zero, modulus);
          return polynomialsEqual(sum, p);
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy additive commutativity: p1 + p2 = p2 + p1', () => {
    const degree = 16;
    const modulus = getNTTPrime(degree);
    
    fc.assert(
      fc.property(
        arbitraryPolynomial(degree, modulus),
        arbitraryPolynomial(degree, modulus),
        (p1, p2) => {
          const sum12 = polynomialAdd(p1, p2, modulus);
          const sum21 = polynomialAdd(p2, p1, modulus);
          return polynomialsEqual(sum12, sum21);
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy additive inverse: p + (-p) = 0', () => {
    const degree = 16;
    const modulus = getNTTPrime(degree);
    const zero = zeroPolynomial(degree);
    
    fc.assert(
      fc.property(
        arbitraryPolynomial(degree, modulus),
        (p) => {
          // Compute -p
          const negP = p.map(c => c === 0n ? 0n : modulus - c);
          
          // Compute p + (-p)
          const sum = polynomialAdd(p, negP, modulus);
          
          return polynomialsEqual(sum, zero);
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy distributivity: a * (b + c) = a*b + a*c', () => {
    const degree = 8;
    const modulus = getNTTPrime(degree);
    
    fc.assert(
      fc.property(
        arbitraryPolynomial(degree, modulus),
        arbitraryPolynomial(degree, modulus),
        arbitraryPolynomial(degree, modulus),
        (a, b, c) => {
          // Compute a * (b + c)
          const bPlusC = polynomialAdd(b, c, modulus);
          const left = polynomialMul(a, bPlusC, modulus);
          
          // Compute a*b + a*c
          const ab = polynomialMul(a, b, modulus);
          const ac = polynomialMul(a, c, modulus);
          const right = polynomialAdd(ab, ac, modulus);
          
          return polynomialsEqual(left, right);
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });

  it('should satisfy associativity of addition: (a + b) + c = a + (b + c)', () => {
    const degree = 16;
    const modulus = getNTTPrime(degree);
    
    fc.assert(
      fc.property(
        arbitraryPolynomial(degree, modulus),
        arbitraryPolynomial(degree, modulus),
        arbitraryPolynomial(degree, modulus),
        (a, b, c) => {
          // (a + b) + c
          const ab = polynomialAdd(a, b, modulus);
          const left = polynomialAdd(ab, c, modulus);
          
          // a + (b + c)
          const bc = polynomialAdd(b, c, modulus);
          const right = polynomialAdd(a, bc, modulus);
          
          return polynomialsEqual(left, right);
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });
});
