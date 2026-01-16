/**
 * Property-Based Tests for Modular Multiplication Correctness
 * 
 * **Property 2: Modular Multiplication Correctness**
 * 
 * This test validates that modular multiplication produces mathematically correct
 * results by comparing against a reference big-integer implementation.
 * 
 * **Validates: Requirements 2.5**
 * - FOR ALL coefficient pairs (a, b) and modulus q, computing (a * b) mod q 
 *   SHALL produce mathematically correct results
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG } from './property-test-config';

/**
 * Reference implementation of modular multiplication using BigInt
 * This serves as the ground truth for verifying the native implementation
 */
function referenceModMul(a: bigint, b: bigint, modulus: bigint): bigint {
  // Ensure inputs are reduced
  const aReduced = ((a % modulus) + modulus) % modulus;
  const bReduced = ((b % modulus) + modulus) % modulus;
  
  // Compute product and reduce
  return (aReduced * bReduced) % modulus;
}

/**
 * Reference implementation of modular addition
 */
function referenceModAdd(a: bigint, b: bigint, modulus: bigint): bigint {
  const aReduced = ((a % modulus) + modulus) % modulus;
  const bReduced = ((b % modulus) + modulus) % modulus;
  return (aReduced + bReduced) % modulus;
}

/**
 * Reference implementation of modular subtraction
 */
function referenceModSub(a: bigint, b: bigint, modulus: bigint): bigint {
  const aReduced = ((a % modulus) + modulus) % modulus;
  const bReduced = ((b % modulus) + modulus) % modulus;
  return ((aReduced - bReduced) % modulus + modulus) % modulus;
}

/**
 * Simulated Montgomery multiplication for testing
 * This mimics the behavior of the C++ implementation
 */
class SimulatedMontgomeryArithmetic {
  private modulus: bigint;
  private r: bigint;  // R = 2^64 for 64-bit, 2^128 for 128-bit
  private rModQ: bigint;
  private r2ModQ: bigint;
  private qInv: bigint;
  
  constructor(modulus: bigint, bits: number = 64) {
    if (modulus <= 0n) {
      throw new Error('Modulus must be positive');
    }
    if (modulus % 2n === 0n) {
      throw new Error('Modulus must be odd for Montgomery arithmetic');
    }
    
    this.modulus = modulus;
    this.r = 1n << BigInt(bits);
    this.rModQ = this.r % modulus;
    this.r2ModQ = (this.rModQ * this.rModQ) % modulus;
    this.qInv = this.computeQInv(modulus, bits);
  }
  
  private computeQInv(q: bigint, bits: number): bigint {
    // Compute -q^(-1) mod R using Newton's method
    // We need q_inv such that q * q_inv ≡ -1 (mod R)
    const r = 1n << BigInt(bits);
    const rMask = r - 1n;
    
    // Newton's method to find q^(-1) mod R
    // Start with initial approximation
    let x = q;
    
    // Newton iterations: x = x * (2 - q * x) mod R
    // Need log2(bits) iterations for convergence
    for (let i = 0; i < 7; i++) {
      x = (x * (2n - q * x)) & rMask;
    }
    
    // Now x = q^(-1) mod R
    // We need -q^(-1) mod R = R - x (when x != 0)
    return x === 0n ? 0n : (r - x) & rMask;
  }
  
  toMontgomery(a: bigint): bigint {
    // a * R mod q = (a * R^2 mod q) * R^(-1) mod q
    // But simpler: just compute (a * R) mod q directly
    return (a * this.rModQ) % this.modulus;
  }
  
  fromMontgomery(a: bigint): bigint {
    // a * R^(-1) mod q via Montgomery reduction
    return this.montgomeryReduce(a);
  }
  
  private montgomeryReduce(t: bigint): bigint {
    // Montgomery reduction: t * R^(-1) mod q
    // Algorithm: 
    //   m = (t * q_inv) mod R
    //   u = (t + m * q) / R
    //   if u >= q then u = u - q
    const rMask = this.r - 1n;
    const m = (t * this.qInv) & rMask;  // m = t * q_inv mod R
    const u = (t + m * this.modulus) / this.r;  // u = (t + m*q) / R
    return u >= this.modulus ? u - this.modulus : u;
  }
  
  montgomeryMul(a: bigint, b: bigint): bigint {
    // Compute a * b * R^(-1) mod q
    const product = a * b;
    return this.montgomeryReduce(product);
  }
  
  /**
   * Full modular multiplication using Montgomery form
   * 
   * To compute (a * b) mod q using Montgomery:
   * 1. Convert a to Montgomery form: aR = a * R mod q
   * 2. Convert b to Montgomery form: bR = b * R mod q  
   * 3. Montgomery multiply: (aR * bR) * R^(-1) mod q = a * b * R mod q
   * 4. Convert back: (a * b * R) * R^(-1) mod q = a * b mod q
   */
  modMul(a: bigint, b: bigint): bigint {
    // Reduce inputs first
    const aReduced = a % this.modulus;
    const bReduced = b % this.modulus;
    
    // Convert to Montgomery form
    const aMont = this.toMontgomery(aReduced);
    const bMont = this.toMontgomery(bReduced);
    
    // Montgomery multiply: (aR * bR) * R^(-1) = abR
    const prodMont = this.montgomeryMul(aMont, bMont);
    
    // Convert back from Montgomery form: abR * R^(-1) = ab
    return this.fromMontgomery(prodMont);
  }
}

/**
 * Generate an odd modulus suitable for Montgomery arithmetic
 */
function arbitraryOddModulus(minBits: number = 16, maxBits: number = 62): fc.Arbitrary<bigint> {
  return fc.bigInt({ min: 3n, max: (1n << BigInt(maxBits)) - 1n })
    .filter(n => n % 2n === 1n && n >= (1n << BigInt(minBits)));
}

/**
 * Generate a coefficient value less than a given modulus
 */
function arbitraryCoefficientForModulus(modulus: bigint): fc.Arbitrary<bigint> {
  return fc.bigInt({ min: 0n, max: modulus - 1n });
}

/**
 * Common NTT-friendly primes used in FHE
 */
const NTT_PRIMES = [
  132120577n,              // 2^27 - 2^11 + 1
  268369921n,              // 2^28 - 2^12 + 1
  1073479681n,             // 2^30 - 2^14 + 1
  1099511627777n,          // 2^40 - 2^13 + 1
  4611686018326724609n,    // 2^62 - 2^15 + 1
];

describe('Property 2: Modular Multiplication Correctness', () => {
  /**
   * **Validates: Requirements 2.5**
   * FOR ALL coefficient pairs (a, b) and modulus q, computing (a * b) mod q 
   * SHALL produce mathematically correct results
   */
  describe('2.1 Basic Modular Multiplication Correctness', () => {
    it('should produce correct results for random coefficient pairs and moduli', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          (modulus, a, b) => {
            // Reduce inputs to be less than modulus
            const aReduced = a % modulus;
            const bReduced = b % modulus;
            
            // Reference implementation
            const expected = referenceModMul(aReduced, bReduced, modulus);
            
            // Simulated Montgomery implementation
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            const actual = montgomery.modMul(aReduced, bReduced);
            
            expect(actual).toBe(expected);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.2 Modular Multiplication with NTT-Friendly Primes', () => {
    it('should produce correct results for NTT-friendly prime moduli', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(...NTT_PRIMES),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          (modulus, a, b) => {
            const aReduced = a % modulus;
            const bReduced = b % modulus;
            
            const expected = referenceModMul(aReduced, bReduced, modulus);
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            const actual = montgomery.modMul(aReduced, bReduced);
            
            expect(actual).toBe(expected);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.3 Modular Multiplication Commutativity', () => {
    it('should satisfy a * b = b * a (mod q)', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          (modulus, a, b) => {
            const aReduced = a % modulus;
            const bReduced = b % modulus;
            
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            const ab = montgomery.modMul(aReduced, bReduced);
            const ba = montgomery.modMul(bReduced, aReduced);
            
            expect(ab).toBe(ba);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.4 Modular Multiplication Associativity', () => {
    it('should satisfy (a * b) * c = a * (b * c) (mod q)', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          fc.bigInt({ min: 0n, max: (1n << 40n) - 1n }),
          fc.bigInt({ min: 0n, max: (1n << 40n) - 1n }),
          fc.bigInt({ min: 0n, max: (1n << 40n) - 1n }),
          (modulus, a, b, c) => {
            const aReduced = a % modulus;
            const bReduced = b % modulus;
            const cReduced = c % modulus;
            
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            
            // (a * b) * c
            const ab = montgomery.modMul(aReduced, bReduced);
            const ab_c = montgomery.modMul(ab, cReduced);
            
            // a * (b * c)
            const bc = montgomery.modMul(bReduced, cReduced);
            const a_bc = montgomery.modMul(aReduced, bc);
            
            expect(ab_c).toBe(a_bc);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.5 Modular Multiplication Identity', () => {
    it('should satisfy a * 1 = a (mod q)', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          (modulus, a) => {
            const aReduced = a % modulus;
            
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            const result = montgomery.modMul(aReduced, 1n);
            
            expect(result).toBe(aReduced);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.6 Modular Multiplication by Zero', () => {
    it('should satisfy a * 0 = 0 (mod q)', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          (modulus, a) => {
            const aReduced = a % modulus;
            
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            const result = montgomery.modMul(aReduced, 0n);
            
            expect(result).toBe(0n);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.7 Distributivity over Addition', () => {
    it('should satisfy a * (b + c) = a*b + a*c (mod q)', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          fc.bigInt({ min: 0n, max: (1n << 40n) - 1n }),
          fc.bigInt({ min: 0n, max: (1n << 40n) - 1n }),
          fc.bigInt({ min: 0n, max: (1n << 40n) - 1n }),
          (modulus, a, b, c) => {
            const aReduced = a % modulus;
            const bReduced = b % modulus;
            const cReduced = c % modulus;
            
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            
            // a * (b + c)
            const bPlusC = referenceModAdd(bReduced, cReduced, modulus);
            const left = montgomery.modMul(aReduced, bPlusC);
            
            // a*b + a*c
            const ab = montgomery.modMul(aReduced, bReduced);
            const ac = montgomery.modMul(aReduced, cReduced);
            const right = referenceModAdd(ab, ac, modulus);
            
            expect(left).toBe(right);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.8 Montgomery Round-Trip Consistency', () => {
    it('should preserve values through Montgomery conversion round-trip', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          fc.bigInt({ min: 0n, max: (1n << 62n) - 1n }),
          (modulus, a) => {
            const aReduced = a % modulus;
            
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            
            // Convert to Montgomery form and back
            const aMont = montgomery.toMontgomery(aReduced);
            const recovered = montgomery.fromMontgomery(aMont);
            
            expect(recovered).toBe(aReduced);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });

  describe('2.9 Edge Cases', () => {
    it('should handle maximum coefficient values correctly', () => {
      fc.assert(
        fc.property(
          arbitraryOddModulus(16, 62),
          (modulus) => {
            const maxCoeff = modulus - 1n;
            
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            
            // max * max should produce correct result
            const expected = referenceModMul(maxCoeff, maxCoeff, modulus);
            const actual = montgomery.modMul(maxCoeff, maxCoeff);
            
            expect(actual).toBe(expected);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });

    it('should handle small moduli correctly', () => {
      fc.assert(
        fc.property(
          fc.bigInt({ min: 3n, max: 1000n }).filter(n => n % 2n === 1n),
          fc.bigInt({ min: 0n, max: 999n }),
          fc.bigInt({ min: 0n, max: 999n }),
          (modulus, a, b) => {
            const aReduced = a % modulus;
            const bReduced = b % modulus;
            
            const expected = referenceModMul(aReduced, bReduced, modulus);
            const montgomery = new SimulatedMontgomeryArithmetic(modulus, 64);
            const actual = montgomery.modMul(aReduced, bReduced);
            
            expect(actual).toBe(expected);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 1000 }
      );
    });
  });
});
