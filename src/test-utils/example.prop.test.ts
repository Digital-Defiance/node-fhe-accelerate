/**
 * Example Property-Based Tests
 * 
 * This file demonstrates how to write property-based tests using fast-check
 * and the test utilities provided by this library.
 * 
 * These examples will be expanded as the native implementation progresses.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  PROPERTY_TEST_CONFIG,
  arbitraryPolyDegree,
  arbitraryNTTPrime,
  arbitraryPolynomialCoeffs,
  arbitraryPlaintext,
  arbitraryParameterPreset,
  arbitrarySecurityLevel,
} from './property-test-config';
import {
  type TestPolynomial,
  polynomialsEqual,
  addPolynomials,
  negatePolynomial,
  createZeroPolynomial,
  createIdentityPolynomial,
  assertPolynomialsEqual,
} from './ciphertext-comparison';

describe('Property-Based Testing Infrastructure', () => {
  describe('Example Property 1: Polynomial Addition Commutativity', () => {
    // Feature: fhe-accelerate, Example Property: Polynomial Addition Commutativity
    it('should satisfy p1 + p2 = p2 + p1 for all polynomials', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 5 }).chain((idx) => {
            const degree = [1024, 2048, 4096, 8192, 16384, 32768][idx];
            const modulus = [1024, 2048, 4096, 8192, 16384, 32768].includes(degree) 
              ? BigInt('1099511627777') 
              : BigInt('1099511627777');
            return fc.tuple(
              fc.constant(degree),
              fc.constant(modulus),
              arbitraryPolynomialCoeffs(degree, modulus),
              arbitraryPolynomialCoeffs(degree, modulus)
            );
          }),
          ([degree, modulus, coeffs1, coeffs2]) => {
            const p1: TestPolynomial = {
              coeffs: coeffs1,
              modulus,
              degree,
              isNTT: false,
            };

            const p2: TestPolynomial = {
              coeffs: coeffs2,
              modulus,
              degree,
              isNTT: false,
            };

            // Compute p1 + p2
            const sum1 = addPolynomials(p1, p2);

            // Compute p2 + p1
            const sum2 = addPolynomials(p2, p1);

            // Assert commutativity
            expect(polynomialsEqual(sum1, sum2)).toBe(true);
          }
        ),
        PROPERTY_TEST_CONFIG
      );
    });
  });

  describe('Example Property 2: Polynomial Addition Identity', () => {
    // Feature: fhe-accelerate, Example Property: Polynomial Addition Identity
    it('should satisfy p + 0 = p for all polynomials', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 5 }).chain((idx) => {
            const degree = [1024, 2048, 4096, 8192, 16384, 32768][idx];
            const modulus = BigInt('1099511627777');
            return fc.tuple(
              fc.constant(degree),
              fc.constant(modulus),
              arbitraryPolynomialCoeffs(degree, modulus)
            );
          }),
          ([degree, modulus, coeffs]) => {
            const p: TestPolynomial = {
              coeffs,
              modulus,
              degree,
              isNTT: false,
            };

            const zero = createZeroPolynomial(degree, modulus);

            // Compute p + 0
            const sum = addPolynomials(p, zero);

            // Assert p + 0 = p
            assertPolynomialsEqual(sum, p, 'Adding zero should preserve polynomial');
          }
        ),
        PROPERTY_TEST_CONFIG
      );
    });
  });

  describe('Example Property 3: Polynomial Additive Inverse', () => {
    // Feature: fhe-accelerate, Example Property: Polynomial Additive Inverse
    it('should satisfy p + (-p) = 0 for all polynomials', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 5 }).chain((idx) => {
            const degree = [1024, 2048, 4096, 8192, 16384, 32768][idx];
            const modulus = BigInt('1099511627777');
            return fc.tuple(
              fc.constant(degree),
              fc.constant(modulus),
              arbitraryPolynomialCoeffs(degree, modulus)
            );
          }),
          ([degree, modulus, coeffs]) => {
            const p: TestPolynomial = {
              coeffs,
              modulus,
              degree,
              isNTT: false,
            };

            const negP = negatePolynomial(p);
            const zero = createZeroPolynomial(degree, modulus);

            // Compute p + (-p)
            const sum = addPolynomials(p, negP);

            // Assert p + (-p) = 0
            assertPolynomialsEqual(sum, zero, 'Adding inverse should give zero');
          }
        ),
        PROPERTY_TEST_CONFIG
      );
    });
  });

  describe('Example Property 4: Plaintext Value Bounds', () => {
    // Feature: fhe-accelerate, Example Property: Plaintext Value Bounds
    it('should generate plaintexts within specified bit range', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 8 }).chain((bits) => 
            fc.tuple(fc.constant(bits), arbitraryPlaintext(bits))
          ),
          ([bits, plaintext]) => {
            const maxValue = (1 << bits) - 1;
            expect(plaintext).toBeGreaterThanOrEqual(0);
            expect(plaintext).toBeLessThanOrEqual(maxValue);
          }
        ),
        PROPERTY_TEST_CONFIG
      );
    });
  });

  describe('Example Property 5: Parameter Preset Validity', () => {
    // Feature: fhe-accelerate, Example Property: Parameter Preset Validity
    it('should generate valid parameter presets', () => {
      fc.assert(
        fc.property(arbitraryParameterPreset(), (preset) => {
          const validPresets = [
            'tfhe-128-fast',
            'tfhe-128-balanced',
            'tfhe-256-secure',
            'bfv-128-simd',
            'ckks-128-ml',
          ];
          expect(validPresets).toContain(preset);
        }),
        PROPERTY_TEST_CONFIG
      );
    });
  });

  describe('Example Property 6: Security Level Validity', () => {
    // Feature: fhe-accelerate, Example Property: Security Level Validity
    it('should generate valid security levels', () => {
      fc.assert(
        fc.property(arbitrarySecurityLevel(), (level) => {
          expect([128, 192, 256]).toContain(level);
        }),
        PROPERTY_TEST_CONFIG
      );
    });
  });

  describe('Example Property 7: Coefficient Modular Reduction', () => {
    // Feature: fhe-accelerate, Example Property: Coefficient Modular Reduction
    it('should ensure coefficients are less than modulus', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 5 }).chain((idx) => {
            const degree = [1024, 2048, 4096, 8192, 16384, 32768][idx];
            const modulus = BigInt('1099511627777');
            return fc.tuple(
              fc.constant(degree),
              fc.constant(modulus),
              arbitraryPolynomialCoeffs(degree, modulus)
            );
          }),
          ([degree, modulus, coeffs]) => {
            // All coefficients should be less than modulus
            for (const coeff of coeffs) {
              expect(coeff).toBeGreaterThanOrEqual(0n);
              expect(coeff).toBeLessThan(modulus);
            }
          }
        ),
        PROPERTY_TEST_CONFIG
      );
    });
  });

  describe('Example Property 8: Identity Polynomial Structure', () => {
    // Feature: fhe-accelerate, Example Property: Identity Polynomial Structure
    it('should have identity polynomial with first coeff = 1, rest = 0', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 5 }).chain((idx) => {
            const degree = [1024, 2048, 4096, 8192, 16384, 32768][idx];
            const modulus = BigInt('1099511627777');
            return fc.tuple(fc.constant(degree), fc.constant(modulus));
          }),
          ([degree, modulus]) => {
            const identity = createIdentityPolynomial(degree, modulus);

            // First coefficient should be 1
            expect(identity.coeffs[0]).toBe(1n);

            // All other coefficients should be 0
            for (let i = 1; i < degree; i++) {
              expect(identity.coeffs[i]).toBe(0n);
            }
          }
        ),
        PROPERTY_TEST_CONFIG
      );
    });
  });
});
