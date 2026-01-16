/**
 * Property-Based Tests for Parameter Validation
 * 
 * **Property 11: Parameter Validation Rejects Insecure Configurations**
 * 
 * This test validates that the parameter validator correctly rejects
 * parameter sets that violate security constraints.
 * 
 * **Validates: Requirements 10.4**
 * - FOR ANY parameter set that violates security constraints, the validator
 *   SHALL reject the configuration with appropriate error messages
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG } from './property-test-config';
import {
  ParameterSet,
  FHEScheme,
  TFHE_128_FAST,
  TFHE_128_BALANCED,
  TFHE_256_SECURE,
  BFV_128_SIMD,
  CKKS_128_ML,
  TFHE_128_VOTING,
  createParameterSet,
  NTT_PRIMES,
} from '../parameters/index';
import {
  validateParameterSet,
  SecurityViolation,
  getMinPolyDegree,
  getMaxLogModulus,
  getMinLweDimension,
  estimateSecurityBits,
} from '../parameters/security-validator';
import { SecurityLevel } from '../index';

/**
 * Generate a valid TFHE parameter set as a baseline
 */
function generateValidTfheParams(): ParameterSet {
  return TFHE_128_FAST();
}

/**
 * Generate a parameter set with polynomial degree too small
 */
function arbitraryInsecurePolyDegree(): fc.Arbitrary<ParameterSet> {
  return fc.record({
    security: fc.constantFrom(128 as SecurityLevel, 192 as SecurityLevel, 256 as SecurityLevel),
    smallDegree: fc.integer({ min: 1, max: 512 }), // Too small for any security level
  }).map(({ security, smallDegree }) => {
    const base = generateValidTfheParams();
    return {
      ...base,
      security,
      polyDegree: smallDegree,
    };
  });
}

/**
 * Generate a parameter set with non-power-of-2 degree
 */
function arbitraryNonPowerOfTwoDegree(): fc.Arbitrary<ParameterSet> {
  return fc.integer({ min: 1025, max: 10000 })
    .filter(n => (n & (n - 1)) !== 0) // Not a power of 2
    .map(degree => {
      const base = generateValidTfheParams();
      return {
        ...base,
        polyDegree: degree,
      };
    });
}

/**
 * Generate a parameter set with modulus too large for the degree
 */
function arbitraryInsecureModulus(): fc.Arbitrary<ParameterSet> {
  return fc.record({
    security: fc.constantFrom(128 as SecurityLevel, 192 as SecurityLevel, 256 as SecurityLevel),
    degree: fc.constantFrom(1024, 2048),
  }).map(({ security, degree }) => {
    const base = generateValidTfheParams();
    // Use very large moduli that exceed security bounds
    const largeModuli = [
      NTT_PRIMES.Q_60_1,
      NTT_PRIMES.Q_60_2,
      NTT_PRIMES.Q_60_3,
      NTT_PRIMES.Q_50_1,
      NTT_PRIMES.Q_50_2,
    ];
    return {
      ...base,
      security,
      polyDegree: degree,
      moduli: largeModuli, // Total log2(q) will be ~270 bits, way too large for N=1024
    };
  });
}

/**
 * Generate a parameter set with LWE dimension too small
 */
function arbitraryInsecureLweDimension(): fc.Arbitrary<ParameterSet> {
  return fc.record({
    security: fc.constantFrom(128 as SecurityLevel, 192 as SecurityLevel, 256 as SecurityLevel),
    smallDim: fc.integer({ min: 100, max: 500 }), // Too small for any security level
  }).map(({ security, smallDim }) => {
    const base = generateValidTfheParams();
    return {
      ...base,
      security,
      lweDimension: smallDim,
    };
  });
}

/**
 * Generate a parameter set with invalid noise standard deviation
 */
function arbitraryInvalidNoiseStd(): fc.Arbitrary<ParameterSet> {
  return fc.oneof(
    // Zero or negative noise
    fc.double({ min: -1, max: 0, noNaN: true }).map(noise => {
      const base = generateValidTfheParams();
      return {
        ...base,
        lweNoiseStd: noise,
      };
    }),
    // Extremely large noise (would cause decryption failures)
    fc.double({ min: 1e10, max: 1e20, noNaN: true }).map(noise => {
      const base = generateValidTfheParams();
      return {
        ...base,
        lweNoiseStd: noise,
      };
    })
  );
}

/**
 * Generate a parameter set with invalid decomposition parameters
 */
function arbitraryInvalidDecomposition(): fc.Arbitrary<ParameterSet> {
  return fc.oneof(
    // Zero decomposition base log
    fc.constant(0).map(baseLog => {
      const base = generateValidTfheParams();
      return {
        ...base,
        decompBaseLog: baseLog,
      };
    }),
    // Zero decomposition level
    fc.constant(0).map(level => {
      const base = generateValidTfheParams();
      return {
        ...base,
        decompLevel: level,
      };
    }),
    // Zero GLWE dimension
    fc.constant(0).map(dim => {
      const base = generateValidTfheParams();
      return {
        ...base,
        glweDimension: dim,
      };
    }),
    // Decomposition base log too large
    fc.integer({ min: 65, max: 128 }).map(baseLog => {
      const base = generateValidTfheParams();
      return {
        ...base,
        decompBaseLog: baseLog,
      };
    })
  );
}

/**
 * Generate a parameter set with empty moduli
 */
function arbitraryEmptyModuli(): fc.Arbitrary<ParameterSet> {
  return fc.constant(null).map(() => {
    const base = generateValidTfheParams();
    return {
      ...base,
      moduli: [],
    };
  });
}

/**
 * Generate a parameter set with modulus less than 2
 */
function arbitraryTooSmallModulus(): fc.Arbitrary<ParameterSet> {
  return fc.integer({ min: 0, max: 1 }).map(mod => {
    const base = generateValidTfheParams();
    return {
      ...base,
      moduli: [BigInt(mod)],
    };
  });
}

describe('Property 11: Parameter Validation Rejects Insecure Configurations', () => {
  /**
   * **Validates: Requirements 10.4**
   * Parameter sets violating security bounds SHALL be rejected
   */
  
  describe('11.1 Valid Presets Pass Validation', () => {
    it('should accept all predefined secure parameter presets', () => {
      const presetFns = [
        { name: 'TFHE_128_FAST', fn: TFHE_128_FAST },
        { name: 'TFHE_128_BALANCED', fn: TFHE_128_BALANCED },
        { name: 'TFHE_256_SECURE', fn: TFHE_256_SECURE },
        { name: 'BFV_128_SIMD', fn: BFV_128_SIMD },
        { name: 'CKKS_128_ML', fn: CKKS_128_ML },
        { name: 'TFHE_128_VOTING', fn: TFHE_128_VOTING },
      ];
      
      for (const { name, fn } of presetFns) {
        const params = fn();
        const result = validateParameterSet(params);
        
        if (!result.isSecure) {
          console.log(`Preset ${name} failed validation:`);
          console.log(`  Violations: ${JSON.stringify(result.violations, null, 2)}`);
          console.log(`  Estimated security: ${result.estimatedSecurityBits}`);
        }
        
        expect(result.isSecure, `Preset ${name} should be secure`).toBe(true);
        expect(result.violations, `Preset ${name} should have no violations`).toHaveLength(0);
        expect(result.estimatedSecurityBits, `Preset ${name} should have positive security`).toBeGreaterThan(0);
      }
    });
  });

  describe('11.2 Polynomial Degree Validation', () => {
    it('should reject parameter sets with polynomial degree too small', () => {
      fc.assert(
        fc.property(
          arbitraryInsecurePolyDegree(),
          (params) => {
            const result = validateParameterSet(params);
            
            // Should be rejected
            expect(result.isSecure).toBe(false);
            
            // Should have appropriate violation
            const hasPolyDegreeViolation = result.violations.some(
              v => v.code === SecurityViolation.POLY_DEGREE_TOO_SMALL ||
                   v.code === SecurityViolation.POLY_DEGREE_NOT_POWER_OF_TWO
            );
            expect(hasPolyDegreeViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });

    it('should reject parameter sets with non-power-of-2 degree', () => {
      fc.assert(
        fc.property(
          arbitraryNonPowerOfTwoDegree(),
          (params) => {
            const result = validateParameterSet(params);
            
            expect(result.isSecure).toBe(false);
            
            const hasViolation = result.violations.some(
              v => v.code === SecurityViolation.POLY_DEGREE_NOT_POWER_OF_TWO
            );
            expect(hasViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });
  });

  describe('11.3 Modulus Validation', () => {
    it('should reject parameter sets with modulus too large for degree', () => {
      fc.assert(
        fc.property(
          arbitraryInsecureModulus(),
          (params) => {
            const result = validateParameterSet(params);
            
            // Should be rejected due to modulus being too large
            expect(result.isSecure).toBe(false);
            
            const hasModulusViolation = result.violations.some(
              v => v.code === SecurityViolation.MODULUS_TOO_LARGE ||
                   v.code === SecurityViolation.SECURITY_LEVEL_NOT_MET
            );
            expect(hasModulusViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });

    it('should reject parameter sets with empty moduli', () => {
      fc.assert(
        fc.property(
          arbitraryEmptyModuli(),
          (params) => {
            const result = validateParameterSet(params);
            
            expect(result.isSecure).toBe(false);
            
            const hasViolation = result.violations.some(
              v => v.code === SecurityViolation.MODULUS_TOO_SMALL
            );
            expect(hasViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 10 }
      );
    });

    it('should reject parameter sets with modulus less than 2', () => {
      fc.assert(
        fc.property(
          arbitraryTooSmallModulus(),
          (params) => {
            const result = validateParameterSet(params);
            
            expect(result.isSecure).toBe(false);
            
            const hasViolation = result.violations.some(
              v => v.code === SecurityViolation.MODULUS_TOO_SMALL
            );
            expect(hasViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 10 }
      );
    });
  });

  describe('11.4 LWE Dimension Validation', () => {
    it('should reject TFHE parameter sets with LWE dimension too small', () => {
      fc.assert(
        fc.property(
          arbitraryInsecureLweDimension(),
          (params) => {
            const result = validateParameterSet(params);
            
            expect(result.isSecure).toBe(false);
            
            const hasLweViolation = result.violations.some(
              v => v.code === SecurityViolation.LWE_DIMENSION_TOO_SMALL
            );
            expect(hasLweViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });
  });

  describe('11.5 Noise Standard Deviation Validation', () => {
    it('should reject parameter sets with invalid noise standard deviation', () => {
      fc.assert(
        fc.property(
          arbitraryInvalidNoiseStd(),
          (params) => {
            const result = validateParameterSet(params);
            
            expect(result.isSecure).toBe(false);
            
            const hasNoiseViolation = result.violations.some(
              v => v.code === SecurityViolation.NOISE_STD_TOO_SMALL ||
                   v.code === SecurityViolation.NOISE_STD_TOO_LARGE
            );
            expect(hasNoiseViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });
  });

  describe('11.6 Decomposition Parameter Validation', () => {
    it('should reject TFHE parameter sets with invalid decomposition parameters', () => {
      fc.assert(
        fc.property(
          arbitraryInvalidDecomposition(),
          (params) => {
            const result = validateParameterSet(params);
            
            expect(result.isSecure).toBe(false);
            
            const hasDecompViolation = result.violations.some(
              v => v.code === SecurityViolation.DECOMP_LEVEL_INVALID ||
                   v.code === SecurityViolation.INVALID_GLWE_DIMENSION
            );
            expect(hasDecompViolation).toBe(true);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });
  });

  describe('11.7 Security Level Bounds', () => {
    it('should correctly identify minimum polynomial degrees for security levels', () => {
      expect(getMinPolyDegree(128)).toBe(1024);
      expect(getMinPolyDegree(192)).toBe(2048);
      expect(getMinPolyDegree(256)).toBe(4096);
    });

    it('should correctly identify minimum LWE dimensions for security levels', () => {
      expect(getMinLweDimension(128)).toBe(630);
      expect(getMinLweDimension(192)).toBe(880);
      expect(getMinLweDimension(256)).toBe(1024);
    });

    it('should provide reasonable max log modulus bounds', () => {
      // For N=1024, 128-bit security
      const maxLog1024_128 = getMaxLogModulus(1024, 128);
      expect(maxLog1024_128).toBeGreaterThan(20);
      expect(maxLog1024_128).toBeLessThan(50);

      // For N=4096, 128-bit security
      const maxLog4096_128 = getMaxLogModulus(4096, 128);
      expect(maxLog4096_128).toBeGreaterThan(100);
      expect(maxLog4096_128).toBeLessThan(150);

      // Higher security should have lower max modulus
      const maxLog1024_256 = getMaxLogModulus(1024, 256);
      expect(maxLog1024_256).toBeLessThan(maxLog1024_128);
    });
  });

  describe('11.8 Security Estimation', () => {
    it('should estimate security bits within reasonable bounds', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 512, max: 2048 }),
          fc.double({ min: 20, max: 60, noNaN: true }),
          fc.double({ min: 1e-15, max: 1e-5, noNaN: true }),
          (n, logQ, noiseStd) => {
            const securityBits = estimateSecurityBits(n, logQ, noiseStd);
            
            // Security should be non-negative
            expect(securityBits).toBeGreaterThanOrEqual(0);
            
            // Security should not exceed 256 bits (our max)
            expect(securityBits).toBeLessThanOrEqual(256);
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
      );
    });

    it('should return 0 security for invalid parameters', () => {
      expect(estimateSecurityBits(0, 40, 1e-10)).toBe(0);
      expect(estimateSecurityBits(1024, 0, 1e-10)).toBe(0);
      expect(estimateSecurityBits(1024, 40, 0)).toBe(0);
      expect(estimateSecurityBits(1024, 40, -1)).toBe(0);
    });

    it('should increase security with larger dimension', () => {
      const sec512 = estimateSecurityBits(512, 40, 1e-10);
      const sec1024 = estimateSecurityBits(1024, 40, 1e-10);
      const sec2048 = estimateSecurityBits(2048, 40, 1e-10);
      
      // Larger dimension should generally provide more security
      // (with same modulus and noise)
      expect(sec1024).toBeGreaterThanOrEqual(sec512);
      expect(sec2048).toBeGreaterThanOrEqual(sec1024);
    });
  });

  describe('11.9 Violation Messages', () => {
    it('should provide detailed violation messages', () => {
      const insecureParams: ParameterSet = {
        ...generateValidTfheParams(),
        polyDegree: 256, // Too small
        lweDimension: 100, // Too small
      };
      
      const result = validateParameterSet(insecureParams);
      
      expect(result.isSecure).toBe(false);
      expect(result.violations.length).toBeGreaterThan(0);
      
      // Each violation should have a message
      for (const violation of result.violations) {
        expect(violation.message).toBeTruthy();
        expect(violation.message.length).toBeGreaterThan(0);
        expect(violation.code).toBeTruthy();
      }
    });

    it('should include parameter names in violations', () => {
      const insecureParams: ParameterSet = {
        ...generateValidTfheParams(),
        polyDegree: 256,
      };
      
      const result = validateParameterSet(insecureParams);
      
      const polyViolation = result.violations.find(
        v => v.code === SecurityViolation.POLY_DEGREE_TOO_SMALL
      );
      
      if (polyViolation) {
        expect(polyViolation.parameterName).toBe('polyDegree');
        expect(polyViolation.actualValue).toBe(256);
      }
    });
  });

  describe('11.10 Edge Cases', () => {
    it('should handle boundary polynomial degrees correctly', () => {
      // Exactly at minimum for 128-bit
      const params1024: ParameterSet = {
        ...generateValidTfheParams(),
        security: 128,
        polyDegree: 1024,
      };
      const result1024 = validateParameterSet(params1024);
      expect(result1024.violations.filter(
        v => v.code === SecurityViolation.POLY_DEGREE_TOO_SMALL
      )).toHaveLength(0);
      
      // Just below minimum for 128-bit
      const params512: ParameterSet = {
        ...generateValidTfheParams(),
        security: 128,
        polyDegree: 512,
      };
      const result512 = validateParameterSet(params512);
      expect(result512.violations.some(
        v => v.code === SecurityViolation.POLY_DEGREE_TOO_SMALL
      )).toBe(true);
    });

    it('should handle very large valid parameters', () => {
      const largeParams: ParameterSet = {
        ...TFHE_256_SECURE(),
        polyDegree: 32768,
      };
      
      const result = validateParameterSet(largeParams);
      // Should not crash and should provide a result
      expect(result).toBeDefined();
      expect(typeof result.isSecure).toBe('boolean');
    });
  });
});
