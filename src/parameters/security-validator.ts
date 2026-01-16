/**
 * @file parameters/security-validator.ts
 * @brief Security constraint validation for FHE parameters
 * 
 * Implements lattice security estimator checks and validates that
 * parameter sets meet the required security levels.
 * 
 * Requirements: 10.4, 10.5
 */

import { SecurityLevel, FHEError, FHEErrorCode } from '../index';
import { ParameterSet, getLogModulus } from './index';

/**
 * Security violation types
 */
export enum SecurityViolation {
  NONE = 'NONE',
  POLY_DEGREE_TOO_SMALL = 'POLY_DEGREE_TOO_SMALL',
  POLY_DEGREE_NOT_POWER_OF_TWO = 'POLY_DEGREE_NOT_POWER_OF_TWO',
  MODULUS_TOO_LARGE = 'MODULUS_TOO_LARGE',
  MODULUS_TOO_SMALL = 'MODULUS_TOO_SMALL',
  LWE_DIMENSION_TOO_SMALL = 'LWE_DIMENSION_TOO_SMALL',
  NOISE_STD_TOO_SMALL = 'NOISE_STD_TOO_SMALL',
  NOISE_STD_TOO_LARGE = 'NOISE_STD_TOO_LARGE',
  DECOMP_LEVEL_INVALID = 'DECOMP_LEVEL_INVALID',
  PLAINTEXT_MODULUS_INVALID = 'PLAINTEXT_MODULUS_INVALID',
  SECURITY_LEVEL_NOT_MET = 'SECURITY_LEVEL_NOT_MET',
  MODULUS_NOT_NTT_FRIENDLY = 'MODULUS_NOT_NTT_FRIENDLY',
  INVALID_GLWE_DIMENSION = 'INVALID_GLWE_DIMENSION',
}

/**
 * Detailed security violation information
 */
export interface SecurityViolationInfo {
  code: SecurityViolation;
  message: string;
  parameterName?: string;
  actualValue?: number;
  requiredValue?: number;
}

/**
 * Security validation result
 */
export interface SecurityValidationResult {
  isSecure: boolean;
  violations: SecurityViolationInfo[];
  estimatedSecurityBits: number;
}

// ========== Security Bounds Tables ==========
// Based on conservative estimates from the lattice estimator

// Minimum polynomial degrees for each security level
const MIN_DEGREE: Record<SecurityLevel, number> = {
  128: 1024,
  192: 2048,
  256: 4096,
};

// Maximum log2(q) for each (degree, security) pair
// Format: MAX_LOG_Q[degree][security]
const MAX_LOG_Q: Record<number, Record<SecurityLevel, number>> = {
  1024: { 128: 27, 192: 19, 256: 14 },
  2048: { 128: 54, 192: 37, 256: 29 },
  4096: { 128: 109, 192: 75, 256: 58 },
  8192: { 128: 218, 192: 152, 256: 118 },
  16384: { 128: 438, 192: 305, 256: 237 },
  32768: { 128: 881, 192: 611, 256: 476 },
};

// Minimum LWE dimensions for each security level
const MIN_LWE_DIM: Record<SecurityLevel, number> = {
  128: 630,
  192: 880,
  256: 1024,
};

/**
 * Check if a number is a power of 2
 */
function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * Miller-Rabin primality test for BigInt
 */
function isPrime(n: bigint): boolean {
  if (n < 2n) return false;
  if (n === 2n || n === 3n) return true;
  if (n % 2n === 0n) return false;

  // Write n-1 as 2^r * d
  let d = n - 1n;
  let r = 0;
  while (d % 2n === 0n) {
    d /= 2n;
    r++;
  }

  // Witnesses sufficient for 64-bit numbers
  const witnesses = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];

  for (const a of witnesses) {
    if (a >= n) continue;

    let x = modPow(a, d, n);

    if (x === 1n || x === n - 1n) continue;

    let composite = true;
    for (let i = 0; i < r - 1; i++) {
      x = modPow(x, 2n, n);
      if (x === n - 1n) {
        composite = false;
        break;
      }
    }

    if (composite) return false;
  }

  return true;
}

/**
 * Modular exponentiation for BigInt
 */
function modPow(base: bigint, exp: bigint, mod: bigint): bigint {
  let result = 1n;
  let b = base % mod;

  while (exp > 0n) {
    if (exp & 1n) {
      result = (result * b) % mod;
    }
    b = (b * b) % mod;
    exp >>= 1n;
  }

  return result;
}

/**
 * Check if a modulus is NTT-friendly for a given degree
 */
export function isNttFriendly(modulus: bigint, degree: number): boolean {
  if (!isPrime(modulus)) {
    return false;
  }

  const twoN = BigInt(2 * degree);
  return modulus % twoN === 1n;
}

/**
 * Get minimum polynomial degree for a security level
 */
export function getMinPolyDegree(security: SecurityLevel): number {
  return MIN_DEGREE[security];
}

/**
 * Get maximum log2(q) for a given degree and security level
 */
export function getMaxLogModulus(degree: number, security: SecurityLevel): number {
  const entry = MAX_LOG_Q[degree];
  if (entry) {
    return entry[security];
  }

  // Interpolate for non-standard degrees
  const logDegree = Math.log2(degree);
  const baseIdx = logDegree - 10; // log2(1024) = 10

  const degrees = [1024, 2048, 4096, 8192, 16384, 32768];
  const lowerIdx = Math.max(0, Math.min(4, Math.floor(baseIdx)));
  const upperIdx = Math.min(lowerIdx + 1, 5);
  const frac = baseIdx - lowerIdx;

  const lowerDegree = degrees[lowerIdx] ?? 1024;
  const upperDegree = degrees[upperIdx] ?? 32768;
  
  const lowerEntry = MAX_LOG_Q[lowerDegree];
  const upperEntry = MAX_LOG_Q[upperDegree];
  
  if (!lowerEntry || !upperEntry) {
    return 27; // Default fallback
  }

  return (
    lowerEntry[security] * (1 - frac) +
    upperEntry[security] * frac
  );
}

/**
 * Get minimum LWE dimension for a security level
 */
export function getMinLweDimension(security: SecurityLevel): number {
  return MIN_LWE_DIM[security];
}

/**
 * Estimate security level in bits for LWE parameters
 * 
 * Uses a simplified lattice security estimator based on the BKZ algorithm.
 */
export function estimateSecurityBits(
  n: number,
  logQ: number,
  noiseStd: number
): number {
  if (n === 0 || logQ <= 0 || noiseStd <= 0) {
    return 0;
  }

  const logSigma = Math.log2(noiseStd);

  // Hermite factor required
  const delta = Math.pow(2, (logQ - logSigma) / (4 * n));

  if (delta <= 1) {
    return 256; // Very secure
  }

  const logDelta = Math.log2(delta);
  if (logDelta <= 0) {
    return 256;
  }

  // Estimate BKZ block size
  const b = 2 * n * logDelta;

  // Convert block size to security bits using core-SVP model
  let security = 0.292 * b;

  // Apply safety margin
  security *= 0.9;

  return Math.max(0, Math.min(256, security));
}

/**
 * Estimate RLWE security for polynomial ring parameters
 * 
 * For BFV/CKKS, security is based on the RLWE problem hardness.
 * Uses conservative estimates from the lattice estimator.
 */
export function estimateRlweSecurityBits(
  polyDegree: number,
  logQ: number,
  _noiseStd: number
): number {
  // For BFV/CKKS, use the standard RLWE security estimation
  // based on polynomial degree and modulus size
  
  // Security bounds from lattice estimator (conservative)
  // These are based on the ratio of log(q) to N
  
  if (polyDegree === 0 || logQ <= 0) {
    return 0;
  }
  
  // Maximum log(q) for each security level at different degrees
  // Based on homomorphicencryption.org security standard
  const securityTable: Record<number, { maxLogQ128: number; maxLogQ192: number; maxLogQ256: number }> = {
    1024: { maxLogQ128: 27, maxLogQ192: 19, maxLogQ256: 14 },
    2048: { maxLogQ128: 54, maxLogQ192: 37, maxLogQ256: 29 },
    4096: { maxLogQ128: 109, maxLogQ192: 75, maxLogQ256: 58 },
    8192: { maxLogQ128: 218, maxLogQ192: 152, maxLogQ256: 118 },
    16384: { maxLogQ128: 438, maxLogQ192: 305, maxLogQ256: 237 },
    32768: { maxLogQ128: 881, maxLogQ192: 611, maxLogQ256: 476 },
  };
  
  // Find the closest degree in the table
  const degrees = [1024, 2048, 4096, 8192, 16384, 32768];
  let closestDegree = degrees[0] ?? 1024;
  for (const d of degrees) {
    if (d <= polyDegree) {
      closestDegree = d;
    }
  }
  
  const bounds = securityTable[closestDegree];
  if (!bounds) {
    return 0;
  }
  
  // Determine security level based on log(q)
  if (logQ <= bounds.maxLogQ256) {
    return 256;
  } else if (logQ <= bounds.maxLogQ192) {
    // Interpolate between 192 and 256
    const ratio = (logQ - bounds.maxLogQ256) / (bounds.maxLogQ192 - bounds.maxLogQ256);
    return 256 - ratio * (256 - 192);
  } else if (logQ <= bounds.maxLogQ128) {
    // Interpolate between 128 and 192
    const ratio = (logQ - bounds.maxLogQ192) / (bounds.maxLogQ128 - bounds.maxLogQ192);
    return 192 - ratio * (192 - 128);
  } else {
    // Below 128-bit security
    // Estimate linearly
    const ratio = logQ / bounds.maxLogQ128;
    return Math.max(0, 128 / ratio);
  }
}

/**
 * Estimate TFHE security based on LWE dimension and polynomial degree
 * 
 * TFHE security is primarily determined by:
 * 1. LWE dimension n (for LWE samples)
 * 2. GLWE dimension k and polynomial degree N (for GLWE samples)
 * 
 * The security is the minimum of LWE and GLWE security.
 * 
 * Based on TFHE-rs parameter selection and lattice estimator results.
 */
export function estimateTfheSecurityBits(
  lweDimension: number,
  polyDegree: number,
  logQ: number
): number {
  // TFHE security estimation based on known secure parameter sets
  // These bounds are derived from the lattice estimator and TFHE-rs defaults
  
  // LWE security: primarily depends on dimension
  // For standard TFHE parameters with appropriate noise:
  // n >= 630 provides ~128-bit security
  // n >= 880 provides ~192-bit security  
  // n >= 1024 provides ~256-bit security
  
  let lweSecurity: number;
  if (lweDimension >= 1024) {
    lweSecurity = 256;
  } else if (lweDimension >= 880) {
    // Linear interpolation between 192 and 256
    lweSecurity = 192 + (lweDimension - 880) * (256 - 192) / (1024 - 880);
  } else if (lweDimension >= 630) {
    // Linear interpolation between 128 and 192
    lweSecurity = 128 + (lweDimension - 630) * (192 - 128) / (880 - 630);
  } else if (lweDimension >= 450) {
    // Below standard, estimate linearly
    lweSecurity = 80 + (lweDimension - 450) * (128 - 80) / (630 - 450);
  } else {
    // Very low dimension
    lweSecurity = Math.max(0, lweDimension * 80 / 450);
  }
  
  // GLWE security: depends on polynomial degree and modulus
  // For TFHE, the GLWE modulus is typically 2^32 or 2^64
  // With N >= 1024 and reasonable modulus, GLWE is secure
  
  let glweSecurity: number;
  if (polyDegree >= 4096) {
    glweSecurity = 256;
  } else if (polyDegree >= 2048) {
    glweSecurity = 192;
  } else if (polyDegree >= 1024) {
    glweSecurity = 128;
  } else if (polyDegree >= 512) {
    glweSecurity = 80;
  } else {
    glweSecurity = Math.max(0, polyDegree * 80 / 512);
  }
  
  // Adjust GLWE security based on modulus size
  // Larger modulus reduces security
  if (logQ > 64) {
    glweSecurity = Math.max(0, glweSecurity - (logQ - 64) * 2);
  }
  
  // Overall security is the minimum
  return Math.min(lweSecurity, glweSecurity);
}

/**
 * Validate polynomial degree
 */
function validatePolyDegree(
  params: ParameterSet,
  result: SecurityValidationResult
): void {
  if (!isPowerOfTwo(params.polyDegree)) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.POLY_DEGREE_NOT_POWER_OF_TWO,
      message: 'Polynomial degree must be a power of 2',
      parameterName: 'polyDegree',
      actualValue: params.polyDegree,
    });
    return;
  }

  const minDegree = getMinPolyDegree(params.security);
  if (params.polyDegree < minDegree) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.POLY_DEGREE_TOO_SMALL,
      message: 'Polynomial degree too small for target security level',
      parameterName: 'polyDegree',
      actualValue: params.polyDegree,
      requiredValue: minDegree,
    });
  }
}

/**
 * Validate modulus chain
 */
function validateModulus(
  params: ParameterSet,
  result: SecurityValidationResult
): void {
  if (params.moduli.length === 0) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.MODULUS_TOO_SMALL,
      message: 'At least one modulus must be specified',
      parameterName: 'moduli',
    });
    return;
  }

  for (const q of params.moduli) {
    if (q < 2n) {
      result.isSecure = false;
      result.violations.push({
        code: SecurityViolation.MODULUS_TOO_SMALL,
        message: 'Modulus must be at least 2',
        parameterName: 'moduli',
        actualValue: Number(q),
        requiredValue: 2,
      });
      return;
    }
  }

  // For TFHE, the modulus bounds are different because security comes from LWE
  // not directly from RLWE. Skip modulus size check for TFHE.
  if (params.scheme === 'TFHE') {
    return;
  }

  const logQ = getLogModulus(params.moduli);
  const maxLogQ = getMaxLogModulus(params.polyDegree, params.security);

  if (logQ > maxLogQ) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.MODULUS_TOO_LARGE,
      message: 'Total modulus too large for polynomial degree and security level',
      parameterName: 'moduli (log2)',
      actualValue: logQ,
      requiredValue: maxLogQ,
    });
  }
}

/**
 * Validate LWE parameters (TFHE only)
 */
function validateLweParameters(
  params: ParameterSet,
  result: SecurityValidationResult
): void {
  if (params.scheme !== 'TFHE') {
    return;
  }

  const minDim = getMinLweDimension(params.security);
  if (params.lweDimension < minDim) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.LWE_DIMENSION_TOO_SMALL,
      message: 'LWE dimension too small for target security level',
      parameterName: 'lweDimension',
      actualValue: params.lweDimension,
      requiredValue: minDim,
    });
  }

  if (params.lweNoiseStd <= 0) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.NOISE_STD_TOO_SMALL,
      message: 'Noise standard deviation must be positive',
      parameterName: 'lweNoiseStd',
      actualValue: params.lweNoiseStd,
      requiredValue: 1e-15,
    });
  }

  // Check upper bound on noise
  const logQ = getLogModulus(params.moduli);
  const maxNoiseStd = Math.pow(2, logQ / 4);

  if (params.lweNoiseStd > maxNoiseStd) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.NOISE_STD_TOO_LARGE,
      message: 'Noise standard deviation too large for correct decryption',
      parameterName: 'lweNoiseStd',
      actualValue: params.lweNoiseStd,
      requiredValue: maxNoiseStd,
    });
  }
}

/**
 * Validate decomposition parameters (TFHE only)
 */
function validateDecomposition(
  params: ParameterSet,
  result: SecurityValidationResult
): void {
  if (params.scheme !== 'TFHE') {
    return;
  }

  if (params.decompBaseLog === 0 || params.decompBaseLog > 64) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.DECOMP_LEVEL_INVALID,
      message: 'Decomposition base log must be between 1 and 64',
      parameterName: 'decompBaseLog',
      actualValue: params.decompBaseLog,
      requiredValue: 23,
    });
  }

  if (params.decompLevel === 0) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.DECOMP_LEVEL_INVALID,
      message: 'Decomposition level must be at least 1',
      parameterName: 'decompLevel',
      actualValue: params.decompLevel,
      requiredValue: 1,
    });
  }

  if (params.glweDimension === 0) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.INVALID_GLWE_DIMENSION,
      message: 'GLWE dimension must be at least 1',
      parameterName: 'glweDimension',
      actualValue: params.glweDimension,
      requiredValue: 1,
    });
  }
}

/**
 * Validate overall security level
 */
function validateSecurityLevel(
  params: ParameterSet,
  result: SecurityValidationResult
): void {
  const logQ = getLogModulus(params.moduli);

  let estimatedBits: number;
  if (params.scheme === 'TFHE') {
    // For TFHE, use LWE security estimate with proper noise handling
    // TFHE uses very small noise (relative to modulus) which is secure
    // because the LWE dimension provides the security
    estimatedBits = estimateTfheSecurityBits(
      params.lweDimension,
      params.polyDegree,
      logQ
    );
  } else {
    // Use RLWE security estimate for BFV/CKKS
    estimatedBits = estimateRlweSecurityBits(
      params.polyDegree,
      logQ,
      params.lweNoiseStd > 0 ? params.lweNoiseStd : 3.2
    );
  }

  result.estimatedSecurityBits = estimatedBits;

  if (estimatedBits < params.security) {
    result.isSecure = false;
    result.violations.push({
      code: SecurityViolation.SECURITY_LEVEL_NOT_MET,
      message: 'Estimated security does not meet target level',
      parameterName: 'security',
      actualValue: estimatedBits,
      requiredValue: params.security,
    });
  }
}

/**
 * Validate a complete parameter set
 * 
 * @param params The parameter set to validate
 * @returns Validation result with any violations
 */
export function validateParameterSet(params: ParameterSet): SecurityValidationResult {
  const result: SecurityValidationResult = {
    isSecure: true,
    violations: [],
    estimatedSecurityBits: 0,
  };

  validatePolyDegree(params, result);
  validateModulus(params, result);
  validateLweParameters(params, result);
  validateDecomposition(params, result);
  validateSecurityLevel(params, result);

  return result;
}

/**
 * Validate and throw if invalid
 * 
 * @param params The parameter set to validate
 * @throws FHEError if validation fails
 */
export function assertValidParameterSet(params: ParameterSet): void {
  const result = validateParameterSet(params);

  if (!result.isSecure) {
    const messages = result.violations.map((v) => {
      let msg = v.message;
      if (v.parameterName) {
        msg += ` (${v.parameterName}`;
        if (v.actualValue !== undefined) {
          msg += `: actual=${v.actualValue}`;
        }
        if (v.requiredValue !== undefined) {
          msg += `, required=${v.requiredValue}`;
        }
        msg += ')';
      }
      return msg;
    });

    throw new FHEError(
      `Parameter validation failed:\n  - ${messages.join('\n  - ')}`,
      FHEErrorCode.INVALID_PARAMETERS,
      {
        violations: result.violations,
        estimatedSecurityBits: result.estimatedSecurityBits,
      }
    );
  }
}

/**
 * Get a human-readable error message from validation result
 */
export function getValidationErrorMessage(result: SecurityValidationResult): string {
  if (result.isSecure) {
    return '';
  }

  const lines = ['Security validation failed:'];
  for (const v of result.violations) {
    let line = `  - ${v.message}`;
    if (v.parameterName) {
      line += ` (parameter: ${v.parameterName}`;
      if (v.actualValue !== undefined) {
        line += `, actual: ${v.actualValue}`;
      }
      if (v.requiredValue !== undefined) {
        line += `, required: ${v.requiredValue}`;
      }
      line += ')';
    }
    lines.push(line);
  }
  lines.push(`Estimated security: ${result.estimatedSecurityBits.toFixed(2)} bits`);

  return lines.join('\n');
}
