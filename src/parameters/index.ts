/**
 * @file parameters/index.ts
 * @brief FHE Parameter Set definitions and presets for TypeScript
 * 
 * This module provides parameter set configurations for various FHE schemes
 * (TFHE, BFV, CKKS) at different security levels, along with validation
 * and derived parameter calculations.
 * 
 * Requirements: 10.1, 10.2, 10.3
 */

import { SecurityLevel, ParameterPreset, CustomParameters, FHEError, FHEErrorCode } from '../index';

/**
 * FHE scheme types
 */
export type FHEScheme = 'TFHE' | 'BFV' | 'CKKS';

/**
 * Complete FHE parameter set with all configuration values
 */
export interface ParameterSet {
  // Core polynomial parameters
  polyDegree: number;
  moduli: bigint[];
  
  // TFHE-specific parameters
  lweDimension: number;
  lweNoiseStd: number;
  glweDimension: number;
  decompBaseLog: number;
  decompLevel: number;
  
  // Security configuration
  security: SecurityLevel;
  scheme: FHEScheme;
  
  // Derived parameters
  plaintextModulus: bigint;
  noiseBudget: number;
  maxMultDepth: number;
}

/**
 * Validation result for parameter sets
 */
export interface ValidationResult {
  isValid: boolean;
  violations: string[];
}

// ========== Common NTT-Friendly Primes ==========

const NTT_PRIMES = {
  // 60-bit primes for high precision
  Q_60_1: 1152921504606584833n,
  Q_60_2: 1152921504598720513n,
  Q_60_3: 1152921504597016577n,
  
  // 50-bit primes for medium precision
  Q_50_1: 1125899906826241n,
  Q_50_2: 1125899906793473n,
  
  // 40-bit primes for TFHE
  Q_40_1: 1099511627777n,
  Q_40_2: 1099511562241n,
  
  // 30-bit primes for fast operations
  Q_30_1: 1073479681n,
  Q_30_2: 1073217537n,
};

// ========== Derived Parameter Calculations ==========

/**
 * Calculate the log2 of the total coefficient modulus
 */
export function getLogModulus(moduli: bigint[]): number {
  let logQ = 0;
  for (const q of moduli) {
    logQ += Math.log2(Number(q));
  }
  return logQ;
}

/**
 * Check if a number is a power of 2
 */
export function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * Calculate derived parameters from base parameters
 */
export function calculateDerivedParameters(params: Partial<ParameterSet>): {
  noiseBudget: number;
  maxMultDepth: number;
} {
  const logQ = getLogModulus(params.moduli || []);
  const logT = Math.log2(Number(params.plaintextModulus || 1n));
  
  let noiseBudget: number;
  
  if (params.scheme === 'TFHE') {
    // TFHE noise budget based on LWE parameters
    const noiseTerm = Math.log2(
      (params.lweNoiseStd || 1) * Math.sqrt(params.lweDimension || 1)
    );
    noiseBudget = logQ - noiseTerm - 10; // Safety margin
  } else {
    // BFV/CKKS noise budget
    noiseBudget = logQ - logT - 20; // Conservative estimate
  }
  
  noiseBudget = Math.max(0, noiseBudget);
  
  // Calculate max multiplication depth
  const noisePerMult = 10; // Approximate bits consumed per multiplication
  let maxMultDepth = Math.floor(noiseBudget / noisePerMult);
  
  // TFHE with bootstrapping has effectively unlimited depth
  if (params.scheme === 'TFHE' && (params.decompLevel || 0) > 0) {
    maxMultDepth = 1000;
  }
  
  return { noiseBudget, maxMultDepth };
}

// ========== Preset Parameter Sets ==========

/**
 * TFHE-128-FAST: Fast bootstrapping with 128-bit security
 * 
 * Optimized for low-latency bootstrapping operations.
 * Suitable for applications requiring frequent bootstrapping.
 */
export function TFHE_128_FAST(): ParameterSet {
  const params: ParameterSet = {
    scheme: 'TFHE',
    security: 128,
    polyDegree: 1024,
    moduli: [NTT_PRIMES.Q_40_1],
    lweDimension: 742,
    lweNoiseStd: 3.2e-11,
    glweDimension: 1,
    decompBaseLog: 23,
    decompLevel: 1,
    plaintextModulus: 4n,
    noiseBudget: 0,
    maxMultDepth: 0,
  };
  
  const derived = calculateDerivedParameters(params);
  params.noiseBudget = derived.noiseBudget;
  params.maxMultDepth = derived.maxMultDepth;
  
  return params;
}

/**
 * TFHE-128-BALANCED: Balanced performance/security for 128-bit
 */
export function TFHE_128_BALANCED(): ParameterSet {
  const params: ParameterSet = {
    scheme: 'TFHE',
    security: 128,
    polyDegree: 2048,
    moduli: [NTT_PRIMES.Q_50_1],
    lweDimension: 830,
    lweNoiseStd: 2.9e-11,
    glweDimension: 1,
    decompBaseLog: 15,
    decompLevel: 2,
    plaintextModulus: 8n,
    noiseBudget: 0,
    maxMultDepth: 0,
  };
  
  const derived = calculateDerivedParameters(params);
  params.noiseBudget = derived.noiseBudget;
  params.maxMultDepth = derived.maxMultDepth;
  
  return params;
}

/**
 * TFHE-256-SECURE: Maximum security with 256-bit level
 */
export function TFHE_256_SECURE(): ParameterSet {
  const params: ParameterSet = {
    scheme: 'TFHE',
    security: 256,
    polyDegree: 4096,
    moduli: [NTT_PRIMES.Q_60_1],
    lweDimension: 1024,
    lweNoiseStd: 2.0e-12,
    glweDimension: 1,
    decompBaseLog: 10,
    decompLevel: 3,
    plaintextModulus: 16n,
    noiseBudget: 0,
    maxMultDepth: 0,
  };
  
  const derived = calculateDerivedParameters(params);
  params.noiseBudget = derived.noiseBudget;
  params.maxMultDepth = derived.maxMultDepth;
  
  return params;
}

/**
 * BFV-128-SIMD: BFV scheme with SIMD packing for 128-bit security
 */
export function BFV_128_SIMD(): ParameterSet {
  const params: ParameterSet = {
    scheme: 'BFV',
    security: 128,
    polyDegree: 8192,  // Larger degree to support the modulus chain
    moduli: [NTT_PRIMES.Q_60_1, NTT_PRIMES.Q_60_2, NTT_PRIMES.Q_60_3],
    lweDimension: 0,
    lweNoiseStd: 3.2,
    glweDimension: 1,
    decompBaseLog: 60,
    decompLevel: 3,
    plaintextModulus: 65537n,
    noiseBudget: 0,
    maxMultDepth: 0,
  };
  
  const derived = calculateDerivedParameters(params);
  params.noiseBudget = derived.noiseBudget;
  params.maxMultDepth = derived.maxMultDepth;
  
  return params;
}

/**
 * CKKS-128-ML: CKKS scheme optimized for ML workloads
 */
export function CKKS_128_ML(): ParameterSet {
  const params: ParameterSet = {
    scheme: 'CKKS',
    security: 128,
    polyDegree: 16384,  // Larger degree to support the modulus chain
    moduli: [
      NTT_PRIMES.Q_60_1,
      NTT_PRIMES.Q_50_1,
      NTT_PRIMES.Q_50_2,
      NTT_PRIMES.Q_40_1,
      NTT_PRIMES.Q_40_2,
    ],
    lweDimension: 0,
    lweNoiseStd: 3.2,
    glweDimension: 1,
    decompBaseLog: 40,
    decompLevel: 5,
    plaintextModulus: 1n << 40n,
    noiseBudget: 0,
    maxMultDepth: 0,
  };
  
  const derived = calculateDerivedParameters(params);
  params.noiseBudget = derived.noiseBudget;
  params.maxMultDepth = derived.maxMultDepth;
  
  return params;
}

/**
 * TFHE-128-VOTING: Optimized for voting applications
 */
export function TFHE_128_VOTING(): ParameterSet {
  const params: ParameterSet = {
    scheme: 'TFHE',
    security: 128,
    polyDegree: 1024,
    moduli: [NTT_PRIMES.Q_40_1],
    lweDimension: 742,
    lweNoiseStd: 3.2e-11,
    glweDimension: 1,
    decompBaseLog: 23,
    decompLevel: 1,
    plaintextModulus: 16n,
    noiseBudget: 0,
    maxMultDepth: 0,
  };
  
  const derived = calculateDerivedParameters(params);
  params.noiseBudget = derived.noiseBudget;
  params.maxMultDepth = derived.maxMultDepth;
  
  return params;
}

// ========== Factory Functions ==========

/**
 * Create a parameter set from a preset name
 */
export function createParameterSet(preset: ParameterPreset): ParameterSet {
  switch (preset) {
    case 'tfhe-128-fast':
      return TFHE_128_FAST();
    case 'tfhe-128-balanced':
      return TFHE_128_BALANCED();
    case 'tfhe-256-secure':
      return TFHE_256_SECURE();
    case 'bfv-128-simd':
      return BFV_128_SIMD();
    case 'ckks-128-ml':
      return CKKS_128_ML();
    default:
      throw new FHEError(
        `Unknown parameter preset: ${preset}`,
        FHEErrorCode.INVALID_PARAMETERS
      );
  }
}

/**
 * Create a parameter set from custom parameters
 */
export function createCustomParameterSet(custom: CustomParameters): ParameterSet {
  const scheme: FHEScheme = custom.lweDimension ? 'TFHE' : 'BFV';
  
  const params: ParameterSet = {
    scheme,
    security: custom.securityLevel,
    polyDegree: custom.polyDegree,
    moduli: custom.moduli,
    lweDimension: custom.lweDimension || 0,
    lweNoiseStd: custom.lweNoiseStd || 3.2e-11,
    glweDimension: custom.glweDimension || 1,
    decompBaseLog: custom.decompBaseLog || 23,
    decompLevel: custom.decompLevel || 1,
    plaintextModulus: 4n,
    noiseBudget: 0,
    maxMultDepth: 0,
  };
  
  const derived = calculateDerivedParameters(params);
  params.noiseBudget = derived.noiseBudget;
  params.maxMultDepth = derived.maxMultDepth;
  
  return params;
}

/**
 * Get list of available preset names
 */
export function getAvailablePresets(): ParameterPreset[] {
  return [
    'tfhe-128-fast',
    'tfhe-128-balanced',
    'tfhe-256-secure',
    'bfv-128-simd',
    'ckks-128-ml',
  ];
}

/**
 * Get a human-readable description of a parameter set
 */
export function parameterSetToString(params: ParameterSet): string {
  const logQ = getLogModulus(params.moduli);
  return `ParameterSet {
  scheme: ${params.scheme}
  security: ${params.security} bits
  polyDegree: ${params.polyDegree}
  moduli: [${params.moduli.map(m => m.toString()).join(', ')}]
  log2(q): ${logQ.toFixed(2)} bits
  lweDimension: ${params.lweDimension}
  lweNoiseStd: ${params.lweNoiseStd}
  glweDimension: ${params.glweDimension}
  decompBaseLog: ${params.decompBaseLog}
  decompLevel: ${params.decompLevel}
  plaintextModulus: ${params.plaintextModulus}
  noiseBudget: ${params.noiseBudget.toFixed(2)} bits
  maxMultDepth: ${params.maxMultDepth}
}`;
}

// Re-export NTT primes for testing
export { NTT_PRIMES };
