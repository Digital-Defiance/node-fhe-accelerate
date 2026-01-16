/**
 * @file api/types.ts
 * @brief Core TypeScript type definitions for FHE operations
 *
 * This module provides comprehensive type definitions for all FHE operations,
 * including encryption, homomorphic operations, key management, and voting.
 *
 * Requirements: 12.1, 12.2, 12.3, 12.6
 */

// ============================================================================
// Opaque Handle Types
// ============================================================================

/**
 * Opaque handle to a native secret key
 */
export interface SecretKey {
  readonly __brand: 'SecretKey';
  readonly handle: bigint;
  readonly keyId: bigint;
}

/**
 * Opaque handle to a native public key
 */
export interface PublicKey {
  readonly __brand: 'PublicKey';
  readonly handle: bigint;
  readonly keyId: bigint;
}

/**
 * Opaque handle to a native evaluation key
 */
export interface EvaluationKey {
  readonly __brand: 'EvaluationKey';
  readonly handle: bigint;
  readonly keyId: bigint;
  readonly decompBaseLog: number;
  readonly decompLevel: number;
}

/**
 * Opaque handle to a native bootstrapping key
 */
export interface BootstrapKey {
  readonly __brand: 'BootstrapKey';
  readonly handle: bigint;
  readonly keyId: bigint;
  readonly lweDimension: number;
}

/**
 * Opaque handle to a native ciphertext
 */
export interface Ciphertext {
  readonly __brand: 'Ciphertext';
  readonly handle: bigint;
  readonly keyId: bigint;
  readonly noiseBudget: number;
  readonly isNtt: boolean;
  readonly degree: number;
}

/**
 * Plaintext structure for FHE
 */
export interface Plaintext {
  readonly __brand: 'Plaintext';
  readonly values: bigint[];
  readonly plaintextModulus: bigint;
  readonly isPacked: boolean;
}

// ============================================================================
// Parameter Types
// ============================================================================

/**
 * Security level in bits
 */
export type SecurityLevel = 128 | 192 | 256;

/**
 * FHE scheme types
 */
export type FHEScheme = 'TFHE' | 'BFV' | 'CKKS';

/**
 * Parameter preset names
 */
export type ParameterPreset =
  | 'tfhe-128-fast'
  | 'tfhe-128-balanced'
  | 'tfhe-256-secure'
  | 'bfv-128-simd'
  | 'ckks-128-ml'
  | 'tfhe-128-voting';

/**
 * Custom parameter configuration
 */
export interface CustomParameters {
  polyDegree: number;
  moduli: bigint[];
  lweDimension?: number;
  lweNoiseStd?: number;
  glweDimension?: number;
  decompBaseLog?: number;
  decompLevel?: number;
  securityLevel: SecurityLevel;
}

/**
 * Complete parameter set
 */
export interface ParameterSet {
  scheme: FHEScheme;
  security: SecurityLevel;
  polyDegree: number;
  moduli: bigint[];
  lweDimension: number;
  lweNoiseStd: number;
  glweDimension: number;
  decompBaseLog: number;
  decompLevel: number;
  plaintextModulus: bigint;
  noiseBudget: number;
  maxMultDepth: number;
}

// ============================================================================
// Error Types
// ============================================================================

/**
 * FHE error codes
 */
export enum FHEErrorCode {
  NOISE_BUDGET_EXHAUSTED = 'NOISE_BUDGET_EXHAUSTED',
  INVALID_PARAMETERS = 'INVALID_PARAMETERS',
  KEY_MISMATCH = 'KEY_MISMATCH',
  HARDWARE_UNAVAILABLE = 'HARDWARE_UNAVAILABLE',
  SERIALIZATION_ERROR = 'SERIALIZATION_ERROR',
  PROOF_VERIFICATION_FAILED = 'PROOF_VERIFICATION_FAILED',
  THRESHOLD_NOT_MET = 'THRESHOLD_NOT_MET',
  INVALID_BALLOT = 'INVALID_BALLOT',
  DUPLICATE_VOTE = 'DUPLICATE_VOTE',
  NATIVE_ERROR = 'NATIVE_ERROR',
}

/**
 * FHE error class with typed error codes
 */
export class FHEError extends Error {
  constructor(
    message: string,
    public readonly code: FHEErrorCode,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'FHEError';
    Object.setPrototypeOf(this, FHEError.prototype);
  }
}

// ============================================================================
// Hardware Types
// ============================================================================

/**
 * Hardware backend types
 */
export type HardwareBackend = 'SME' | 'Metal' | 'NEON' | 'AMX' | 'NeuralEngine' | 'Fallback';

/**
 * Hardware capabilities
 */
export interface HardwareCapabilities {
  hasSme: boolean;
  hasMetal: boolean;
  hasNeon: boolean;
  hasAmx: boolean;
  hasNeuralEngine: boolean;
  metalGpuCores: number;
  unifiedMemorySize: bigint;
}

// ============================================================================
// Key Management Types
// ============================================================================

/**
 * Secret key distribution types
 */
export type SecretKeyDistribution = 'TERNARY' | 'GAUSSIAN' | 'BINARY' | 'UNIFORM';

/**
 * Key generation options
 */
export interface KeyGenerationOptions {
  distribution?: SecretKeyDistribution;
  decompBaseLog?: number;
  decompLevel?: number;
}

/**
 * Threshold key configuration
 */
export interface ThresholdConfig {
  threshold: number; // M required shares
  totalShares: number; // N total shares
}

/**
 * Secret key share for threshold decryption
 */
export interface SecretKeyShare {
  readonly shareId: number;
  readonly handle: bigint;
  readonly commitment: Uint8Array;
  readonly keyId: bigint;
}

/**
 * Threshold key set
 */
export interface ThresholdKeys {
  shares: SecretKeyShare[];
  publicKey: PublicKey;
  threshold: number;
  totalShares: number;
}

/**
 * Partial decryption result
 */
export interface PartialDecryption {
  shareId: number;
  partialResult: bigint[];
  proof?: Uint8Array;
}

// ============================================================================
// Encryption Types
// ============================================================================

/**
 * Encryption result with optional proof
 */
export interface EncryptionResult {
  ciphertext: Ciphertext;
  proof?: Uint8Array;
}

/**
 * Decryption result
 */
export interface DecryptionResult {
  plaintext: Plaintext;
  remainingNoiseBudget: number;
  success: boolean;
  errorMessage?: string;
}

/**
 * Batch encryption options
 */
export interface BatchEncryptionOptions {
  generateProofs?: boolean;
  useGpu?: boolean;
  batchSize?: number;
}

/**
 * Batch encryption result
 */
export interface BatchEncryptionResult {
  ciphertexts: Ciphertext[];
  proofs?: Uint8Array[];
  elapsedMs: number;
  throughputPerSecond: number;
}

// ============================================================================
// Progress Callback Types
// ============================================================================

/**
 * Progress information
 */
export interface ProgressInfo {
  stage: string;
  current: number;
  total: number;
  elapsedMs: number;
  estimatedRemainingMs?: number;
  progressPercent: number;
}

/**
 * Progress callback type
 */
export type ProgressCallback = (progress: ProgressInfo) => void;

// ============================================================================
// Serialization Types
// ============================================================================

/**
 * Serialization format
 */
export type SerializationFormat = 'binary' | 'json' | 'compressed';

/**
 * Serialization options
 */
export interface SerializationOptions {
  format?: SerializationFormat;
  includeMetadata?: boolean;
  compress?: boolean;
}

/**
 * Serialized key data
 */
export interface SerializedKey {
  data: Uint8Array;
  format: SerializationFormat;
  keyType: 'secret' | 'public' | 'evaluation' | 'bootstrap';
  version: number;
  checksum: Uint8Array;
}

/**
 * Serialized ciphertext data
 */
export interface SerializedCiphertext {
  data: Uint8Array;
  format: SerializationFormat;
  keyId: bigint;
  version: number;
  checksum: Uint8Array;
}
