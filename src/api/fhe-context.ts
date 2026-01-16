/**
 * @file api/fhe-context.ts
 * @brief High-level convenience API for FHE operations
 *
 * This module provides FHEContext for simplified FHE usage with
 * automatic key management and parameter presets.
 *
 * Requirements: 12.1
 */

import type {
  SecretKey,
  PublicKey,
  EvaluationKey,
  BootstrapKey,
  Ciphertext,
  ParameterSet,
  ParameterPreset,
  CustomParameters,
  ThresholdConfig,
  ThresholdKeys,
  HardwareCapabilities,
  ProgressCallback,
} from './types';
import type { FHEEngine } from './fhe-engine';
import { createFHEEngine } from './fhe-engine';

/**
 * FHE Context configuration options
 */
export interface FHEContextOptions {
  /** Generate evaluation key on initialization */
  generateEvalKey?: boolean;
  /** Generate bootstrap key on initialization */
  generateBootstrapKey?: boolean;
  /** Threshold configuration for distributed decryption */
  thresholdConfig?: ThresholdConfig;
}

/**
 * High-level FHE Context for simplified usage
 *
 * Provides automatic key management and convenient methods for
 * common FHE operations.
 *
 * @example
 * ```typescript
 * const ctx = await FHEContext.create('tfhe-128-fast');
 * const ct1 = await ctx.encrypt(BigInt(5));
 * const ct2 = await ctx.encrypt(BigInt(3));
 * const sum = await ctx.add(ct1, ct2);
 * const result = await ctx.decrypt(sum); // 8n
 * ctx.dispose();
 * ```
 */
export class FHEContext {
  private engine: FHEEngine;
  private sk: SecretKey;
  private pk: PublicKey;
  private ek?: EvaluationKey;
  private bk?: BootstrapKey;
  private thresholdKeys?: ThresholdKeys;
  private disposed = false;

  private constructor(
    engine: FHEEngine,
    sk: SecretKey,
    pk: PublicKey,
    ek: EvaluationKey | undefined,
    bk: BootstrapKey | undefined,
    thresholdKeys: ThresholdKeys | undefined
  ) {
    this.engine = engine;
    this.sk = sk;
    this.pk = pk;
    if (ek !== undefined) {
      this.ek = ek;
    }
    if (bk !== undefined) {
      this.bk = bk;
    }
    if (thresholdKeys !== undefined) {
      this.thresholdKeys = thresholdKeys;
    }
  }

  /**
   * Create a new FHE context with the specified parameters
   */
  static async create(
    params: ParameterPreset | CustomParameters,
    options: FHEContextOptions = {}
  ): Promise<FHEContext> {
    const engine = await createFHEEngine(params);
    const sk = await engine.generateSecretKey();
    const pk = await engine.generatePublicKey(sk);

    let ek: EvaluationKey | undefined;
    let bk: BootstrapKey | undefined;
    let thresholdKeys: ThresholdKeys | undefined;

    if (options.generateEvalKey !== false) {
      ek = await engine.generateEvalKey(sk);
    }

    if (options.generateBootstrapKey === true) {
      bk = await engine.generateBootstrapKey(sk);
    }

    if (options.thresholdConfig) {
      thresholdKeys = await engine.generateThresholdKeys(options.thresholdConfig);
    }

    return new FHEContext(engine, sk, pk, ek, bk, thresholdKeys);
  }

  private checkDisposed(): void {
    if (this.disposed) {
      throw new Error('FHEContext has been disposed');
    }
  }

  // ========================================================================
  // Encryption/Decryption
  // ========================================================================

  /** Encrypt a single value */
  async encrypt(value: bigint): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.encryptValue(value, this.pk);
  }

  /** Encrypt multiple values with SIMD packing */
  async encryptPacked(values: bigint[]): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.encryptPacked(values, this.pk);
  }

  /** Decrypt a ciphertext to a single value */
  async decrypt(ct: Ciphertext): Promise<bigint> {
    this.checkDisposed();
    return this.engine.decryptValue(ct, this.sk);
  }

  /** Decrypt a packed ciphertext */
  async decryptPacked(ct: Ciphertext, numValues: number): Promise<bigint[]> {
    this.checkDisposed();
    return this.engine.decryptPacked(ct, this.sk, numValues);
  }

  // ========================================================================
  // Homomorphic Operations
  // ========================================================================

  /** Add two ciphertexts */
  async add(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.add(ct1, ct2);
  }

  /** Add a scalar to a ciphertext */
  async addScalar(ct: Ciphertext, value: bigint): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.addScalar(ct, value);
  }

  /** Subtract two ciphertexts */
  async subtract(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.subtract(ct1, ct2);
  }

  /** Negate a ciphertext */
  async negate(ct: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.negate(ct);
  }

  /** Multiply two ciphertexts (with automatic relinearization) */
  async multiply(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    if (this.ek) {
      return this.engine.multiplyRelin(ct1, ct2, this.ek);
    }
    return this.engine.multiply(ct1, ct2);
  }

  /** Multiply a ciphertext by a scalar */
  async multiplyScalar(ct: Ciphertext, scalar: bigint): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.multiplyScalar(ct, scalar);
  }

  /** Square a ciphertext (with automatic relinearization) */
  async square(ct: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    if (this.ek) {
      return this.engine.squareRelin(ct, this.ek);
    }
    return this.engine.square(ct);
  }

  /** Sum multiple ciphertexts */
  async sum(cts: Ciphertext[], progress?: ProgressCallback): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.batchAdd(cts, progress);
  }

  // ========================================================================
  // Bootstrapping
  // ========================================================================

  /** Bootstrap a ciphertext to refresh noise */
  async bootstrap(ct: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    if (!this.bk) {
      throw new Error(
        'Bootstrap key not generated. Create context with generateBootstrapKey: true'
      );
    }
    return this.engine.bootstrap(ct, this.bk);
  }

  /** Programmable bootstrapping with lookup table */
  async programmableBootstrap(ct: Ciphertext, lookupTable: bigint[]): Promise<Ciphertext> {
    this.checkDisposed();
    if (!this.bk) {
      throw new Error(
        'Bootstrap key not generated. Create context with generateBootstrapKey: true'
      );
    }
    return this.engine.programmableBootstrap(ct, this.bk, lookupTable);
  }

  // ========================================================================
  // Utilities
  // ========================================================================

  /** Get the encryption of zero */
  async zero(): Promise<Ciphertext> {
    this.checkDisposed();
    return this.engine.getZeroCiphertext(this.pk);
  }

  /** Get the remaining noise budget of a ciphertext */
  async getNoiseBudget(ct: Ciphertext): Promise<number> {
    this.checkDisposed();
    return this.engine.getNoiseBudget(ct, this.sk);
  }

  /** Estimate noise budget without secret key */
  estimateNoiseBudget(ct: Ciphertext): number {
    this.checkDisposed();
    return this.engine.estimateNoiseBudget(ct);
  }

  /** Get the parameter set */
  getParams(): ParameterSet {
    this.checkDisposed();
    return this.engine.getParams();
  }

  /** Get hardware capabilities */
  getHardwareCapabilities(): HardwareCapabilities {
    this.checkDisposed();
    return this.engine.getHardwareCapabilities();
  }

  /** Get the public key */
  getPublicKey(): PublicKey {
    this.checkDisposed();
    return this.pk;
  }

  /** Get the evaluation key (if generated) */
  getEvaluationKey(): EvaluationKey | undefined {
    this.checkDisposed();
    return this.ek;
  }

  /** Get the bootstrap key (if generated) */
  getBootstrapKey(): BootstrapKey | undefined {
    this.checkDisposed();
    return this.bk;
  }

  /** Get threshold keys (if generated) */
  getThresholdKeys(): ThresholdKeys | undefined {
    this.checkDisposed();
    return this.thresholdKeys;
  }

  /** Get the underlying FHE engine */
  getEngine(): FHEEngine {
    this.checkDisposed();
    return this.engine;
  }

  /** Dispose of the context and free resources */
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
      this.engine.dispose();
    }
  }
}

// ============================================================================
// Parameter Preset Helpers
// ============================================================================

/**
 * Create a fast TFHE context with 128-bit security
 * Optimized for low-latency bootstrapping
 */
export async function createFastContext(options?: FHEContextOptions): Promise<FHEContext> {
  return FHEContext.create('tfhe-128-fast', options);
}

/**
 * Create a balanced TFHE context with 128-bit security
 * Good balance between performance and noise budget
 */
export async function createBalancedContext(options?: FHEContextOptions): Promise<FHEContext> {
  return FHEContext.create('tfhe-128-balanced', options);
}

/**
 * Create a secure TFHE context with 256-bit security
 * Maximum security for sensitive applications
 */
export async function createSecureContext(options?: FHEContextOptions): Promise<FHEContext> {
  return FHEContext.create('tfhe-256-secure', options);
}

/**
 * Create a BFV context with SIMD packing for 128-bit security
 * Optimized for batch operations on integers
 */
export async function createSIMDContext(options?: FHEContextOptions): Promise<FHEContext> {
  return FHEContext.create('bfv-128-simd', options);
}

/**
 * Create a CKKS context for ML workloads with 128-bit security
 * Optimized for approximate arithmetic on real numbers
 */
export async function createMLContext(options?: FHEContextOptions): Promise<FHEContext> {
  return FHEContext.create('ckks-128-ml', options);
}

/**
 * Create a voting-optimized TFHE context
 * Optimized for encrypted voting applications
 */
export async function createVotingContext(options?: FHEContextOptions): Promise<FHEContext> {
  return FHEContext.create('tfhe-128-voting', options);
}
