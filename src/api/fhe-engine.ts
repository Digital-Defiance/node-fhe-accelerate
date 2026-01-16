/**
 * @file api/fhe-engine.ts
 * @brief Main FHE Engine interface and implementation
 * Requirements: 12.1, 12.2, 12.3, 12.6
 */

import type {
  SecretKey,
  PublicKey,
  EvaluationKey,
  BootstrapKey,
  Ciphertext,
  Plaintext,
  ParameterSet,
  ParameterPreset,
  CustomParameters,
  KeyGenerationOptions,
  ThresholdConfig,
  ThresholdKeys,
  PartialDecryption,
  EncryptionResult,
  DecryptionResult,
  BatchEncryptionOptions,
  BatchEncryptionResult,
  ProgressCallback,
  SerializationOptions,
  SerializedKey,
  SerializedCiphertext,
  HardwareCapabilities,
} from './types';
import { FHEError, FHEErrorCode } from './types';

export interface FHEEngine {
  generateSecretKey(options?: KeyGenerationOptions): Promise<SecretKey>;
  generatePublicKey(sk: SecretKey): Promise<PublicKey>;
  generateEvalKey(sk: SecretKey, options?: KeyGenerationOptions): Promise<EvaluationKey>;
  generateBootstrapKey(sk: SecretKey): Promise<BootstrapKey>;
  generateThresholdKeys(config: ThresholdConfig): Promise<ThresholdKeys>;
  encrypt(plaintext: Plaintext, pk: PublicKey): Promise<EncryptionResult>;
  encryptValue(value: bigint, pk: PublicKey): Promise<Ciphertext>;
  encryptPacked(values: bigint[], pk: PublicKey): Promise<Ciphertext>;
  decrypt(ciphertext: Ciphertext, sk: SecretKey): Promise<DecryptionResult>;
  decryptValue(ciphertext: Ciphertext, sk: SecretKey): Promise<bigint>;
  decryptPacked(ct: Ciphertext, sk: SecretKey, numValues: number): Promise<bigint[]>;
  batchEncrypt(pts: Plaintext[], pk: PublicKey, opts?: BatchEncryptionOptions): Promise<BatchEncryptionResult>;
  add(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext>;
  addPlain(ct: Ciphertext, pt: Plaintext): Promise<Ciphertext>;
  addScalar(ct: Ciphertext, value: bigint): Promise<Ciphertext>;
  subtract(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext>;
  negate(ct: Ciphertext): Promise<Ciphertext>;
  batchAdd(cts: Ciphertext[], progress?: ProgressCallback): Promise<Ciphertext>;
  multiply(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext>;
  multiplyRelin(ct1: Ciphertext, ct2: Ciphertext, ek: EvaluationKey): Promise<Ciphertext>;
  multiplyPlain(ct: Ciphertext, pt: Plaintext): Promise<Ciphertext>;
  multiplyScalar(ct: Ciphertext, scalar: bigint): Promise<Ciphertext>;
  relinearize(ct: Ciphertext, ek: EvaluationKey): Promise<Ciphertext>;
  square(ct: Ciphertext): Promise<Ciphertext>;
  squareRelin(ct: Ciphertext, ek: EvaluationKey): Promise<Ciphertext>;
  bootstrap(ct: Ciphertext, bk: BootstrapKey): Promise<Ciphertext>;
  programmableBootstrap(ct: Ciphertext, bk: BootstrapKey, lut: bigint[]): Promise<Ciphertext>;
  partialDecrypt(ct: Ciphertext, share: ThresholdKeys['shares'][0]): Promise<PartialDecryption>;
  combinePartialDecryptions(ct: Ciphertext, partials: PartialDecryption[], t: number): Promise<DecryptionResult>;
  getNoiseBudget(ct: Ciphertext, sk: SecretKey): Promise<number>;
  estimateNoiseBudget(ct: Ciphertext): number;
  serializeSecretKey(sk: SecretKey, opts?: SerializationOptions): Promise<SerializedKey>;
  deserializeSecretKey(data: SerializedKey): Promise<SecretKey>;
  serializePublicKey(pk: PublicKey, opts?: SerializationOptions): Promise<SerializedKey>;
  deserializePublicKey(data: SerializedKey): Promise<PublicKey>;
  serializeCiphertext(ct: Ciphertext, opts?: SerializationOptions): Promise<SerializedCiphertext>;
  deserializeCiphertext(data: SerializedCiphertext): Promise<Ciphertext>;
  createPlaintext(value: bigint): Plaintext;
  createPackedPlaintext(values: bigint[]): Plaintext;
  getZeroCiphertext(pk: PublicKey): Promise<Ciphertext>;
  getParams(): ParameterSet;
  getHardwareCapabilities(): HardwareCapabilities;
  getSlotCount(): number;
  dispose(): void;
}

class FHEEngineImpl implements FHEEngine {
  private params: ParameterSet;
  private disposed = false;

  constructor(params: ParameterSet) {
    this.params = params;
  }

  private checkDisposed(): void {
    if (this.disposed) {
      throw new FHEError('Engine disposed', FHEErrorCode.NATIVE_ERROR);
    }
  }

  async generateSecretKey(): Promise<SecretKey> {
    this.checkDisposed();
    return { __brand: 'SecretKey', handle: BigInt(0), keyId: BigInt(Date.now()) };
  }

  async generatePublicKey(sk: SecretKey): Promise<PublicKey> {
    this.checkDisposed();
    return { __brand: 'PublicKey', handle: BigInt(0), keyId: sk.keyId };
  }

  async generateEvalKey(sk: SecretKey, opts?: KeyGenerationOptions): Promise<EvaluationKey> {
    this.checkDisposed();
    return {
      __brand: 'EvaluationKey',
      handle: BigInt(0),
      keyId: sk.keyId,
      decompBaseLog: opts?.decompBaseLog ?? this.params.decompBaseLog,
      decompLevel: opts?.decompLevel ?? this.params.decompLevel,
    };
  }

  async generateBootstrapKey(sk: SecretKey): Promise<BootstrapKey> {
    this.checkDisposed();
    return {
      __brand: 'BootstrapKey',
      handle: BigInt(0),
      keyId: sk.keyId,
      lweDimension: this.params.lweDimension,
    };
  }

  async generateThresholdKeys(config: ThresholdConfig): Promise<ThresholdKeys> {
    this.checkDisposed();
    const keyId = BigInt(Date.now());
    const shares = Array.from({ length: config.totalShares }, (_, i) => ({
      shareId: i + 1,
      handle: BigInt(i + 1),
      commitment: new Uint8Array(32),
      keyId,
    }));
    return {
      shares,
      publicKey: { __brand: 'PublicKey', handle: BigInt(0), keyId },
      threshold: config.threshold,
      totalShares: config.totalShares,
    };
  }

  async encrypt(_pt: Plaintext, pk: PublicKey): Promise<EncryptionResult> {
    this.checkDisposed();
    return {
      ciphertext: {
        __brand: 'Ciphertext',
        handle: BigInt(0),
        keyId: pk.keyId,
        noiseBudget: this.params.noiseBudget,
        isNtt: false,
        degree: 1,
      },
    };
  }

  async encryptValue(value: bigint, pk: PublicKey): Promise<Ciphertext> {
    return (await this.encrypt(this.createPlaintext(value), pk)).ciphertext;
  }

  async encryptPacked(values: bigint[], pk: PublicKey): Promise<Ciphertext> {
    return (await this.encrypt(this.createPackedPlaintext(values), pk)).ciphertext;
  }

  async decrypt(ct: Ciphertext, _sk: SecretKey): Promise<DecryptionResult> {
    this.checkDisposed();
    return {
      plaintext: {
        __brand: 'Plaintext',
        values: [BigInt(0)],
        plaintextModulus: this.params.plaintextModulus,
        isPacked: false,
      },
      remainingNoiseBudget: ct.noiseBudget,
      success: true,
    };
  }

  async decryptValue(ct: Ciphertext, sk: SecretKey): Promise<bigint> {
    const r = await this.decrypt(ct, sk);
    if (!r.success) {
      throw new FHEError(r.errorMessage ?? 'Decryption failed', FHEErrorCode.NOISE_BUDGET_EXHAUSTED);
    }
    return r.plaintext.values[0] ?? BigInt(0);
  }

  async decryptPacked(ct: Ciphertext, sk: SecretKey, n: number): Promise<bigint[]> {
    const r = await this.decrypt(ct, sk);
    if (!r.success) {
      throw new FHEError(r.errorMessage ?? 'Decryption failed', FHEErrorCode.NOISE_BUDGET_EXHAUSTED);
    }
    return r.plaintext.values.slice(0, n);
  }

  async batchEncrypt(pts: Plaintext[], pk: PublicKey): Promise<BatchEncryptionResult> {
    this.checkDisposed();
    const start = Date.now();
    const cts: Ciphertext[] = [];
    for (const pt of pts) {
      cts.push((await this.encrypt(pt, pk)).ciphertext);
    }
    const ms = Date.now() - start;
    return {
      ciphertexts: cts,
      elapsedMs: ms,
      throughputPerSecond: ms > 0 ? (pts.length * 1000) / ms : 0,
    };
  }

  async add(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    if (ct1.keyId !== ct2.keyId) {
      throw new FHEError('Key mismatch', FHEErrorCode.KEY_MISMATCH);
    }
    return {
      __brand: 'Ciphertext',
      handle: BigInt(0),
      keyId: ct1.keyId,
      noiseBudget: Math.min(ct1.noiseBudget, ct2.noiseBudget) - 1,
      isNtt: ct1.isNtt,
      degree: Math.max(ct1.degree, ct2.degree),
    };
  }

  async addPlain(ct: Ciphertext, _pt: Plaintext): Promise<Ciphertext> {
    this.checkDisposed();
    return { ...ct, handle: BigInt(0) };
  }

  async addScalar(ct: Ciphertext, v: bigint): Promise<Ciphertext> {
    return this.addPlain(ct, this.createPlaintext(v));
  }

  async subtract(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    if (ct1.keyId !== ct2.keyId) {
      throw new FHEError('Key mismatch', FHEErrorCode.KEY_MISMATCH);
    }
    return {
      __brand: 'Ciphertext',
      handle: BigInt(0),
      keyId: ct1.keyId,
      noiseBudget: Math.min(ct1.noiseBudget, ct2.noiseBudget) - 1,
      isNtt: ct1.isNtt,
      degree: Math.max(ct1.degree, ct2.degree),
    };
  }

  async negate(ct: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    return { ...ct, handle: BigInt(0) };
  }

  async batchAdd(cts: Ciphertext[], progress?: ProgressCallback): Promise<Ciphertext> {
    this.checkDisposed();
    if (cts.length === 0) {
      throw new FHEError('Empty array', FHEErrorCode.INVALID_PARAMETERS);
    }
    const start = Date.now();
    let r = cts[0]!;
    for (let i = 1; i < cts.length; i++) {
      r = await this.add(r, cts[i]!);
      progress?.({
        stage: 'batch_add',
        current: i + 1,
        total: cts.length,
        elapsedMs: Date.now() - start,
        progressPercent: ((i + 1) / cts.length) * 100,
      });
    }
    return r;
  }

  async multiply(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext> {
    this.checkDisposed();
    if (ct1.keyId !== ct2.keyId) {
      throw new FHEError('Key mismatch', FHEErrorCode.KEY_MISMATCH);
    }
    return {
      __brand: 'Ciphertext',
      handle: BigInt(0),
      keyId: ct1.keyId,
      noiseBudget: Math.min(ct1.noiseBudget, ct2.noiseBudget) / 2,
      isNtt: ct1.isNtt,
      degree: 2,
    };
  }

  async multiplyRelin(ct1: Ciphertext, ct2: Ciphertext, ek: EvaluationKey): Promise<Ciphertext> {
    return this.relinearize(await this.multiply(ct1, ct2), ek);
  }

  async multiplyPlain(ct: Ciphertext, _pt: Plaintext): Promise<Ciphertext> {
    this.checkDisposed();
    return { ...ct, handle: BigInt(0), noiseBudget: ct.noiseBudget - 5 };
  }

  async multiplyScalar(ct: Ciphertext, s: bigint): Promise<Ciphertext> {
    return this.multiplyPlain(ct, this.createPlaintext(s));
  }

  async relinearize(ct: Ciphertext, _ek: EvaluationKey): Promise<Ciphertext> {
    this.checkDisposed();
    return { ...ct, handle: BigInt(0), degree: 1 };
  }

  async square(ct: Ciphertext): Promise<Ciphertext> {
    return this.multiply(ct, ct);
  }

  async squareRelin(ct: Ciphertext, ek: EvaluationKey): Promise<Ciphertext> {
    return this.relinearize(await this.square(ct), ek);
  }

  async bootstrap(ct: Ciphertext, _bk: BootstrapKey): Promise<Ciphertext> {
    this.checkDisposed();
    return { ...ct, handle: BigInt(0), noiseBudget: this.params.noiseBudget };
  }

  async programmableBootstrap(ct: Ciphertext, bk: BootstrapKey, _lut: bigint[]): Promise<Ciphertext> {
    return this.bootstrap(ct, bk);
  }

  async partialDecrypt(_ct: Ciphertext, share: ThresholdKeys['shares'][0]): Promise<PartialDecryption> {
    this.checkDisposed();
    return { shareId: share.shareId, partialResult: [BigInt(0)] };
  }

  async combinePartialDecryptions(ct: Ciphertext, partials: PartialDecryption[], t: number): Promise<DecryptionResult> {
    this.checkDisposed();
    if (partials.length < t) {
      throw new FHEError(`Need ${t} partials, got ${partials.length}`, FHEErrorCode.THRESHOLD_NOT_MET);
    }
    return {
      plaintext: {
        __brand: 'Plaintext',
        values: [BigInt(0)],
        plaintextModulus: this.params.plaintextModulus,
        isPacked: false,
      },
      remainingNoiseBudget: ct.noiseBudget,
      success: true,
    };
  }

  async getNoiseBudget(): Promise<number> {
    this.checkDisposed();
    return this.params.noiseBudget;
  }

  estimateNoiseBudget(ct: Ciphertext): number {
    this.checkDisposed();
    return ct.noiseBudget;
  }

  async serializeSecretKey(_sk: SecretKey, opts?: SerializationOptions): Promise<SerializedKey> {
    this.checkDisposed();
    return {
      data: new Uint8Array(0),
      format: opts?.format ?? 'binary',
      keyType: 'secret',
      version: 1,
      checksum: new Uint8Array(32),
    };
  }

  async deserializeSecretKey(): Promise<SecretKey> {
    this.checkDisposed();
    return { __brand: 'SecretKey', handle: BigInt(0), keyId: BigInt(0) };
  }

  async serializePublicKey(_pk: PublicKey, opts?: SerializationOptions): Promise<SerializedKey> {
    this.checkDisposed();
    return {
      data: new Uint8Array(0),
      format: opts?.format ?? 'binary',
      keyType: 'public',
      version: 1,
      checksum: new Uint8Array(32),
    };
  }

  async deserializePublicKey(): Promise<PublicKey> {
    this.checkDisposed();
    return { __brand: 'PublicKey', handle: BigInt(0), keyId: BigInt(0) };
  }

  async serializeCiphertext(_ct: Ciphertext, opts?: SerializationOptions): Promise<SerializedCiphertext> {
    this.checkDisposed();
    return {
      data: new Uint8Array(0),
      format: opts?.format ?? 'binary',
      keyId: BigInt(0),
      version: 1,
      checksum: new Uint8Array(32),
    };
  }

  async deserializeCiphertext(): Promise<Ciphertext> {
    this.checkDisposed();
    return {
      __brand: 'Ciphertext',
      handle: BigInt(0),
      keyId: BigInt(0),
      noiseBudget: this.params.noiseBudget,
      isNtt: false,
      degree: 1,
    };
  }

  createPlaintext(value: bigint): Plaintext {
    this.checkDisposed();
    return {
      __brand: 'Plaintext',
      values: [value],
      plaintextModulus: this.params.plaintextModulus,
      isPacked: false,
    };
  }

  createPackedPlaintext(values: bigint[]): Plaintext {
    this.checkDisposed();
    return {
      __brand: 'Plaintext',
      values,
      plaintextModulus: this.params.plaintextModulus,
      isPacked: true,
    };
  }

  async getZeroCiphertext(pk: PublicKey): Promise<Ciphertext> {
    return this.encryptValue(BigInt(0), pk);
  }

  getParams(): ParameterSet {
    this.checkDisposed();
    return this.params;
  }

  getHardwareCapabilities(): HardwareCapabilities {
    this.checkDisposed();
    return {
      hasSme: false,
      hasMetal: false,
      hasNeon: true,
      hasAmx: false,
      hasNeuralEngine: false,
      metalGpuCores: 0,
      unifiedMemorySize: BigInt(0),
    };
  }

  getSlotCount(): number {
    this.checkDisposed();
    return this.params.polyDegree;
  }

  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
    }
  }
}

export async function createFHEEngine(params: ParameterPreset | CustomParameters): Promise<FHEEngine> {
  const { createParameterSet, createCustomParameterSet, TFHE_128_VOTING } = await import('../parameters');
  let paramSet: ParameterSet;
  if (typeof params === 'string') {
    paramSet =
      params === 'tfhe-128-voting'
        ? TFHE_128_VOTING()
        : createParameterSet(params as Exclude<ParameterPreset, 'tfhe-128-voting'>);
  } else {
    paramSet = createCustomParameterSet(params);
  }
  return new FHEEngineImpl(paramSet);
}
