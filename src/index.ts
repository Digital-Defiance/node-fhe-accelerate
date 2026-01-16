/**
 * @digitaldefiance/node-fhe-accelerate
 * 
 * High-performance Fully Homomorphic Encryption (FHE) acceleration library
 * optimized for Apple M4 Max hardware.
 * 
 * This library provides hardware-accelerated FHE operations targeting sub-100ms
 * latency for typical operations, making privacy-preserving computation feel as
 * fast as standard database calls.
 * 
 * @module @digitaldefiance/node-fhe-accelerate
 */

// Placeholder exports - will be implemented in subsequent tasks
export const version = '0.1.0';

// Type definitions for opaque handles to native objects
export type SecretKey = { readonly __brand: 'SecretKey'; readonly handle: bigint };
export type PublicKey = { readonly __brand: 'PublicKey'; readonly handle: bigint };
export type EvaluationKey = { readonly __brand: 'EvaluationKey'; readonly handle: bigint };
export type BootstrapKey = { readonly __brand: 'BootstrapKey'; readonly handle: bigint };
export type Ciphertext = { readonly __brand: 'Ciphertext'; readonly handle: bigint };
export type Plaintext = { readonly __brand: 'Plaintext'; readonly handle: bigint };

// Parameter configuration types
export type SecurityLevel = 128 | 192 | 256;

export type ParameterPreset = 
    | 'tfhe-128-fast'
    | 'tfhe-128-balanced'
    | 'tfhe-256-secure'
    | 'bfv-128-simd'
    | 'ckks-128-ml';

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

// Error types
export enum FHEErrorCode {
    NOISE_BUDGET_EXHAUSTED = 'NOISE_BUDGET_EXHAUSTED',
    INVALID_PARAMETERS = 'INVALID_PARAMETERS',
    KEY_MISMATCH = 'KEY_MISMATCH',
    HARDWARE_UNAVAILABLE = 'HARDWARE_UNAVAILABLE',
    SERIALIZATION_ERROR = 'SERIALIZATION_ERROR',
}

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

// Streaming interface
export interface CiphertextStream extends AsyncIterable<Ciphertext> {
    pipe(transform: (ct: Ciphertext) => Promise<Ciphertext>): CiphertextStream;
    collect(): Promise<Ciphertext[]>;
}

// Progress callback
export type ProgressCallback = (progress: {
    stage: string;
    current: number;
    total: number;
    elapsedMs: number;
}) => void;

// Main FHE Engine interface (to be implemented)
export interface FHEEngine {
    // Key generation
    generateSecretKey(): Promise<SecretKey>;
    generatePublicKey(sk: SecretKey): Promise<PublicKey>;
    generateEvalKey(sk: SecretKey, decompBase?: number): Promise<EvaluationKey>;
    generateBootstrapKey(sk: SecretKey): Promise<BootstrapKey>;
    
    // Encryption/Decryption
    encrypt(plaintext: Plaintext, pk: PublicKey): Promise<Ciphertext>;
    decrypt(ciphertext: Ciphertext, sk: SecretKey): Promise<Plaintext>;
    
    // Homomorphic operations
    add(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext>;
    addPlain(ct: Ciphertext, pt: Plaintext): Promise<Ciphertext>;
    multiply(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext>;
    multiplyPlain(ct: Ciphertext, pt: Plaintext): Promise<Ciphertext>;
    
    // Maintenance
    relinearize(ct: Ciphertext, ek: EvaluationKey): Promise<Ciphertext>;
    bootstrap(ct: Ciphertext, bk: BootstrapKey): Promise<Ciphertext>;
    
    // Utilities
    getNoiseBudget(ct: Ciphertext, sk: SecretKey): Promise<number>;
}

// Factory function (to be implemented)
export function createEngine(_params: ParameterPreset | CustomParameters): Promise<FHEEngine> {
    return Promise.reject(new Error('Not yet implemented - native addon required'));
}

// Re-export API modules
export * from './api/types';
export * from './api/voting-types';
export * from './api/zk-types';
export * from './api/zk-proofs';
export * from './api/audit-trail';
export * from './api/fhe-context';
export * from './api/tally-streaming';
export * from './api/voting-example';

// Re-export verification tools
export * from './verification';
