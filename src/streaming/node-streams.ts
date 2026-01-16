/**
 * Node.js Stream Interfaces for FHE Ciphertext Streaming
 * 
 * This module provides Node.js Readable/Writable/Transform stream interfaces
 * for ciphertext operations, enabling pipe() for operation chaining.
 * 
 * Requirements: 11.3, 12.4
 */

import { Readable, Writable, Transform, TransformCallback } from 'stream';
import type { Ciphertext, Plaintext, PublicKey } from '../index';
import type { StreamingProgressCallback, StreamingConfig } from './index';
import { DEFAULT_STREAMING_CONFIG } from './index';

/**
 * Options for ciphertext streams
 */
export interface CiphertextStreamOptions {
    config?: Partial<StreamingConfig>;
    progress?: StreamingProgressCallback;
    highWaterMark?: number;
}

/**
 * Readable stream that emits ciphertexts
 */
export class CiphertextReadableStream extends Readable {
    private ciphertexts: Ciphertext[];
    private index: number = 0;
    private _config: StreamingConfig;
    private progressCallback: StreamingProgressCallback | undefined;
    private startTime: number;
    
    constructor(ciphertexts: Ciphertext[], options: CiphertextStreamOptions = {}) {
        super({
            objectMode: true,
            highWaterMark: options.highWaterMark ?? 16
        });
        this.ciphertexts = ciphertexts;
        this._config = { ...DEFAULT_STREAMING_CONFIG, ...options.config };
        this.progressCallback = options.progress;
        this.startTime = Date.now();
    }
    
    override _read(): void {
        if (this.index >= this.ciphertexts.length) {
            this.push(null);
            return;
        }
        
        const ct = this.ciphertexts[this.index++];
        
        if (this.progressCallback) {
            const elapsed = Date.now() - this.startTime;
            const estimatedRemaining = this.index > 0 
                ? (elapsed / this.index) * (this.ciphertexts.length - this.index)
                : 0;
            
            this.progressCallback({
                stage: 'reading',
                current: this.index,
                total: this.ciphertexts.length,
                elapsedMs: elapsed,
                estimatedRemainingMs: estimatedRemaining,
                progressPercent: (this.index / this.ciphertexts.length) * 100
            });
        }
        
        this.push(ct);
    }
    
    getProcessedCount(): number {
        return this.index;
    }
    
    getConfig(): StreamingConfig {
        return this._config;
    }
}

/**
 * Writable stream that collects ciphertexts
 */
export class CiphertextWritableStream extends Writable {
    private collected: Ciphertext[] = [];
    private _config: StreamingConfig;
    private progressCallback: StreamingProgressCallback | undefined;
    private startTime: number;
    
    constructor(options: CiphertextStreamOptions = {}) {
        super({
            objectMode: true,
            highWaterMark: options.highWaterMark ?? 16
        });
        this._config = { ...DEFAULT_STREAMING_CONFIG, ...options.config };
        this.progressCallback = options.progress;
        this.startTime = Date.now();
    }
    
    override _write(chunk: Ciphertext, _encoding: string, callback: (error?: Error | null) => void): void {
        this.collected.push(chunk);
        
        if (this.progressCallback) {
            const elapsed = Date.now() - this.startTime;
            this.progressCallback({
                stage: 'writing',
                current: this.collected.length,
                total: this.collected.length,
                elapsedMs: elapsed,
                estimatedRemainingMs: 0,
                progressPercent: 0
            });
        }
        
        callback();
    }
    
    getCollected(): Ciphertext[] {
        return this.collected;
    }
    
    getProcessedCount(): number {
        return this.collected.length;
    }
    
    getConfig(): StreamingConfig {
        return this._config;
    }
}

/**
 * Transform stream that applies a function to each ciphertext
 */
export class CiphertextTransformStream extends Transform {
    private transformFn: (ct: Ciphertext) => Promise<Ciphertext>;
    private _config: StreamingConfig;
    private progressCallback: StreamingProgressCallback | undefined;
    private processedCount: number = 0;
    private startTime: number;
    
    constructor(
        transformFn: (ct: Ciphertext) => Promise<Ciphertext>,
        options: CiphertextStreamOptions = {}
    ) {
        super({
            objectMode: true,
            highWaterMark: options.highWaterMark ?? 16
        });
        this.transformFn = transformFn;
        this._config = { ...DEFAULT_STREAMING_CONFIG, ...options.config };
        this.progressCallback = options.progress;
        this.startTime = Date.now();
    }
    
    override async _transform(
        chunk: Ciphertext,
        _encoding: string,
        callback: TransformCallback
    ): Promise<void> {
        try {
            const transformed = await this.transformFn(chunk);
            this.processedCount++;
            
            if (this.progressCallback) {
                const elapsed = Date.now() - this.startTime;
                this.progressCallback({
                    stage: 'transforming',
                    current: this.processedCount,
                    total: this.processedCount,
                    elapsedMs: elapsed,
                    estimatedRemainingMs: 0,
                    progressPercent: 0
                });
            }
            
            callback(null, transformed);
        } catch (error) {
            callback(error instanceof Error ? error : new Error(String(error)));
        }
    }
    
    getProcessedCount(): number {
        return this.processedCount;
    }
    
    getConfig(): StreamingConfig {
        return this._config;
    }
}

/**
 * Transform stream that adds ciphertexts to produce running totals
 */
export class CiphertextAdditionStream extends Transform {
    private addFn: (ct1: Ciphertext, ct2: Ciphertext) => Promise<Ciphertext>;
    private accumulator?: Ciphertext;
    private _config: StreamingConfig;
    private progressCallback: StreamingProgressCallback | undefined;
    private processedCount: number = 0;
    private startTime: number;
    private emitRunningTotals: boolean;
    
    constructor(
        addFn: (ct1: Ciphertext, ct2: Ciphertext) => Promise<Ciphertext>,
        options: CiphertextStreamOptions & { emitRunningTotals?: boolean } = {}
    ) {
        super({
            objectMode: true,
            highWaterMark: options.highWaterMark ?? 16
        });
        this.addFn = addFn;
        this._config = { ...DEFAULT_STREAMING_CONFIG, ...options.config };
        this.progressCallback = options.progress;
        this.startTime = Date.now();
        this.emitRunningTotals = options.emitRunningTotals ?? false;
    }
    
    override async _transform(
        chunk: Ciphertext,
        _encoding: string,
        callback: TransformCallback
    ): Promise<void> {
        try {
            if (!this.accumulator) {
                this.accumulator = chunk;
            } else {
                this.accumulator = await this.addFn(this.accumulator, chunk);
            }
            
            this.processedCount++;
            
            if (this.progressCallback) {
                const elapsed = Date.now() - this.startTime;
                this.progressCallback({
                    stage: 'adding',
                    current: this.processedCount,
                    total: this.processedCount,
                    elapsedMs: elapsed,
                    estimatedRemainingMs: 0,
                    progressPercent: 0
                });
            }
            
            // Emit running total if configured, otherwise just continue
            if (this.emitRunningTotals) {
                callback(null, this.accumulator);
            } else {
                callback();
            }
        } catch (error) {
            callback(error instanceof Error ? error : new Error(String(error)));
        }
    }
    
    override _flush(callback: TransformCallback): void {
        // Emit final accumulator if not emitting running totals
        if (!this.emitRunningTotals && this.accumulator) {
            callback(null, this.accumulator);
        } else {
            callback();
        }
    }
    
    getAccumulator(): Ciphertext | undefined {
        return this.accumulator;
    }
    
    getProcessedCount(): number {
        return this.processedCount;
    }
    
    getConfig(): StreamingConfig {
        return this._config;
    }
}

/**
 * Transform stream that encrypts plaintexts to ciphertexts
 */
export class EncryptionStream extends Transform {
    private encryptFn: (pt: Plaintext, pk: PublicKey) => Promise<Ciphertext>;
    private publicKey: PublicKey;
    private _config: StreamingConfig;
    private progressCallback: StreamingProgressCallback | undefined;
    private processedCount: number = 0;
    private startTime: number;
    
    constructor(
        encryptFn: (pt: Plaintext, pk: PublicKey) => Promise<Ciphertext>,
        publicKey: PublicKey,
        options: CiphertextStreamOptions = {}
    ) {
        super({
            objectMode: true,
            highWaterMark: options.highWaterMark ?? 16
        });
        this.encryptFn = encryptFn;
        this.publicKey = publicKey;
        this._config = { ...DEFAULT_STREAMING_CONFIG, ...options.config };
        this.progressCallback = options.progress;
        this.startTime = Date.now();
    }
    
    override async _transform(
        chunk: Plaintext,
        _encoding: string,
        callback: TransformCallback
    ): Promise<void> {
        try {
            const encrypted = await this.encryptFn(chunk, this.publicKey);
            this.processedCount++;
            
            if (this.progressCallback) {
                const elapsed = Date.now() - this.startTime;
                this.progressCallback({
                    stage: 'encrypting',
                    current: this.processedCount,
                    total: this.processedCount,
                    elapsedMs: elapsed,
                    estimatedRemainingMs: 0,
                    progressPercent: 0
                });
            }
            
            callback(null, encrypted);
        } catch (error) {
            callback(error instanceof Error ? error : new Error(String(error)));
        }
    }
    
    getProcessedCount(): number {
        return this.processedCount;
    }
    
    getConfig(): StreamingConfig {
        return this._config;
    }
}

/**
 * Utility to convert async iterable to Node.js Readable stream
 */
export function asyncIterableToReadable<T>(
    iterable: AsyncIterable<T>
): Readable {
    return Readable.from(iterable, { objectMode: true });
}

/**
 * Utility to convert Node.js Readable stream to async iterable
 */
export async function* readableToAsyncIterable<T>(
    readable: Readable
): AsyncGenerator<T> {
    for await (const chunk of readable) {
        yield chunk as T;
    }
}
