/**
 * Streaming Operations for FHE Ciphertexts
 * 
 * This module provides streaming interfaces for processing large ciphertexts
 * in chunks, maintaining correctness across chunk boundaries, and supporting
 * async iteration with progress callbacks.
 * 
 * Requirements: 11.1, 11.2, 11.3, 11.6
 */

import type { Ciphertext } from '../index';

/**
 * Chunk metadata for tracking processing state
 */
export interface ChunkMetadata {
    chunkIndex: number;
    totalChunks: number;
    startOffset: number;
    chunkSize: number;
    isFirst: boolean;
    isLast: boolean;
}

/**
 * Ciphertext chunk for streaming operations
 */
export interface CiphertextChunk {
    data: Ciphertext;
    metadata: ChunkMetadata;
    leftBoundary?: bigint[];
    rightBoundary?: bigint[];
}

/**
 * Progress information for streaming operations
 */
export interface StreamingProgress {
    stage: string;
    current: number;
    total: number;
    elapsedMs: number;
    estimatedRemainingMs: number;
    progressPercent: number;
}

/**
 * Progress callback type
 */
export type StreamingProgressCallback = (progress: StreamingProgress) => void;

/**
 * Streaming configuration options
 */
export interface StreamingConfig {
    chunkSize: number;
    boundaryOverlap: number;
    maxMemoryBytes: number;
    prefetchChunks: number;
    useMemoryMapping: boolean;
    enableCompression: boolean;
}

/**
 * Default streaming configuration
 */
export const DEFAULT_STREAMING_CONFIG: StreamingConfig = {
    chunkSize: 4096,
    boundaryOverlap: 64,
    maxMemoryBytes: 256 * 1024 * 1024, // 256 MB
    prefetchChunks: 2,
    useMemoryMapping: true,
    enableCompression: false,
};

/**
 * Result of a streaming operation
 */
export interface StreamingResult {
    success: boolean;
    errorMessage?: string;
    chunksProcessed: number;
    totalTimeMs: number;
    throughputChunksPerSec: number;
}

/**
 * Async iterable ciphertext stream interface
 */
export interface CiphertextStream extends AsyncIterable<Ciphertext> {
    /**
     * Apply a transformation to each ciphertext in the stream
     */
    pipe(transform: (ct: Ciphertext) => Promise<Ciphertext>): CiphertextStream;
    
    /**
     * Collect all ciphertexts from the stream into an array
     */
    collect(): Promise<Ciphertext[]>;
    
    /**
     * Get the number of items processed so far
     */
    getProcessedCount(): number;
    
    /**
     * Cancel the stream processing
     */
    cancel(): void;
    
    /**
     * Async iterator implementation
     */
    [Symbol.asyncIterator](): AsyncIterator<Ciphertext>;
}

/**
 * Create a ciphertext stream from an array
 */
export function createCiphertextStream(
    ciphertexts: Ciphertext[],
    config: Partial<StreamingConfig> = {}
): CiphertextStream {
    const fullConfig = { ...DEFAULT_STREAMING_CONFIG, ...config };
    let index = 0;
    let cancelled = false;
    let processedCount = 0;
    
    const stream: CiphertextStream = {
        [Symbol.asyncIterator](): AsyncIterator<Ciphertext> {
            return {
                async next(): Promise<IteratorResult<Ciphertext>> {
                    if (index < ciphertexts.length && !cancelled) {
                        const ct = ciphertexts[index++];
                        if (ct !== undefined) {
                            processedCount++;
                            return { value: ct, done: false };
                        }
                    }
                    return { value: undefined as unknown as Ciphertext, done: true };
                }
            };
        },
        
        pipe(transform: (ct: Ciphertext) => Promise<Ciphertext>): CiphertextStream {
            return createTransformedStream(stream, transform, fullConfig);
        },
        
        async collect(): Promise<Ciphertext[]> {
            const results: Ciphertext[] = [];
            for await (const ct of stream) {
                results.push(ct);
            }
            return results;
        },
        
        getProcessedCount(): number {
            return processedCount;
        },
        
        cancel(): void {
            cancelled = true;
        }
    };
    
    return stream;
}

/**
 * Create a transformed ciphertext stream
 */
function createTransformedStream(
    source: CiphertextStream,
    transform: (ct: Ciphertext) => Promise<Ciphertext>,
    config: StreamingConfig
): CiphertextStream {
    let cancelled = false;
    let processedCount = 0;
    
    const stream: CiphertextStream = {
        [Symbol.asyncIterator](): AsyncIterator<Ciphertext> {
            const sourceIterator = source[Symbol.asyncIterator]();
            return {
                async next(): Promise<IteratorResult<Ciphertext>> {
                    if (cancelled) {
                        return { value: undefined as unknown as Ciphertext, done: true };
                    }
                    const result = await sourceIterator.next();
                    if (result.done) {
                        return { value: undefined as unknown as Ciphertext, done: true };
                    }
                    const transformed = await transform(result.value);
                    processedCount++;
                    return { value: transformed, done: false };
                }
            };
        },
        
        pipe(nextTransform: (ct: Ciphertext) => Promise<Ciphertext>): CiphertextStream {
            return createTransformedStream(stream, nextTransform, config);
        },
        
        async collect(): Promise<Ciphertext[]> {
            const results: Ciphertext[] = [];
            for await (const ct of stream) {
                results.push(ct);
            }
            return results;
        },
        
        getProcessedCount(): number {
            return processedCount;
        },
        
        cancel(): void {
            cancelled = true;
            source.cancel();
        }
    };
    
    return stream;
}

/**
 * Create a ciphertext stream from an async generator
 */
export function createAsyncCiphertextStream(
    generator: () => AsyncGenerator<Ciphertext>,
    config: Partial<StreamingConfig> = {}
): CiphertextStream {
    const fullConfig = { ...DEFAULT_STREAMING_CONFIG, ...config };
    let cancelled = false;
    let processedCount = 0;
    let gen: AsyncGenerator<Ciphertext> | null = null;
    
    const stream: CiphertextStream = {
        [Symbol.asyncIterator](): AsyncIterator<Ciphertext> {
            gen = generator();
            return {
                async next(): Promise<IteratorResult<Ciphertext>> {
                    if (cancelled || !gen) {
                        return { value: undefined as unknown as Ciphertext, done: true };
                    }
                    const result = await gen.next();
                    if (result.done) {
                        return { value: undefined as unknown as Ciphertext, done: true };
                    }
                    processedCount++;
                    return { value: result.value, done: false };
                }
            };
        },
        
        pipe(transform: (ct: Ciphertext) => Promise<Ciphertext>): CiphertextStream {
            return createTransformedStream(stream, transform, fullConfig);
        },
        
        async collect(): Promise<Ciphertext[]> {
            const results: Ciphertext[] = [];
            for await (const ct of stream) {
                results.push(ct);
            }
            return results;
        },
        
        getProcessedCount(): number {
            return processedCount;
        },
        
        cancel(): void {
            cancelled = true;
            if (gen) {
                gen.return(undefined);
            }
        }
    };
    
    return stream;
}

/**
 * Chunked Ciphertext Processor
 * 
 * Handles splitting large ciphertexts into processable chunks and
 * reassembling them while maintaining correctness across chunk boundaries.
 * 
 * Requirements: 11.1, 11.2
 */
export class ChunkedCiphertextProcessor {
    private config: StreamingConfig;
    
    constructor(config: Partial<StreamingConfig> = {}) {
        this.config = { ...DEFAULT_STREAMING_CONFIG, ...config };
    }
    
    /**
     * Calculate the number of chunks needed for a ciphertext
     */
    calculateChunkCount(polynomialDegree: number): number {
        if (polynomialDegree <= 0) {
            return 1;
        }
        if (polynomialDegree <= this.config.chunkSize) {
            return 1;
        }
        return Math.ceil(polynomialDegree / this.config.chunkSize);
    }
    
    /**
     * Split polynomial coefficients into chunks
     * 
     * For streaming operations that are element-wise (like addition, scalar multiplication),
     * we don't need overlap - each coefficient can be processed independently.
     * The overlap is only needed for operations that span coefficients (like convolution).
     * 
     * For simplicity and correctness, we use non-overlapping chunks for element-wise operations.
     */
    splitCoefficients(coefficients: bigint[]): bigint[][] {
        const chunks: bigint[][] = [];
        const chunkSize = this.config.chunkSize;
        
        if (coefficients.length === 0) {
            return [[]];
        }
        
        if (coefficients.length <= chunkSize) {
            return [[...coefficients]];
        }
        
        // Use non-overlapping chunks for element-wise operations
        for (let i = 0; i < coefficients.length; i += chunkSize) {
            const endOffset = Math.min(i + chunkSize, coefficients.length);
            chunks.push(coefficients.slice(i, endOffset));
        }
        
        return chunks;
    }
    
    /**
     * Merge coefficient chunks back into a single array
     */
    mergeCoefficients(chunks: bigint[][], totalSize: number): bigint[] {
        if (chunks.length === 0) {
            return [];
        }
        
        if (chunks.length === 1) {
            // Return a copy, padded or truncated to totalSize
            const firstChunk = chunks[0];
            if (!firstChunk) {
                return new Array(totalSize).fill(BigInt(0));
            }
            const result = [...firstChunk];
            while (result.length < totalSize) {
                result.push(BigInt(0));
            }
            return result.slice(0, totalSize);
        }
        
        const result: bigint[] = [];
        
        // Simply concatenate all chunks
        for (const chunk of chunks) {
            if (chunk) {
                result.push(...chunk);
            }
        }
        
        // Ensure result is exactly totalSize
        while (result.length < totalSize) {
            result.push(BigInt(0));
        }
        
        return result.slice(0, totalSize);
    }
    
    /**
     * Process coefficients in chunks with a transformation function
     * 
     * For element-wise operations, each chunk is processed independently
     * and the results are concatenated.
     */
    async processChunked(
        coefficients: bigint[],
        transform: (chunk: bigint[]) => Promise<bigint[]>,
        progress?: StreamingProgressCallback
    ): Promise<bigint[]> {
        const startTime = Date.now();
        const chunks = this.splitCoefficients(coefficients);
        const processedChunks: bigint[][] = [];
        
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            if (!chunk) continue;
            const processed = await transform(chunk);
            processedChunks.push(processed);
            
            if (progress) {
                const elapsed = Date.now() - startTime;
                const estimatedRemaining = (elapsed / (i + 1)) * (chunks.length - i - 1);
                progress({
                    stage: 'processing',
                    current: i + 1,
                    total: chunks.length,
                    elapsedMs: elapsed,
                    estimatedRemainingMs: estimatedRemaining,
                    progressPercent: ((i + 1) / chunks.length) * 100
                });
            }
        }
        
        // Merge preserving the original length
        return this.mergeCoefficients(processedChunks, coefficients.length);
    }
    
    getConfig(): StreamingConfig {
        return { ...this.config };
    }
    
    setConfig(config: Partial<StreamingConfig>): void {
        this.config = { ...this.config, ...config };
    }
}

/**
 * Streaming Ciphertext Processor
 * 
 * Provides async streaming interface for ciphertext operations with
 * support for progress callbacks and backpressure handling.
 * 
 * Requirements: 11.3
 */
export class StreamingCiphertextProcessor {
    private config: StreamingConfig;
    private chunkedProcessor: ChunkedCiphertextProcessor;
    private active: boolean = false;
    private cancelled: boolean = false;
    
    constructor(config: Partial<StreamingConfig> = {}) {
        this.config = { ...DEFAULT_STREAMING_CONFIG, ...config };
        this.chunkedProcessor = new ChunkedCiphertextProcessor(this.config);
    }
    
    /**
     * Process a stream of ciphertexts with a transformation
     */
    async processStream(
        inputStream: CiphertextStream,
        transform: (ct: Ciphertext) => Promise<Ciphertext>,
        progress?: StreamingProgressCallback
    ): Promise<StreamingResult> {
        const startTime = Date.now();
        this.active = true;
        this.cancelled = false;
        
        let processed = 0;
        const results: Ciphertext[] = [];
        
        try {
            for await (const ct of inputStream) {
                if (this.cancelled) break;
                
                const transformed = await transform(ct);
                results.push(transformed);
                processed++;
                
                if (progress) {
                    const elapsed = Date.now() - startTime;
                    progress({
                        stage: 'processing',
                        current: processed,
                        total: processed, // Unknown total for streams
                        elapsedMs: elapsed,
                        estimatedRemainingMs: 0,
                        progressPercent: 0
                    });
                }
            }
            
            const totalTime = Date.now() - startTime;
            return {
                success: !this.cancelled,
                chunksProcessed: processed,
                totalTimeMs: totalTime,
                throughputChunksPerSec: totalTime > 0 ? (processed * 1000) / totalTime : 0
            };
        } catch (error) {
            return {
                success: false,
                errorMessage: error instanceof Error ? error.message : String(error),
                chunksProcessed: processed,
                totalTimeMs: Date.now() - startTime,
                throughputChunksPerSec: 0
            };
        } finally {
            this.active = false;
        }
    }
    
    /**
     * Stream addition of ciphertexts (running total)
     */
    async streamAdd(
        inputStream: CiphertextStream,
        addFn: (ct1: Ciphertext, ct2: Ciphertext) => Promise<Ciphertext>,
        progress?: StreamingProgressCallback
    ): Promise<{ result: StreamingResult; accumulator: Ciphertext | undefined }> {
        const startTime = Date.now();
        this.active = true;
        this.cancelled = false;
        
        let processed = 0;
        let accumulator: Ciphertext | undefined;
        
        try {
            for await (const ct of inputStream) {
                if (this.cancelled) break;
                
                if (!accumulator) {
                    accumulator = ct;
                } else {
                    accumulator = await addFn(accumulator, ct);
                }
                
                processed++;
                
                if (progress) {
                    const elapsed = Date.now() - startTime;
                    progress({
                        stage: 'stream_add',
                        current: processed,
                        total: processed,
                        elapsedMs: elapsed,
                        estimatedRemainingMs: 0,
                        progressPercent: 0
                    });
                }
            }
            
            const totalTime = Date.now() - startTime;
            return {
                result: {
                    success: !this.cancelled,
                    chunksProcessed: processed,
                    totalTimeMs: totalTime,
                    throughputChunksPerSec: totalTime > 0 ? (processed * 1000) / totalTime : 0
                },
                accumulator
            };
        } catch (error) {
            return {
                result: {
                    success: false,
                    errorMessage: error instanceof Error ? error.message : String(error),
                    chunksProcessed: processed,
                    totalTimeMs: Date.now() - startTime,
                    throughputChunksPerSec: 0
                },
                accumulator
            };
        } finally {
            this.active = false;
        }
    }
    
    /**
     * Cancel ongoing streaming operation
     */
    cancel(): void {
        this.cancelled = true;
    }
    
    /**
     * Check if streaming is currently active
     */
    isActive(): boolean {
        return this.active;
    }
    
    getConfig(): StreamingConfig {
        return { ...this.config };
    }
    
    setConfig(config: Partial<StreamingConfig>): void {
        this.config = { ...this.config, ...config };
        this.chunkedProcessor.setConfig(this.config);
    }
}

/**
 * Verify streaming equivalence - that streaming and non-streaming
 * operations produce identical results.
 * 
 * Requirements: 11.6
 */
export async function verifyStreamingEquivalence(
    coefficients: bigint[],
    transform: (coeffs: bigint[]) => Promise<bigint[]>,
    config: Partial<StreamingConfig> = {}
): Promise<boolean> {
    const processor = new ChunkedCiphertextProcessor(config);
    
    // Non-streaming result
    const nonStreamingResult = await transform(coefficients);
    
    // Streaming result
    const streamingResult = await processor.processChunked(coefficients, transform);
    
    // Compare results
    if (nonStreamingResult.length !== streamingResult.length) {
        return false;
    }
    
    for (let i = 0; i < nonStreamingResult.length; i++) {
        if (nonStreamingResult[i] !== streamingResult[i]) {
            return false;
        }
    }
    
    return true;
}

/**
 * Compare two coefficient arrays for equality
 */
export function coefficientsEqual(a: bigint[], b: bigint[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}


// Re-export Node.js stream interfaces
export * from './node-streams';
