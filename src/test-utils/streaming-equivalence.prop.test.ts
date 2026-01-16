/**
 * Property-Based Tests for Streaming Equivalence
 * 
 * **Property 12: Streaming Equivalence**
 * Process data via streaming and non-streaming
 * Verify bit-identical results
 * 
 * **Validates: Requirements 11.6**
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG } from './property-test-config';
import {
    ChunkedCiphertextProcessor,
    StreamingCiphertextProcessor,
    verifyStreamingEquivalence,
    coefficientsEqual,
    createCiphertextStream,
    DEFAULT_STREAMING_CONFIG,
    type StreamingConfig
} from '../streaming';

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * Generate random coefficients for testing
 */
function generateRandomCoefficients(length: number, modulus: bigint): bigint[] {
    const coeffs: bigint[] = [];
    for (let i = 0; i < length; i++) {
        // Generate random bigint in range [0, modulus)
        const randomValue = BigInt(Math.floor(Math.random() * Number(modulus % BigInt(Number.MAX_SAFE_INTEGER))));
        coeffs.push(randomValue);
    }
    return coeffs;
}

/**
 * Simple modular addition for testing
 */
function modAdd(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
    const result: bigint[] = [];
    const len = Math.max(a.length, b.length);
    for (let i = 0; i < len; i++) {
        const ai = i < a.length ? a[i] : 0n;
        const bi = i < b.length ? b[i] : 0n;
        result.push((ai + bi) % modulus);
    }
    return result;
}

/**
 * Simple modular scalar multiplication for testing
 */
function modScalarMul(coeffs: bigint[], scalar: bigint, modulus: bigint): bigint[] {
    return coeffs.map(c => (c * scalar) % modulus);
}

/**
 * Identity transform for testing
 */
async function identityTransform(coeffs: bigint[]): Promise<bigint[]> {
    return coeffs;
}

/**
 * Modular doubling transform for testing
 */
function createDoublingTransform(modulus: bigint) {
    return async (coeffs: bigint[]): Promise<bigint[]> => {
        return coeffs.map(c => (c * 2n) % modulus);
    };
}

/**
 * Modular addition transform for testing
 */
function createAdditionTransform(addend: bigint[], modulus: bigint) {
    return async (coeffs: bigint[]): Promise<bigint[]> => {
        return modAdd(coeffs, addend, modulus);
    };
}

// ============================================================================
// Property Tests
// ============================================================================

describe('Property 12: Streaming Equivalence', () => {
    const TEST_MODULUS = 65537n; // Small prime for testing
    
    describe('ChunkedCiphertextProcessor', () => {
        it('should calculate correct chunk count for various polynomial sizes', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: 1, max: 32768 }),
                    fc.integer({ min: 64, max: 4096 }),
                    (polyDegree, chunkSize) => {
                        const processor = new ChunkedCiphertextProcessor({ chunkSize });
                        const chunkCount = processor.calculateChunkCount(polyDegree);
                        
                        // Chunk count should be at least 1
                        if (chunkCount < 1) return false;
                        
                        // If poly fits in one chunk, count should be 1
                        if (polyDegree <= chunkSize && chunkCount !== 1) return false;
                        
                        return true;
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
            );
        });
        
        it('should preserve coefficients through split and merge (identity)', () => {
            /**
             * **Property 12.1: Split-Merge Identity**
             * For any coefficient array, splitting and merging should produce
             * the original array (bit-identical).
             */
            fc.assert(
                fc.property(
                    fc.integer({ min: 64, max: 4096 }),
                    fc.integer({ min: 32, max: 512 }),
                    (polyDegree, chunkSize) => {
                        const processor = new ChunkedCiphertextProcessor({ 
                            chunkSize,
                            boundaryOverlap: Math.min(16, Math.floor(chunkSize / 4))
                        });
                        
                        // Generate random coefficients
                        const original = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                        
                        // Split into chunks
                        const chunks = processor.splitCoefficients(original);
                        
                        // Merge back
                        const merged = processor.mergeCoefficients(chunks, polyDegree);
                        
                        // Verify equality
                        return coefficientsEqual(original, merged);
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
            );
        });
        
        it('should produce identical results for streaming vs non-streaming identity transform', async () => {
            /**
             * **Property 12.2: Streaming Identity Equivalence**
             * Applying identity transform via streaming should produce
             * bit-identical results to non-streaming.
             */
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 64, max: 2048 }),
                    async (polyDegree) => {
                        const coefficients = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                        
                        const isEquivalent = await verifyStreamingEquivalence(
                            coefficients,
                            identityTransform,
                            { chunkSize: 256, boundaryOverlap: 32 }
                        );
                        
                        return isEquivalent;
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
            );
        });
        
        it('should produce identical results for streaming vs non-streaming doubling', async () => {
            /**
             * **Property 12.3: Streaming Doubling Equivalence**
             * Applying coefficient doubling via streaming should produce
             * bit-identical results to non-streaming.
             */
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 64, max: 2048 }),
                    async (polyDegree) => {
                        const coefficients = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                        const doublingTransform = createDoublingTransform(TEST_MODULUS);
                        
                        const isEquivalent = await verifyStreamingEquivalence(
                            coefficients,
                            doublingTransform,
                            { chunkSize: 256, boundaryOverlap: 32 }
                        );
                        
                        return isEquivalent;
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
            );
        });
        
        it('should produce identical results for streaming vs non-streaming addition', async () => {
            /**
             * **Property 12.4: Streaming Addition Equivalence**
             * Applying coefficient addition via streaming should produce
             * bit-identical results to non-streaming.
             * 
             * Note: For addition, we need to ensure the addend is processed
             * the same way as the coefficients. We use a fixed addend that
             * doesn't depend on chunk boundaries.
             */
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 64, max: 2048 }),
                    async (polyDegree) => {
                        const coefficients = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                        
                        // Use a simple scalar addition instead of element-wise addition
                        // to avoid issues with addend array chunking
                        const scalarAddend = 42n;
                        const scalarAdditionTransform = async (chunk: bigint[]): Promise<bigint[]> => {
                            return chunk.map(c => (c + scalarAddend) % TEST_MODULUS);
                        };
                        
                        const isEquivalent = await verifyStreamingEquivalence(
                            coefficients,
                            scalarAdditionTransform,
                            { chunkSize: 256, boundaryOverlap: 32 }
                        );
                        
                        return isEquivalent;
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
            );
        });
    });
    
    describe('Chunk Boundary Handling', () => {
        it('should handle coefficients at chunk boundaries correctly', () => {
            /**
             * **Property 12.5: Boundary Coefficient Preservation**
             * Coefficients at chunk boundaries should be preserved correctly
             * after split and merge operations.
             */
            fc.assert(
                fc.property(
                    fc.integer({ min: 128, max: 1024 }),
                    fc.integer({ min: 32, max: 128 }),
                    (polyDegree, chunkSize) => {
                        const processor = new ChunkedCiphertextProcessor({ 
                            chunkSize,
                            boundaryOverlap: Math.min(16, Math.floor(chunkSize / 4))
                        });
                        
                        // Generate coefficients with known values at boundaries
                        const original = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                        
                        // Mark boundary positions with special values
                        const effectiveChunkSize = chunkSize - processor.getConfig().boundaryOverlap;
                        for (let i = 0; i < polyDegree; i += effectiveChunkSize) {
                            if (i < polyDegree) {
                                original[i] = BigInt(i); // Mark boundary start
                            }
                            const boundaryEnd = Math.min(i + chunkSize - 1, polyDegree - 1);
                            if (boundaryEnd < polyDegree) {
                                original[boundaryEnd] = BigInt(boundaryEnd); // Mark boundary end
                            }
                        }
                        
                        // Split and merge
                        const chunks = processor.splitCoefficients(original);
                        const merged = processor.mergeCoefficients(chunks, polyDegree);
                        
                        // Verify boundary values are preserved
                        return coefficientsEqual(original, merged);
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
            );
        });
        
        it('should handle single-chunk case correctly', async () => {
            /**
             * **Property 12.6: Single Chunk Equivalence**
             * When data fits in a single chunk, streaming should be
             * equivalent to non-streaming.
             */
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 16, max: 256 }),
                    async (polyDegree) => {
                        const processor = new ChunkedCiphertextProcessor({ 
                            chunkSize: 512, // Larger than polyDegree
                            boundaryOverlap: 32
                        });
                        
                        const coefficients = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                        
                        // Should produce single chunk
                        const chunks = processor.splitCoefficients(coefficients);
                        if (chunks.length !== 1) return false;
                        
                        // Merge should be identity
                        const merged = processor.mergeCoefficients(chunks, polyDegree);
                        return coefficientsEqual(coefficients, merged);
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
            );
        });
    });
    
    describe('Configuration Variations', () => {
        it('should produce consistent results across different chunk sizes', async () => {
            /**
             * **Property 12.7: Chunk Size Independence**
             * The final result should be independent of chunk size
             * (within reasonable bounds).
             */
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 256, max: 1024 }),
                    async (polyDegree) => {
                        const coefficients = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                        const transform = createDoublingTransform(TEST_MODULUS);
                        
                        // Process with different chunk sizes
                        const processor64 = new ChunkedCiphertextProcessor({ 
                            chunkSize: 64, boundaryOverlap: 8 
                        });
                        const processor128 = new ChunkedCiphertextProcessor({ 
                            chunkSize: 128, boundaryOverlap: 16 
                        });
                        const processor256 = new ChunkedCiphertextProcessor({ 
                            chunkSize: 256, boundaryOverlap: 32 
                        });
                        
                        const result64 = await processor64.processChunked(coefficients, transform);
                        const result128 = await processor128.processChunked(coefficients, transform);
                        const result256 = await processor256.processChunked(coefficients, transform);
                        
                        // All results should be identical
                        return coefficientsEqual(result64, result128) && 
                               coefficientsEqual(result128, result256);
                    }
                ),
                { ...PROPERTY_TEST_CONFIG, numRuns: 30 }
            );
        });
        
        it('should handle edge case polynomial degrees', () => {
            /**
             * **Property 12.8: Edge Case Handling**
             * Streaming should handle edge cases like power-of-2 degrees,
             * prime degrees, and degrees that don't divide evenly into chunks.
             */
            const edgeCases = [
                1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                127, 128, 129, 255, 256, 257, 511, 512, 513,
                1023, 1024, 1025
            ];
            
            for (const polyDegree of edgeCases) {
                const processor = new ChunkedCiphertextProcessor({ 
                    chunkSize: 64, boundaryOverlap: 8 
                });
                
                const coefficients = generateRandomCoefficients(polyDegree, TEST_MODULUS);
                const chunks = processor.splitCoefficients(coefficients);
                const merged = processor.mergeCoefficients(chunks, polyDegree);
                
                expect(coefficientsEqual(coefficients, merged)).toBe(true);
            }
        });
    });
    
    describe('Progress Callback', () => {
        it('should report progress correctly during chunked processing', async () => {
            /**
             * **Property 12.9: Progress Reporting Correctness**
             * Progress callbacks should report monotonically increasing
             * progress values.
             */
            const polyDegree = 1024;
            const processor = new ChunkedCiphertextProcessor({ 
                chunkSize: 128, boundaryOverlap: 16 
            });
            
            const coefficients = generateRandomCoefficients(polyDegree, TEST_MODULUS);
            const progressValues: number[] = [];
            
            await processor.processChunked(
                coefficients,
                identityTransform,
                (progress) => {
                    progressValues.push(progress.current);
                }
            );
            
            // Progress should be monotonically increasing
            for (let i = 1; i < progressValues.length; i++) {
                expect(progressValues[i]).toBeGreaterThanOrEqual(progressValues[i - 1]);
            }
            
            // Final progress should equal total chunks
            const expectedChunks = processor.calculateChunkCount(polyDegree);
            expect(progressValues[progressValues.length - 1]).toBe(expectedChunks);
        });
    });
    
    describe('Async Stream Interface', () => {
        it('should process all items in async stream', async () => {
            /**
             * **Property 12.10: Async Stream Completeness**
             * All items in an async stream should be processed.
             */
            // Create mock ciphertexts (using simple objects for testing)
            const mockCiphertexts = Array.from({ length: 100 }, (_, i) => ({
                __brand: 'Ciphertext' as const,
                handle: BigInt(i)
            }));
            
            const stream = createCiphertextStream(mockCiphertexts);
            const collected = await stream.collect();
            
            expect(collected.length).toBe(mockCiphertexts.length);
            for (let i = 0; i < collected.length; i++) {
                expect(collected[i].handle).toBe(BigInt(i));
            }
        });
        
        it('should support stream cancellation', async () => {
            /**
             * **Property 12.11: Stream Cancellation**
             * Cancelled streams should stop processing.
             */
            const mockCiphertexts = Array.from({ length: 1000 }, (_, i) => ({
                __brand: 'Ciphertext' as const,
                handle: BigInt(i)
            }));
            
            const stream = createCiphertextStream(mockCiphertexts);
            const collected: typeof mockCiphertexts = [];
            
            for await (const ct of stream) {
                collected.push(ct);
                if (collected.length >= 50) {
                    stream.cancel();
                    break;
                }
            }
            
            // Should have stopped at or near 50
            expect(collected.length).toBeLessThanOrEqual(51);
            expect(collected.length).toBeGreaterThanOrEqual(50);
        });
        
        it('should support stream piping with transformations', async () => {
            /**
             * **Property 12.12: Stream Piping**
             * Piped transformations should be applied correctly.
             */
            const mockCiphertexts = Array.from({ length: 10 }, (_, i) => ({
                __brand: 'Ciphertext' as const,
                handle: BigInt(i)
            }));
            
            const stream = createCiphertextStream(mockCiphertexts);
            
            // Apply transformation that doubles the handle
            const transformedStream = stream.pipe(async (ct) => ({
                ...ct,
                handle: ct.handle * 2n
            }));
            
            const collected = await transformedStream.collect();
            
            expect(collected.length).toBe(mockCiphertexts.length);
            for (let i = 0; i < collected.length; i++) {
                expect(collected[i].handle).toBe(BigInt(i * 2));
            }
        });
    });
});

describe('Streaming Equivalence Edge Cases', () => {
    const TEST_MODULUS = 65537n;
    
    it('should handle empty coefficient arrays', () => {
        const processor = new ChunkedCiphertextProcessor();
        const empty: bigint[] = [];
        
        const chunks = processor.splitCoefficients(empty);
        expect(chunks.length).toBe(1);
        expect(chunks[0].length).toBe(0);
        
        const merged = processor.mergeCoefficients(chunks, 0);
        expect(merged.length).toBe(0);
    });
    
    it('should handle single coefficient', () => {
        const processor = new ChunkedCiphertextProcessor();
        const single = [42n];
        
        const chunks = processor.splitCoefficients(single);
        const merged = processor.mergeCoefficients(chunks, 1);
        
        expect(coefficientsEqual(single, merged)).toBe(true);
    });
    
    it('should handle coefficients equal to chunk size', () => {
        const chunkSize = 64;
        const processor = new ChunkedCiphertextProcessor({ chunkSize, boundaryOverlap: 8 });
        const coefficients = generateRandomCoefficients(chunkSize, TEST_MODULUS);
        
        const chunks = processor.splitCoefficients(coefficients);
        const merged = processor.mergeCoefficients(chunks, chunkSize);
        
        expect(coefficientsEqual(coefficients, merged)).toBe(true);
    });
    
    it('should handle coefficients one less than chunk size', () => {
        const chunkSize = 64;
        const processor = new ChunkedCiphertextProcessor({ chunkSize, boundaryOverlap: 8 });
        const coefficients = generateRandomCoefficients(chunkSize - 1, TEST_MODULUS);
        
        const chunks = processor.splitCoefficients(coefficients);
        const merged = processor.mergeCoefficients(chunks, chunkSize - 1);
        
        expect(coefficientsEqual(coefficients, merged)).toBe(true);
    });
    
    it('should handle coefficients one more than chunk size', () => {
        const chunkSize = 64;
        const processor = new ChunkedCiphertextProcessor({ chunkSize, boundaryOverlap: 8 });
        const coefficients = generateRandomCoefficients(chunkSize + 1, TEST_MODULUS);
        
        const chunks = processor.splitCoefficients(coefficients);
        const merged = processor.mergeCoefficients(chunks, chunkSize + 1);
        
        expect(coefficientsEqual(coefficients, merged)).toBe(true);
    });
    
    it('should handle very large coefficient arrays', async () => {
        const processor = new ChunkedCiphertextProcessor({ 
            chunkSize: 1024, 
            boundaryOverlap: 64 
        });
        const largeSize = 16384;
        const coefficients = generateRandomCoefficients(largeSize, TEST_MODULUS);
        
        const isEquivalent = await verifyStreamingEquivalence(
            coefficients,
            identityTransform,
            { chunkSize: 1024, boundaryOverlap: 64 }
        );
        
        expect(isEquivalent).toBe(true);
    });
});
