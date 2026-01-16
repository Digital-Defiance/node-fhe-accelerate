/**
 * @file streaming_processor.h
 * @brief Streaming Operations for Large Ciphertexts
 * 
 * This file defines the streaming infrastructure for processing large ciphertexts
 * in chunks, maintaining correctness across chunk boundaries, and providing
 * async iteration with progress callbacks.
 * 
 * Requirements: 11.1, 11.2, 11.3, 11.6
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "encryption.h"
#include "parameter_set.h"

namespace fhe_accelerate {

/**
 * Chunk metadata for tracking processing state
 */
struct ChunkMetadata {
    size_t chunk_index;         // Index of this chunk in the sequence
    size_t total_chunks;        // Total number of chunks
    size_t start_offset;        // Starting offset in the original data
    size_t chunk_size;          // Size of this chunk
    bool is_first;              // Whether this is the first chunk
    bool is_last;               // Whether this is the last chunk
    
    ChunkMetadata(size_t idx = 0, size_t total = 1, size_t offset = 0, 
                  size_t size = 0, bool first = true, bool last = true)
        : chunk_index(idx), total_chunks(total), start_offset(offset),
          chunk_size(size), is_first(first), is_last(last) {}
};

/**
 * Ciphertext chunk for streaming operations
 * 
 * Represents a portion of a large ciphertext that can be processed independently
 * while maintaining correctness across chunk boundaries.
 */
struct CiphertextChunk {
    Ciphertext data;            // The actual ciphertext data for this chunk
    ChunkMetadata metadata;     // Metadata about this chunk's position
    
    // Boundary data for maintaining correctness across chunks
    std::vector<uint64_t> left_boundary;   // Coefficients needed from previous chunk
    std::vector<uint64_t> right_boundary;  // Coefficients needed for next chunk
    
    CiphertextChunk(Ciphertext&& ct, ChunkMetadata meta)
        : data(std::move(ct)), metadata(meta) {}
    
    CiphertextChunk(const CiphertextChunk&) = default;
    CiphertextChunk(CiphertextChunk&&) = default;
    CiphertextChunk& operator=(const CiphertextChunk&) = default;
    CiphertextChunk& operator=(CiphertextChunk&&) = default;
};

/**
 * Progress information for streaming operations
 */
struct StreamingProgress {
    std::string stage;          // Current processing stage
    size_t current;             // Current progress (e.g., chunks processed)
    size_t total;               // Total items to process
    double elapsed_ms;          // Elapsed time in milliseconds
    double estimated_remaining_ms;  // Estimated remaining time
    
    StreamingProgress(const std::string& s = "", size_t cur = 0, size_t tot = 0,
                      double elapsed = 0.0, double remaining = 0.0)
        : stage(s), current(cur), total(tot), elapsed_ms(elapsed),
          estimated_remaining_ms(remaining) {}
    
    double progress_percent() const {
        return total > 0 ? (100.0 * current / total) : 0.0;
    }
};

/**
 * Progress callback type for streaming operations
 */
using StreamingProgressCallback = std::function<void(const StreamingProgress&)>;

/**
 * Streaming configuration options
 */
struct StreamingConfig {
    size_t chunk_size;              // Number of coefficients per chunk
    size_t boundary_overlap;        // Overlap size for boundary handling
    size_t max_memory_bytes;        // Maximum memory to use for buffering
    size_t prefetch_chunks;         // Number of chunks to prefetch
    bool use_memory_mapping;        // Use memory-mapped I/O for large data
    bool enable_compression;        // Enable compression for chunks
    
    StreamingConfig()
        : chunk_size(4096)
        , boundary_overlap(64)
        , max_memory_bytes(256 * 1024 * 1024)  // 256 MB default
        , prefetch_chunks(2)
        , use_memory_mapping(true)
        , enable_compression(false) {}
    
    // Calculate optimal chunk size based on polynomial degree
    static StreamingConfig optimal_for_degree(uint32_t poly_degree) {
        StreamingConfig config;
        // Use polynomial degree as chunk size for natural alignment
        config.chunk_size = poly_degree;
        // Boundary overlap should be enough for NTT butterfly operations
        config.boundary_overlap = std::min(static_cast<size_t>(64), 
                                           static_cast<size_t>(poly_degree / 16));
        return config;
    }
};

/**
 * Result of a streaming operation
 */
struct StreamingResult {
    bool success;
    std::string error_message;
    size_t chunks_processed;
    double total_time_ms;
    double throughput_chunks_per_sec;
    
    StreamingResult()
        : success(true), chunks_processed(0), total_time_ms(0),
          throughput_chunks_per_sec(0) {}
    
    static StreamingResult failure(const std::string& msg) {
        StreamingResult result;
        result.success = false;
        result.error_message = msg;
        return result;
    }
};

/**
 * Chunked Ciphertext Processor
 * 
 * Handles splitting large ciphertexts into processable chunks and
 * reassembling them while maintaining correctness across chunk boundaries.
 * 
 * Requirements: 11.1, 11.2
 */
class ChunkedCiphertextProcessor {
public:
    /**
     * Construct processor with given parameters and configuration
     */
    explicit ChunkedCiphertextProcessor(const ParameterSet& params,
                                        const StreamingConfig& config = StreamingConfig());
    ~ChunkedCiphertextProcessor();
    
    // ========================================================================
    // Chunking Operations (Requirement 11.1)
    // ========================================================================
    
    /**
     * Split a ciphertext into processable chunks
     * 
     * Divides a large ciphertext into smaller chunks that can be processed
     * independently. Each chunk includes boundary data needed for operations
     * that span chunk boundaries.
     * 
     * @param ct The ciphertext to split
     * @return Vector of ciphertext chunks
     */
    std::vector<CiphertextChunk> split_ciphertext(const Ciphertext& ct);
    
    /**
     * Merge chunks back into a single ciphertext
     * 
     * Reassembles chunks into a complete ciphertext, handling boundary
     * corrections to ensure bit-identical results with non-chunked processing.
     * 
     * @param chunks Vector of ciphertext chunks (must be in order)
     * @return Merged ciphertext
     */
    Ciphertext merge_chunks(const std::vector<CiphertextChunk>& chunks);
    
    /**
     * Calculate the number of chunks needed for a ciphertext
     * 
     * @param ct The ciphertext to analyze
     * @return Number of chunks needed
     */
    size_t calculate_chunk_count(const Ciphertext& ct) const;
    
    // ========================================================================
    // Chunked Homomorphic Operations (Requirement 11.2)
    // ========================================================================
    
    /**
     * Add two ciphertexts using chunked processing
     * 
     * Performs homomorphic addition on large ciphertexts by processing
     * chunks independently and maintaining correctness across boundaries.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param progress Optional progress callback
     * @return Result ciphertext
     */
    Ciphertext chunked_add(const Ciphertext& ct1, const Ciphertext& ct2,
                           StreamingProgressCallback progress = nullptr);
    
    /**
     * Multiply two ciphertexts using chunked processing
     * 
     * Performs homomorphic multiplication on large ciphertexts.
     * Note: Multiplication requires more careful boundary handling due to
     * polynomial multiplication spanning coefficients.
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param ek Evaluation key for relinearization
     * @param progress Optional progress callback
     * @return Result ciphertext
     */
    Ciphertext chunked_multiply(const Ciphertext& ct1, const Ciphertext& ct2,
                                const EvaluationKey& ek,
                                StreamingProgressCallback progress = nullptr);
    
    /**
     * Add a chunk to an accumulator (for streaming aggregation)
     * 
     * Efficiently adds a single chunk to a running total, useful for
     * streaming ballot aggregation.
     * 
     * @param accumulator Current accumulated result
     * @param chunk Chunk to add
     * @return Updated accumulator
     */
    CiphertextChunk add_chunk_to_accumulator(const CiphertextChunk& accumulator,
                                              const CiphertextChunk& chunk);
    
    // ========================================================================
    // Boundary Handling
    // ========================================================================
    
    /**
     * Extract boundary coefficients from a polynomial
     * 
     * @param poly The polynomial
     * @param start Start index
     * @param count Number of coefficients to extract
     * @return Vector of boundary coefficients
     */
    std::vector<uint64_t> extract_boundary(const Polynomial& poly,
                                           size_t start, size_t count) const;
    
    /**
     * Apply boundary corrections after chunk operations
     * 
     * Ensures that operations spanning chunk boundaries produce correct results.
     * 
     * @param chunk The chunk to correct
     * @param left_boundary Boundary data from previous chunk
     * @param right_boundary Boundary data for next chunk
     */
    void apply_boundary_correction(CiphertextChunk& chunk,
                                   const std::vector<uint64_t>& left_boundary,
                                   const std::vector<uint64_t>& right_boundary);
    
    // ========================================================================
    // Configuration
    // ========================================================================
    
    const StreamingConfig& get_config() const { return config_; }
    void set_config(const StreamingConfig& config) { config_ = config; }
    
    const ParameterSet& get_params() const { return params_; }
    
private:
    ParameterSet params_;
    StreamingConfig config_;
    std::unique_ptr<EncryptionEngine> engine_;
    
    // Helper to create a chunk from polynomial data
    CiphertextChunk create_chunk(const Ciphertext& ct, size_t chunk_idx,
                                 size_t total_chunks, size_t start_offset);
};

/**
 * Async Streaming Interface
 * 
 * Provides AsyncIterable-style interface for ciphertext streams with
 * support for progress callbacks and backpressure handling.
 * 
 * Requirements: 11.3
 */
class CiphertextStreamProcessor {
public:
    /**
     * Construct stream processor with given parameters
     */
    explicit CiphertextStreamProcessor(const ParameterSet& params,
                                       const StreamingConfig& config = StreamingConfig());
    ~CiphertextStreamProcessor();
    
    // ========================================================================
    // Async Streaming Interface (Requirement 11.3)
    // ========================================================================
    
    /**
     * Start streaming processing of ciphertexts
     * 
     * Begins async processing of a stream of ciphertexts. Results are
     * delivered via the output callback as they become available.
     * 
     * @param input_stream Function that provides input ciphertexts (returns nullopt when done)
     * @param transform Operation to apply to each ciphertext
     * @param output Callback to receive processed ciphertexts
     * @param progress Optional progress callback
     * @return Future that completes when streaming is done
     */
    std::future<StreamingResult> process_stream(
        std::function<std::optional<Ciphertext>()> input_stream,
        std::function<Ciphertext(const Ciphertext&)> transform,
        std::function<void(Ciphertext&&)> output,
        StreamingProgressCallback progress = nullptr
    );
    
    /**
     * Stream addition of multiple ciphertexts
     * 
     * Adds ciphertexts from a stream, producing running totals.
     * 
     * @param input_stream Function that provides input ciphertexts
     * @param output Callback to receive running totals
     * @param progress Optional progress callback
     * @return Future with final result
     */
    std::future<StreamingResult> stream_add(
        std::function<std::optional<Ciphertext>()> input_stream,
        std::function<void(Ciphertext&&)> output,
        StreamingProgressCallback progress = nullptr
    );
    
    /**
     * Stream batch encryption
     * 
     * Encrypts plaintexts from a stream, producing ciphertexts.
     * 
     * @param input_stream Function that provides input plaintexts
     * @param pk Public key for encryption
     * @param output Callback to receive encrypted ciphertexts
     * @param progress Optional progress callback
     * @return Future with result
     */
    std::future<StreamingResult> stream_encrypt(
        std::function<std::optional<Plaintext>()> input_stream,
        const PublicKey& pk,
        std::function<void(Ciphertext&&)> output,
        StreamingProgressCallback progress = nullptr
    );
    
    // ========================================================================
    // Stream Control
    // ========================================================================
    
    /**
     * Cancel ongoing streaming operation
     */
    void cancel();
    
    /**
     * Check if streaming is currently active
     */
    bool is_active() const { return active_.load(); }
    
    /**
     * Wait for current streaming operation to complete
     */
    void wait();
    
    // ========================================================================
    // Configuration
    // ========================================================================
    
    const StreamingConfig& get_config() const { return config_; }
    void set_config(const StreamingConfig& config) { config_ = config; }
    
private:
    ParameterSet params_;
    StreamingConfig config_;
    std::unique_ptr<EncryptionEngine> engine_;
    std::unique_ptr<ChunkedCiphertextProcessor> chunked_processor_;
    
    std::atomic<bool> active_;
    std::atomic<bool> cancelled_;
    
    // Internal buffer for streaming
    std::queue<Ciphertext> buffer_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    
    // Helper to update progress
    void update_progress(StreamingProgressCallback& callback,
                         const std::string& stage, size_t current, size_t total,
                         std::chrono::steady_clock::time_point start_time);
};

/**
 * Streaming Equivalence Verifier
 * 
 * Utility class for verifying that streaming and non-streaming operations
 * produce bit-identical results.
 * 
 * Requirements: 11.6
 */
class StreamingEquivalenceVerifier {
public:
    explicit StreamingEquivalenceVerifier(const ParameterSet& params);
    
    /**
     * Verify that chunked addition produces identical results to non-chunked
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @return True if results are bit-identical
     */
    bool verify_add_equivalence(const Ciphertext& ct1, const Ciphertext& ct2);
    
    /**
     * Verify that chunked multiplication produces identical results
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param ek Evaluation key
     * @return True if results are bit-identical
     */
    bool verify_multiply_equivalence(const Ciphertext& ct1, const Ciphertext& ct2,
                                     const EvaluationKey& ek);
    
    /**
     * Verify that streaming encryption produces valid ciphertexts
     * 
     * @param plaintexts Vector of plaintexts
     * @param pk Public key
     * @param sk Secret key for verification
     * @return True if all ciphertexts decrypt correctly
     */
    bool verify_stream_encrypt_equivalence(const std::vector<Plaintext>& plaintexts,
                                           const PublicKey& pk,
                                           const SecretKey& sk);
    
    /**
     * Compare two ciphertexts for bit-identical equality
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @return True if ciphertexts are bit-identical
     */
    static bool ciphertexts_equal(const Ciphertext& ct1, const Ciphertext& ct2);
    
private:
    ParameterSet params_;
    std::unique_ptr<EncryptionEngine> engine_;
    std::unique_ptr<ChunkedCiphertextProcessor> chunked_processor_;
};

// Factory functions
std::unique_ptr<ChunkedCiphertextProcessor> create_chunked_processor(
    const ParameterSet& params,
    const StreamingConfig& config = StreamingConfig()
);

std::unique_ptr<CiphertextStreamProcessor> create_stream_processor(
    const ParameterSet& params,
    const StreamingConfig& config = StreamingConfig()
);

} // namespace fhe_accelerate
