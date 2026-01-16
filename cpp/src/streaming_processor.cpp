/**
 * @file streaming_processor.cpp
 * @brief Streaming Operations Implementation
 * 
 * Implements chunked ciphertext processing, async streaming interface,
 * and streaming equivalence verification.
 * 
 * Requirements: 11.1, 11.2, 11.3, 11.6
 */

#include "streaming_processor.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <thread>

namespace fhe_accelerate {

// ============================================================================
// ChunkedCiphertextProcessor Implementation
// ============================================================================

ChunkedCiphertextProcessor::ChunkedCiphertextProcessor(
    const ParameterSet& params,
    const StreamingConfig& config
)
    : params_(params)
    , config_(config)
{
    engine_ = std::make_unique<EncryptionEngine>(params);
}

ChunkedCiphertextProcessor::~ChunkedCiphertextProcessor() = default;

size_t ChunkedCiphertextProcessor::calculate_chunk_count(const Ciphertext& ct) const {
    size_t poly_degree = ct.c0.degree();
    if (poly_degree <= config_.chunk_size) {
        return 1;  // No chunking needed
    }
    
    // Calculate number of chunks needed
    size_t effective_chunk_size = config_.chunk_size - config_.boundary_overlap;
    return (poly_degree + effective_chunk_size - 1) / effective_chunk_size;
}

std::vector<CiphertextChunk> ChunkedCiphertextProcessor::split_ciphertext(
    const Ciphertext& ct
) {
    std::vector<CiphertextChunk> chunks;
    
    size_t poly_degree = ct.c0.degree();
    size_t total_chunks = calculate_chunk_count(ct);
    
    if (total_chunks == 1) {
        // No chunking needed - return single chunk with full ciphertext
        ChunkMetadata meta(0, 1, 0, poly_degree, true, true);
        chunks.emplace_back(
            Ciphertext(ct.c0.clone(), ct.c1.clone(), ct.noise_budget, ct.key_id, ct.is_ntt),
            meta
        );
        return chunks;
    }
    
    // Calculate effective chunk size (accounting for overlap)
    size_t effective_chunk_size = config_.chunk_size - config_.boundary_overlap;
    uint64_t modulus = ct.c0.modulus();
    
    for (size_t i = 0; i < total_chunks; ++i) {
        size_t start_offset = i * effective_chunk_size;
        size_t end_offset = std::min(start_offset + config_.chunk_size, poly_degree);
        size_t chunk_size = end_offset - start_offset;
        
        bool is_first = (i == 0);
        bool is_last = (i == total_chunks - 1);
        
        ChunkMetadata meta(i, total_chunks, start_offset, chunk_size, is_first, is_last);
        
        // Extract coefficients for this chunk
        std::vector<uint64_t> c0_coeffs(chunk_size);
        std::vector<uint64_t> c1_coeffs(chunk_size);
        
        for (size_t j = 0; j < chunk_size; ++j) {
            size_t src_idx = start_offset + j;
            if (src_idx < poly_degree) {
                c0_coeffs[j] = ct.c0[src_idx];
                c1_coeffs[j] = ct.c1[src_idx];
            } else {
                c0_coeffs[j] = 0;
                c1_coeffs[j] = 0;
            }
        }
        
        Polynomial c0_chunk(std::move(c0_coeffs), modulus, ct.is_ntt);
        Polynomial c1_chunk(std::move(c1_coeffs), modulus, ct.is_ntt);
        
        CiphertextChunk chunk(
            Ciphertext(std::move(c0_chunk), std::move(c1_chunk), 
                       ct.noise_budget, ct.key_id, ct.is_ntt),
            meta
        );
        
        // Extract boundary data
        if (!is_first) {
            chunk.left_boundary = extract_boundary(ct.c0, start_offset, 
                                                   config_.boundary_overlap);
        }
        if (!is_last) {
            size_t right_start = end_offset - config_.boundary_overlap;
            chunk.right_boundary = extract_boundary(ct.c0, right_start,
                                                    config_.boundary_overlap);
        }
        
        chunks.push_back(std::move(chunk));
    }
    
    return chunks;
}

Ciphertext ChunkedCiphertextProcessor::merge_chunks(
    const std::vector<CiphertextChunk>& chunks
) {
    if (chunks.empty()) {
        throw std::invalid_argument("Cannot merge empty chunk vector");
    }
    
    if (chunks.size() == 1) {
        // Single chunk - just return a copy
        const auto& chunk = chunks[0];
        return Ciphertext(chunk.data.c0.clone(), chunk.data.c1.clone(),
                          chunk.data.noise_budget, chunk.data.key_id, chunk.data.is_ntt);
    }
    
    // Calculate total size
    size_t total_size = 0;
    for (const auto& chunk : chunks) {
        if (chunk.metadata.is_last) {
            total_size = chunk.metadata.start_offset + chunk.metadata.chunk_size;
        }
    }
    
    // Adjust for overlap
    size_t effective_chunk_size = config_.chunk_size - config_.boundary_overlap;
    total_size = chunks[0].metadata.chunk_size + 
                 (chunks.size() - 1) * effective_chunk_size;
    
    // Use the original polynomial degree from params
    total_size = params_.poly_degree;
    
    uint64_t modulus = chunks[0].data.c0.modulus();
    bool is_ntt = chunks[0].data.is_ntt;
    uint64_t key_id = chunks[0].data.key_id;
    double noise_budget = chunks[0].data.noise_budget;
    
    // Merge coefficients
    std::vector<uint64_t> c0_merged(total_size, 0);
    std::vector<uint64_t> c1_merged(total_size, 0);
    
    for (const auto& chunk : chunks) {
        size_t start = chunk.metadata.start_offset;
        size_t chunk_size = chunk.metadata.chunk_size;
        
        // Handle overlap by averaging or taking the non-overlapping portion
        size_t copy_start = chunk.metadata.is_first ? 0 : config_.boundary_overlap / 2;
        size_t copy_end = chunk.metadata.is_last ? chunk_size : 
                          chunk_size - config_.boundary_overlap / 2;
        
        for (size_t j = copy_start; j < copy_end && (start + j) < total_size; ++j) {
            size_t dst_idx = start + j - (chunk.metadata.is_first ? 0 : config_.boundary_overlap / 2);
            if (dst_idx < total_size && j < chunk.data.c0.degree()) {
                c0_merged[dst_idx] = chunk.data.c0[j];
                c1_merged[dst_idx] = chunk.data.c1[j];
            }
        }
        
        // Update noise budget (take minimum)
        noise_budget = std::min(noise_budget, chunk.data.noise_budget);
    }
    
    Polynomial c0_poly(std::move(c0_merged), modulus, is_ntt);
    Polynomial c1_poly(std::move(c1_merged), modulus, is_ntt);
    
    return Ciphertext(std::move(c0_poly), std::move(c1_poly), 
                      noise_budget, key_id, is_ntt);
}

std::vector<uint64_t> ChunkedCiphertextProcessor::extract_boundary(
    const Polynomial& poly,
    size_t start,
    size_t count
) const {
    std::vector<uint64_t> boundary(count);
    size_t poly_size = poly.degree();
    
    for (size_t i = 0; i < count; ++i) {
        size_t idx = start + i;
        boundary[i] = (idx < poly_size) ? poly[idx] : 0;
    }
    
    return boundary;
}

void ChunkedCiphertextProcessor::apply_boundary_correction(
    CiphertextChunk& chunk,
    const std::vector<uint64_t>& left_boundary,
    const std::vector<uint64_t>& right_boundary
) {
    // For addition operations, boundary correction is straightforward
    // For multiplication, we need to handle polynomial convolution at boundaries
    
    // This is a simplified implementation for addition-like operations
    // More complex operations would need specialized boundary handling
    
    // Left boundary correction
    if (!left_boundary.empty() && !chunk.metadata.is_first) {
        // Blend left boundary coefficients
        size_t blend_count = std::min(left_boundary.size(), 
                                      static_cast<size_t>(chunk.data.c0.degree()));
        for (size_t i = 0; i < blend_count; ++i) {
            // For addition, we don't need to modify - boundaries are independent
            // This is a placeholder for more complex operations
        }
    }
    
    // Right boundary correction
    if (!right_boundary.empty() && !chunk.metadata.is_last) {
        // Similar handling for right boundary
    }
}

Ciphertext ChunkedCiphertextProcessor::chunked_add(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    StreamingProgressCallback progress
) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Split both ciphertexts into chunks
    auto chunks1 = split_ciphertext(ct1);
    auto chunks2 = split_ciphertext(ct2);
    
    if (chunks1.size() != chunks2.size()) {
        throw std::invalid_argument("Ciphertexts must have same chunk count for addition");
    }
    
    size_t total_chunks = chunks1.size();
    std::vector<CiphertextChunk> result_chunks;
    result_chunks.reserve(total_chunks);
    
    // Process each chunk pair
    for (size_t i = 0; i < total_chunks; ++i) {
        // Add corresponding chunks
        Ciphertext chunk_sum = engine_->add(chunks1[i].data, chunks2[i].data);
        
        CiphertextChunk result_chunk(std::move(chunk_sum), chunks1[i].metadata);
        result_chunks.push_back(std::move(result_chunk));
        
        // Report progress
        if (progress) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(now - start_time).count();
            double estimated_remaining = (elapsed / (i + 1)) * (total_chunks - i - 1);
            
            progress(StreamingProgress("chunked_add", i + 1, total_chunks, 
                                       elapsed, estimated_remaining));
        }
    }
    
    // Merge chunks back into single ciphertext
    return merge_chunks(result_chunks);
}

Ciphertext ChunkedCiphertextProcessor::chunked_multiply(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    const EvaluationKey& ek,
    StreamingProgressCallback progress
) {
    auto start_time = std::chrono::steady_clock::now();
    
    // For multiplication, we need special handling because polynomial
    // multiplication causes coefficients to interact across the entire polynomial.
    // 
    // Strategy: For large polynomials, we use NTT-based multiplication which
    // is naturally parallelizable. The chunking here is for memory management
    // rather than independent processing.
    
    // For now, fall back to non-chunked multiplication for correctness
    // A full implementation would use overlap-save or overlap-add methods
    
    if (progress) {
        progress(StreamingProgress("multiply_start", 0, 1, 0, 0));
    }
    
    Ciphertext result = engine_->multiply_relin(ct1, ct2, ek);
    
    if (progress) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(now - start_time).count();
        progress(StreamingProgress("multiply_complete", 1, 1, elapsed, 0));
    }
    
    return result;
}

CiphertextChunk ChunkedCiphertextProcessor::add_chunk_to_accumulator(
    const CiphertextChunk& accumulator,
    const CiphertextChunk& chunk
) {
    // Verify chunks are compatible
    if (accumulator.metadata.chunk_index != chunk.metadata.chunk_index) {
        throw std::invalid_argument("Chunk indices must match for accumulation");
    }
    
    // Add the chunks
    Ciphertext sum = engine_->add(accumulator.data, chunk.data);
    
    // Create result chunk with updated metadata
    ChunkMetadata result_meta = accumulator.metadata;
    
    return CiphertextChunk(std::move(sum), result_meta);
}

CiphertextChunk ChunkedCiphertextProcessor::create_chunk(
    const Ciphertext& ct,
    size_t chunk_idx,
    size_t total_chunks,
    size_t start_offset
) {
    size_t poly_degree = ct.c0.degree();
    size_t chunk_size = std::min(config_.chunk_size, poly_degree - start_offset);
    
    bool is_first = (chunk_idx == 0);
    bool is_last = (chunk_idx == total_chunks - 1);
    
    ChunkMetadata meta(chunk_idx, total_chunks, start_offset, chunk_size, is_first, is_last);
    
    // Extract coefficients
    uint64_t modulus = ct.c0.modulus();
    std::vector<uint64_t> c0_coeffs(chunk_size);
    std::vector<uint64_t> c1_coeffs(chunk_size);
    
    for (size_t j = 0; j < chunk_size; ++j) {
        c0_coeffs[j] = ct.c0[start_offset + j];
        c1_coeffs[j] = ct.c1[start_offset + j];
    }
    
    Polynomial c0_chunk(std::move(c0_coeffs), modulus, ct.is_ntt);
    Polynomial c1_chunk(std::move(c1_coeffs), modulus, ct.is_ntt);
    
    return CiphertextChunk(
        Ciphertext(std::move(c0_chunk), std::move(c1_chunk),
                   ct.noise_budget, ct.key_id, ct.is_ntt),
        meta
    );
}

// ============================================================================
// CiphertextStreamProcessor Implementation
// ============================================================================

CiphertextStreamProcessor::CiphertextStreamProcessor(
    const ParameterSet& params,
    const StreamingConfig& config
)
    : params_(params)
    , config_(config)
    , active_(false)
    , cancelled_(false)
{
    engine_ = std::make_unique<EncryptionEngine>(params);
    chunked_processor_ = std::make_unique<ChunkedCiphertextProcessor>(params, config);
}

CiphertextStreamProcessor::~CiphertextStreamProcessor() {
    cancel();
    wait();
}

void CiphertextStreamProcessor::update_progress(
    StreamingProgressCallback& callback,
    const std::string& stage,
    size_t current,
    size_t total,
    std::chrono::steady_clock::time_point start_time
) {
    if (!callback) return;
    
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(now - start_time).count();
    double estimated_remaining = 0;
    
    if (current > 0 && total > current) {
        estimated_remaining = (elapsed / current) * (total - current);
    }
    
    callback(StreamingProgress(stage, current, total, elapsed, estimated_remaining));
}

std::future<StreamingResult> CiphertextStreamProcessor::process_stream(
    std::function<std::optional<Ciphertext>()> input_stream,
    std::function<Ciphertext(const Ciphertext&)> transform,
    std::function<void(Ciphertext&&)> output,
    StreamingProgressCallback progress
) {
    return std::async(std::launch::async, [this, input_stream, transform, output, progress]() {
        StreamingResult result;
        auto start_time = std::chrono::steady_clock::now();
        
        active_.store(true);
        cancelled_.store(false);
        
        size_t processed = 0;
        
        try {
            while (!cancelled_.load()) {
                auto input = input_stream();
                if (!input.has_value()) {
                    break;  // End of stream
                }
                
                // Apply transform
                Ciphertext transformed = transform(input.value());
                
                // Output result
                output(std::move(transformed));
                
                processed++;
                
                // Update progress (estimate total based on processed so far)
                if (progress) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double, std::milli>(
                        now - start_time).count();
                    progress(StreamingProgress("processing", processed, processed, elapsed, 0));
                }
            }
            
            result.success = !cancelled_.load();
            result.chunks_processed = processed;
            
        } catch (const std::exception& e) {
            result = StreamingResult::failure(e.what());
            result.chunks_processed = processed;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        result.total_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        if (result.total_time_ms > 0) {
            result.throughput_chunks_per_sec = 
                (processed * 1000.0) / result.total_time_ms;
        }
        
        active_.store(false);
        return result;
    });
}

std::future<StreamingResult> CiphertextStreamProcessor::stream_add(
    std::function<std::optional<Ciphertext>()> input_stream,
    std::function<void(Ciphertext&&)> output,
    StreamingProgressCallback progress
) {
    return std::async(std::launch::async, [this, input_stream, output, progress]() {
        StreamingResult result;
        auto start_time = std::chrono::steady_clock::now();
        
        active_.store(true);
        cancelled_.store(false);
        
        size_t processed = 0;
        std::optional<Ciphertext> accumulator;
        
        try {
            while (!cancelled_.load()) {
                auto input = input_stream();
                if (!input.has_value()) {
                    break;  // End of stream
                }
                
                if (!accumulator.has_value()) {
                    // First ciphertext becomes the accumulator
                    accumulator = std::move(input.value());
                } else {
                    // Add to accumulator
                    accumulator = engine_->add(accumulator.value(), input.value());
                }
                
                // Output running total
                output(Ciphertext(accumulator->c0.clone(), accumulator->c1.clone(),
                                  accumulator->noise_budget, accumulator->key_id,
                                  accumulator->is_ntt));
                
                processed++;
                
                // Update progress
                if (progress) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double, std::milli>(
                        now - start_time).count();
                    progress(StreamingProgress("stream_add", processed, processed, elapsed, 0));
                }
            }
            
            result.success = !cancelled_.load();
            result.chunks_processed = processed;
            
        } catch (const std::exception& e) {
            result = StreamingResult::failure(e.what());
            result.chunks_processed = processed;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        result.total_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        if (result.total_time_ms > 0) {
            result.throughput_chunks_per_sec = 
                (processed * 1000.0) / result.total_time_ms;
        }
        
        active_.store(false);
        return result;
    });
}

std::future<StreamingResult> CiphertextStreamProcessor::stream_encrypt(
    std::function<std::optional<Plaintext>()> input_stream,
    const PublicKey& pk,
    std::function<void(Ciphertext&&)> output,
    StreamingProgressCallback progress
) {
    return std::async(std::launch::async, [this, input_stream, &pk, output, progress]() {
        StreamingResult result;
        auto start_time = std::chrono::steady_clock::now();
        
        active_.store(true);
        cancelled_.store(false);
        
        size_t processed = 0;
        
        try {
            while (!cancelled_.load()) {
                auto input = input_stream();
                if (!input.has_value()) {
                    break;  // End of stream
                }
                
                // Encrypt the plaintext
                Ciphertext ct = engine_->encrypt(input.value(), pk);
                
                // Output encrypted ciphertext
                output(std::move(ct));
                
                processed++;
                
                // Update progress
                if (progress) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double, std::milli>(
                        now - start_time).count();
                    progress(StreamingProgress("stream_encrypt", processed, processed, 
                                               elapsed, 0));
                }
            }
            
            result.success = !cancelled_.load();
            result.chunks_processed = processed;
            
        } catch (const std::exception& e) {
            result = StreamingResult::failure(e.what());
            result.chunks_processed = processed;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        result.total_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        if (result.total_time_ms > 0) {
            result.throughput_chunks_per_sec = 
                (processed * 1000.0) / result.total_time_ms;
        }
        
        active_.store(false);
        return result;
    });
}

void CiphertextStreamProcessor::cancel() {
    cancelled_.store(true);
}

void CiphertextStreamProcessor::wait() {
    // Wait for active_ to become false
    while (active_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// ============================================================================
// StreamingEquivalenceVerifier Implementation
// ============================================================================

StreamingEquivalenceVerifier::StreamingEquivalenceVerifier(const ParameterSet& params)
    : params_(params)
{
    engine_ = std::make_unique<EncryptionEngine>(params);
    chunked_processor_ = std::make_unique<ChunkedCiphertextProcessor>(params);
}

bool StreamingEquivalenceVerifier::ciphertexts_equal(
    const Ciphertext& ct1,
    const Ciphertext& ct2
) {
    // Check basic properties
    if (ct1.key_id != ct2.key_id) return false;
    if (ct1.is_ntt != ct2.is_ntt) return false;
    if (ct1.is_degree_2() != ct2.is_degree_2()) return false;
    
    // Check c0 coefficients
    if (ct1.c0.degree() != ct2.c0.degree()) return false;
    for (size_t i = 0; i < ct1.c0.degree(); ++i) {
        if (ct1.c0[i] != ct2.c0[i]) return false;
    }
    
    // Check c1 coefficients
    if (ct1.c1.degree() != ct2.c1.degree()) return false;
    for (size_t i = 0; i < ct1.c1.degree(); ++i) {
        if (ct1.c1[i] != ct2.c1[i]) return false;
    }
    
    // Check c2 if present
    if (ct1.is_degree_2()) {
        const auto& c2_1 = ct1.c2.value();
        const auto& c2_2 = ct2.c2.value();
        if (c2_1.degree() != c2_2.degree()) return false;
        for (size_t i = 0; i < c2_1.degree(); ++i) {
            if (c2_1[i] != c2_2[i]) return false;
        }
    }
    
    return true;
}

bool StreamingEquivalenceVerifier::verify_add_equivalence(
    const Ciphertext& ct1,
    const Ciphertext& ct2
) {
    // Compute non-chunked result
    Ciphertext non_chunked_result = engine_->add(ct1, ct2);
    
    // Compute chunked result
    Ciphertext chunked_result = chunked_processor_->chunked_add(ct1, ct2);
    
    // Compare results
    return ciphertexts_equal(non_chunked_result, chunked_result);
}

bool StreamingEquivalenceVerifier::verify_multiply_equivalence(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    const EvaluationKey& ek
) {
    // Compute non-chunked result
    Ciphertext non_chunked_result = engine_->multiply_relin(ct1, ct2, ek);
    
    // Compute chunked result
    Ciphertext chunked_result = chunked_processor_->chunked_multiply(ct1, ct2, ek);
    
    // Compare results
    return ciphertexts_equal(non_chunked_result, chunked_result);
}

bool StreamingEquivalenceVerifier::verify_stream_encrypt_equivalence(
    const std::vector<Plaintext>& plaintexts,
    const PublicKey& pk,
    const SecretKey& sk
) {
    // Encrypt using streaming
    std::vector<Ciphertext> stream_results;
    size_t idx = 0;
    
    CiphertextStreamProcessor stream_processor(params_);
    
    auto input_stream = [&]() -> std::optional<Plaintext> {
        if (idx < plaintexts.size()) {
            return plaintexts[idx++];
        }
        return std::nullopt;
    };
    
    auto output = [&](Ciphertext&& ct) {
        stream_results.push_back(std::move(ct));
    };
    
    auto future = stream_processor.stream_encrypt(input_stream, pk, output);
    auto result = future.get();
    
    if (!result.success) {
        return false;
    }
    
    // Verify all ciphertexts decrypt correctly
    for (size_t i = 0; i < plaintexts.size(); ++i) {
        auto decrypted = engine_->decrypt_value(stream_results[i], sk);
        if (!decrypted.has_value()) {
            return false;
        }
        if (decrypted.value() != plaintexts[i].value()) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<ChunkedCiphertextProcessor> create_chunked_processor(
    const ParameterSet& params,
    const StreamingConfig& config
) {
    return std::make_unique<ChunkedCiphertextProcessor>(params, config);
}

std::unique_ptr<CiphertextStreamProcessor> create_stream_processor(
    const ParameterSet& params,
    const StreamingConfig& config
) {
    return std::make_unique<CiphertextStreamProcessor>(params, config);
}

} // namespace fhe_accelerate
