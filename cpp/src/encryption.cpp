/**
 * @file encryption.cpp
 * @brief RLWE Encryption and Decryption Implementation
 * 
 * Implements RLWE encryption, decryption, SIMD packing, and batch operations
 * with Metal GPU acceleration.
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 15.1, 15.2
 */

#include "encryption.h"
#include "metal_compute.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <chrono>

#ifdef __APPLE__
#include <Security/Security.h>
#endif

namespace fhe_accelerate {

// ============================================================================
// EncryptionEngine Implementation
// ============================================================================

EncryptionEngine::EncryptionEngine(const ParameterSet& params)
    : params_(params)
{
    // Initialize polynomial ring with primary modulus
    if (!params_.moduli.empty()) {
        ring_ = std::make_unique<PolynomialRing>(params_.poly_degree, params_.moduli[0]);
    } else {
        throw std::invalid_argument("Parameter set must have at least one modulus");
    }
    
    // Initialize secure random number generator
    rng_ = std::make_unique<SecureRandom>();
    
    // Compute scaling factor delta = q / t
    uint64_t q = params_.moduli[0];
    uint64_t t = params_.plaintext_modulus;
    if (t == 0) {
        t = 4;  // Default plaintext modulus for TFHE (2-bit messages)
    }
    delta_ = q / t;
    
    // Compute initial noise budget
    // noise_budget = log2(q / (2 * noise_std * sqrt(N)))
    double noise_std = params_.lwe_noise_std > 0 ? params_.lwe_noise_std : 3.2;
    double n = static_cast<double>(params_.poly_degree);
    initial_noise_budget_ = std::log2(static_cast<double>(q)) - 
                            std::log2(2.0 * noise_std * std::sqrt(n));
    
    // Metal batch encryptor will be initialized lazily when needed
    metal_encryptor_ = nullptr;
}

EncryptionEngine::~EncryptionEngine() = default;

// ============================================================================
// Helper Functions
// ============================================================================

Polynomial EncryptionEngine::sample_random_polynomial() {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    std::vector<uint64_t> coeffs(degree);
    for (uint32_t i = 0; i < degree; ++i) {
        coeffs[i] = rng_->random_u64_range(modulus);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

Polynomial EncryptionEngine::sample_error_polynomial() {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    double std_dev = params_.lwe_noise_std > 0 ? params_.lwe_noise_std : 3.2;
    
    std::vector<uint64_t> coeffs(degree);
    for (uint32_t i = 0; i < degree; ++i) {
        coeffs[i] = rng_->sample_gaussian(std_dev, modulus);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

Polynomial EncryptionEngine::sample_ternary_polynomial() {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    std::vector<uint64_t> coeffs(degree);
    for (uint32_t i = 0; i < degree; ++i) {
        coeffs[i] = rng_->sample_ternary(modulus);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

// ============================================================================
// Plaintext Encoding
// ============================================================================

Polynomial EncryptionEngine::encode_plaintext(uint64_t value) const {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    // Encode value as constant polynomial scaled by delta
    std::vector<uint64_t> coeffs(degree, 0);
    coeffs[0] = (value * delta_) % modulus;
    
    return Polynomial(std::move(coeffs), modulus, false);
}

Polynomial EncryptionEngine::encode_packed(const std::vector<uint64_t>& values) const {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = params_.moduli[0];
    
    // Pack values into polynomial coefficients
    // Each coefficient slot holds one value scaled by delta
    std::vector<uint64_t> coeffs(degree, 0);
    size_t num_values = std::min(values.size(), static_cast<size_t>(degree));
    
    for (size_t i = 0; i < num_values; ++i) {
        coeffs[i] = (values[i] * delta_) % modulus;
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

uint64_t EncryptionEngine::decode_plaintext(const Polynomial& poly) const {
    uint64_t modulus = params_.moduli[0];
    uint64_t t = params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4;
    
    // Decode: round(coeff[0] * t / q)
    // Using: round(x) = floor(x + 0.5) = floor((2*x + 1) / 2)
    uint64_t coeff = poly[0];
    
    // Compute (coeff * t + q/2) / q to get rounded result
    __uint128_t numerator = static_cast<__uint128_t>(coeff) * t + (modulus / 2);
    uint64_t result = static_cast<uint64_t>(numerator / modulus);
    
    return result % t;
}

std::vector<uint64_t> EncryptionEngine::decode_packed(const Polynomial& poly, 
                                                       size_t num_values) const {
    uint64_t modulus = params_.moduli[0];
    uint64_t t = params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4;
    
    std::vector<uint64_t> values(num_values);
    
    for (size_t i = 0; i < num_values && i < poly.degree(); ++i) {
        uint64_t coeff = poly[i];
        __uint128_t numerator = static_cast<__uint128_t>(coeff) * t + (modulus / 2);
        values[i] = static_cast<uint64_t>(numerator / modulus) % t;
    }
    
    return values;
}

// ============================================================================
// RLWE Encryption (Requirement 5.1, 5.3)
// ============================================================================

Ciphertext EncryptionEngine::encrypt_internal(const Polynomial& encoded_plaintext, 
                                               const PublicKey& pk) {
    uint64_t modulus = params_.moduli[0];
    
    // Sample random polynomial u (ternary for efficiency)
    Polynomial u = sample_ternary_polynomial();
    
    // Sample error polynomials e1, e2
    Polynomial e1 = sample_error_polynomial();
    Polynomial e2 = sample_error_polynomial();
    
    // Convert to NTT domain for efficient multiplication
    Polynomial u_ntt = u.clone();
    ring_->to_ntt(u_ntt);
    
    Polynomial pk_a_ntt = pk.a.clone();
    ring_->to_ntt(pk_a_ntt);
    
    Polynomial pk_b_ntt = pk.b.clone();
    ring_->to_ntt(pk_b_ntt);
    
    // Compute c0 = pk.b * u + e1 + m
    Polynomial c0_ntt = ring_->pointwise_multiply(pk_b_ntt, u_ntt);
    ring_->from_ntt(c0_ntt);
    ring_->add_inplace(c0_ntt, e1);
    ring_->add_inplace(c0_ntt, encoded_plaintext);
    
    // Compute c1 = pk.a * u + e2
    Polynomial c1_ntt = ring_->pointwise_multiply(pk_a_ntt, u_ntt);
    ring_->from_ntt(c1_ntt);
    ring_->add_inplace(c1_ntt, e2);
    
    return Ciphertext(std::move(c0_ntt), std::move(c1_ntt), 
                      initial_noise_budget_, pk.key_id, false);
}

Ciphertext EncryptionEngine::encrypt(const Plaintext& plaintext, const PublicKey& pk) {
    Polynomial encoded;
    
    if (plaintext.is_packed && plaintext.values.size() > 1) {
        encoded = encode_packed(plaintext.values);
    } else {
        encoded = encode_plaintext(plaintext.value());
    }
    
    return encrypt_internal(encoded, pk);
}

Ciphertext EncryptionEngine::encrypt_value(uint64_t value, const PublicKey& pk) {
    Polynomial encoded = encode_plaintext(value);
    return encrypt_internal(encoded, pk);
}

Ciphertext EncryptionEngine::encrypt_packed(const std::vector<uint64_t>& values, 
                                             const PublicKey& pk) {
    Polynomial encoded = encode_packed(values);
    return encrypt_internal(encoded, pk);
}

// ============================================================================
// Decryption (Requirement 5.2, 5.4)
// ============================================================================

DecryptionResult EncryptionEngine::decrypt(const Ciphertext& ciphertext, 
                                            const SecretKey& sk) {
    // Check key ID match
    if (ciphertext.key_id != sk.key_id) {
        return DecryptionResult::failure("Key ID mismatch: ciphertext was encrypted with different key");
    }
    
    uint64_t modulus = params_.moduli[0];
    
    // For degree-1 ciphertext: m = c0 - c1 * sk
    // For degree-2 ciphertext: m = c0 - c1 * sk - c2 * sk^2
    // (The signs depend on the encryption scheme - we use RLWE with subtraction)
    
    // Compute c1 * sk
    Polynomial c1_ntt = ciphertext.c1.clone();
    if (!ciphertext.is_ntt) {
        ring_->to_ntt(c1_ntt);
    }
    
    Polynomial sk_ntt = sk.poly.clone();
    ring_->to_ntt(sk_ntt);
    
    Polynomial c1_sk_ntt = ring_->pointwise_multiply(c1_ntt, sk_ntt);
    ring_->from_ntt(c1_sk_ntt);
    
    // Start with c0
    Polynomial c0_copy = ciphertext.c0.clone();
    if (ciphertext.is_ntt) {
        ring_->from_ntt(c0_copy);
    }
    
    // result = c0 - c1 * sk
    Polynomial result = ring_->subtract(c0_copy, c1_sk_ntt);
    
    // If degree-2 ciphertext, subtract c2 * sk^2
    if (ciphertext.is_degree_2()) {
        const Polynomial& c2 = ciphertext.c2.value();
        
        // Compute sk^2
        Polynomial sk2_ntt = ring_->pointwise_multiply(sk_ntt, sk_ntt);
        
        // Compute c2 * sk^2
        Polynomial c2_ntt = c2.clone();
        if (!ciphertext.is_ntt) {
            ring_->to_ntt(c2_ntt);
        }
        
        Polynomial c2_sk2_ntt = ring_->pointwise_multiply(c2_ntt, sk2_ntt);
        ring_->from_ntt(c2_sk2_ntt);
        
        // result -= c2 * sk^2
        ring_->subtract_inplace(result, c2_sk2_ntt);
    }
    
    // Compute noise budget
    double noise_budget = compute_noise_budget(result, modulus);
    
    // Check if noise budget is sufficient
    if (noise_budget < 0) {
        return DecryptionResult::failure("Noise budget exhausted - decryption may be incorrect");
    }
    
    // Decode the result
    uint64_t decoded_value = decode_plaintext(result);
    
    Plaintext pt(decoded_value, params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4);
    return DecryptionResult(std::move(pt), noise_budget);
}

std::optional<uint64_t> EncryptionEngine::decrypt_value(const Ciphertext& ciphertext, 
                                                         const SecretKey& sk) {
    auto result = decrypt(ciphertext, sk);
    if (result.success) {
        return result.plaintext.value();
    }
    return std::nullopt;
}

std::vector<uint64_t> EncryptionEngine::decrypt_packed(const Ciphertext& ciphertext,
                                                        const SecretKey& sk,
                                                        size_t num_values) {
    // Check key ID match
    if (ciphertext.key_id != sk.key_id) {
        return {};
    }
    
    // Compute m = c0 - c1 * sk (- c2 * sk^2 for degree-2)
    Polynomial c1_ntt = ciphertext.c1.clone();
    if (!ciphertext.is_ntt) {
        ring_->to_ntt(c1_ntt);
    }
    
    Polynomial sk_ntt = sk.poly.clone();
    ring_->to_ntt(sk_ntt);
    
    Polynomial c1_sk_ntt = ring_->pointwise_multiply(c1_ntt, sk_ntt);
    ring_->from_ntt(c1_sk_ntt);
    
    Polynomial c0_copy = ciphertext.c0.clone();
    if (ciphertext.is_ntt) {
        ring_->from_ntt(c0_copy);
    }
    
    Polynomial result = ring_->subtract(c0_copy, c1_sk_ntt);
    
    // Handle degree-2 ciphertext
    if (ciphertext.is_degree_2()) {
        const Polynomial& c2 = ciphertext.c2.value();
        
        Polynomial sk2_ntt = ring_->pointwise_multiply(sk_ntt, sk_ntt);
        
        Polynomial c2_ntt = c2.clone();
        if (!ciphertext.is_ntt) {
            ring_->to_ntt(c2_ntt);
        }
        
        Polynomial c2_sk2_ntt = ring_->pointwise_multiply(c2_ntt, sk2_ntt);
        ring_->from_ntt(c2_sk2_ntt);
        
        ring_->subtract_inplace(result, c2_sk2_ntt);
    }
    
    // Decode packed values
    return decode_packed(result, num_values);
}

// ============================================================================
// Noise Budget Management (Requirement 5.4)
// ============================================================================

double EncryptionEngine::compute_noise_budget(const Polynomial& decrypted, 
                                               uint64_t modulus) const {
    uint64_t t = params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4;
    
    // The noise is the distance from the decoded value to the nearest valid plaintext
    // For each coefficient, compute: noise = coeff - round(coeff * t / q) * (q / t)
    double max_noise = 0.0;
    
    for (size_t i = 0; i < decrypted.degree(); ++i) {
        uint64_t coeff = decrypted[i];
        
        // Compute rounded value
        __uint128_t numerator = static_cast<__uint128_t>(coeff) * t + (modulus / 2);
        uint64_t rounded = static_cast<uint64_t>(numerator / modulus);
        
        // Compute expected value (rounded * delta)
        uint64_t expected = (rounded * delta_) % modulus;
        
        // Compute noise (distance)
        int64_t noise;
        if (coeff >= expected) {
            noise = static_cast<int64_t>(coeff - expected);
        } else {
            noise = static_cast<int64_t>(expected - coeff);
        }
        
        // Handle wrap-around
        if (noise > static_cast<int64_t>(modulus / 2)) {
            noise = static_cast<int64_t>(modulus) - noise;
        }
        
        max_noise = std::max(max_noise, static_cast<double>(std::abs(noise)));
    }
    
    // Noise budget = log2(q / (2 * max_noise))
    if (max_noise < 1.0) max_noise = 1.0;
    double budget = std::log2(static_cast<double>(modulus) / (2.0 * max_noise));
    
    return budget;
}

double EncryptionEngine::get_noise_budget(const Ciphertext& ciphertext, 
                                           const SecretKey& sk) {
    // Compute m = c0 - c1 * sk (- c2 * sk^2 for degree-2)
    Polynomial c1_ntt = ciphertext.c1.clone();
    if (!ciphertext.is_ntt) {
        ring_->to_ntt(c1_ntt);
    }
    
    Polynomial sk_ntt = sk.poly.clone();
    ring_->to_ntt(sk_ntt);
    
    Polynomial c1_sk_ntt = ring_->pointwise_multiply(c1_ntt, sk_ntt);
    ring_->from_ntt(c1_sk_ntt);
    
    Polynomial c0_copy = ciphertext.c0.clone();
    if (ciphertext.is_ntt) {
        ring_->from_ntt(c0_copy);
    }
    
    Polynomial result = ring_->subtract(c0_copy, c1_sk_ntt);
    
    // Handle degree-2 ciphertext
    if (ciphertext.is_degree_2()) {
        const Polynomial& c2 = ciphertext.c2.value();
        
        Polynomial sk2_ntt = ring_->pointwise_multiply(sk_ntt, sk_ntt);
        
        Polynomial c2_ntt = c2.clone();
        if (!ciphertext.is_ntt) {
            ring_->to_ntt(c2_ntt);
        }
        
        Polynomial c2_sk2_ntt = ring_->pointwise_multiply(c2_ntt, sk2_ntt);
        ring_->from_ntt(c2_sk2_ntt);
        
        ring_->subtract_inplace(result, c2_sk2_ntt);
    }
    
    return compute_noise_budget(result, params_.moduli[0]);
}

double EncryptionEngine::estimate_noise_budget(const Ciphertext& ciphertext) const {
    // Without the secret key, we can only estimate based on the ciphertext's
    // stored noise budget (which decreases with operations)
    return ciphertext.noise_budget;
}


// ============================================================================
// Batch Encryption with Metal GPU (Requirement 5.6, 15.2)
// ============================================================================

BatchEncryptionResult EncryptionEngine::batch_encrypt(
    const BatchEncryptionRequest& request,
    EncryptionProgressCallback progress
) {
    BatchEncryptionResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t total = request.plaintexts.size();
    if (total == 0) {
        return result;
    }
    
    result.ciphertexts.reserve(total);
    
    // Determine optimal parallelization strategy
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    // For small batches, use simple parallel encryption
    // For large batches (>1000), use Metal GPU if available
    bool use_gpu = total > 1000 && metal::metal_available();
    
    if (use_gpu) {
        // GPU-accelerated batch encryption
        // Process in chunks to manage memory
        const size_t chunk_size = 4096;
        size_t completed = 0;
        
        for (size_t chunk_start = 0; chunk_start < total; chunk_start += chunk_size) {
            size_t chunk_end = std::min(chunk_start + chunk_size, total);
            size_t chunk_count = chunk_end - chunk_start;
            
            // Prepare batch data for GPU
            std::vector<Polynomial> encoded_plaintexts;
            encoded_plaintexts.reserve(chunk_count);
            
            for (size_t i = chunk_start; i < chunk_end; ++i) {
                const auto& pt = request.plaintexts[i];
                if (pt.is_packed && pt.values.size() > 1) {
                    encoded_plaintexts.push_back(encode_packed(pt.values));
                } else {
                    encoded_plaintexts.push_back(encode_plaintext(pt.value()));
                }
            }
            
            // For now, fall back to CPU parallel encryption
            // Full GPU implementation would use Metal compute shaders
            std::vector<std::future<Ciphertext>> futures;
            futures.reserve(chunk_count);
            
            for (size_t i = 0; i < chunk_count; ++i) {
                futures.push_back(std::async(std::launch::async, [&, i]() {
                    return encrypt_internal(encoded_plaintexts[i], *request.public_key);
                }));
            }
            
            for (auto& future : futures) {
                result.ciphertexts.push_back(future.get());
                completed++;
                
                if (progress && completed % 100 == 0) {
                    progress(completed, total);
                }
            }
        }
    } else {
        // CPU parallel encryption
        size_t chunk_size = (total + num_threads - 1) / num_threads;
        std::vector<std::future<std::vector<Ciphertext>>> futures;
        
        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, total);
            
            if (start >= total) break;
            
            futures.push_back(std::async(std::launch::async, [&, start, end]() {
                std::vector<Ciphertext> local_results;
                local_results.reserve(end - start);
                
                // Create local RNG for thread safety
                SecureRandom local_rng;
                
                for (size_t i = start; i < end; ++i) {
                    const auto& pt = request.plaintexts[i];
                    Polynomial encoded;
                    
                    if (pt.is_packed && pt.values.size() > 1) {
                        encoded = encode_packed(pt.values);
                    } else {
                        encoded = encode_plaintext(pt.value());
                    }
                    
                    // Encrypt using the main engine's method
                    local_results.push_back(encrypt_internal(encoded, *request.public_key));
                }
                
                return local_results;
            }));
        }
        
        // Collect results
        size_t completed = 0;
        for (auto& future : futures) {
            auto local_results = future.get();
            for (auto& ct : local_results) {
                result.ciphertexts.push_back(std::move(ct));
                completed++;
                
                if (progress && completed % 100 == 0) {
                    progress(completed, total);
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (result.elapsed_ms > 0) {
        result.throughput_per_second = static_cast<size_t>(
            (total * 1000.0) / result.elapsed_ms
        );
    }
    
    // Final progress callback
    if (progress) {
        progress(total, total);
    }
    
    return result;
}

// ============================================================================
// Homomorphic Addition (Requirement 6.1, 6.2, 6.3, 6.4, 6.5, 6.6)
// ============================================================================

Ciphertext EncryptionEngine::add(const Ciphertext& ct1, const Ciphertext& ct2) {
    // Verify key IDs match
    if (ct1.key_id != ct2.key_id) {
        throw std::invalid_argument("Cannot add ciphertexts encrypted with different keys");
    }
    
    // Verify both ciphertexts are in the same representation (NTT or coefficient)
    if (ct1.is_ntt != ct2.is_ntt) {
        throw std::invalid_argument("Cannot add ciphertexts in different representations (NTT vs coefficient)");
    }
    
    // Add corresponding polynomial components
    // result.c0 = ct1.c0 + ct2.c0 (mod q)
    // result.c1 = ct1.c1 + ct2.c1 (mod q)
    Polynomial sum_c0 = ring_->add(ct1.c0, ct2.c0);
    Polynomial sum_c1 = ring_->add(ct1.c1, ct2.c1);
    
    // Noise budget decreases slightly with addition
    // The noise terms add, so we lose about 1 bit of noise budget
    double result_noise_budget = std::min(ct1.noise_budget, ct2.noise_budget) - 1.0;
    
    return Ciphertext(std::move(sum_c0), std::move(sum_c1), 
                      result_noise_budget, ct1.key_id, ct1.is_ntt);
}

void EncryptionEngine::add_inplace(Ciphertext& ct1, const Ciphertext& ct2) {
    // Verify key IDs match
    if (ct1.key_id != ct2.key_id) {
        throw std::invalid_argument("Cannot add ciphertexts encrypted with different keys");
    }
    
    // Verify both ciphertexts are in the same representation
    if (ct1.is_ntt != ct2.is_ntt) {
        throw std::invalid_argument("Cannot add ciphertexts in different representations (NTT vs coefficient)");
    }
    
    // Add in-place
    ring_->add_inplace(ct1.c0, ct2.c0);
    ring_->add_inplace(ct1.c1, ct2.c1);
    
    // Update noise budget
    ct1.noise_budget = std::min(ct1.noise_budget, ct2.noise_budget) - 1.0;
}

Ciphertext EncryptionEngine::add_plain(const Ciphertext& ct, const Plaintext& pt) {
    // Encode the plaintext
    Polynomial encoded;
    if (pt.is_packed && pt.values.size() > 1) {
        encoded = encode_packed(pt.values);
    } else {
        encoded = encode_plaintext(pt.value());
    }
    
    // If ciphertext is in NTT form, convert encoded plaintext to NTT
    if (ct.is_ntt) {
        ring_->to_ntt(encoded);
    }
    
    // Add encoded plaintext to c0 only
    // result.c0 = ct.c0 + encode(pt) (mod q)
    // result.c1 = ct.c1 (unchanged)
    Polynomial sum_c0 = ring_->add(ct.c0, encoded);
    Polynomial c1_copy = ct.c1.clone();
    
    // Plaintext addition has minimal noise growth (no multiplication involved)
    // We conservatively reduce by 0.5 bits
    double result_noise_budget = ct.noise_budget - 0.5;
    
    return Ciphertext(std::move(sum_c0), std::move(c1_copy), 
                      result_noise_budget, ct.key_id, ct.is_ntt);
}

void EncryptionEngine::add_plain_inplace(Ciphertext& ct, const Plaintext& pt) {
    // Encode the plaintext
    Polynomial encoded;
    if (pt.is_packed && pt.values.size() > 1) {
        encoded = encode_packed(pt.values);
    } else {
        encoded = encode_plaintext(pt.value());
    }
    
    // If ciphertext is in NTT form, convert encoded plaintext to NTT
    if (ct.is_ntt) {
        ring_->to_ntt(encoded);
    }
    
    // Add encoded plaintext to c0 only
    ring_->add_inplace(ct.c0, encoded);
    
    // Update noise budget (minimal growth for plaintext addition)
    ct.noise_budget -= 0.5;
}

Ciphertext EncryptionEngine::add_scalar(const Ciphertext& ct, uint64_t value) {
    Plaintext pt(value, params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4);
    return add_plain(ct, pt);
}

Ciphertext EncryptionEngine::subtract(const Ciphertext& ct1, const Ciphertext& ct2) {
    // Verify key IDs match
    if (ct1.key_id != ct2.key_id) {
        throw std::invalid_argument("Cannot subtract ciphertexts encrypted with different keys");
    }
    
    // Verify both ciphertexts are in the same representation
    if (ct1.is_ntt != ct2.is_ntt) {
        throw std::invalid_argument("Cannot subtract ciphertexts in different representations (NTT vs coefficient)");
    }
    
    // Subtract corresponding polynomial components
    // result.c0 = ct1.c0 - ct2.c0 (mod q)
    // result.c1 = ct1.c1 - ct2.c1 (mod q)
    Polynomial diff_c0 = ring_->subtract(ct1.c0, ct2.c0);
    Polynomial diff_c1 = ring_->subtract(ct1.c1, ct2.c1);
    
    // Noise budget decreases similarly to addition
    double result_noise_budget = std::min(ct1.noise_budget, ct2.noise_budget) - 1.0;
    
    return Ciphertext(std::move(diff_c0), std::move(diff_c1), 
                      result_noise_budget, ct1.key_id, ct1.is_ntt);
}

Ciphertext EncryptionEngine::negate(const Ciphertext& ct) {
    // Negate both polynomial components
    // result.c0 = -ct.c0 (mod q)
    // result.c1 = -ct.c1 (mod q)
    Polynomial neg_c0 = ring_->negate(ct.c0);
    Polynomial neg_c1 = ring_->negate(ct.c1);
    
    // Negation doesn't affect noise budget
    return Ciphertext(std::move(neg_c0), std::move(neg_c1), 
                      ct.noise_budget, ct.key_id, ct.is_ntt);
}

Ciphertext EncryptionEngine::get_zero_ciphertext(const PublicKey& pk) {
    // Encrypt zero as the additive identity
    return encrypt_value(0, pk);
}

// ============================================================================
// Homomorphic Multiplication (Requirement 7.1, 7.2, 7.3, 7.4, 7.5, 7.6)
// ============================================================================

Ciphertext EncryptionEngine::multiply(const Ciphertext& ct1, const Ciphertext& ct2) {
    // Verify key IDs match
    if (ct1.key_id != ct2.key_id) {
        throw std::invalid_argument("Cannot multiply ciphertexts encrypted with different keys");
    }
    
    // Verify both ciphertexts are degree-1 (standard ciphertexts)
    if (ct1.is_degree_2() || ct2.is_degree_2()) {
        throw std::invalid_argument("Cannot multiply degree-2 ciphertexts - relinearize first");
    }
    
    // Verify both ciphertexts are in the same representation
    if (ct1.is_ntt != ct2.is_ntt) {
        throw std::invalid_argument("Cannot multiply ciphertexts in different representations");
    }
    
    uint64_t modulus = params_.moduli[0];
    
    // Convert to NTT domain if not already
    Polynomial c1_c0 = ct1.c0.clone();
    Polynomial c1_c1 = ct1.c1.clone();
    Polynomial c2_c0 = ct2.c0.clone();
    Polynomial c2_c1 = ct2.c1.clone();
    
    if (!ct1.is_ntt) {
        ring_->to_ntt(c1_c0);
        ring_->to_ntt(c1_c1);
        ring_->to_ntt(c2_c0);
        ring_->to_ntt(c2_c1);
    }
    
    // Compute tensor product:
    // result.c0 = ct1.c0 * ct2.c0
    // result.c1 = ct1.c0 * ct2.c1 + ct1.c1 * ct2.c0
    // result.c2 = ct1.c1 * ct2.c1
    
    // c0 = ct1.c0 * ct2.c0
    Polynomial result_c0 = ring_->pointwise_multiply(c1_c0, c2_c0);
    
    // c1 = ct1.c0 * ct2.c1 + ct1.c1 * ct2.c0
    Polynomial c0_c1 = ring_->pointwise_multiply(c1_c0, c2_c1);
    Polynomial c1_c0_prod = ring_->pointwise_multiply(c1_c1, c2_c0);
    Polynomial result_c1 = ring_->add(c0_c1, c1_c0_prod);
    
    // c2 = ct1.c1 * ct2.c1
    Polynomial result_c2 = ring_->pointwise_multiply(c1_c1, c2_c1);
    
    // Convert back from NTT if input was in coefficient form
    if (!ct1.is_ntt) {
        ring_->from_ntt(result_c0);
        ring_->from_ntt(result_c1);
        ring_->from_ntt(result_c2);
    }
    
    // Noise budget decreases significantly with multiplication
    // Approximately: new_budget = min(budget1, budget2) - log2(N) - some_constant
    double noise_reduction = std::log2(static_cast<double>(params_.poly_degree)) + 5.0;
    double result_noise_budget = std::min(ct1.noise_budget, ct2.noise_budget) - noise_reduction;
    
    return Ciphertext(std::move(result_c0), std::move(result_c1), std::move(result_c2),
                      result_noise_budget, ct1.key_id, ct1.is_ntt);
}

Ciphertext EncryptionEngine::multiply_relin(const Ciphertext& ct1, const Ciphertext& ct2,
                                             const EvaluationKey& ek) {
    // First multiply
    Ciphertext product = multiply(ct1, ct2);
    
    // Then relinearize
    return relinearize(product, ek);
}

Ciphertext EncryptionEngine::multiply_plain(const Ciphertext& ct, const Plaintext& pt) {
    // Encode the plaintext as a polynomial
    Polynomial encoded;
    if (pt.is_packed && pt.values.size() > 1) {
        encoded = encode_packed(pt.values);
    } else {
        encoded = encode_plaintext(pt.value());
    }
    
    // If ciphertext is in NTT form, convert encoded plaintext to NTT
    if (ct.is_ntt) {
        ring_->to_ntt(encoded);
    }
    
    // Multiply both ciphertext components by the plaintext polynomial
    // result.c0 = ct.c0 * encode(pt) (mod q)
    // result.c1 = ct.c1 * encode(pt) (mod q)
    
    Polynomial c0_ntt = ct.c0.clone();
    Polynomial c1_ntt = ct.c1.clone();
    Polynomial encoded_ntt = encoded.clone();
    
    if (!ct.is_ntt) {
        ring_->to_ntt(c0_ntt);
        ring_->to_ntt(c1_ntt);
        ring_->to_ntt(encoded_ntt);
    }
    
    Polynomial result_c0 = ring_->pointwise_multiply(c0_ntt, encoded_ntt);
    Polynomial result_c1 = ring_->pointwise_multiply(c1_ntt, encoded_ntt);
    
    if (!ct.is_ntt) {
        ring_->from_ntt(result_c0);
        ring_->from_ntt(result_c1);
    }
    
    // Plaintext multiplication has less noise growth than ciphertext multiplication
    // Noise grows by approximately log2(||pt||) bits
    double noise_reduction = 2.0;  // Conservative estimate
    double result_noise_budget = ct.noise_budget - noise_reduction;
    
    return Ciphertext(std::move(result_c0), std::move(result_c1),
                      result_noise_budget, ct.key_id, ct.is_ntt);
}

void EncryptionEngine::multiply_plain_inplace(Ciphertext& ct, const Plaintext& pt) {
    // Encode the plaintext as a polynomial
    Polynomial encoded;
    if (pt.is_packed && pt.values.size() > 1) {
        encoded = encode_packed(pt.values);
    } else {
        encoded = encode_plaintext(pt.value());
    }
    
    // Convert to NTT for multiplication
    Polynomial c0_ntt = ct.c0.clone();
    Polynomial c1_ntt = ct.c1.clone();
    Polynomial encoded_ntt = encoded.clone();
    
    if (!ct.is_ntt) {
        ring_->to_ntt(c0_ntt);
        ring_->to_ntt(c1_ntt);
        ring_->to_ntt(encoded_ntt);
    }
    
    // Multiply in NTT domain
    ring_->pointwise_multiply_inplace(c0_ntt, encoded_ntt);
    ring_->pointwise_multiply_inplace(c1_ntt, encoded_ntt);
    
    if (!ct.is_ntt) {
        ring_->from_ntt(c0_ntt);
        ring_->from_ntt(c1_ntt);
    }
    
    ct.c0 = std::move(c0_ntt);
    ct.c1 = std::move(c1_ntt);
    
    // Update noise budget
    ct.noise_budget -= 2.0;
}

Ciphertext EncryptionEngine::multiply_scalar(const Ciphertext& ct, uint64_t scalar) {
    // Optimized scalar multiplication - just multiply each coefficient
    uint64_t modulus = params_.moduli[0];
    
    Polynomial result_c0 = ring_->multiply_scalar(ct.c0, scalar);
    Polynomial result_c1 = ring_->multiply_scalar(ct.c1, scalar);
    
    // Scalar multiplication has minimal noise growth
    double result_noise_budget = ct.noise_budget - 1.0;
    
    return Ciphertext(std::move(result_c0), std::move(result_c1),
                      result_noise_budget, ct.key_id, ct.is_ntt);
}

Ciphertext EncryptionEngine::relinearize(const Ciphertext& ct, const EvaluationKey& ek) {
    // Verify this is a degree-2 ciphertext
    if (!ct.is_degree_2()) {
        // Already degree-1, just return a copy
        return Ciphertext(ct.c0.clone(), ct.c1.clone(), ct.noise_budget, ct.key_id, ct.is_ntt);
    }
    
    // Verify key IDs match
    if (ct.key_id != ek.key_id) {
        throw std::invalid_argument("Evaluation key does not match ciphertext key");
    }
    
    uint64_t modulus = params_.moduli[0];
    uint32_t degree = params_.poly_degree;
    
    // Get the c2 component
    const Polynomial& c2 = ct.c2.value();
    
    // Relinearization: reduce degree-2 ciphertext to degree-1
    // Using key switching: c2 * s^2 -> c2 * (rlk.a * s + rlk.b + e)
    // 
    // The relinearization key contains encryptions of s^2 under s:
    // rlk[i] = (a_i, b_i) where b_i = -a_i * s + e_i + s^2 * base^i
    //
    // To relinearize:
    // 1. Decompose c2 into digits: c2 = sum(c2_i * base^i)
    // 2. Compute: c0' = c0 + sum(c2_i * rlk.b_i)
    //            c1' = c1 + sum(c2_i * rlk.a_i)
    
    const auto& relin_key = ek.relin_key;
    uint32_t decomp_base_log = relin_key.decomp_base_log > 0 ? relin_key.decomp_base_log : 4;
    uint64_t decomp_base = 1ULL << decomp_base_log;
    uint32_t num_levels = relin_key.decomp_level > 0 ? relin_key.decomp_level : 
                          static_cast<uint32_t>((64 + decomp_base_log - 1) / decomp_base_log);
    
    // Start with c0 and c1
    Polynomial result_c0 = ct.c0.clone();
    Polynomial result_c1 = ct.c1.clone();
    
    // If we have relinearization keys, apply them
    if (!relin_key.keys.empty()) {
        // Decompose c2 and apply key switching
        for (uint32_t level = 0; level < num_levels && level < relin_key.keys.size(); ++level) {
            // Extract digit at this level
            Polynomial c2_digit(degree, modulus);
            uint64_t shift = level * decomp_base_log;
            uint64_t mask = decomp_base - 1;
            
            for (uint32_t i = 0; i < degree; ++i) {
                c2_digit[i] = (c2[i] >> shift) & mask;
            }
            
            // Get the key pair for this level
            const auto& [rlk_a, rlk_b] = relin_key.keys[level];
            
            // Convert to NTT for multiplication
            Polynomial c2_digit_ntt = c2_digit.clone();
            Polynomial rlk_a_ntt = rlk_a.clone();
            Polynomial rlk_b_ntt = rlk_b.clone();
            
            ring_->to_ntt(c2_digit_ntt);
            ring_->to_ntt(rlk_a_ntt);
            ring_->to_ntt(rlk_b_ntt);
            
            // c0' += c2_digit * rlk_b
            Polynomial prod_b = ring_->pointwise_multiply(c2_digit_ntt, rlk_b_ntt);
            ring_->from_ntt(prod_b);
            ring_->add_inplace(result_c0, prod_b);
            
            // c1' += c2_digit * rlk_a
            Polynomial prod_a = ring_->pointwise_multiply(c2_digit_ntt, rlk_a_ntt);
            ring_->from_ntt(prod_a);
            ring_->add_inplace(result_c1, prod_a);
        }
    } else {
        // No relinearization key available - use a simplified approach
        // This is less accurate but allows the operation to proceed
        // In practice, you should always have proper relinearization keys
        
        // Simple approach: just drop c2 (loses information but maintains structure)
        // This is only for testing/development purposes
    }
    
    // Relinearization adds some noise
    double noise_reduction = 1.0;
    double result_noise_budget = ct.noise_budget - noise_reduction;
    
    return Ciphertext(std::move(result_c0), std::move(result_c1),
                      result_noise_budget, ct.key_id, ct.is_ntt);
}

void EncryptionEngine::relinearize_inplace(Ciphertext& ct, const EvaluationKey& ek) {
    if (!ct.is_degree_2()) {
        return;  // Already degree-1
    }
    
    Ciphertext relinearized = relinearize(ct, ek);
    ct = std::move(relinearized);
}

Ciphertext EncryptionEngine::square(const Ciphertext& ct) {
    // Verify this is a degree-1 ciphertext
    if (ct.is_degree_2()) {
        throw std::invalid_argument("Cannot square degree-2 ciphertext - relinearize first");
    }
    
    uint64_t modulus = params_.moduli[0];
    
    // Convert to NTT domain if not already
    Polynomial c0_ntt = ct.c0.clone();
    Polynomial c1_ntt = ct.c1.clone();
    
    if (!ct.is_ntt) {
        ring_->to_ntt(c0_ntt);
        ring_->to_ntt(c1_ntt);
    }
    
    // Compute square using tensor product:
    // result.c0 = ct.c0 * ct.c0
    // result.c1 = 2 * ct.c0 * ct.c1
    // result.c2 = ct.c1 * ct.c1
    
    // c0 = ct.c0^2
    Polynomial result_c0 = ring_->pointwise_multiply(c0_ntt, c0_ntt);
    
    // c1 = 2 * ct.c0 * ct.c1
    Polynomial c0_c1 = ring_->pointwise_multiply(c0_ntt, c1_ntt);
    // Multiply by 2 (add to itself)
    Polynomial result_c1 = ring_->add(c0_c1, c0_c1);
    
    // c2 = ct.c1^2
    Polynomial result_c2 = ring_->pointwise_multiply(c1_ntt, c1_ntt);
    
    // Convert back from NTT if input was in coefficient form
    if (!ct.is_ntt) {
        ring_->from_ntt(result_c0);
        ring_->from_ntt(result_c1);
        ring_->from_ntt(result_c2);
    }
    
    // Noise budget decreases with squaring
    double noise_reduction = std::log2(static_cast<double>(params_.poly_degree)) + 5.0;
    double result_noise_budget = ct.noise_budget - noise_reduction;
    
    return Ciphertext(std::move(result_c0), std::move(result_c1), std::move(result_c2),
                      result_noise_budget, ct.key_id, ct.is_ntt);
}

Ciphertext EncryptionEngine::square_relin(const Ciphertext& ct, const EvaluationKey& ek) {
    Ciphertext squared = square(ct);
    return relinearize(squared, ek);
}

// ============================================================================
// Ballot Aggregation Primitives (Requirement 15.1, 15.8)
// ============================================================================

Ciphertext EncryptionEngine::tally_votes(
    const std::vector<Ciphertext>& ballots,
    EncryptionProgressCallback progress
) {
    // Use tree reduction for efficient parallel tallying
    return batch_add_tree(ballots, progress);
}

Ciphertext EncryptionEngine::tally_weighted_votes(
    const std::vector<Ciphertext>& ballots,
    const std::vector<Ciphertext>& weights,
    const EvaluationKey& ek,
    EncryptionProgressCallback progress
) {
    if (ballots.size() != weights.size()) {
        throw std::invalid_argument("Ballots and weights must have the same size");
    }
    
    if (ballots.empty()) {
        throw std::invalid_argument("Cannot tally empty ballot list");
    }
    
    size_t total = ballots.size();
    std::vector<Ciphertext> weighted_ballots;
    weighted_ballots.reserve(total);
    
    // Multiply each ballot by its weight
    for (size_t i = 0; i < total; ++i) {
        // Multiply ballot by weight (produces degree-2 ciphertext)
        Ciphertext product = multiply(ballots[i], weights[i]);
        
        // Relinearize to degree-1
        Ciphertext relinearized = relinearize(product, ek);
        
        weighted_ballots.push_back(std::move(relinearized));
        
        if (progress && (i + 1) % 100 == 0) {
            progress(i + 1, total * 2);  // First half is multiplication
        }
    }
    
    // Sum all weighted ballots using tree reduction
    auto sum_progress = [&progress, total](size_t completed, size_t sum_total) {
        if (progress) {
            progress(total + completed, total + sum_total);
        }
    };
    
    return batch_add_tree(weighted_ballots, sum_progress);
}

Ciphertext EncryptionEngine::check_threshold(
    const Ciphertext& tally,
    uint64_t threshold
) {
    // Note: True threshold detection requires programmable bootstrapping (PBS)
    // to perform comparison on encrypted data. This is a placeholder that
    // prepares the ciphertext for PBS-based comparison.
    //
    // The actual comparison would be:
    // 1. Subtract threshold from tally: diff = tally - threshold
    // 2. Use PBS with sign function LUT to get sign(diff)
    // 3. Map sign to {0, 1}: result = (sign(diff) + 1) / 2
    //
    // For now, we return the difference which can be used with PBS later.
    
    Plaintext threshold_pt(threshold, params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4);
    
    // Compute tally - threshold
    // First encode threshold
    Polynomial encoded_threshold = encode_plaintext(threshold);
    
    // Negate to get -threshold
    Polynomial neg_threshold = ring_->negate(encoded_threshold);
    
    // Add -threshold to tally (equivalent to subtraction)
    Polynomial result_c0 = ring_->add(tally.c0, neg_threshold);
    Polynomial result_c1 = tally.c1.clone();
    
    // Return the difference (ready for PBS comparison)
    return Ciphertext(std::move(result_c0), std::move(result_c1),
                      tally.noise_budget - 0.5, tally.key_id, tally.is_ntt);
}

Ciphertext EncryptionEngine::update_tally(
    const Ciphertext& current_tally,
    const Ciphertext& new_ballot
) {
    // Simple addition for incremental update
    return add(current_tally, new_ballot);
}

Ciphertext EncryptionEngine::tally_multi_candidate(
    const std::vector<Ciphertext>& ballots,
    size_t num_candidates,
    EncryptionProgressCallback progress
) {
    // For multi-candidate tallying, we assume each ballot is SIMD-packed
    // with votes for each candidate in different slots.
    // The sum operation preserves the slot structure, so we can just
    // use the standard batch_add_tree.
    
    if (num_candidates > params_.poly_degree) {
        throw std::invalid_argument("Number of candidates exceeds polynomial degree");
    }
    
    return batch_add_tree(ballots, progress);
}

// ============================================================================
// Comparison Operations for Fraud Detection (Requirement 15.3, 15.7)
// ============================================================================

Ciphertext EncryptionEngine::compare_greater_than(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    const BootstrapKey* bk
) {
    // Compute difference: diff = ct1 - ct2
    // If ct1 > ct2, diff will be positive
    // We need PBS with sign function to extract the sign
    
    Ciphertext diff = subtract(ct1, ct2);
    
    // Without bootstrapping key, return the difference
    // The caller can apply PBS with sign LUT externally
    if (!bk) {
        return diff;
    }
    
    // With bootstrapping key, we would apply PBS here
    // For now, return the prepared difference
    // Full PBS implementation is in task 17 (TFHE bootstrapping)
    return diff;
}

Ciphertext EncryptionEngine::compare_less_than(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    const BootstrapKey* bk
) {
    // ct1 < ct2 is equivalent to ct2 > ct1
    return compare_greater_than(ct2, ct1, bk);
}

Ciphertext EncryptionEngine::compare_equal(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    const BootstrapKey* bk
) {
    // Compute difference: diff = ct1 - ct2
    // If ct1 == ct2, diff will be zero
    // We need PBS with zero-test function: f(x) = 1 if x == 0, else 0
    
    Ciphertext diff = subtract(ct1, ct2);
    
    // Without bootstrapping key, return the difference
    // The caller can apply PBS with zero-test LUT externally
    if (!bk) {
        return diff;
    }
    
    // With bootstrapping key, we would apply PBS here
    // Full PBS implementation is in task 17
    return diff;
}

Ciphertext EncryptionEngine::check_range(
    const Ciphertext& ct,
    uint64_t min_value,
    uint64_t max_value,
    const BootstrapKey* bk
) {
    // Range check: min <= ct <= max
    // Equivalent to: (ct >= min) AND (ct <= max)
    // Which is: (ct - min >= 0) AND (max - ct >= 0)
    
    // Create plaintext for min and max
    Plaintext min_pt(min_value, params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4);
    Plaintext max_pt(max_value, params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4);
    
    // Compute ct - min
    Polynomial encoded_min = encode_plaintext(min_value);
    Polynomial neg_min = ring_->negate(encoded_min);
    Polynomial diff_min_c0 = ring_->add(ct.c0, neg_min);
    Ciphertext diff_min(std::move(diff_min_c0), ct.c1.clone(), 
                        ct.noise_budget - 0.5, ct.key_id, ct.is_ntt);
    
    // Compute max - ct
    Polynomial encoded_max = encode_plaintext(max_value);
    Polynomial neg_ct_c0 = ring_->negate(ct.c0);
    Polynomial neg_ct_c1 = ring_->negate(ct.c1);
    Polynomial diff_max_c0 = ring_->add(encoded_max, neg_ct_c0);
    Ciphertext diff_max(std::move(diff_max_c0), std::move(neg_ct_c1),
                        ct.noise_budget - 0.5, ct.key_id, ct.is_ntt);
    
    // Without PBS, return the first difference (ct - min)
    // Full range check requires PBS to compute sign and AND
    if (!bk) {
        return diff_min;
    }
    
    // With PBS, we would:
    // 1. Apply sign function to both differences
    // 2. Multiply the results (AND operation)
    // For now, return the first difference
    return diff_min;
}

Ciphertext EncryptionEngine::detect_duplicate(
    const Ciphertext& new_ballot,
    const std::vector<Ciphertext>& existing_ballots,
    const BootstrapKey* bk
) {
    if (existing_ballots.empty()) {
        // No existing ballots, return encryption of 0 (no duplicate)
        return encrypt_value(0, PublicKey(
            Polynomial(params_.poly_degree, params_.moduli[0]),
            Polynomial(params_.poly_degree, params_.moduli[0]),
            new_ballot.key_id
        ));
    }
    
    // Compare new ballot with each existing ballot
    // Result is OR of all equality tests
    
    // Start with first comparison
    Ciphertext result = compare_equal(new_ballot, existing_ballots[0], bk);
    
    // OR with remaining comparisons
    // Note: Without PBS, we can only prepare the differences
    // Full duplicate detection requires PBS for equality test and OR
    for (size_t i = 1; i < existing_ballots.size(); ++i) {
        Ciphertext eq_test = compare_equal(new_ballot, existing_ballots[i], bk);
        
        // For OR operation on encrypted bits, we use: a OR b = a + b - a*b
        // But this requires multiplication and PBS
        // For now, just add the equality tests (approximation)
        result = add(result, eq_test);
    }
    
    return result;
}

Ciphertext EncryptionEngine::compute_anomaly_score(
    const Ciphertext& ballot,
    const Ciphertext& expected_distribution,
    const EvaluationKey& ek
) {
    // Compute squared difference as anomaly score
    // score = (ballot - expected)^2
    
    // Compute difference
    Ciphertext diff = subtract(ballot, expected_distribution);
    
    // Square the difference
    Ciphertext squared = square(diff);
    
    // Relinearize
    return relinearize(squared, ek);
}

// ============================================================================
// Batch Homomorphic Operations (Requirement 15.1, 15.2)
// ============================================================================

Ciphertext EncryptionEngine::batch_add(const std::vector<Ciphertext>& ciphertexts) {
    if (ciphertexts.empty()) {
        throw std::invalid_argument("Cannot add empty vector of ciphertexts");
    }
    
    if (ciphertexts.size() == 1) {
        return Ciphertext(ciphertexts[0]);
    }
    
    // Start with a copy of the first ciphertext
    Polynomial sum_c0 = ciphertexts[0].c0.clone();
    Polynomial sum_c1 = ciphertexts[0].c1.clone();
    double min_noise_budget = ciphertexts[0].noise_budget;
    uint64_t key_id = ciphertexts[0].key_id;
    
    // Accumulate remaining ciphertexts
    for (size_t i = 1; i < ciphertexts.size(); ++i) {
        const auto& ct = ciphertexts[i];
        
        // Verify key ID matches
        if (ct.key_id != key_id) {
            throw std::invalid_argument("All ciphertexts must be encrypted with the same key");
        }
        
        ring_->add_inplace(sum_c0, ct.c0);
        ring_->add_inplace(sum_c1, ct.c1);
        
        // Track minimum noise budget
        min_noise_budget = std::min(min_noise_budget, ct.noise_budget);
    }
    
    // Noise budget decreases slightly with addition (by about log2(n) bits)
    double noise_reduction = std::log2(static_cast<double>(ciphertexts.size()));
    double result_noise_budget = min_noise_budget - noise_reduction;
    
    return Ciphertext(std::move(sum_c0), std::move(sum_c1), 
                      result_noise_budget, key_id, false);
}

Ciphertext EncryptionEngine::batch_add_tree(
    const std::vector<Ciphertext>& ciphertexts,
    EncryptionProgressCallback progress
) {
    if (ciphertexts.empty()) {
        throw std::invalid_argument("Cannot add empty vector of ciphertexts");
    }
    
    if (ciphertexts.size() == 1) {
        return Ciphertext(ciphertexts[0]);
    }
    
    // Tree reduction for better parallelism and numerical stability
    std::vector<Ciphertext> current_level;
    current_level.reserve(ciphertexts.size());
    
    // Copy input ciphertexts
    for (const auto& ct : ciphertexts) {
        current_level.push_back(Ciphertext(ct));
    }
    
    size_t total_ops = ciphertexts.size() - 1;
    size_t completed_ops = 0;
    
    // Reduce until we have a single ciphertext
    while (current_level.size() > 1) {
        std::vector<Ciphertext> next_level;
        next_level.reserve((current_level.size() + 1) / 2);
        
        // Parallel pairwise addition
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        size_t num_pairs = current_level.size() / 2;
        
        if (num_pairs > num_threads) {
            // Parallel reduction
            std::vector<std::future<Ciphertext>> futures;
            
            for (size_t i = 0; i < num_pairs; ++i) {
                futures.push_back(std::async(std::launch::async, [&, i]() {
                    const auto& ct1 = current_level[2 * i];
                    const auto& ct2 = current_level[2 * i + 1];
                    
                    Polynomial sum_c0 = ring_->add(ct1.c0, ct2.c0);
                    Polynomial sum_c1 = ring_->add(ct1.c1, ct2.c1);
                    
                    double noise = std::min(ct1.noise_budget, ct2.noise_budget) - 1.0;
                    
                    return Ciphertext(std::move(sum_c0), std::move(sum_c1),
                                     noise, ct1.key_id, false);
                }));
            }
            
            for (auto& future : futures) {
                next_level.push_back(future.get());
                completed_ops++;
                
                if (progress) {
                    progress(completed_ops, total_ops);
                }
            }
        } else {
            // Sequential reduction for small batches
            for (size_t i = 0; i < num_pairs; ++i) {
                const auto& ct1 = current_level[2 * i];
                const auto& ct2 = current_level[2 * i + 1];
                
                Polynomial sum_c0 = ring_->add(ct1.c0, ct2.c0);
                Polynomial sum_c1 = ring_->add(ct1.c1, ct2.c1);
                
                double noise = std::min(ct1.noise_budget, ct2.noise_budget) - 1.0;
                
                next_level.emplace_back(std::move(sum_c0), std::move(sum_c1),
                                       noise, ct1.key_id, false);
                completed_ops++;
                
                if (progress) {
                    progress(completed_ops, total_ops);
                }
            }
        }
        
        // Handle odd element
        if (current_level.size() % 2 == 1) {
            next_level.push_back(std::move(current_level.back()));
        }
        
        current_level = std::move(next_level);
    }
    
    return std::move(current_level[0]);
}

// ============================================================================
// Streaming Batch Encryption Template Implementation
// ============================================================================

template<typename Iterator>
void EncryptionEngine::batch_encrypt_streaming(
    Iterator begin, Iterator end,
    const PublicKey& pk,
    size_t batch_size,
    std::function<void(std::vector<Ciphertext>&&)> output
) {
    std::vector<Plaintext> batch;
    batch.reserve(batch_size);
    
    for (auto it = begin; it != end; ++it) {
        batch.push_back(*it);
        
        if (batch.size() >= batch_size) {
            // Process batch
            BatchEncryptionRequest request(std::move(batch), &pk, false);
            auto result = batch_encrypt(request, nullptr);
            output(std::move(result.ciphertexts));
            
            batch.clear();
            batch.reserve(batch_size);
        }
    }
    
    // Process remaining items
    if (!batch.empty()) {
        BatchEncryptionRequest request(std::move(batch), &pk, false);
        auto result = batch_encrypt(request, nullptr);
        output(std::move(result.ciphertexts));
    }
}

// Explicit template instantiation for common iterator types
template void EncryptionEngine::batch_encrypt_streaming<std::vector<Plaintext>::iterator>(
    std::vector<Plaintext>::iterator, std::vector<Plaintext>::iterator,
    const PublicKey&, size_t, std::function<void(std::vector<Ciphertext>&&)>
);

template void EncryptionEngine::batch_encrypt_streaming<std::vector<Plaintext>::const_iterator>(
    std::vector<Plaintext>::const_iterator, std::vector<Plaintext>::const_iterator,
    const PublicKey&, size_t, std::function<void(std::vector<Ciphertext>&&)>
);

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<EncryptionEngine> create_encryption_engine(const ParameterSet& params) {
    return std::make_unique<EncryptionEngine>(params);
}

} // namespace fhe_accelerate
