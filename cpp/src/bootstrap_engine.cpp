/**
 * @file bootstrap_engine.cpp
 * @brief TFHE Bootstrapping Engine Implementation
 * 
 * Implements TFHE bootstrapping operations including bootstrapping key generation,
 * blind rotate, sample extract, key switching, and programmable bootstrapping.
 * 
 * Requirements: 4.4, 8.1, 8.3, 8.5, 8.6
 */

#include "bootstrap_engine.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <chrono>

namespace fhe_accelerate {

// ============================================================================
// BootstrapEngine Implementation
// ============================================================================

BootstrapEngine::BootstrapEngine(const ParameterSet& params, HardwareDispatcher* dispatcher)
    : params_(params), dispatcher_(dispatcher)
{
    // Initialize polynomial ring with primary modulus
    if (!params_.moduli.empty()) {
        ring_ = std::make_unique<PolynomialRing>(params_.poly_degree, params_.moduli[0]);
    } else {
        throw std::invalid_argument("Parameter set must have at least one modulus");
    }
    
    // Initialize secure random number generator
    rng_ = std::make_unique<SecureRandom>();
    
    // Set moduli
    glwe_modulus_ = params_.moduli[0];
    lwe_modulus_ = glwe_modulus_;  // Can be different in some configurations
    
    // Compute scaling factor delta = q / t
    uint64_t t = params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4;
    delta_ = glwe_modulus_ / t;
    
    // Initialize default test polynomial
    init_default_test_poly();
}

BootstrapEngine::~BootstrapEngine() = default;


// ============================================================================
// Helper Functions
// ============================================================================

void BootstrapEngine::init_default_test_poly() {
    // Create default test polynomial for identity bootstrapping
    // The test polynomial encodes the identity function
    uint32_t N = params_.poly_degree;
    uint64_t q = glwe_modulus_;
    uint64_t t = params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4;
    
    std::vector<uint64_t> coeffs(N, 0);
    
    // For identity bootstrapping, the test polynomial is:
    // sum_{i=0}^{N-1} (i * delta / N) * X^i
    // This encodes the identity function when evaluated at rotation points
    for (uint32_t i = 0; i < N; ++i) {
        // Compute the value at this rotation
        // The rotation by i corresponds to plaintext value (i * t / (2N))
        uint64_t value = (i * t) / (2 * N);
        coeffs[i] = (value * delta_) % q;
    }
    
    default_test_poly_ = std::make_unique<Polynomial>(std::move(coeffs), q, false);
}

Polynomial BootstrapEngine::sample_random_polynomial() {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = glwe_modulus_;
    
    std::vector<uint64_t> coeffs(degree);
    for (uint32_t i = 0; i < degree; ++i) {
        coeffs[i] = rng_->random_u64_range(modulus);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

Polynomial BootstrapEngine::sample_error_polynomial() {
    uint32_t degree = params_.poly_degree;
    uint64_t modulus = glwe_modulus_;
    double std_dev = params_.lwe_noise_std > 0 ? params_.lwe_noise_std : 3.2;
    
    std::vector<uint64_t> coeffs(degree);
    for (uint32_t i = 0; i < degree; ++i) {
        coeffs[i] = rng_->sample_gaussian(std_dev, modulus);
    }
    
    return Polynomial(std::move(coeffs), modulus, false);
}

std::vector<int64_t> BootstrapEngine::sample_lwe_error(size_t dimension) {
    double std_dev = params_.lwe_noise_std > 0 ? params_.lwe_noise_std : 3.2;
    std::vector<int64_t> error(dimension);
    
    for (size_t i = 0; i < dimension; ++i) {
        // Sample from discrete Gaussian
        uint64_t sample = rng_->sample_gaussian(std_dev, lwe_modulus_);
        // Convert to signed representation
        if (sample > lwe_modulus_ / 2) {
            error[i] = static_cast<int64_t>(sample) - static_cast<int64_t>(lwe_modulus_);
        } else {
            error[i] = static_cast<int64_t>(sample);
        }
    }
    
    return error;
}

Polynomial BootstrapEngine::rotate_polynomial(const Polynomial& poly, int32_t rotation) {
    uint32_t N = poly.degree();
    uint64_t q = poly.modulus();
    
    // Normalize rotation to [0, 2N)
    int32_t rot = ((rotation % (2 * static_cast<int32_t>(N))) + 2 * static_cast<int32_t>(N)) 
                  % (2 * static_cast<int32_t>(N));
    
    std::vector<uint64_t> result(N, 0);
    
    for (uint32_t i = 0; i < N; ++i) {
        int32_t new_idx = (static_cast<int32_t>(i) + rot) % (2 * static_cast<int32_t>(N));
        
        if (new_idx < static_cast<int32_t>(N)) {
            // No sign change
            result[new_idx] = poly[i];
        } else {
            // Sign change due to X^N = -1
            result[new_idx - N] = (q - poly[i]) % q;
        }
    }
    
    return Polynomial(std::move(result), q, false);
}


// ============================================================================
// Decomposition Helpers
// ============================================================================

std::vector<Polynomial> BootstrapEngine::decompose_polynomial(
    const Polynomial& poly,
    uint32_t base_log,
    uint32_t level
) {
    uint32_t N = poly.degree();
    uint64_t q = poly.modulus();
    uint64_t base = 1ULL << base_log;
    uint64_t mask = base - 1;
    
    std::vector<Polynomial> decomposed;
    decomposed.reserve(level);
    
    for (uint32_t l = 0; l < level; ++l) {
        std::vector<uint64_t> coeffs(N);
        uint32_t shift = (level - 1 - l) * base_log;
        
        for (uint32_t i = 0; i < N; ++i) {
            // Extract digit at level l
            uint64_t digit = (poly[i] >> shift) & mask;
            
            // Center the digit around 0 for better noise
            if (digit > base / 2) {
                coeffs[i] = (q - (base - digit)) % q;
            } else {
                coeffs[i] = digit;
            }
        }
        
        decomposed.emplace_back(std::move(coeffs), q, false);
    }
    
    return decomposed;
}

// ============================================================================
// GLWE Operations
// ============================================================================

GLWECiphertext BootstrapEngine::encrypt_glwe_zero(const SecretKey& sk) {
    uint32_t k = params_.glwe_dimension > 0 ? params_.glwe_dimension : 1;
    uint32_t N = params_.poly_degree;
    uint64_t q = glwe_modulus_;
    
    // Sample random mask polynomials
    std::vector<Polynomial> mask;
    mask.reserve(k);
    for (uint32_t i = 0; i < k; ++i) {
        mask.push_back(sample_random_polynomial());
    }
    
    // Sample error polynomial
    Polynomial e = sample_error_polynomial();
    
    // Compute body = sum(mask[i] * sk) + e
    // For k=1, body = mask[0] * sk + e
    Polynomial body(N, q);
    body.set_zero();
    
    // Convert to NTT for multiplication
    Polynomial sk_ntt = sk.poly.clone();
    ring_->to_ntt(sk_ntt);
    
    for (uint32_t i = 0; i < k; ++i) {
        Polynomial mask_ntt = mask[i].clone();
        ring_->to_ntt(mask_ntt);
        
        Polynomial product = ring_->pointwise_multiply(mask_ntt, sk_ntt);
        ring_->from_ntt(product);
        
        ring_->add_inplace(body, product);
    }
    
    // Add error
    ring_->add_inplace(body, e);
    
    return GLWECiphertext(std::move(mask), std::move(body), sk.key_id, false);
}

void BootstrapEngine::add_glwe_inplace(GLWECiphertext& ct1, const GLWECiphertext& ct2) {
    // Add mask polynomials
    for (size_t i = 0; i < ct1.mask.size(); ++i) {
        ring_->add_inplace(ct1.mask[i], ct2.mask[i]);
    }
    // Add body
    ring_->add_inplace(ct1.body, ct2.body);
}

void BootstrapEngine::subtract_glwe_inplace(GLWECiphertext& ct1, const GLWECiphertext& ct2) {
    // Subtract mask polynomials
    for (size_t i = 0; i < ct1.mask.size(); ++i) {
        ring_->subtract_inplace(ct1.mask[i], ct2.mask[i]);
    }
    // Subtract body
    ring_->subtract_inplace(ct1.body, ct2.body);
}

GLWECiphertext BootstrapEngine::multiply_glwe_by_monomial(const GLWECiphertext& ct, int32_t exponent) {
    // Multiply each polynomial by X^exponent mod (X^N + 1)
    std::vector<Polynomial> new_mask;
    new_mask.reserve(ct.mask.size());
    
    for (const auto& m : ct.mask) {
        new_mask.push_back(rotate_polynomial(m, exponent));
    }
    
    Polynomial new_body = rotate_polynomial(ct.body, exponent);
    
    return GLWECiphertext(std::move(new_mask), std::move(new_body), ct.key_id, ct.is_ntt);
}


// ============================================================================
// Bootstrapping Key Generation (Requirement 4.4)
// ============================================================================

GGSWCiphertext BootstrapEngine::encrypt_ggsw(int64_t value, const SecretKey& glwe_sk) {
    uint32_t k = params_.glwe_dimension > 0 ? params_.glwe_dimension : 1;
    uint32_t N = params_.poly_degree;
    uint64_t q = glwe_modulus_;
    uint32_t base_log = params_.decomp_base_log > 0 ? params_.decomp_base_log : 4;
    uint32_t level = params_.decomp_level > 0 ? params_.decomp_level : 3;
    
    // GGSW has (k+1) * level rows
    size_t num_rows = (k + 1) * level;
    std::vector<GLWECiphertext> matrix;
    matrix.reserve(num_rows);
    
    // For each row, create a GLWE encryption of 0, then add the appropriate gadget term
    for (uint32_t row = 0; row < k + 1; ++row) {
        for (uint32_t l = 0; l < level; ++l) {
            // Encrypt zero
            GLWECiphertext ct = encrypt_glwe_zero(glwe_sk);
            
            // Compute gadget value: value * q / base^(l+1)
            uint64_t gadget_value = (static_cast<uint64_t>(std::abs(value)) * q) >> ((l + 1) * base_log);
            if (value < 0) {
                gadget_value = (q - gadget_value) % q;
            }
            
            // Add gadget term to appropriate position
            if (row < k) {
                // Add to mask[row]
                ct.mask[row][0] = (ct.mask[row][0] + gadget_value) % q;
            } else {
                // Add to body
                ct.body[0] = (ct.body[0] + gadget_value) % q;
            }
            
            matrix.push_back(std::move(ct));
        }
    }
    
    return GGSWCiphertext(std::move(matrix), base_log, level, glwe_sk.key_id);
}

std::unique_ptr<ExtendedBootstrapKey> BootstrapEngine::generate_bootstrap_key(
    const std::vector<int64_t>& lwe_sk,
    const SecretKey& glwe_sk,
    BootstrapProgressCallback progress
) {
    auto bsk = std::make_unique<ExtendedBootstrapKey>();
    
    size_t n = lwe_sk.size();
    bsk->lwe_dimension = static_cast<uint32_t>(n);
    bsk->glwe_dimension = params_.glwe_dimension > 0 ? params_.glwe_dimension : 1;
    bsk->poly_degree = params_.poly_degree;
    bsk->decomp_base_log = params_.decomp_base_log > 0 ? params_.decomp_base_log : 4;
    bsk->decomp_level = params_.decomp_level > 0 ? params_.decomp_level : 3;
    bsk->key_id = glwe_sk.key_id;
    
    // Generate GGSW encryptions of each LWE secret key bit
    // Parallelize across available cores
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    bsk->bsk.resize(n);
    
    // Use parallel execution for key generation
    std::vector<std::future<GGSWCiphertext>> futures;
    futures.reserve(n);
    
    size_t completed = 0;
    size_t chunk_size = (n + num_threads - 1) / num_threads;
    
    for (size_t chunk_start = 0; chunk_start < n; chunk_start += chunk_size) {
        size_t chunk_end = std::min(chunk_start + chunk_size, n);
        
        for (size_t i = chunk_start; i < chunk_end; ++i) {
            // For now, generate sequentially (thread-safe RNG needed for parallel)
            bsk->bsk[i] = encrypt_ggsw(lwe_sk[i], glwe_sk);
            
            completed++;
            if (progress && completed % 10 == 0) {
                progress(completed, n, "Generating GGSW encryptions");
            }
        }
    }
    
    if (progress) {
        progress(n, n, "GGSW generation complete");
    }
    
    // Generate key switching key
    if (progress) {
        progress(0, 1, "Generating key switching key");
    }
    bsk->ksk = generate_key_switch_key(glwe_sk, lwe_sk);
    if (progress) {
        progress(1, 1, "Key switching key complete");
    }
    
    return bsk;
}

KeySwitchKey BootstrapEngine::generate_key_switch_key(
    const SecretKey& glwe_sk,
    const std::vector<int64_t>& lwe_sk
) {
    KeySwitchKey ksk;
    
    uint32_t N = params_.poly_degree;
    uint64_t q = glwe_modulus_;
    uint32_t base_log = params_.decomp_base_log > 0 ? params_.decomp_base_log : 4;
    uint32_t level = params_.decomp_level > 0 ? params_.decomp_level : 3;
    
    ksk.decomp_base_log = base_log;
    ksk.decomp_level = level;
    ksk.key_id = glwe_sk.key_id;
    
    // For each coefficient of the GLWE secret key, generate key switching elements
    // KSK encrypts s_glwe[i] under s_lwe for each i
    
    // Extract GLWE secret key coefficients
    const auto& sk_coeffs = glwe_sk.poly.coefficients();
    
    // For each coefficient and each decomposition level
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t l = 0; l < level; ++l) {
            // Sample random a
            std::vector<uint64_t> a(lwe_sk.size());
            for (size_t j = 0; j < lwe_sk.size(); ++j) {
                a[j] = rng_->random_u64_range(q);
            }
            
            // Compute b = <a, s_lwe> + e + sk_coeff * q / base^(l+1)
            int64_t inner_product = 0;
            for (size_t j = 0; j < lwe_sk.size(); ++j) {
                inner_product += static_cast<int64_t>(a[j]) * lwe_sk[j];
            }
            
            // Sample error
            int64_t e = static_cast<int64_t>(rng_->sample_gaussian(
                params_.lwe_noise_std > 0 ? params_.lwe_noise_std : 3.2, q));
            if (e > static_cast<int64_t>(q / 2)) {
                e -= static_cast<int64_t>(q);
            }
            
            // Gadget value
            uint64_t gadget = (sk_coeffs[i] * q) >> ((l + 1) * base_log);
            
            uint64_t b = (static_cast<uint64_t>((inner_product + e) % static_cast<int64_t>(q)) + gadget) % q;
            
            // Store as polynomial pair (for compatibility with existing structure)
            Polynomial a_poly(std::vector<uint64_t>(a), q, false);
            Polynomial b_poly(std::vector<uint64_t>{b}, q, false);
            
            ksk.keys.emplace_back(std::move(a_poly), std::move(b_poly));
        }
    }
    
    return ksk;
}


// ============================================================================
// External Product and CMux
// ============================================================================

GLWECiphertext BootstrapEngine::external_product(
    const GLWECiphertext& glwe,
    const GGSWCiphertext& ggsw
) {
    uint32_t k = glwe.glwe_dimension();
    uint32_t N = glwe.poly_degree();
    uint64_t q = glwe.body.modulus();
    uint32_t level = ggsw.decomp_level;
    uint32_t base_log = ggsw.decomp_base_log;
    
    // Initialize result as zero GLWE
    GLWECiphertext result(k, N, q, glwe.key_id);
    for (auto& m : result.mask) {
        m.set_zero();
    }
    result.body.set_zero();
    
    // Decompose each polynomial in the GLWE ciphertext
    std::vector<std::vector<Polynomial>> decomposed_mask;
    for (const auto& m : glwe.mask) {
        decomposed_mask.push_back(decompose_polynomial(m, base_log, level));
    }
    auto decomposed_body = decompose_polynomial(glwe.body, base_log, level);
    
    // Compute external product
    // result = sum over rows of (decomposed_glwe[row] * ggsw.matrix[row])
    
    size_t row_idx = 0;
    
    // Process mask polynomials
    for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t l = 0; l < level; ++l) {
            const auto& ggsw_row = ggsw.matrix[row_idx++];
            
            // Convert decomposed polynomial to NTT
            Polynomial decomp_ntt = decomposed_mask[i][l].clone();
            ring_->to_ntt(decomp_ntt);
            
            // Multiply with each component of GGSW row
            for (uint32_t j = 0; j < k; ++j) {
                Polynomial ggsw_mask_ntt = ggsw_row.mask[j].clone();
                ring_->to_ntt(ggsw_mask_ntt);
                
                Polynomial product = ring_->pointwise_multiply(decomp_ntt, ggsw_mask_ntt);
                ring_->from_ntt(product);
                
                ring_->add_inplace(result.mask[j], product);
            }
            
            // Body
            Polynomial ggsw_body_ntt = ggsw_row.body.clone();
            ring_->to_ntt(ggsw_body_ntt);
            
            Polynomial body_product = ring_->pointwise_multiply(decomp_ntt, ggsw_body_ntt);
            ring_->from_ntt(body_product);
            
            ring_->add_inplace(result.body, body_product);
        }
    }
    
    // Process body polynomial
    for (uint32_t l = 0; l < level; ++l) {
        const auto& ggsw_row = ggsw.matrix[row_idx++];
        
        Polynomial decomp_ntt = decomposed_body[l].clone();
        ring_->to_ntt(decomp_ntt);
        
        for (uint32_t j = 0; j < k; ++j) {
            Polynomial ggsw_mask_ntt = ggsw_row.mask[j].clone();
            ring_->to_ntt(ggsw_mask_ntt);
            
            Polynomial product = ring_->pointwise_multiply(decomp_ntt, ggsw_mask_ntt);
            ring_->from_ntt(product);
            
            ring_->add_inplace(result.mask[j], product);
        }
        
        Polynomial ggsw_body_ntt = ggsw_row.body.clone();
        ring_->to_ntt(ggsw_body_ntt);
        
        Polynomial body_product = ring_->pointwise_multiply(decomp_ntt, ggsw_body_ntt);
        ring_->from_ntt(body_product);
        
        ring_->add_inplace(result.body, body_product);
    }
    
    return result;
}

GLWECiphertext BootstrapEngine::cmux(
    const GGSWCiphertext& ggsw,
    const GLWECiphertext& ct0,
    const GLWECiphertext& ct1
) {
    // CMux(ggsw, ct0, ct1) = ct0 + ggsw * (ct1 - ct0)
    // If ggsw encrypts 0, returns ct0
    // If ggsw encrypts 1, returns ct1
    
    // Compute ct1 - ct0
    GLWECiphertext diff = ct1;  // Copy ct1
    subtract_glwe_inplace(diff, ct0);
    
    // Compute ggsw * diff
    GLWECiphertext product = external_product(diff, ggsw);
    
    // Add ct0
    add_glwe_inplace(product, ct0);
    
    return product;
}


// ============================================================================
// Blind Rotate Operation (Requirement 8.1, 8.3)
// ============================================================================

void BootstrapEngine::blind_rotate(
    GLWECiphertext& acc,
    const LWECiphertext& lwe_ct,
    const ExtendedBootstrapKey& bsk
) {
    uint32_t N = params_.poly_degree;
    uint64_t q = lwe_modulus_;
    size_t n = lwe_ct.dimension();
    
    // Initial rotation by -b (the body of the LWE ciphertext)
    // Compute rotation amount: round(b * 2N / q)
    int32_t b_rotation = -static_cast<int32_t>((lwe_ct.b * 2 * N + q / 2) / q);
    acc = multiply_glwe_by_monomial(acc, b_rotation);
    
    // For each coefficient a[i] of the LWE ciphertext
    for (size_t i = 0; i < n; ++i) {
        // Compute rotation amount for this coefficient
        int32_t a_rotation = static_cast<int32_t>((lwe_ct.a[i] * 2 * N + q / 2) / q);
        
        if (a_rotation == 0) {
            continue;  // No rotation needed
        }
        
        // Compute X^{a_rotation} * acc
        GLWECiphertext rotated_acc = multiply_glwe_by_monomial(acc, a_rotation);
        
        // CMux: if s[i] = 1, use rotated_acc; if s[i] = 0, use acc
        // This is done via external product with GGSW encryption of s[i]
        acc = cmux(bsk.bsk[i], acc, rotated_acc);
    }
}

void BootstrapEngine::blind_rotate_sme(
    GLWECiphertext& acc,
    const LWECiphertext& lwe_ct,
    const ExtendedBootstrapKey& bsk
) {
    // SME-accelerated version
    // For now, fall back to standard implementation
    // TODO: Implement SME matrix operations for polynomial multiplication
    blind_rotate(acc, lwe_ct, bsk);
}

// ============================================================================
// Sample Extract and Key Switching (Requirement 8.1)
// ============================================================================

LWECiphertext BootstrapEngine::sample_extract(const GLWECiphertext& glwe) {
    uint32_t N = glwe.poly_degree();
    uint32_t k = glwe.glwe_dimension();
    uint64_t q = glwe.body.modulus();
    
    // Extract the constant coefficient as an LWE ciphertext
    // The LWE dimension is k * N
    size_t lwe_dim = k * N;
    
    std::vector<uint64_t> a(lwe_dim);
    
    // For each mask polynomial, extract coefficients
    // a[i*N + j] = mask[i][N-j] for j > 0, mask[i][0] for j = 0
    // (with sign changes due to X^N = -1)
    for (uint32_t i = 0; i < k; ++i) {
        const auto& mask = glwe.mask[i];
        
        // First coefficient
        a[i * N] = mask[0];
        
        // Remaining coefficients with negation
        for (uint32_t j = 1; j < N; ++j) {
            a[i * N + j] = (q - mask[N - j]) % q;
        }
    }
    
    // Body is just the constant coefficient
    uint64_t b = glwe.body[0];
    
    return LWECiphertext(std::move(a), b, q, glwe.key_id);
}

LWECiphertext BootstrapEngine::key_switch(
    const LWECiphertext& lwe,
    const KeySwitchKey& ksk
) {
    uint64_t q = lwe.modulus;
    uint32_t base_log = ksk.decomp_base_log;
    uint32_t level = ksk.decomp_level;
    uint64_t base = 1ULL << base_log;
    uint64_t mask = base - 1;
    
    // Output LWE dimension (from KSK structure)
    size_t out_dim = ksk.keys.size() > 0 ? ksk.keys[0].first.degree() : 0;
    
    std::vector<uint64_t> result_a(out_dim, 0);
    uint64_t result_b = lwe.b;
    
    // For each input coefficient
    size_t ksk_idx = 0;
    for (size_t i = 0; i < lwe.dimension(); ++i) {
        uint64_t coeff = lwe.a[i];
        
        // Decompose coefficient
        for (uint32_t l = 0; l < level; ++l) {
            uint32_t shift = (level - 1 - l) * base_log;
            uint64_t digit = (coeff >> shift) & mask;
            
            if (digit == 0) {
                ksk_idx++;
                continue;
            }
            
            // Subtract digit * ksk[i][l]
            const auto& ksk_entry = ksk.keys[ksk_idx++];
            
            for (size_t j = 0; j < out_dim && j < ksk_entry.first.degree(); ++j) {
                result_a[j] = (result_a[j] + q - (digit * ksk_entry.first[j]) % q) % q;
            }
            
            result_b = (result_b + q - (digit * ksk_entry.second[0]) % q) % q;
        }
    }
    
    return LWECiphertext(std::move(result_a), result_b, q, ksk.key_id);
}


// ============================================================================
// Bootstrapping Operations (Requirement 8.5)
// ============================================================================

LWECiphertext BootstrapEngine::bootstrap(
    const LWECiphertext& lwe,
    const ExtendedBootstrapKey& bsk
) {
    // Use default test polynomial (identity function)
    return bootstrap_with_test_poly(lwe, bsk, *default_test_poly_);
}

LWECiphertext BootstrapEngine::bootstrap_with_test_poly(
    const LWECiphertext& lwe,
    const ExtendedBootstrapKey& bsk,
    const Polynomial& test_poly
) {
    uint32_t k = bsk.glwe_dimension;
    uint32_t N = bsk.poly_degree;
    uint64_t q = glwe_modulus_;
    
    // Initialize accumulator with test polynomial
    // acc = (0, ..., 0, test_poly)
    GLWECiphertext acc(k, N, q, bsk.key_id);
    for (auto& m : acc.mask) {
        m.set_zero();
    }
    acc.body = test_poly.clone();
    
    // Perform blind rotate
    blind_rotate(acc, lwe, bsk);
    
    // Sample extract
    LWECiphertext extracted = sample_extract(acc);
    
    // Key switch back to original LWE dimension
    LWECiphertext result = key_switch(extracted, bsk.ksk);
    
    return result;
}

// ============================================================================
// Programmable Bootstrapping (Requirement 8.6)
// ============================================================================

LWECiphertext BootstrapEngine::programmable_bootstrap(
    const LWECiphertext& lwe,
    const ExtendedBootstrapKey& bsk,
    const LookupTable& lut
) {
    return bootstrap_with_test_poly(lwe, bsk, lut.table);
}

LookupTable BootstrapEngine::create_lookup_table(
    std::function<uint64_t(uint64_t)> func,
    uint64_t input_modulus,
    uint64_t output_modulus,
    const std::string& name
) {
    uint32_t N = params_.poly_degree;
    uint64_t q = glwe_modulus_;
    
    // Compute delta for output encoding
    uint64_t delta_out = q / output_modulus;
    
    std::vector<uint64_t> coeffs(N, 0);
    
    // The test polynomial encodes the function
    // For input value v, the rotation by v * 2N / input_modulus
    // should give f(v) * delta_out in the constant coefficient
    
    for (uint32_t i = 0; i < N; ++i) {
        // Compute the input value corresponding to this rotation
        // rotation i corresponds to input (i * input_modulus) / (2 * N)
        uint64_t input_val = (i * input_modulus + N) / (2 * N);
        input_val = input_val % input_modulus;
        
        // Apply function
        uint64_t output_val = func(input_val) % output_modulus;
        
        // Encode output
        coeffs[i] = (output_val * delta_out) % q;
    }
    
    return LookupTable(Polynomial(std::move(coeffs), q, false), 
                       input_modulus, output_modulus, name);
}

LookupTable BootstrapEngine::create_negation_lut(uint64_t modulus) {
    return create_lookup_table(
        [modulus](uint64_t x) { return (modulus - x) % modulus; },
        modulus, modulus, "negation"
    );
}

LookupTable BootstrapEngine::create_threshold_lut(uint64_t threshold, uint64_t modulus) {
    return create_lookup_table(
        [threshold](uint64_t x) { return x >= threshold ? 1ULL : 0ULL; },
        modulus, 2, "threshold_" + std::to_string(threshold)
    );
}

LookupTable BootstrapEngine::create_identity_lut(uint64_t modulus) {
    return create_lookup_table(
        [](uint64_t x) { return x; },
        modulus, modulus, "identity"
    );
}


// ============================================================================
// LWE Operations
// ============================================================================

LWECiphertext BootstrapEngine::encrypt_lwe(uint64_t value, const std::vector<int64_t>& sk) {
    size_t n = sk.size();
    uint64_t q = lwe_modulus_;
    uint64_t t = params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4;
    uint64_t delta = q / t;
    
    // Sample random mask
    std::vector<uint64_t> a(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = rng_->random_u64_range(q);
    }
    
    // Compute inner product <a, s>
    int64_t inner_product = 0;
    for (size_t i = 0; i < n; ++i) {
        inner_product += static_cast<int64_t>(a[i]) * sk[i];
    }
    
    // Sample error
    int64_t e = static_cast<int64_t>(rng_->sample_gaussian(
        params_.lwe_noise_std > 0 ? params_.lwe_noise_std : 3.2, q));
    if (e > static_cast<int64_t>(q / 2)) {
        e -= static_cast<int64_t>(q);
    }
    
    // Compute body: b = <a, s> + e + m * delta
    uint64_t m_encoded = (value * delta) % q;
    uint64_t b = (static_cast<uint64_t>((inner_product + e) % static_cast<int64_t>(q)) + m_encoded) % q;
    
    return LWECiphertext(std::move(a), b, q, 0);
}

uint64_t BootstrapEngine::decrypt_lwe(const LWECiphertext& lwe, const std::vector<int64_t>& sk) {
    uint64_t q = lwe.modulus;
    uint64_t t = params_.plaintext_modulus > 0 ? params_.plaintext_modulus : 4;
    
    // Compute inner product <a, s>
    int64_t inner_product = 0;
    for (size_t i = 0; i < lwe.dimension(); ++i) {
        inner_product += static_cast<int64_t>(lwe.a[i]) * sk[i];
    }
    
    // Compute phase: b - <a, s>
    int64_t phase = static_cast<int64_t>(lwe.b) - inner_product;
    phase = ((phase % static_cast<int64_t>(q)) + static_cast<int64_t>(q)) % static_cast<int64_t>(q);
    
    // Decode: round(phase * t / q)
    uint64_t result = (static_cast<uint64_t>(phase) * t + q / 2) / q;
    return result % t;
}

LWECiphertext BootstrapEngine::add_lwe(const LWECiphertext& lwe1, const LWECiphertext& lwe2) {
    if (lwe1.dimension() != lwe2.dimension()) {
        throw std::invalid_argument("LWE ciphertexts must have same dimension");
    }
    
    uint64_t q = lwe1.modulus;
    size_t n = lwe1.dimension();
    
    std::vector<uint64_t> a(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (lwe1.a[i] + lwe2.a[i]) % q;
    }
    
    uint64_t b = (lwe1.b + lwe2.b) % q;
    
    return LWECiphertext(std::move(a), b, q, lwe1.key_id);
}

LWECiphertext BootstrapEngine::negate_lwe(const LWECiphertext& lwe) {
    uint64_t q = lwe.modulus;
    size_t n = lwe.dimension();
    
    std::vector<uint64_t> a(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (q - lwe.a[i]) % q;
    }
    
    uint64_t b = (q - lwe.b) % q;
    
    return LWECiphertext(std::move(a), b, q, lwe.key_id);
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<BootstrapEngine> create_bootstrap_engine(
    const ParameterSet& params,
    HardwareDispatcher* dispatcher
) {
    return std::make_unique<BootstrapEngine>(params, dispatcher);
}

} // namespace fhe_accelerate
