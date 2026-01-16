/**
 * @file test_key_manager.cpp
 * @brief Tests for FHE Key Generation and Management
 * 
 * Tests key generation, serialization, and threshold decryption.
 */

#include "test_harness.h"
#include "../include/key_manager.h"
#include "../include/key_serializer.h"
#include "../include/parameter_set.h"
#include <iostream>
#include <sstream>
#include <chrono>

using namespace fhe_accelerate;

// ============================================================================
// Test Utilities
// ============================================================================

ParameterSet create_test_params() {
    // Use TFHE-128-FAST preset for testing
    return TFHE_128_FAST();
}

// ============================================================================
// Secret Key Generation Tests
// ============================================================================

void test_secret_key_generation_ternary() {
    std::cout << "Testing secret key generation (ternary)..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    
    auto sk = key_manager->generate_secret_key(SecretKeyDistribution::TERNARY);
    
    // Verify key properties
    TEST_ASSERT(sk != nullptr, "Secret key should not be null");
    TEST_ASSERT(sk->poly.degree() == params.poly_degree, "Polynomial degree should match params");
    TEST_ASSERT(sk->distribution == SecretKeyDistribution::TERNARY, "Distribution should be ternary");
    TEST_ASSERT(sk->key_id > 0, "Key ID should be positive");
    
    // Verify coefficients are in {-1, 0, 1} (represented as {q-1, 0, 1})
    uint64_t modulus = params.moduli[0];
    bool all_ternary = true;
    for (uint32_t i = 0; i < sk->poly.degree(); ++i) {
        uint64_t coeff = sk->poly[i];
        if (coeff != 0 && coeff != 1 && coeff != modulus - 1) {
            all_ternary = false;
            break;
        }
    }
    TEST_ASSERT(all_ternary, "All coefficients should be ternary");
    
    std::cout << "  PASSED: Secret key generation (ternary)" << std::endl;
}

void test_secret_key_generation_binary() {
    std::cout << "Testing secret key generation (binary)..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    
    auto sk = key_manager->generate_secret_key(SecretKeyDistribution::BINARY);
    
    TEST_ASSERT(sk != nullptr, "Secret key should not be null");
    TEST_ASSERT(sk->distribution == SecretKeyDistribution::BINARY, "Distribution should be binary");
    
    // Verify coefficients are in {0, 1}
    bool all_binary = true;
    for (uint32_t i = 0; i < sk->poly.degree(); ++i) {
        if (sk->poly[i] != 0 && sk->poly[i] != 1) {
            all_binary = false;
            break;
        }
    }
    TEST_ASSERT(all_binary, "All coefficients should be binary");
    
    std::cout << "  PASSED: Secret key generation (binary)" << std::endl;
}

// ============================================================================
// Public Key Generation Tests
// ============================================================================

void test_public_key_generation() {
    std::cout << "Testing public key generation..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    
    auto sk = key_manager->generate_secret_key();
    
    auto start = std::chrono::high_resolution_clock::now();
    auto pk = key_manager->generate_public_key(*sk);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    TEST_ASSERT(pk != nullptr, "Public key should not be null");
    TEST_ASSERT(pk->a.degree() == params.poly_degree, "Polynomial a degree should match");
    TEST_ASSERT(pk->b.degree() == params.poly_degree, "Polynomial b degree should match");
    TEST_ASSERT(pk->key_id == sk->key_id, "Key IDs should match");
    
    std::cout << "  Public key generation time: " << duration.count() << "ms" << std::endl;
    
    // Target is < 100ms for 128-bit security
    // Note: This may not pass on all hardware, but we log the time
    if (duration.count() < 100) {
        std::cout << "  PASSED: Public key generation (< 100ms target met)" << std::endl;
    } else {
        std::cout << "  PASSED: Public key generation (target not met: " 
                  << duration.count() << "ms > 100ms)" << std::endl;
    }
}

// ============================================================================
// Evaluation Key Generation Tests
// ============================================================================

void test_eval_key_generation() {
    std::cout << "Testing evaluation key generation..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    
    auto sk = key_manager->generate_secret_key();
    auto ek = key_manager->generate_eval_key(*sk);
    
    TEST_ASSERT(ek != nullptr, "Evaluation key should not be null");
    TEST_ASSERT(ek->key_id == sk->key_id, "Key IDs should match");
    TEST_ASSERT(ek->relin_key.decomp_base_log > 0, "Decomposition base log should be positive");
    TEST_ASSERT(ek->relin_key.decomp_level > 0, "Decomposition level should be positive");
    TEST_ASSERT(!ek->relin_key.keys.empty(), "Relinearization keys should not be empty");
    
    std::cout << "  Decomposition base log: " << ek->relin_key.decomp_base_log << std::endl;
    std::cout << "  Decomposition levels: " << ek->relin_key.decomp_level << std::endl;
    std::cout << "  Number of key pairs: " << ek->relin_key.keys.size() << std::endl;
    
    std::cout << "  PASSED: Evaluation key generation" << std::endl;
}

// ============================================================================
// Bootstrapping Key Generation Tests
// ============================================================================

void test_bootstrap_key_generation() {
    std::cout << "Testing bootstrapping key generation..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    
    auto sk = key_manager->generate_secret_key();
    
    auto start = std::chrono::high_resolution_clock::now();
    auto bk = key_manager->generate_bootstrap_key(*sk);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    TEST_ASSERT(bk != nullptr, "Bootstrap key should not be null");
    TEST_ASSERT(bk->key_id == sk->key_id, "Key IDs should match");
    TEST_ASSERT(!bk->bsk.empty(), "BSK should not be empty");
    TEST_ASSERT(!bk->ksk.keys.empty(), "KSK should not be empty");
    
    std::cout << "  Bootstrap key generation time: " << duration.count() << "ms" << std::endl;
    std::cout << "  BSK size: " << bk->bsk.size() << " entries" << std::endl;
    std::cout << "  KSK size: " << bk->ksk.keys.size() << " key pairs" << std::endl;
    
    std::cout << "  PASSED: Bootstrapping key generation" << std::endl;
}

// ============================================================================
// Threshold Key Generation Tests
// ============================================================================

void test_threshold_key_generation() {
    std::cout << "Testing threshold key generation (3-of-5)..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    
    auto threshold_keys = key_manager->generate_threshold_keys(3, 5);
    
    TEST_ASSERT(threshold_keys != nullptr, "Threshold keys should not be null");
    TEST_ASSERT(threshold_keys->threshold == 3, "Threshold should be 3");
    TEST_ASSERT(threshold_keys->total_shares == 5, "Total shares should be 5");
    TEST_ASSERT(threshold_keys->shares.size() == 5, "Should have 5 shares");
    
    // Verify each share has unique ID
    std::set<uint32_t> share_ids;
    for (const auto& share : threshold_keys->shares) {
        share_ids.insert(share.share_id);
    }
    TEST_ASSERT(share_ids.size() == 5, "All share IDs should be unique");
    
    std::cout << "  PASSED: Threshold key generation" << std::endl;
}

// ============================================================================
// Key Serialization Tests
// ============================================================================

void test_secret_key_serialization() {
    std::cout << "Testing secret key serialization round-trip..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    auto serializer = create_key_serializer();
    
    auto sk = key_manager->generate_secret_key();
    
    // Serialize
    std::vector<uint8_t> serialized;
    auto ser_result = serializer->serialize_secret_key(*sk, serialized);
    
    TEST_ASSERT(ser_result.success, "Serialization should succeed");
    TEST_ASSERT(!serialized.empty(), "Serialized data should not be empty");
    
    std::cout << "  Serialized size: " << serialized.size() << " bytes" << std::endl;
    
    // Deserialize
    auto deser_result = serializer->deserialize_secret_key(serialized, params.moduli[0]);
    
    TEST_ASSERT(deser_result.success, "Deserialization should succeed");
    TEST_ASSERT(deser_result.value != nullptr, "Deserialized key should not be null");
    
    // Verify round-trip
    auto& sk2 = *deser_result.value;
    TEST_ASSERT(sk2.key_id == sk->key_id, "Key IDs should match");
    TEST_ASSERT(sk2.distribution == sk->distribution, "Distributions should match");
    TEST_ASSERT(sk2.poly.degree() == sk->poly.degree(), "Degrees should match");
    
    bool coeffs_match = true;
    for (uint32_t i = 0; i < sk->poly.degree(); ++i) {
        if (sk2.poly[i] != sk->poly[i]) {
            coeffs_match = false;
            break;
        }
    }
    TEST_ASSERT(coeffs_match, "Coefficients should match after round-trip");
    
    std::cout << "  PASSED: Secret key serialization round-trip" << std::endl;
}

void test_public_key_serialization() {
    std::cout << "Testing public key serialization round-trip..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    auto serializer = create_key_serializer();
    
    auto sk = key_manager->generate_secret_key();
    auto pk = key_manager->generate_public_key(*sk);
    
    // Serialize
    std::vector<uint8_t> serialized;
    auto ser_result = serializer->serialize_public_key(*pk, serialized);
    
    TEST_ASSERT(ser_result.success, "Serialization should succeed");
    
    std::cout << "  Serialized size: " << serialized.size() << " bytes" << std::endl;
    
    // Deserialize
    auto deser_result = serializer->deserialize_public_key(serialized);
    
    TEST_ASSERT(deser_result.success, "Deserialization should succeed");
    
    auto& pk2 = *deser_result.value;
    TEST_ASSERT(pk2.key_id == pk->key_id, "Key IDs should match");
    TEST_ASSERT(pk2.a.degree() == pk->a.degree(), "Polynomial a degrees should match");
    TEST_ASSERT(pk2.b.degree() == pk->b.degree(), "Polynomial b degrees should match");
    
    std::cout << "  PASSED: Public key serialization round-trip" << std::endl;
}

void test_eval_key_serialization() {
    std::cout << "Testing evaluation key serialization round-trip..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    auto serializer = create_key_serializer();
    
    auto sk = key_manager->generate_secret_key();
    auto ek = key_manager->generate_eval_key(*sk);
    
    // Serialize
    std::vector<uint8_t> serialized;
    auto ser_result = serializer->serialize_eval_key(*ek, serialized);
    
    TEST_ASSERT(ser_result.success, "Serialization should succeed");
    
    std::cout << "  Serialized size: " << serialized.size() << " bytes" << std::endl;
    
    // Deserialize
    auto deser_result = serializer->deserialize_eval_key(serialized);
    
    TEST_ASSERT(deser_result.success, "Deserialization should succeed");
    
    auto& ek2 = *deser_result.value;
    TEST_ASSERT(ek2.key_id == ek->key_id, "Key IDs should match");
    TEST_ASSERT(ek2.relin_key.decomp_base_log == ek->relin_key.decomp_base_log, 
                "Decomposition base log should match");
    TEST_ASSERT(ek2.relin_key.keys.size() == ek->relin_key.keys.size(), 
                "Number of key pairs should match");
    
    std::cout << "  PASSED: Evaluation key serialization round-trip" << std::endl;
}

void test_integrity_verification() {
    std::cout << "Testing integrity verification..." << std::endl;
    
    auto params = create_test_params();
    auto key_manager = create_key_manager(params);
    auto serializer = create_key_serializer();
    
    auto sk = key_manager->generate_secret_key();
    
    std::vector<uint8_t> serialized;
    serializer->serialize_secret_key(*sk, serialized);
    
    // Verify integrity of valid data
    TEST_ASSERT(serializer->verify_integrity(serialized), "Valid data should pass integrity check");
    
    // Corrupt data and verify it fails
    if (serialized.size() > 50) {
        serialized[50] ^= 0xFF;  // Flip bits
        TEST_ASSERT(!serializer->verify_integrity(serialized), 
                    "Corrupted data should fail integrity check");
    }
    
    std::cout << "  PASSED: Integrity verification" << std::endl;
}

// ============================================================================
// Ballot Serialization Tests
// ============================================================================

void test_ballot_serialization() {
    std::cout << "Testing ballot serialization..." << std::endl;
    
    auto params = create_test_params();
    uint32_t degree = params.poly_degree;
    uint64_t modulus = params.moduli[0];
    
    // Create mock encrypted choices (3 choices for 3 candidates)
    std::vector<std::pair<Polynomial, Polynomial>> choices;
    for (int i = 0; i < 3; ++i) {
        std::vector<uint64_t> a_coeffs(degree, i + 1);
        std::vector<uint64_t> b_coeffs(degree, i + 2);
        Polynomial a(std::move(a_coeffs), modulus, false);
        Polynomial b(std::move(b_coeffs), modulus, false);
        choices.emplace_back(std::move(a), std::move(b));
    }
    
    auto ballot_serializer = create_ballot_serializer();
    
    std::vector<uint8_t> serialized;
    uint64_t timestamp = 1234567890;
    auto result = ballot_serializer->serialize_ballot(choices, timestamp, serialized);
    
    TEST_ASSERT(result.success, "Ballot serialization should succeed");
    
    std::cout << "  Serialized ballot size: " << serialized.size() << " bytes" << std::endl;
    
    // Check if under 10KB target (for small degree)
    size_t estimated = BallotSerializer::estimate_ballot_size(3, degree);
    std::cout << "  Estimated size: " << estimated << " bytes" << std::endl;
    
    // Deserialize
    auto deser_result = ballot_serializer->deserialize_ballot(serialized, degree, modulus);
    
    TEST_ASSERT(deser_result.success, "Ballot deserialization should succeed");
    TEST_ASSERT(deser_result.value->timestamp == timestamp, "Timestamp should match");
    TEST_ASSERT(deser_result.value->encrypted_choices.size() == 3, "Should have 3 choices");
    
    std::cout << "  PASSED: Ballot serialization" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FHE Key Manager Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // Secret Key Generation Tests
    try {
        test_secret_key_generation_ternary();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_secret_key_generation_ternary - " << e.what() << std::endl;
        failed++;
    }
    
    try {
        test_secret_key_generation_binary();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_secret_key_generation_binary - " << e.what() << std::endl;
        failed++;
    }
    
    // Public Key Generation Tests
    try {
        test_public_key_generation();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_public_key_generation - " << e.what() << std::endl;
        failed++;
    }
    
    // Evaluation Key Generation Tests
    try {
        test_eval_key_generation();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_eval_key_generation - " << e.what() << std::endl;
        failed++;
    }
    
    // Bootstrapping Key Generation Tests
    try {
        test_bootstrap_key_generation();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_bootstrap_key_generation - " << e.what() << std::endl;
        failed++;
    }
    
    // Threshold Key Generation Tests
    try {
        test_threshold_key_generation();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_threshold_key_generation - " << e.what() << std::endl;
        failed++;
    }
    
    // Serialization Tests
    try {
        test_secret_key_serialization();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_secret_key_serialization - " << e.what() << std::endl;
        failed++;
    }
    
    try {
        test_public_key_serialization();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_public_key_serialization - " << e.what() << std::endl;
        failed++;
    }
    
    try {
        test_eval_key_serialization();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_eval_key_serialization - " << e.what() << std::endl;
        failed++;
    }
    
    try {
        test_integrity_verification();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_integrity_verification - " << e.what() << std::endl;
        failed++;
    }
    
    try {
        test_ballot_serialization();
        passed++;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: test_ballot_serialization - " << e.what() << std::endl;
        failed++;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return failed > 0 ? 1 : 0;
}
