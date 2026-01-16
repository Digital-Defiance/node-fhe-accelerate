/**
 * Tests for Unconventional Hardware Acceleration
 * 
 * Tests AMX, Neural Engine, and other creative hardware exploitation.
 */

#include "test_harness.h"
#include "../include/unconventional_accel.h"
#include <iostream>
#include <chrono>
#include <cstring>

using namespace fhe_accelerate;
using namespace fhe_accelerate::unconventional;
using namespace fhe_accelerate::test;

// Test modulus
static const uint64_t TEST_MODULUS = 132120577ULL;

void test_amx_availability() {
    std::cout << "Testing AMX availability...\n";
    
    bool available = AMXAccelerator::is_available();
    std::cout << "  AMX available: " << (available ? "YES" : "NO") << "\n";
    
    if (available) {
        // Try to create accelerator
        AMXAccelerator amx;
        std::cout << "  AMX accelerator created successfully\n";
    }
    
    std::cout << "  PASSED\n";
}

void test_amx_batch_modmul() {
    std::cout << "Testing AMX batch modular multiplication...\n";
    
    if (!AMXAccelerator::is_available()) {
        std::cout << "  SKIPPED (AMX not available)\n";
        return;
    }
    
    AMXAccelerator amx;
    TestRandom rng(42);
    
    const size_t n = 1024;
    std::vector<uint64_t> a(n), b(n), result(n), expected(n);
    
    // Generate random inputs
    for (size_t i = 0; i < n; i++) {
        a[i] = rng.next_coefficient(TEST_MODULUS);
        b[i] = rng.next_coefficient(TEST_MODULUS);
    }
    
    // Compute expected result
    for (size_t i = 0; i < n; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        expected[i] = product % TEST_MODULUS;
    }
    
    // Compute using AMX
    amx.batch_modmul_amx(a.data(), b.data(), result.data(), n, TEST_MODULUS);
    
    // Verify
    for (size_t i = 0; i < n; i++) {
        TestAssert::assert_equal(result[i], expected[i], 
            "AMX batch modmul mismatch at index " + std::to_string(i));
    }
    
    std::cout << "  PASSED\n";
}

void test_amx_poly_mul_toeplitz() {
    std::cout << "Testing AMX Toeplitz polynomial multiplication...\n";
    
    if (!AMXAccelerator::is_available()) {
        std::cout << "  SKIPPED (AMX not available)\n";
        return;
    }
    
    AMXAccelerator amx;
    
    // Small test case
    const size_t n = 8;
    std::vector<uint64_t> a = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> b = {8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<uint64_t> result(2 * n - 1, 0);
    
    // Compute using AMX
    amx.poly_mul_toeplitz(a.data(), b.data(), result.data(), n, TEST_MODULUS);
    
    // Compute expected (schoolbook multiplication)
    std::vector<uint64_t> expected(2 * n - 1, 0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            __uint128_t product = static_cast<__uint128_t>(a[i]) * b[j];
            __uint128_t sum = static_cast<__uint128_t>(expected[i + j]) + product;
            expected[i + j] = sum % TEST_MODULUS;
        }
    }
    
    // Verify
    for (size_t i = 0; i < 2 * n - 1; i++) {
        TestAssert::assert_equal(result[i], expected[i],
            "Toeplitz poly mul mismatch at index " + std::to_string(i));
    }
    
    std::cout << "  PASSED\n";
}

void benchmark_amx_vs_scalar() {
    std::cout << "\n=== AMX vs Scalar Benchmark ===\n";
    
    if (!AMXAccelerator::is_available()) {
        std::cout << "  SKIPPED (AMX not available)\n";
        return;
    }
    
    AMXAccelerator amx;
    TestRandom rng(42);
    
    const size_t n = 4096;
    std::vector<uint64_t> a(n), b(n), result(n);
    
    for (size_t i = 0; i < n; i++) {
        a[i] = rng.next_coefficient(TEST_MODULUS);
        b[i] = rng.next_coefficient(TEST_MODULUS);
    }
    
    // Benchmark scalar
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        for (size_t i = 0; i < n; i++) {
            __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
            result[i] = product % TEST_MODULUS;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Benchmark AMX
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000; iter++) {
        amx.batch_modmul_amx(a.data(), b.data(), result.data(), n, TEST_MODULUS);
    }
    end = std::chrono::high_resolution_clock::now();
    auto amx_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "  Scalar: " << scalar_time / 1000.0 << " µs per batch\n";
    std::cout << "  AMX:    " << amx_time / 1000.0 << " µs per batch\n";
    std::cout << "  Speedup: " << static_cast<double>(scalar_time) / amx_time << "x\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Unconventional Hardware Acceleration Tests\n";
    std::cout << "========================================\n\n";
    
    try {
        test_amx_availability();
        test_amx_batch_modmul();
        test_amx_poly_mul_toeplitz();
        benchmark_amx_vs_scalar();
        
        std::cout << "\n========================================\n";
        std::cout << "All tests PASSED!\n";
        std::cout << "========================================\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n========================================\n";
        std::cerr << "TEST FAILED: " << e.what() << "\n";
        std::cerr << "========================================\n";
        return 1;
    }
}
