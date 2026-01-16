/**
 * SME Accelerator Tests and Benchmarks
 */

#include "sme_accelerator.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>

using namespace fhe_accelerate::sme;

// Reference implementation
void reference_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus) {
    for (size_t i = 0; i < count; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = product % modulus;
    }
}

bool verify_results(const uint64_t* expected, const uint64_t* actual, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (expected[i] != actual[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename Func>
double benchmark(Func&& func, int iterations) {
    // Warmup
    for (int i = 0; i < 5; i++) {
        func();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return duration.count() / 1000.0 / iterations;
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           SME ACCELERATOR TESTS AND BENCHMARKS               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    // Check SME availability
    std::cout << "Hardware Detection:\n";
    std::cout << "  SME:  " << (sme_available() ? "YES" : "NO") << "\n";
    std::cout << "  SME2: " << (sme2_available() ? "YES" : "NO") << "\n";
    std::cout << "  SME Vector Length: " << sme_vector_length() << " bytes\n\n";
    
    // Test parameters
    uint64_t modulus = 132120577ULL;
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536};
    
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, modulus - 1);
    
    // ========================================================================
    // Test 1: Batch Modular Multiplication
    // ========================================================================
    std::cout << "Test 1: Batch Modular Multiplication\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Size"
              << std::right << std::setw(15) << "Reference (µs)"
              << std::setw(15) << "SME-style (µs)"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "Correct"
              << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t n : sizes) {
        std::vector<uint64_t> a(n), b(n);
        std::vector<uint64_t> result_ref(n), result_sme(n);
        
        for (size_t i = 0; i < n; i++) {
            a[i] = dist(gen);
            b[i] = dist(gen);
        }
        
        int iterations = n <= 4096 ? 1000 : 100;
        
        double ref_time = benchmark([&]() {
            reference_modmul(a.data(), b.data(), result_ref.data(), n, modulus);
        }, iterations);
        
        double sme_time = benchmark([&]() {
            sme_batch_modmul(a.data(), b.data(), result_sme.data(), n, modulus);
        }, iterations);
        
        bool correct = verify_results(result_ref.data(), result_sme.data(), n);
        double speedup = ref_time / sme_time;
        
        std::cout << std::left << std::setw(12) << n
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << ref_time
                  << std::setw(15) << sme_time
                  << std::setw(12) << std::setprecision(2) << speedup << "x"
                  << std::setw(10) << (correct ? "YES" : "NO")
                  << "\n";
    }
    
    // ========================================================================
    // Test 2: NTT Butterfly Stage
    // ========================================================================
    std::cout << "\nTest 2: NTT Butterfly Stage\n";
    std::cout << std::string(60, '-') << "\n";
    
    // Generate twiddle factors
    auto generate_twiddles = [](size_t n, uint64_t modulus) -> std::vector<uint64_t> {
        std::vector<uint64_t> twiddles(n);
        uint64_t g = 3;  // Primitive root
        uint64_t order = modulus - 1;
        
        // Compute w = g^(order/n) mod modulus
        uint64_t exp = order / n;
        uint64_t w = 1;
        uint64_t base = g;
        while (exp > 0) {
            if (exp & 1) {
                w = static_cast<uint64_t>((static_cast<__uint128_t>(w) * base) % modulus);
            }
            base = static_cast<uint64_t>((static_cast<__uint128_t>(base) * base) % modulus);
            exp >>= 1;
        }
        
        twiddles[0] = 1;
        for (size_t i = 1; i < n; i++) {
            twiddles[i] = static_cast<uint64_t>((static_cast<__uint128_t>(twiddles[i-1]) * w) % modulus);
        }
        return twiddles;
    };
    
    for (size_t degree : {1024, 4096}) {
        auto twiddles = generate_twiddles(degree, modulus);
        
        std::vector<uint64_t> coeffs(degree);
        for (size_t i = 0; i < degree; i++) {
            coeffs[i] = dist(gen);
        }
        
        // Test each stage
        std::cout << "Degree " << degree << ":\n";
        
        size_t log_degree = 0;
        size_t temp = degree;
        while (temp > 1) { temp >>= 1; log_degree++; }
        
        double total_time = 0;
        for (size_t stage = 0; stage < log_degree; stage++) {
            std::vector<uint64_t> test_coeffs = coeffs;
            
            double stage_time = benchmark([&]() {
                sme_ntt_butterfly_stage(test_coeffs.data(), degree, stage, modulus, twiddles.data());
            }, 100);
            
            total_time += stage_time;
        }
        
        std::cout << "  Total NTT time (all stages): " << std::fixed << std::setprecision(2) 
                  << total_time << " µs\n";
        std::cout << "  Throughput: " << std::scientific << std::setprecision(2) 
                  << (degree / (total_time / 1e6)) << " coefficients/sec\n";
    }
    
    // ========================================================================
    // Test 3: Batch Polynomial Addition
    // ========================================================================
    std::cout << "\nTest 3: Batch Polynomial Addition\n";
    std::cout << std::string(60, '-') << "\n";
    
    size_t degree = 1024;
    std::vector<size_t> batch_sizes = {1, 10, 100, 1000};
    
    for (size_t batch : batch_sizes) {
        size_t total = degree * batch;
        std::vector<uint64_t> a(total), b(total), result(total);
        
        for (size_t i = 0; i < total; i++) {
            a[i] = dist(gen);
            b[i] = dist(gen);
        }
        
        double time = benchmark([&]() {
            sme_batch_poly_add(a.data(), b.data(), result.data(), degree, batch, modulus);
        }, 100);
        
        std::cout << "  Batch size " << std::setw(4) << batch << ": " 
                  << std::fixed << std::setprecision(2) << time << " µs "
                  << "(" << std::scientific << std::setprecision(2) 
                  << (total / (time / 1e6)) << " ops/sec)\n";
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    SUMMARY                                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "SME Status: " << (sme_available() ? "Available but using NEON fallback" : "Not available") << "\n";
    std::cout << "\nNote: True SME intrinsics are not yet exposed by Apple's toolchain.\n";
    std::cout << "Current implementation uses SME-style 8x unrolled NEON operations\n";
    std::cout << "that mimic SME's tile-based approach for cache efficiency.\n";
    std::cout << "\nWhen Apple exposes SME intrinsics, the implementation can be\n";
    std::cout << "upgraded to use true SME streaming mode and ZA tile registers.\n";
    
    return 0;
}
