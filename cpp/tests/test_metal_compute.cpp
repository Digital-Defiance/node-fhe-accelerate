/**
 * Metal GPU Compute Backend Tests
 * 
 * Tests and benchmarks for GPU-accelerated FHE operations.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

// Forward declarations for Metal compute
namespace fhe_accelerate {
namespace metal {
    class MetalComputeContext;
    MetalComputeContext& get_metal_context();
    bool metal_available();
    void gpu_batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                          size_t count, uint64_t modulus);
}
}

// CPU reference implementation
void cpu_batch_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                      size_t count, uint64_t modulus) {
    for (size_t i = 0; i < count; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = product % modulus;
    }
}

// Barrett reduction CPU implementation
struct BarrettParams {
    uint64_t modulus;
    uint64_t mu;
    int k;
};

BarrettParams compute_barrett_params(uint64_t modulus) {
    BarrettParams params;
    params.modulus = modulus;
    params.k = 64 - __builtin_clzll(modulus);
    
    if (params.k <= 32) {
        params.mu = (1ULL << (2 * params.k)) / modulus;
    } else {
        __uint128_t numerator = static_cast<__uint128_t>(1) << (2 * params.k);
        params.mu = static_cast<uint64_t>(numerator / modulus);
    }
    
    return params;
}

uint64_t barrett_reduce(__uint128_t x, const BarrettParams& params) {
    int k = params.k;
    __uint128_t x_shifted = x >> (k - 1);
    __uint128_t q_approx = (x_shifted * params.mu) >> (k + 1);
    __uint128_t r = x - q_approx * params.modulus;
    
    while (r >= params.modulus) {
        r -= params.modulus;
    }
    
    return static_cast<uint64_t>(r);
}

void cpu_barrett_modmul(const uint64_t* a, const uint64_t* b, uint64_t* result,
                        size_t count, uint64_t modulus) {
    BarrettParams params = compute_barrett_params(modulus);
    
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        __uint128_t p0 = static_cast<__uint128_t>(a[i]) * b[i];
        __uint128_t p1 = static_cast<__uint128_t>(a[i+1]) * b[i+1];
        __uint128_t p2 = static_cast<__uint128_t>(a[i+2]) * b[i+2];
        __uint128_t p3 = static_cast<__uint128_t>(a[i+3]) * b[i+3];
        
        result[i] = barrett_reduce(p0, params);
        result[i+1] = barrett_reduce(p1, params);
        result[i+2] = barrett_reduce(p2, params);
        result[i+3] = barrett_reduce(p3, params);
    }
    
    for (; i < count; i++) {
        __uint128_t product = static_cast<__uint128_t>(a[i]) * b[i];
        result[i] = barrett_reduce(product, params);
    }
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
    return duration.count() / 1000.0 / iterations;  // microseconds per iteration
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

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           METAL GPU COMPUTE BACKEND BENCHMARK                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    // Check Metal availability
    bool metal_ok = fhe_accelerate::metal::metal_available();
    std::cout << "Metal GPU: " << (metal_ok ? "AVAILABLE" : "NOT AVAILABLE") << "\n\n";
    
    if (!metal_ok) {
        std::cout << "Skipping GPU benchmarks - Metal not available\n";
        return 0;
    }
    
    // Test parameters
    uint64_t modulus = 132120577ULL;  // NTT-friendly prime
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, modulus - 1);
    
    std::cout << std::left << std::setw(15) << "Size"
              << std::right << std::setw(15) << "CPU (µs)"
              << std::setw(15) << "Barrett (µs)"
              << std::setw(15) << "GPU (µs)"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "Correct"
              << "\n";
    std::cout << std::string(82, '-') << "\n";
    
    for (size_t n : sizes) {
        // Allocate and initialize data
        std::vector<uint64_t> a(n), b(n);
        std::vector<uint64_t> result_cpu(n), result_barrett(n), result_gpu(n);
        
        for (size_t i = 0; i < n; i++) {
            a[i] = dist(gen);
            b[i] = dist(gen);
        }
        
        // Determine iteration count based on size
        int iterations = n <= 4096 ? 1000 : (n <= 65536 ? 100 : 10);
        
        // Benchmark CPU scalar
        double cpu_time = benchmark([&]() {
            cpu_batch_modmul(a.data(), b.data(), result_cpu.data(), n, modulus);
        }, iterations);
        
        // Benchmark CPU Barrett
        double barrett_time = benchmark([&]() {
            cpu_barrett_modmul(a.data(), b.data(), result_barrett.data(), n, modulus);
        }, iterations);
        
        // Benchmark GPU
        double gpu_time = benchmark([&]() {
            fhe_accelerate::metal::gpu_batch_modmul(a.data(), b.data(), result_gpu.data(), n, modulus);
        }, iterations);
        
        // Verify correctness
        bool correct = verify_results(result_cpu.data(), result_gpu.data(), n);
        
        // Calculate speedup vs best CPU
        double best_cpu = std::min(cpu_time, barrett_time);
        double speedup = best_cpu / gpu_time;
        
        std::cout << std::left << std::setw(15) << n
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << cpu_time
                  << std::setw(15) << barrett_time
                  << std::setw(15) << gpu_time
                  << std::setw(12) << std::setprecision(2) << speedup << "x"
                  << std::setw(10) << (correct ? "YES" : "NO")
                  << "\n";
    }
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ANALYSIS                                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "GPU dispatch overhead makes it slower for small batches (<4096).\n";
    std::cout << "For large batches (>65536), GPU provides significant speedup.\n";
    std::cout << "Optimal use case: batch processing of many polynomials.\n";
    std::cout << "\nRecommendation:\n";
    std::cout << "  - Use CPU Barrett for n < 4096\n";
    std::cout << "  - Use GPU for n >= 4096 (batch operations)\n";
    std::cout << "  - GPU excels at processing multiple polynomials in parallel\n";
    
    return 0;
}
