/**
 * Neural Engine Tests and Benchmarks
 */

#include "neural_engine_accel.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace fhe_accelerate::neural_engine;

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           NEURAL ENGINE EXPLORATION                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Hardware Detection:\n";
    std::cout << "  Neural Engine: " << (neural_engine_available() ? "YES" : "NO") << "\n";
    std::cout << "  TOPS: " << neural_engine_tops() << "\n\n";
    
    std::cout << "Accelerate Framework Benchmarks:\n";
    std::cout << std::string(60, '-') << "\n";
    
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144};
    
    std::cout << std::left << std::setw(12) << "Size"
              << std::right << std::setw(15) << "CPU (µs)"
              << std::setw(15) << "vDSP (µs)"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t n : sizes) {
        auto result = benchmark_neural_engine(n);
        std::cout << std::left << std::setw(12) << n
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << result.cpu_time_us
                  << std::setw(15) << result.ane_time_us
                  << std::setw(12) << std::setprecision(2) << result.speedup << "x"
                  << "\n";
    }
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ANALYSIS                                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Neural Engine Findings:\n";
    std::cout << "  - ANE is optimized for neural network inference (INT8, FP16)\n";
    std::cout << "  - Direct use requires CoreML model compilation\n";
    std::cout << "  - Integer modular arithmetic doesn't map well to ANE\n\n";
    
    std::cout << "Accelerate Framework Findings:\n";
    std::cout << "  - vDSP provides highly optimized SIMD operations\n";
    std::cout << "  - BLAS useful for matrix-form polynomial operations\n";
    std::cout << "  - Sparse BLAS available for sparse ciphertext ops\n\n";
    
    std::cout << "Recommendation:\n";
    std::cout << "  - Use vDSP for float operations (hash trees, etc.)\n";
    std::cout << "  - Use BLAS for matrix-form NTT/polynomial ops\n";
    std::cout << "  - ANE best suited for ML-based FHE optimizations\n";
    std::cout << "    (e.g., learned parameter selection, noise estimation)\n";
    
    return 0;
}
