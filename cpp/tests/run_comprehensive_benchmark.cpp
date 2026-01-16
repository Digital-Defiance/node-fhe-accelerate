/**
 * Comprehensive Hardware Benchmark Runner
 * 
 * Run this to benchmark ALL hardware features and find the fastest
 * implementation for each FHE operation.
 * 
 * Compile: clang++ -std=c++17 -O3 -arch arm64 -I./cpp/include \
 *          cpp/tests/run_comprehensive_benchmark.cpp \
 *          cpp/src/comprehensive_benchmark.cpp \
 *          cpp/src/hardware_benchmark.cpp \
 *          cpp/src/neural_engine_accel.mm \
 *          cpp/src/ray_tracing_accel.mm \
 *          cpp/src/texture_sampling_accel.mm \
 *          cpp/src/memory_optimizer.mm \
 *          -framework Metal -framework Foundation -framework IOSurface \
 *          -framework Accelerate -framework CoreML \
 *          -o run_comprehensive_benchmark
 * 
 * Run: ./run_comprehensive_benchmark [output_directory]
 */

#include "comprehensive_benchmark.h"
#include <iostream>
#include <string>

using namespace fhe_accelerate::comprehensive_benchmark;

int main(int argc, char* argv[]) {
    std::cout << "Starting comprehensive hardware benchmark...\n\n";
    
    ComprehensiveBenchmarkRunner runner;
    
    // Configure benchmark parameters
    runner.set_iterations(100);
    runner.set_warmup(10);
    
    // Run full benchmark suite
    runner.run_full_suite();
    
    // Export results if output directory specified
    if (argc > 1) {
        std::string output_dir = argv[1];
        runner.export_results(output_dir);
    } else {
        // Default export to current directory
        runner.export_results(".");
    }
    
    std::cout << "\nBenchmark complete!\n";
    return 0;
}
