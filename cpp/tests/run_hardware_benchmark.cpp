/**
 * Hardware Benchmark Runner
 * 
 * Run this to benchmark ALL hardware features and find the fastest
 * implementation for each FHE operation.
 * 
 * Compile: clang++ -std=c++17 -O3 -arch arm64 -I./cpp/include \
 *          cpp/tests/run_hardware_benchmark.cpp cpp/src/hardware_benchmark.cpp \
 *          -o run_benchmark
 * 
 * Run: ./run_benchmark
 */

#include "hardware_benchmark.h"
#include <iostream>

using namespace fhe_accelerate::benchmark;

int main(int argc, char* argv[]) {
    std::cout << "Starting comprehensive hardware benchmark...\n";
    
    BenchmarkRunner runner;
    runner.run_full_benchmark_suite();
    
    std::cout << "\nBenchmark complete!\n";
    return 0;
}
