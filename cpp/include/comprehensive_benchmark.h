/**
 * Comprehensive Hardware Benchmarking Suite
 * 
 * This module provides exhaustive benchmarking of ALL hardware features
 * on Apple M4 Max for FHE operations.
 * 
 * Requirements: 21.1-21.12
 * 
 * Features:
 * - BenchmarkResult struct with latency, throughput, bandwidth, power
 * - Statistical analysis (mean, stddev, significance testing)
 * - Benchmark report generator
 * - Tests for all hardware paths: SME, AMX, Metal GPU, NEON, Neural Engine,
 *   Ray Tracing, Texture Sampling, CPU fallback
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>

namespace fhe_accelerate {
namespace comprehensive_benchmark {

// ============================================================================
// Hardware Backend Enumeration
// ============================================================================

enum class HardwareBackend {
    CPU_SCALAR,      // Pure scalar CPU implementation
    NEON,            // ARM NEON SIMD
    NEON_UNROLLED,   // NEON with loop unrolling
    SME,             // Scalable Matrix Extension
    SME2,            // SME2 with predication
    AMX,             // Apple Matrix Coprocessor via Accelerate
    METAL_GPU,       // Metal GPU compute
    NEURAL_ENGINE,   // Apple Neural Engine
    RAY_TRACING,     // Metal ray tracing hardware
    TEXTURE_SAMPLING,// Metal texture sampling
    HYBRID,          // Combination of multiple backends
    COUNT            // Number of backends
};

const char* backend_name(HardwareBackend backend);

// ============================================================================
// Operation Types
// ============================================================================

enum class OperationType {
    MODULAR_MUL,           // Modular multiplication
    MODULAR_ADD,           // Modular addition
    NTT_FORWARD,           // Forward NTT
    NTT_INVERSE,           // Inverse NTT
    POLY_MUL,              // Polynomial multiplication
    POLY_ADD,              // Polynomial addition
    KEY_SWITCH,            // Key switching
    BOOTSTRAP,             // TFHE bootstrapping
    DECOMPOSITION,         // Gadget decomposition
    MERKLE_TREE,           // Merkle tree operations
    POSEIDON_HASH,         // Poseidon hash
    LUT_EVAL,              // Lookup table evaluation
    MEMORY_COPY,           // Memory bandwidth test
    COUNT
};

const char* operation_name(OperationType op);

// ============================================================================
// Benchmark Result Structure
// ============================================================================

struct BenchmarkResult {
    // Identification
    std::string operation;
    std::string method_name;
    HardwareBackend backend;
    
    // Data parameters
    size_t data_size;           // Number of elements
    size_t element_size_bytes;  // Size of each element
    
    // Timing results (in microseconds)
    double latency_us;          // Average latency per operation
    double latency_min_us;      // Minimum latency
    double latency_max_us;      // Maximum latency
    double latency_stddev_us;   // Standard deviation
    
    // Throughput
    double throughput_ops_sec;  // Operations per second
    double throughput_elements_sec; // Elements per second
    
    // Bandwidth
    double bandwidth_gbps;      // GB/s for memory operations
    
    // Power (if measurable)
    std::optional<double> power_watts;
    std::optional<double> energy_per_op_nj; // Nanojoules per operation
    
    // Comparison
    double speedup_vs_baseline; // Speedup vs CPU scalar
    
    // Correctness
    bool correctness_verified;
    std::string correctness_notes;
    
    // Additional info
    std::string notes;
    int num_iterations;
    int num_warmup_iterations;
    
    // Constructor
    BenchmarkResult();
};

// ============================================================================
// Statistical Analysis
// ============================================================================

struct StatisticalSummary {
    double mean;
    double median;
    double stddev;
    double variance;
    double min;
    double max;
    double percentile_95;
    double percentile_99;
    size_t sample_count;
    double confidence_interval_95_low;
    double confidence_interval_95_high;
};

class StatisticalAnalyzer {
public:
    /**
     * Compute statistical summary from samples
     */
    static StatisticalSummary analyze(const std::vector<double>& samples);
    
    /**
     * Compute mean
     */
    static double mean(const std::vector<double>& samples);
    
    /**
     * Compute standard deviation
     */
    static double stddev(const std::vector<double>& samples);
    
    /**
     * Compute variance
     */
    static double variance(const std::vector<double>& samples);
    
    /**
     * Compute percentile
     */
    static double percentile(std::vector<double> samples, double p);
    
    /**
     * Two-sample t-test for significance
     * Returns p-value
     */
    static double t_test(const std::vector<double>& a, const std::vector<double>& b);
    
    /**
     * Check if difference is statistically significant (p < 0.05)
     */
    static bool is_significant(const std::vector<double>& a, const std::vector<double>& b);
    
    /**
     * Compute 95% confidence interval
     */
    static std::pair<double, double> confidence_interval_95(const std::vector<double>& samples);
};

// ============================================================================
// Benchmark Suite
// ============================================================================

struct BenchmarkSuite {
    std::string suite_name;
    OperationType operation;
    std::vector<BenchmarkResult> results;
    
    // Winner information
    std::string winner_method;
    HardwareBackend winner_backend;
    double best_latency_us;
    double best_throughput;
    
    // Statistical comparison
    bool winner_is_significant;  // Statistically significant vs second best
    
    // Methods
    void determine_winner();
    void print_report() const;
    void export_csv(const std::string& filename) const;
    void export_json(const std::string& filename) const;
};

// ============================================================================
// Benchmark Report Generator
// ============================================================================

class BenchmarkReportGenerator {
public:
    BenchmarkReportGenerator();
    ~BenchmarkReportGenerator();
    
    /**
     * Add a benchmark suite to the report
     */
    void add_suite(const BenchmarkSuite& suite);
    
    /**
     * Generate text report
     */
    std::string generate_text_report() const;
    
    /**
     * Generate markdown report
     */
    std::string generate_markdown_report() const;
    
    /**
     * Generate CSV report
     */
    void export_csv(const std::string& directory) const;
    
    /**
     * Generate JSON report
     */
    void export_json(const std::string& filename) const;
    
    /**
     * Generate HTML report with charts
     */
    void export_html(const std::string& filename) const;
    
    /**
     * Get optimal backend for operation at given size
     */
    HardwareBackend get_optimal_backend(OperationType op, size_t data_size) const;
    
    /**
     * Get summary of all winners
     */
    std::map<std::string, std::string> get_winners_summary() const;
    
private:
    std::vector<BenchmarkSuite> suites_;
    
    std::string format_time(double us) const;
    std::string format_throughput(double ops_sec) const;
    std::string format_bandwidth(double gbps) const;
};

// ============================================================================
// Timing Utilities
// ============================================================================

class BenchmarkTimer {
public:
    BenchmarkTimer();
    
    /**
     * Time a function with warmup and multiple iterations
     * Returns vector of individual timings in microseconds
     */
    template<typename Func>
    std::vector<double> time_operation(Func&& func, int iterations, int warmup = 10);
    
    /**
     * Time a function and return BenchmarkResult
     */
    template<typename Func>
    BenchmarkResult benchmark(
        const std::string& operation,
        const std::string& method_name,
        HardwareBackend backend,
        size_t data_size,
        size_t element_size,
        Func&& func,
        int iterations = 100,
        int warmup = 10
    );
    
    /**
     * Set baseline for speedup calculation
     */
    void set_baseline(double baseline_time_us);
    
private:
    double baseline_time_us_;
    
    template<typename Func>
    double time_single(Func&& func);
};

// ============================================================================
// Comprehensive Benchmark Runner
// ============================================================================

class ComprehensiveBenchmarkRunner {
public:
    ComprehensiveBenchmarkRunner();
    ~ComprehensiveBenchmarkRunner();
    
    // ========== Configuration ==========
    
    /**
     * Set number of iterations for benchmarks
     */
    void set_iterations(int iterations) { iterations_ = iterations; }
    
    /**
     * Set warmup iterations
     */
    void set_warmup(int warmup) { warmup_ = warmup; }
    
    /**
     * Enable/disable specific backends
     */
    void enable_backend(HardwareBackend backend, bool enabled);
    
    /**
     * Set data sizes to test
     */
    void set_data_sizes(const std::vector<size_t>& sizes) { data_sizes_ = sizes; }
    
    /**
     * Set polynomial degrees to test
     */
    void set_poly_degrees(const std::vector<size_t>& degrees) { poly_degrees_ = degrees; }
    
    // ========== Individual Benchmarks ==========
    
    /**
     * Benchmark NTT on all hardware paths
     * Requirements: 21.1, 21.3
     */
    BenchmarkSuite benchmark_ntt(size_t degree, uint64_t modulus);
    
    /**
     * Benchmark modular multiplication on all hardware paths
     * Requirements: 21.1, 21.4
     */
    BenchmarkSuite benchmark_modmul(size_t n, uint64_t modulus);
    
    /**
     * Benchmark polynomial multiplication on all hardware paths
     * Requirements: 21.1, 21.3
     */
    BenchmarkSuite benchmark_poly_mul(size_t degree, uint64_t modulus);
    
    /**
     * Benchmark Neural Engine operations
     * Requirements: 21.4, 21.5
     */
    BenchmarkSuite benchmark_neural_engine(size_t batch_size, uint64_t modulus);
    
    /**
     * Benchmark ray tracing operations
     * Requirements: 21.6
     */
    BenchmarkSuite benchmark_ray_tracing(uint32_t base, uint32_t levels, size_t count);
    
    /**
     * Benchmark texture sampling operations
     * Requirements: 21.6
     */
    BenchmarkSuite benchmark_texture_sampling(size_t degree, size_t num_points);
    
    /**
     * Benchmark memory system
     * Requirements: 21.9, 21.2
     */
    BenchmarkSuite benchmark_memory(size_t size);
    
    /**
     * Benchmark pipelined operations
     * Requirements: 21.10, 21.11
     */
    BenchmarkSuite benchmark_pipeline(size_t degree, uint64_t modulus);
    
    // ========== Full Benchmark Suite ==========
    
    /**
     * Run all benchmarks
     */
    void run_full_suite();
    
    /**
     * Get report generator with all results
     */
    BenchmarkReportGenerator& get_report() { return report_; }
    
    /**
     * Export all results
     */
    void export_results(const std::string& directory);
    
private:
    int iterations_;
    int warmup_;
    std::vector<size_t> data_sizes_;
    std::vector<size_t> poly_degrees_;
    std::map<HardwareBackend, bool> backend_enabled_;
    BenchmarkReportGenerator report_;
    BenchmarkTimer timer_;
    
    // Hardware detection
    bool has_sme_;
    bool has_sme2_;
    bool has_amx_;
    bool has_metal_;
    bool has_neural_engine_;
    bool has_ray_tracing_;
    
    void detect_hardware();
    
    // Verification helpers
    bool verify_modmul(const uint64_t* a, const uint64_t* b, 
                       const uint64_t* result, size_t n, uint64_t modulus);
    bool verify_ntt(const uint64_t* original, const uint64_t* transformed,
                    size_t n, uint64_t modulus);
};

// ============================================================================
// Template Implementations
// ============================================================================

template<typename Func>
std::vector<double> BenchmarkTimer::time_operation(Func&& func, int iterations, int warmup) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        func();
    }
    
    // Actual timing
    std::vector<double> timings;
    timings.reserve(iterations);
    
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        timings.push_back(us);
    }
    
    return timings;
}

template<typename Func>
double BenchmarkTimer::time_single(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

template<typename Func>
BenchmarkResult BenchmarkTimer::benchmark(
    const std::string& operation,
    const std::string& method_name,
    HardwareBackend backend,
    size_t data_size,
    size_t element_size,
    Func&& func,
    int iterations,
    int warmup
) {
    BenchmarkResult result;
    result.operation = operation;
    result.method_name = method_name;
    result.backend = backend;
    result.data_size = data_size;
    result.element_size_bytes = element_size;
    result.num_iterations = iterations;
    result.num_warmup_iterations = warmup;
    
    // Time the operation
    auto timings = time_operation(std::forward<Func>(func), iterations, warmup);
    
    // Compute statistics
    auto stats = StatisticalAnalyzer::analyze(timings);
    
    result.latency_us = stats.mean;
    result.latency_min_us = stats.min;
    result.latency_max_us = stats.max;
    result.latency_stddev_us = stats.stddev;
    
    // Compute throughput
    result.throughput_ops_sec = 1e6 / stats.mean;  // ops/sec
    result.throughput_elements_sec = data_size * 1e6 / stats.mean;
    
    // Compute bandwidth
    size_t total_bytes = data_size * element_size;
    result.bandwidth_gbps = (total_bytes / 1e9) / (stats.mean / 1e6);
    
    // Compute speedup
    if (baseline_time_us_ > 0) {
        result.speedup_vs_baseline = baseline_time_us_ / stats.mean;
    } else {
        result.speedup_vs_baseline = 1.0;
    }
    
    return result;
}

} // namespace comprehensive_benchmark
} // namespace fhe_accelerate
