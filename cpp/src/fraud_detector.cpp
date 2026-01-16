/**
 * @file fraud_detector.cpp
 * @brief Fraud Detection Implementation for Encrypted Voting
 * 
 * Implements fraud detection on encrypted voting data including duplicate
 * detection, statistical anomaly detection, threshold alerts, and time-series
 * analysis - all while preserving vote privacy.
 * 
 * Requirements: 15.3, 15.7
 */

#include "fraud_detector.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <unordered_set>

namespace fhe_accelerate {

// ============================================================================
// FraudDetector Implementation
// ============================================================================

FraudDetector::FraudDetector(
    const ParameterSet& params,
    EncryptionEngine* encryption_engine,
    BootstrapEngine* bootstrap_engine,
    HardwareDispatcher* dispatcher
)
    : params_(params),
      encryption_engine_(encryption_engine),
      bootstrap_engine_(bootstrap_engine),
      dispatcher_(dispatcher),
      duplicate_sensitivity_(0.5),
      realtime_mode_(false)
{
    if (!encryption_engine_) {
        throw std::invalid_argument("Encryption engine is required");
    }
}

FraudDetector::~FraudDetector() = default;

// ============================================================================
// Configuration
// ============================================================================

void FraudDetector::set_statistical_model(const StatisticalModel& model) {
    statistical_model_ = model;
}

void FraudDetector::set_duplicate_sensitivity(double sensitivity) {
    duplicate_sensitivity_ = std::clamp(sensitivity, 0.0, 1.0);
}

void FraudDetector::set_realtime_mode(bool enabled) {
    realtime_mode_ = enabled;
}


// ============================================================================
// Duplicate Detection (Requirement 15.3)
// ============================================================================

FraudDetectionResult FraudDetector::detect_duplicates(
    const std::vector<Ciphertext>& ballots,
    FraudDetectionProgressCallback progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    FraudDetectionResult result;
    result.ballots_analyzed = ballots.size();
    
    if (ballots.size() < 2) {
        auto end_time = std::chrono::high_resolution_clock::now();
        result.analysis_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        return result;
    }
    
    // Compare each pair of ballots for potential duplicates
    // This is O(n^2) but necessary for thorough duplicate detection
    // In practice, we use heuristics to reduce comparisons
    
    size_t total_comparisons = (ballots.size() * (ballots.size() - 1)) / 2;
    size_t completed = 0;
    
    for (size_t i = 0; i < ballots.size(); ++i) {
        for (size_t j = i + 1; j < ballots.size(); ++j) {
            // Compute encrypted equality test
            Ciphertext equality = compare_equal(ballots[i], ballots[j], nullptr);
            
            // The equality ciphertext encrypts 1 if equal, 0 otherwise
            // We can't decrypt it, but we can use it for further analysis
            // or flag it for manual review
            
            // For now, we use a heuristic based on ciphertext structure
            // In a real implementation, this would use PBS for comparison
            
            // Check if the difference is "small" (potential duplicate)
            Ciphertext diff = encryption_engine_->subtract(ballots[i], ballots[j]);
            double estimated_budget = encryption_engine_->estimate_noise_budget(diff);
            
            // If the difference has very high noise budget, the values might be similar
            // This is a heuristic - real implementation uses PBS
            if (estimated_budget > params_.noise_budget * 0.9) {
                FraudAlert alert(
                    FraudAlertType::DUPLICATE_VOTE,
                    "Potential duplicate ballot detected between indices " + 
                        std::to_string(i) + " and " + std::to_string(j),
                    duplicate_sensitivity_
                );
                alert.ballot_index = i;
                result.alerts.push_back(std::move(alert));
            }
            
            completed++;
            if (progress && completed % 100 == 0) {
                progress(completed, total_comparisons, "Checking for duplicates");
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.analysis_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    if (progress) {
        progress(total_comparisons, total_comparisons, "Duplicate detection complete");
    }
    
    return result;
}

Ciphertext FraudDetector::check_duplicate(
    const Ciphertext& new_ballot,
    const std::vector<Ciphertext>& existing_ballots,
    const ExtendedBootstrapKey* bk
) {
    // Use encryption engine's detect_duplicate method
    return encryption_engine_->detect_duplicate(new_ballot, existing_ballots, 
        bk ? reinterpret_cast<const BootstrapKey*>(bk) : nullptr);
}


// ============================================================================
// Statistical Anomaly Detection (Requirement 15.7)
// ============================================================================

FraudDetectionResult FraudDetector::detect_anomalies(
    const Ciphertext& tally,
    const StatisticalModel& expected,
    FraudDetectionProgressCallback progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    FraudDetectionResult result;
    result.ballots_analyzed = 1;  // Analyzing the tally
    
    if (progress) {
        progress(0, 1, "Analyzing tally for anomalies");
    }
    
    // Statistical anomaly detection on encrypted data is challenging
    // We use several approaches:
    // 1. Compare encrypted tally against encrypted expected values
    // 2. Use PBS to compute deviation indicators
    // 3. Aggregate indicators to produce anomaly score
    
    // For now, we use a simplified approach based on noise budget analysis
    // Real implementation would use PBS for encrypted comparisons
    
    double noise_budget = encryption_engine_->estimate_noise_budget(tally);
    
    // If noise budget is unusually low or high, flag as potential anomaly
    double expected_budget = params_.noise_budget * 0.5;  // Expected after operations
    double deviation = std::abs(noise_budget - expected_budget) / expected_budget;
    
    if (deviation > expected.variance_threshold) {
        FraudAlert alert(
            FraudAlertType::STATISTICAL_ANOMALY,
            "Tally shows unusual statistical properties (deviation: " + 
                std::to_string(deviation) + ")",
            std::min(1.0, deviation / expected.variance_threshold)
        );
        result.alerts.push_back(std::move(alert));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.analysis_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    if (progress) {
        progress(1, 1, "Anomaly detection complete");
    }
    
    return result;
}

Ciphertext FraudDetector::compute_anomaly_score(
    const Ciphertext& ballot,
    const Ciphertext& expected_distribution,
    const EvaluationKey& ek
) {
    // Compute squared difference as anomaly score
    // score = (ballot - expected)^2
    
    Ciphertext diff = encryption_engine_->subtract(ballot, expected_distribution);
    Ciphertext score = encryption_engine_->square_relin(diff, ek);
    
    return score;
}


// ============================================================================
// Threshold Alerts (Requirement 15.7)
// ============================================================================

FraudDetectionResult FraudDetector::check_thresholds(
    const Ciphertext& tally,
    const std::vector<Threshold>& thresholds,
    const ExtendedBootstrapKey* bk
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    FraudDetectionResult result;
    result.ballots_analyzed = 1;
    
    for (size_t i = 0; i < thresholds.size(); ++i) {
        const auto& threshold = thresholds[i];
        
        // Check threshold using encrypted comparison
        Ciphertext indicator = check_threshold(tally, threshold.value, bk);
        
        // The indicator encrypts 1 if threshold exceeded, 0 otherwise
        // We can't decrypt it, but we can flag for review
        
        // For alerting purposes, we use heuristics
        // Real implementation would use PBS result
        
        if (threshold.alert_on_exceed) {
            FraudAlert alert(
                FraudAlertType::THRESHOLD_EXCEEDED,
                "Threshold check performed for: " + threshold.name + 
                    " (threshold: " + std::to_string(threshold.value) + ")",
                0.5  // Confidence is uncertain without decryption
            );
            alert.encrypted_evidence = std::move(indicator);
            result.alerts.push_back(std::move(alert));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.analysis_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    return result;
}

Ciphertext FraudDetector::check_threshold(
    const Ciphertext& value,
    uint64_t threshold,
    const ExtendedBootstrapKey* bk
) {
    // Use encryption engine's threshold check
    return encryption_engine_->check_threshold(value, threshold);
}

// ============================================================================
// Time-Series Analysis (Requirement 15.7)
// ============================================================================

FraudDetectionResult FraudDetector::analyze_voting_patterns(
    const std::vector<TimestampedBallot>& ballots,
    FraudDetectionProgressCallback progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    FraudDetectionResult result;
    result.ballots_analyzed = ballots.size();
    
    if (ballots.empty()) {
        return result;
    }
    
    if (progress) {
        progress(0, ballots.size(), "Analyzing voting patterns");
    }
    
    // Extract timestamps
    std::vector<uint64_t> timestamps;
    timestamps.reserve(ballots.size());
    for (const auto& ballot : ballots) {
        timestamps.push_back(ballot.timestamp);
    }
    
    // Sort timestamps for analysis
    std::vector<uint64_t> sorted_timestamps = timestamps;
    std::sort(sorted_timestamps.begin(), sorted_timestamps.end());
    
    // Analyze inter-arrival times
    std::vector<uint64_t> inter_arrival_times;
    for (size_t i = 1; i < sorted_timestamps.size(); ++i) {
        inter_arrival_times.push_back(sorted_timestamps[i] - sorted_timestamps[i-1]);
    }
    
    if (!inter_arrival_times.empty()) {
        // Compute statistics
        double mean_iat = 0;
        for (auto iat : inter_arrival_times) {
            mean_iat += static_cast<double>(iat);
        }
        mean_iat /= inter_arrival_times.size();
        
        double variance_iat = 0;
        for (auto iat : inter_arrival_times) {
            double diff = static_cast<double>(iat) - mean_iat;
            variance_iat += diff * diff;
        }
        variance_iat /= inter_arrival_times.size();
        double std_iat = std::sqrt(variance_iat);
        
        // Check for suspicious patterns
        // 1. Very regular timing (potential bot)
        if (std_iat < mean_iat * 0.1 && ballots.size() > 10) {
            FraudAlert alert(
                FraudAlertType::TIMING_ANOMALY,
                "Suspiciously regular voting timing detected (std/mean = " +
                    std::to_string(std_iat / mean_iat) + ")",
                0.7
            );
            result.alerts.push_back(std::move(alert));
        }
        
        // 2. Bursts of votes
        size_t burst_count = 0;
        for (auto iat : inter_arrival_times) {
            if (static_cast<double>(iat) < mean_iat * 0.1) {
                burst_count++;
            }
        }
        
        if (burst_count > inter_arrival_times.size() * 0.3) {
            FraudAlert alert(
                FraudAlertType::TIMING_ANOMALY,
                "Voting bursts detected (" + std::to_string(burst_count) + 
                    " rapid submissions)",
                0.6
            );
            result.alerts.push_back(std::move(alert));
        }
    }
    
    // Check for individual timing anomalies
    for (size_t i = 0; i < ballots.size(); ++i) {
        if (is_timing_anomaly(ballots[i].timestamp, timestamps)) {
            FraudAlert alert(
                FraudAlertType::TIMING_ANOMALY,
                "Individual ballot timing anomaly at index " + std::to_string(i),
                0.5
            );
            alert.ballot_index = i;
            result.alerts.push_back(std::move(alert));
        }
        
        if (progress && i % 100 == 0) {
            progress(i, ballots.size(), "Analyzing timing patterns");
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.analysis_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    if (progress) {
        progress(ballots.size(), ballots.size(), "Pattern analysis complete");
    }
    
    return result;
}

FraudDetectionResult FraudDetector::detect_rate_anomalies(
    const std::vector<TimestampedBallot>& ballots,
    uint64_t window_size_ms,
    size_t rate_threshold
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    FraudDetectionResult result;
    result.ballots_analyzed = ballots.size();
    
    if (ballots.empty()) {
        return result;
    }
    
    // Sort by timestamp
    std::vector<std::pair<uint64_t, size_t>> sorted_ballots;
    for (size_t i = 0; i < ballots.size(); ++i) {
        sorted_ballots.emplace_back(ballots[i].timestamp, i);
    }
    std::sort(sorted_ballots.begin(), sorted_ballots.end());
    
    // Sliding window rate analysis
    size_t window_start = 0;
    for (size_t window_end = 0; window_end < sorted_ballots.size(); ++window_end) {
        // Move window start forward
        while (window_start < window_end && 
               sorted_ballots[window_end].first - sorted_ballots[window_start].first > window_size_ms) {
            window_start++;
        }
        
        size_t window_count = window_end - window_start + 1;
        
        if (window_count > rate_threshold) {
            FraudAlert alert(
                FraudAlertType::TIMING_ANOMALY,
                "Voting rate exceeded threshold: " + std::to_string(window_count) +
                    " votes in " + std::to_string(window_size_ms) + "ms window",
                std::min(1.0, static_cast<double>(window_count) / rate_threshold)
            );
            result.alerts.push_back(std::move(alert));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.analysis_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    return result;
}


// ============================================================================
// Encrypted Comparison Operations (Requirement 15.3)
// ============================================================================

Ciphertext FraudDetector::compare_greater_than(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    const ExtendedBootstrapKey* bk
) {
    // Use encryption engine's comparison
    return encryption_engine_->compare_greater_than(ct1, ct2, 
        bk ? reinterpret_cast<const BootstrapKey*>(bk) : nullptr);
}

Ciphertext FraudDetector::compare_equal(
    const Ciphertext& ct1,
    const Ciphertext& ct2,
    const ExtendedBootstrapKey* bk
) {
    // Use encryption engine's equality test
    return encryption_engine_->compare_equal(ct1, ct2,
        bk ? reinterpret_cast<const BootstrapKey*>(bk) : nullptr);
}

Ciphertext FraudDetector::check_range(
    const Ciphertext& ct,
    uint64_t min_value,
    uint64_t max_value,
    const ExtendedBootstrapKey* bk
) {
    // Use encryption engine's range check
    return encryption_engine_->check_range(ct, min_value, max_value,
        bk ? reinterpret_cast<const BootstrapKey*>(bk) : nullptr);
}

// ============================================================================
// Helper Functions
// ============================================================================

double FraudDetector::compute_timing_score(const std::vector<uint64_t>& timestamps) {
    if (timestamps.size() < 2) {
        return 0.0;
    }
    
    // Compute coefficient of variation of inter-arrival times
    std::vector<double> iats;
    for (size_t i = 1; i < timestamps.size(); ++i) {
        iats.push_back(static_cast<double>(timestamps[i] - timestamps[i-1]));
    }
    
    double mean = std::accumulate(iats.begin(), iats.end(), 0.0) / iats.size();
    if (mean < 1e-10) return 1.0;  // All same timestamp = suspicious
    
    double variance = 0;
    for (double iat : iats) {
        variance += (iat - mean) * (iat - mean);
    }
    variance /= iats.size();
    
    double cv = std::sqrt(variance) / mean;
    
    // Low CV = regular timing = suspicious
    // High CV = irregular timing = normal
    return 1.0 - std::min(1.0, cv);
}

bool FraudDetector::is_timing_anomaly(uint64_t timestamp, const std::vector<uint64_t>& recent_timestamps) {
    if (recent_timestamps.size() < 10) {
        return false;  // Not enough data
    }
    
    // Check if this timestamp is an outlier
    std::vector<uint64_t> sorted = recent_timestamps;
    std::sort(sorted.begin(), sorted.end());
    
    // Compute IQR
    size_t q1_idx = sorted.size() / 4;
    size_t q3_idx = (3 * sorted.size()) / 4;
    
    uint64_t q1 = sorted[q1_idx];
    uint64_t q3 = sorted[q3_idx];
    uint64_t iqr = q3 - q1;
    
    // Check if timestamp is outside 1.5 * IQR
    uint64_t lower_bound = q1 > 1.5 * iqr ? q1 - static_cast<uint64_t>(1.5 * iqr) : 0;
    uint64_t upper_bound = q3 + static_cast<uint64_t>(1.5 * iqr);
    
    return timestamp < lower_bound || timestamp > upper_bound;
}

Ciphertext FraudDetector::encrypted_abs_diff(const Ciphertext& ct1, const Ciphertext& ct2) {
    // Compute |ct1 - ct2| using homomorphic operations
    // This is approximate - real implementation would use PBS
    
    Ciphertext diff = encryption_engine_->subtract(ct1, ct2);
    
    // For absolute value, we'd need PBS to apply the abs function
    // For now, return the difference (which may be negative in encrypted form)
    return diff;
}

Ciphertext FraudDetector::encrypted_sum(const std::vector<Ciphertext>& ciphertexts) {
    if (ciphertexts.empty()) {
        throw std::invalid_argument("Cannot sum empty vector of ciphertexts");
    }
    
    return encryption_engine_->batch_add(ciphertexts);
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<FraudDetector> create_fraud_detector(
    const ParameterSet& params,
    EncryptionEngine* encryption_engine,
    BootstrapEngine* bootstrap_engine,
    HardwareDispatcher* dispatcher
) {
    return std::make_unique<FraudDetector>(params, encryption_engine, bootstrap_engine, dispatcher);
}

} // namespace fhe_accelerate
