/**
 * @file fraud_detector.h
 * @brief Fraud Detection on Encrypted Voting Data
 * 
 * This file defines the fraud detection infrastructure for encrypted voting,
 * including duplicate detection, statistical anomaly detection, threshold alerts,
 * and time-series analysis - all operating on encrypted data without revealing
 * individual votes.
 * 
 * Requirements: 15.3, 15.7
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <string>
#include <chrono>
#include "encryption.h"
#include "bootstrap_engine.h"
#include "parameter_set.h"

namespace fhe_accelerate {

// Forward declarations
class HardwareDispatcher;

/**
 * Fraud alert types
 */
enum class FraudAlertType {
    DUPLICATE_VOTE,         // Potential duplicate ballot detected
    STATISTICAL_ANOMALY,    // Voting pattern deviates from expected
    TIMING_ANOMALY,         // Suspicious timing pattern
    THRESHOLD_EXCEEDED,     // Vote count exceeds threshold
    PATTERN_ANOMALY         // Unusual voting pattern detected
};

/**
 * Fraud alert structure
 * 
 * Contains information about a detected anomaly while preserving
 * privacy of individual votes.
 */
struct FraudAlert {
    FraudAlertType type;
    std::string description;
    uint64_t timestamp;
    double confidence;              // 0.0 to 1.0
    Ciphertext encrypted_evidence;  // Evidence remains encrypted
    std::optional<size_t> ballot_index;  // Index of suspicious ballot (if applicable)
    
    FraudAlert(FraudAlertType t, const std::string& desc, double conf)
        : type(t), description(desc), 
          timestamp(std::chrono::system_clock::now().time_since_epoch().count()),
          confidence(conf), encrypted_evidence(Polynomial(1, 1), Polynomial(1, 1), 0, 0) {}
    
    FraudAlert(FraudAlertType t, const std::string& desc, double conf, Ciphertext&& evidence)
        : type(t), description(desc),
          timestamp(std::chrono::system_clock::now().time_since_epoch().count()),
          confidence(conf), encrypted_evidence(std::move(evidence)) {}
};

/**
 * Statistical model for expected voting patterns
 */
struct StatisticalModel {
    std::vector<double> expected_distribution;  // Expected vote distribution per candidate
    double variance_threshold;                   // Threshold for anomaly detection
    size_t min_sample_size;                      // Minimum ballots before detection
    
    StatisticalModel() : variance_threshold(2.0), min_sample_size(100) {}
    
    StatisticalModel(std::vector<double>&& dist, double var_thresh, size_t min_samples)
        : expected_distribution(std::move(dist)), 
          variance_threshold(var_thresh),
          min_sample_size(min_samples) {}
};

/**
 * Threshold configuration for alerts
 */
struct Threshold {
    std::string name;
    uint64_t value;
    bool alert_on_exceed;   // Alert when exceeded
    bool alert_on_below;    // Alert when below
    
    Threshold(const std::string& n, uint64_t v, bool exceed = true, bool below = false)
        : name(n), value(v), alert_on_exceed(exceed), alert_on_below(below) {}
};

/**
 * Timestamped ballot for time-series analysis
 */
struct TimestampedBallot {
    Ciphertext ballot;
    uint64_t timestamp;
    std::optional<std::string> region;  // Optional geographic region
    
    TimestampedBallot(Ciphertext&& b, uint64_t ts)
        : ballot(std::move(b)), timestamp(ts) {}
    
    TimestampedBallot(Ciphertext&& b, uint64_t ts, const std::string& r)
        : ballot(std::move(b)), timestamp(ts), region(r) {}
};


/**
 * Fraud detection result
 */
struct FraudDetectionResult {
    std::vector<FraudAlert> alerts;
    size_t ballots_analyzed;
    double analysis_time_ms;
    bool privacy_preserved;  // Confirms no individual votes were revealed
    
    FraudDetectionResult() 
        : ballots_analyzed(0), analysis_time_ms(0), privacy_preserved(true) {}
    
    bool has_alerts() const { return !alerts.empty(); }
    size_t alert_count() const { return alerts.size(); }
};

/**
 * Progress callback for fraud detection operations
 */
using FraudDetectionProgressCallback = std::function<void(size_t completed, size_t total, const std::string& stage)>;

/**
 * Fraud Detector
 * 
 * Implements fraud detection on encrypted voting data without revealing
 * individual votes. Uses homomorphic operations and programmable bootstrapping
 * for encrypted comparisons.
 * 
 * Requirements: 15.3, 15.7
 */
class FraudDetector {
public:
    /**
     * Construct fraud detector with given parameters
     * 
     * @param params FHE parameter set
     * @param encryption_engine Encryption engine for homomorphic operations
     * @param bootstrap_engine Bootstrap engine for PBS (optional)
     * @param dispatcher Hardware dispatcher for acceleration (optional)
     */
    FraudDetector(
        const ParameterSet& params,
        EncryptionEngine* encryption_engine,
        BootstrapEngine* bootstrap_engine = nullptr,
        HardwareDispatcher* dispatcher = nullptr
    );
    ~FraudDetector();
    
    // ========================================================================
    // Duplicate Detection (Requirement 15.3)
    // ========================================================================
    
    /**
     * Detect duplicate voting patterns in encrypted ballots
     * 
     * Uses encrypted comparison to identify potential duplicates without
     * revealing individual vote contents.
     * 
     * @param ballots Vector of encrypted ballots to analyze
     * @param progress Optional progress callback
     * @return Fraud detection result with any duplicate alerts
     */
    FraudDetectionResult detect_duplicates(
        const std::vector<Ciphertext>& ballots,
        FraudDetectionProgressCallback progress = nullptr
    );
    
    /**
     * Check if a new ballot is a duplicate of any existing ballot
     * 
     * @param new_ballot New ballot to check
     * @param existing_ballots List of existing ballots
     * @param bk Bootstrapping key for encrypted comparison (optional)
     * @return Ciphertext encrypting 1 if duplicate found, 0 otherwise
     */
    Ciphertext check_duplicate(
        const Ciphertext& new_ballot,
        const std::vector<Ciphertext>& existing_ballots,
        const ExtendedBootstrapKey* bk = nullptr
    );
    
    // ========================================================================
    // Statistical Anomaly Detection (Requirement 15.7)
    // ========================================================================
    
    /**
     * Detect statistical anomalies in encrypted tally
     * 
     * Compares the encrypted tally against expected distribution to
     * identify suspicious patterns.
     * 
     * @param tally Encrypted tally (sum of votes per candidate)
     * @param expected Expected statistical model
     * @param progress Optional progress callback
     * @return Fraud detection result with any anomaly alerts
     */
    FraudDetectionResult detect_anomalies(
        const Ciphertext& tally,
        const StatisticalModel& expected,
        FraudDetectionProgressCallback progress = nullptr
    );
    
    /**
     * Compute anomaly score for a ballot
     * 
     * Returns an encrypted score indicating how anomalous the ballot is
     * compared to expected patterns.
     * 
     * @param ballot Encrypted ballot
     * @param expected_distribution Expected vote distribution (encrypted)
     * @param ek Evaluation key for multiplication
     * @return Ciphertext encrypting the anomaly score
     */
    Ciphertext compute_anomaly_score(
        const Ciphertext& ballot,
        const Ciphertext& expected_distribution,
        const EvaluationKey& ek
    );
    
    // ========================================================================
    // Threshold Alerts (Requirement 15.7)
    // ========================================================================
    
    /**
     * Check thresholds on encrypted tally without decryption
     * 
     * @param tally Encrypted tally
     * @param thresholds Vector of thresholds to check
     * @param bk Bootstrapping key for encrypted comparison (optional)
     * @return Fraud detection result with any threshold alerts
     */
    FraudDetectionResult check_thresholds(
        const Ciphertext& tally,
        const std::vector<Threshold>& thresholds,
        const ExtendedBootstrapKey* bk = nullptr
    );
    
    /**
     * Check if encrypted value exceeds threshold
     * 
     * Uses programmable bootstrapping for encrypted comparison.
     * 
     * @param value Encrypted value to check
     * @param threshold Plaintext threshold
     * @param bk Bootstrapping key
     * @return Ciphertext encrypting 1 if value >= threshold, 0 otherwise
     */
    Ciphertext check_threshold(
        const Ciphertext& value,
        uint64_t threshold,
        const ExtendedBootstrapKey* bk = nullptr
    );
    
    // ========================================================================
    // Time-Series Analysis (Requirement 15.7)
    // ========================================================================
    
    /**
     * Analyze voting patterns over time
     * 
     * Detects suspicious timing patterns such as:
     * - Sudden spikes in voting rate
     * - Unusual voting times
     * - Coordinated voting patterns
     * 
     * @param ballots Timestamped encrypted ballots
     * @param progress Optional progress callback
     * @return Fraud detection result with any timing alerts
     */
    FraudDetectionResult analyze_voting_patterns(
        const std::vector<TimestampedBallot>& ballots,
        FraudDetectionProgressCallback progress = nullptr
    );
    
    /**
     * Detect voting rate anomalies
     * 
     * @param ballots Timestamped ballots
     * @param window_size_ms Time window for rate calculation
     * @param rate_threshold Maximum expected votes per window
     * @return Fraud detection result with rate anomaly alerts
     */
    FraudDetectionResult detect_rate_anomalies(
        const std::vector<TimestampedBallot>& ballots,
        uint64_t window_size_ms,
        size_t rate_threshold
    );
    
    // ========================================================================
    // Encrypted Comparison Operations (Requirement 15.3)
    // ========================================================================
    
    /**
     * Encrypted greater-than comparison
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param bk Bootstrapping key (optional)
     * @return Ciphertext encrypting 1 if ct1 > ct2, 0 otherwise
     */
    Ciphertext compare_greater_than(
        const Ciphertext& ct1,
        const Ciphertext& ct2,
        const ExtendedBootstrapKey* bk = nullptr
    );
    
    /**
     * Encrypted equality test
     * 
     * @param ct1 First ciphertext
     * @param ct2 Second ciphertext
     * @param bk Bootstrapping key (optional)
     * @return Ciphertext encrypting 1 if ct1 == ct2, 0 otherwise
     */
    Ciphertext compare_equal(
        const Ciphertext& ct1,
        const Ciphertext& ct2,
        const ExtendedBootstrapKey* bk = nullptr
    );
    
    /**
     * Encrypted range check
     * 
     * @param ct Ciphertext to check
     * @param min_value Minimum value (plaintext)
     * @param max_value Maximum value (plaintext)
     * @param bk Bootstrapping key (optional)
     * @return Ciphertext encrypting 1 if min <= ct <= max, 0 otherwise
     */
    Ciphertext check_range(
        const Ciphertext& ct,
        uint64_t min_value,
        uint64_t max_value,
        const ExtendedBootstrapKey* bk = nullptr
    );
    
    // ========================================================================
    // Configuration
    // ========================================================================
    
    /**
     * Set the statistical model for anomaly detection
     */
    void set_statistical_model(const StatisticalModel& model);
    
    /**
     * Set sensitivity for duplicate detection
     * Higher values = more sensitive (more false positives)
     */
    void set_duplicate_sensitivity(double sensitivity);
    
    /**
     * Enable/disable real-time detection mode
     */
    void set_realtime_mode(bool enabled);
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const ParameterSet& get_params() const { return params_; }
    bool is_realtime_mode() const { return realtime_mode_; }
    
private:
    ParameterSet params_;
    EncryptionEngine* encryption_engine_;
    BootstrapEngine* bootstrap_engine_;
    HardwareDispatcher* dispatcher_;
    
    StatisticalModel statistical_model_;
    double duplicate_sensitivity_;
    bool realtime_mode_;
    
    // Helper functions
    double compute_timing_score(const std::vector<uint64_t>& timestamps);
    bool is_timing_anomaly(uint64_t timestamp, const std::vector<uint64_t>& recent_timestamps);
    
    // Encrypted operations helpers
    Ciphertext encrypted_abs_diff(const Ciphertext& ct1, const Ciphertext& ct2);
    Ciphertext encrypted_sum(const std::vector<Ciphertext>& ciphertexts);
};

// Factory function
std::unique_ptr<FraudDetector> create_fraud_detector(
    const ParameterSet& params,
    EncryptionEngine* encryption_engine,
    BootstrapEngine* bootstrap_engine = nullptr,
    HardwareDispatcher* dispatcher = nullptr
);

} // namespace fhe_accelerate
