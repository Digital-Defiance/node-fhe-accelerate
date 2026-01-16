/**
 * @file key_serializer.h
 * @brief FHE Key Serialization and Deserialization
 * 
 * This file defines the serialization infrastructure for FHE keys,
 * including binary format, integrity checksums, streaming support,
 * and versioning for protocol upgrades.
 * 
 * Requirements: 4.5, 4.6, 16
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <functional>
#include "key_manager.h"

namespace fhe_accelerate {

/**
 * Serialization format version
 */
constexpr uint32_t SERIALIZATION_VERSION = 1;

/**
 * Magic bytes for file format identification
 */
constexpr uint32_t MAGIC_SECRET_KEY = 0x46484553;    // "FHES"
constexpr uint32_t MAGIC_PUBLIC_KEY = 0x46484550;    // "FHEP"
constexpr uint32_t MAGIC_EVAL_KEY = 0x46484545;      // "FHEE"
constexpr uint32_t MAGIC_BOOTSTRAP_KEY = 0x46484542; // "FHEB"
constexpr uint32_t MAGIC_BALLOT = 0x46484556;        // "FHEV" (vote)

/**
 * Checksum algorithm types
 */
enum class ChecksumType : uint8_t {
    NONE = 0,
    CRC32 = 1,
    SHA256 = 2
};

/**
 * Compression types
 */
enum class CompressionType : uint8_t {
    NONE = 0,
    ZLIB = 1,
    LZ4 = 2
};

/**
 * Serialization header structure
 */
struct SerializationHeader {
    uint32_t magic;             // Magic bytes for format identification
    uint32_t version;           // Serialization format version
    uint32_t key_type;          // Type of key (0=secret, 1=public, 2=eval, 3=bootstrap)
    uint64_t key_id;            // Key identifier
    uint32_t poly_degree;       // Polynomial degree
    uint64_t modulus;           // Primary modulus
    uint32_t data_size;         // Size of serialized data (excluding header)
    ChecksumType checksum_type; // Checksum algorithm used
    CompressionType compression;// Compression algorithm used
    uint8_t reserved[7];        // Reserved for future use
    uint32_t checksum;          // Checksum of data
    
    SerializationHeader()
        : magic(0), version(SERIALIZATION_VERSION), key_type(0), key_id(0)
        , poly_degree(0), modulus(0), data_size(0)
        , checksum_type(ChecksumType::CRC32)
        , compression(CompressionType::NONE)
        , checksum(0)
    {
        std::memset(reserved, 0, sizeof(reserved));
    }
};

/**
 * Serialization result
 */
struct SerializationResult {
    bool success;
    std::string error_message;
    size_t bytes_written;
    
    SerializationResult() : success(false), bytes_written(0) {}
    
    static SerializationResult ok(size_t bytes) {
        SerializationResult r;
        r.success = true;
        r.bytes_written = bytes;
        return r;
    }
    
    static SerializationResult error(const std::string& msg) {
        SerializationResult r;
        r.success = false;
        r.error_message = msg;
        return r;
    }
};

/**
 * Deserialization result template
 */
template<typename T>
struct DeserializationResult {
    bool success;
    std::string error_message;
    std::unique_ptr<T> value;
    size_t bytes_read;
    
    DeserializationResult() : success(false), bytes_read(0) {}
    
    static DeserializationResult<T> ok(std::unique_ptr<T> val, size_t bytes) {
        DeserializationResult<T> r;
        r.success = true;
        r.value = std::move(val);
        r.bytes_read = bytes;
        return r;
    }
    
    static DeserializationResult<T> error(const std::string& msg) {
        DeserializationResult<T> r;
        r.success = false;
        r.error_message = msg;
        return r;
    }
};

/**
 * Progress callback for streaming operations
 */
using SerializationProgressCallback = std::function<void(size_t bytes_processed, size_t total_bytes)>;

/**
 * Key Serializer
 * 
 * Handles serialization and deserialization of FHE keys with:
 * - Binary format for efficiency
 * - Integrity checksums (CRC32, SHA256)
 * - Streaming support for large keys
 * - Versioning for protocol upgrades
 * - Partial deserialization for verification
 */
class KeySerializer {
public:
    KeySerializer();
    ~KeySerializer();
    
    // ========================================================================
    // Configuration
    // ========================================================================
    
    /**
     * Set checksum algorithm
     */
    void set_checksum_type(ChecksumType type) { checksum_type_ = type; }
    
    /**
     * Set compression algorithm
     */
    void set_compression_type(CompressionType type) { compression_type_ = type; }
    
    /**
     * Set progress callback for streaming operations
     */
    void set_progress_callback(SerializationProgressCallback callback) {
        progress_callback_ = std::move(callback);
    }
    
    // ========================================================================
    // Secret Key Serialization
    // ========================================================================
    
    /**
     * Serialize secret key to byte vector
     */
    SerializationResult serialize_secret_key(
        const SecretKey& sk,
        std::vector<uint8_t>& output
    );
    
    /**
     * Serialize secret key to stream
     */
    SerializationResult serialize_secret_key(
        const SecretKey& sk,
        std::ostream& output
    );
    
    /**
     * Deserialize secret key from byte vector
     */
    DeserializationResult<SecretKey> deserialize_secret_key(
        const std::vector<uint8_t>& input,
        uint64_t modulus
    );
    
    /**
     * Deserialize secret key from stream
     */
    DeserializationResult<SecretKey> deserialize_secret_key(
        std::istream& input,
        uint64_t modulus
    );
    
    // ========================================================================
    // Public Key Serialization
    // ========================================================================
    
    /**
     * Serialize public key to byte vector
     */
    SerializationResult serialize_public_key(
        const PublicKey& pk,
        std::vector<uint8_t>& output
    );
    
    /**
     * Serialize public key to stream
     */
    SerializationResult serialize_public_key(
        const PublicKey& pk,
        std::ostream& output
    );
    
    /**
     * Deserialize public key from byte vector
     */
    DeserializationResult<PublicKey> deserialize_public_key(
        const std::vector<uint8_t>& input
    );
    
    /**
     * Deserialize public key from stream
     */
    DeserializationResult<PublicKey> deserialize_public_key(
        std::istream& input
    );
    
    // ========================================================================
    // Evaluation Key Serialization
    // ========================================================================
    
    /**
     * Serialize evaluation key to byte vector
     */
    SerializationResult serialize_eval_key(
        const EvaluationKey& ek,
        std::vector<uint8_t>& output
    );
    
    /**
     * Serialize evaluation key to stream (streaming for large keys)
     */
    SerializationResult serialize_eval_key(
        const EvaluationKey& ek,
        std::ostream& output
    );
    
    /**
     * Deserialize evaluation key from byte vector
     */
    DeserializationResult<EvaluationKey> deserialize_eval_key(
        const std::vector<uint8_t>& input
    );
    
    /**
     * Deserialize evaluation key from stream
     */
    DeserializationResult<EvaluationKey> deserialize_eval_key(
        std::istream& input
    );
    
    // ========================================================================
    // Bootstrapping Key Serialization
    // ========================================================================
    
    /**
     * Serialize bootstrapping key to byte vector
     */
    SerializationResult serialize_bootstrap_key(
        const BootstrapKey& bk,
        std::vector<uint8_t>& output
    );
    
    /**
     * Serialize bootstrapping key to stream (streaming for large keys)
     */
    SerializationResult serialize_bootstrap_key(
        const BootstrapKey& bk,
        std::ostream& output
    );
    
    /**
     * Deserialize bootstrapping key from byte vector
     */
    DeserializationResult<BootstrapKey> deserialize_bootstrap_key(
        const std::vector<uint8_t>& input
    );
    
    /**
     * Deserialize bootstrapping key from stream
     */
    DeserializationResult<BootstrapKey> deserialize_bootstrap_key(
        std::istream& input
    );
    
    // ========================================================================
    // Partial Deserialization (for verification)
    // ========================================================================
    
    /**
     * Read only the header from serialized data
     * Useful for quick verification without full deserialization
     */
    DeserializationResult<SerializationHeader> read_header(
        const std::vector<uint8_t>& input
    );
    
    /**
     * Read only the header from stream
     */
    DeserializationResult<SerializationHeader> read_header(
        std::istream& input
    );
    
    /**
     * Verify integrity of serialized data without full deserialization
     */
    bool verify_integrity(const std::vector<uint8_t>& input);
    
    /**
     * Verify integrity from stream
     */
    bool verify_integrity(std::istream& input);
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /**
     * Compute CRC32 checksum
     */
    static uint32_t compute_crc32(const uint8_t* data, size_t length);
    
    /**
     * Compute SHA256 hash (returns first 4 bytes as uint32_t)
     */
    static uint32_t compute_sha256_truncated(const uint8_t* data, size_t length);
    
    /**
     * Get estimated serialized size for a key
     */
    static size_t estimate_size(const SecretKey& sk);
    static size_t estimate_size(const PublicKey& pk);
    static size_t estimate_size(const EvaluationKey& ek);
    static size_t estimate_size(const BootstrapKey& bk);
    
private:
    ChecksumType checksum_type_;
    CompressionType compression_type_;
    SerializationProgressCallback progress_callback_;
    
    // Helper functions
    void write_header(std::ostream& out, const SerializationHeader& header);
    bool read_header_impl(std::istream& in, SerializationHeader& header);
    
    void write_polynomial(std::ostream& out, const Polynomial& poly);
    Polynomial read_polynomial(std::istream& in, uint32_t degree, uint64_t modulus);
    
    uint32_t compute_checksum(const uint8_t* data, size_t length);
    bool verify_checksum(const uint8_t* data, size_t length, uint32_t expected, ChecksumType type);
    
    void report_progress(size_t bytes_processed, size_t total_bytes);
};

/**
 * Ballot Serializer
 * 
 * Specialized serializer for encrypted ballots with:
 * - Compact format (<10KB per ballot)
 * - Integrity verification
 * - Versioning support
 */
class BallotSerializer {
public:
    BallotSerializer();
    ~BallotSerializer();
    
    /**
     * Serialize encrypted ballot to compact format
     * Target: <10KB per ballot
     */
    SerializationResult serialize_ballot(
        const std::vector<std::pair<Polynomial, Polynomial>>& encrypted_choices,
        uint64_t timestamp,
        std::vector<uint8_t>& output
    );
    
    /**
     * Deserialize encrypted ballot
     */
    struct BallotData {
        std::vector<std::pair<Polynomial, Polynomial>> encrypted_choices;
        uint64_t timestamp;
        uint32_t version;
    };
    
    DeserializationResult<BallotData> deserialize_ballot(
        const std::vector<uint8_t>& input,
        uint32_t poly_degree,
        uint64_t modulus
    );
    
    /**
     * Get estimated ballot size
     */
    static size_t estimate_ballot_size(uint32_t num_choices, uint32_t poly_degree);
    
private:
    ChecksumType checksum_type_;
};

// Factory functions
std::unique_ptr<KeySerializer> create_key_serializer();
std::unique_ptr<BallotSerializer> create_ballot_serializer();

} // namespace fhe_accelerate
