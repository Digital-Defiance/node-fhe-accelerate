/**
 * Zero-Knowledge Cryptographic Hash Functions
 * 
 * Implements hash functions optimized for ZK proof systems:
 * - Poseidon hash (ZK-friendly, algebraic)
 * - Blake2s (for Bulletproofs)
 * 
 * Uses Metal GPU for parallel hash tree construction.
 * Optimized for M4 Max cache hierarchy.
 * 
 * Requirements: 19, 20.4
 */

#pragma once

#include "zk_field_arithmetic.h"
#include <vector>
#include <array>
#include <memory>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Poseidon Hash (ZK-Friendly)
// ============================================================================

/**
 * Poseidon hash parameters
 * 
 * Poseidon is an algebraic hash function designed for ZK circuits.
 * It operates over a prime field and uses:
 * - S-box: x^alpha (typically alpha=5 for BN254/BLS12-381)
 * - MDS matrix for diffusion
 * - Round constants for security
 */
struct PoseidonParams {
    size_t t;                    // State width (rate + capacity)
    size_t rate;                 // Rate (number of input elements per permutation)
    size_t capacity;             // Capacity (security parameter)
    size_t full_rounds;          // Number of full rounds
    size_t partial_rounds;       // Number of partial rounds
    uint64_t alpha;              // S-box exponent (typically 5)
    
    // Precomputed constants
    std::vector<FieldElement256> round_constants;
    std::vector<std::vector<FieldElement256>> mds_matrix;
    
    PoseidonParams() : t(3), rate(2), capacity(1), 
                       full_rounds(8), partial_rounds(57), alpha(5) {}
};

/**
 * Poseidon hash function over BN254 scalar field
 */
class PoseidonHash {
public:
    /**
     * Construct Poseidon hasher with given parameters
     */
    explicit PoseidonHash(const PoseidonParams& params);
    
    /**
     * Construct Poseidon hasher with default parameters for BN254
     */
    PoseidonHash();
    
    /**
     * Hash a single field element
     */
    FieldElement256 hash(const FieldElement256& input) const;
    
    /**
     * Hash two field elements (common case for Merkle trees)
     */
    FieldElement256 hash2(const FieldElement256& left, 
                          const FieldElement256& right) const;
    
    /**
     * Hash multiple field elements
     */
    FieldElement256 hash_many(const FieldElement256* inputs, size_t count) const;
    
    /**
     * Hash bytes (converts to field elements first)
     */
    FieldElement256 hash_bytes(const uint8_t* data, size_t len) const;
    
    /**
     * Build Merkle tree using Poseidon hash
     * Returns root and stores tree layers
     */
    FieldElement256 build_merkle_tree(const std::vector<FieldElement256>& leaves,
                                       std::vector<std::vector<FieldElement256>>& tree) const;
    
    /**
     * GPU-accelerated Merkle tree construction
     */
    FieldElement256 build_merkle_tree_gpu(const std::vector<FieldElement256>& leaves,
                                           std::vector<std::vector<FieldElement256>>& tree) const;
    
    /**
     * Get Merkle authentication path
     */
    std::vector<FieldElement256> get_merkle_path(
        const std::vector<std::vector<FieldElement256>>& tree,
        size_t leaf_index) const;
    
    /**
     * Verify Merkle authentication path
     */
    bool verify_merkle_path(const FieldElement256& root,
                            const FieldElement256& leaf,
                            size_t leaf_index,
                            const std::vector<FieldElement256>& path) const;
    
    /**
     * Batch hash for parallel processing
     */
    void batch_hash2(const FieldElement256* lefts, const FieldElement256* rights,
                     FieldElement256* outputs, size_t count) const;
    
    /**
     * GPU-accelerated batch hash
     */
    void batch_hash2_gpu(const FieldElement256* lefts, const FieldElement256* rights,
                         FieldElement256* outputs, size_t count) const;
    
    /**
     * Get parameters
     */
    const PoseidonParams& params() const { return params_; }
    
private:
    PoseidonParams params_;
    const Field256& field_;
    
    // Poseidon permutation
    void permutation(std::vector<FieldElement256>& state) const;
    
    // S-box: x^alpha
    FieldElement256 sbox(const FieldElement256& x) const;
    
    // MDS matrix multiplication
    void mds_multiply(std::vector<FieldElement256>& state) const;
    
    // Add round constants
    void add_round_constants(std::vector<FieldElement256>& state, size_t round) const;
    
    // Full round (S-box on all elements)
    void full_round(std::vector<FieldElement256>& state, size_t round) const;
    
    // Partial round (S-box on first element only)
    void partial_round(std::vector<FieldElement256>& state, size_t round) const;
    
    // Generate round constants (deterministic from seed)
    void generate_round_constants();
    
    // Generate MDS matrix
    void generate_mds_matrix();
};

// ============================================================================
// Blake2s Hash (for Bulletproofs)
// ============================================================================

/**
 * Blake2s hash function
 * 
 * Blake2s is a fast cryptographic hash function optimized for 32-bit platforms.
 * Used in Bulletproofs for non-algebraic hashing.
 */
class Blake2sHash {
public:
    static constexpr size_t DIGEST_SIZE = 32;
    static constexpr size_t BLOCK_SIZE = 64;
    
    /**
     * Hash arbitrary data
     */
    static std::array<uint8_t, DIGEST_SIZE> hash(const uint8_t* data, size_t len);
    
    /**
     * Hash with personalization string
     */
    static std::array<uint8_t, DIGEST_SIZE> hash_with_personal(
        const uint8_t* data, size_t len,
        const uint8_t* personal, size_t personal_len);
    
    /**
     * Hash two 32-byte values (for Merkle trees)
     */
    static std::array<uint8_t, DIGEST_SIZE> hash2(
        const std::array<uint8_t, DIGEST_SIZE>& left,
        const std::array<uint8_t, DIGEST_SIZE>& right);
    
    /**
     * Convert hash output to field element
     */
    static FieldElement256 to_field_element(const std::array<uint8_t, DIGEST_SIZE>& hash);
    
    /**
     * Build Merkle tree using Blake2s
     */
    static std::array<uint8_t, DIGEST_SIZE> build_merkle_tree(
        const std::vector<std::array<uint8_t, DIGEST_SIZE>>& leaves,
        std::vector<std::vector<std::array<uint8_t, DIGEST_SIZE>>>& tree);
    
    /**
     * GPU-accelerated Merkle tree construction
     */
    static std::array<uint8_t, DIGEST_SIZE> build_merkle_tree_gpu(
        const std::vector<std::array<uint8_t, DIGEST_SIZE>>& leaves,
        std::vector<std::vector<std::array<uint8_t, DIGEST_SIZE>>>& tree);
    
    /**
     * Batch hash for parallel processing
     */
    static void batch_hash(const uint8_t* const* inputs, const size_t* lengths,
                           std::array<uint8_t, DIGEST_SIZE>* outputs, size_t count);
    
    /**
     * Incremental hashing interface
     */
    class Hasher {
    public:
        Hasher();
        void update(const uint8_t* data, size_t len);
        std::array<uint8_t, DIGEST_SIZE> finalize();
        void reset();
        
    private:
        std::array<uint32_t, 8> h_;      // State
        std::array<uint8_t, BLOCK_SIZE> buf_;  // Buffer
        size_t buf_len_;
        uint64_t total_len_;
        
        void compress(const uint8_t* block, bool is_last);
    };
    
private:
    // Blake2s constants
    static constexpr std::array<uint32_t, 8> IV = {
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    };
    
    static constexpr std::array<std::array<uint8_t, 16>, 10> SIGMA = {{
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
        {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
        {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
        {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
        {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
        {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
        {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
        {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
        {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
    }};
    
    static void G(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                  uint32_t x, uint32_t y);
    static uint32_t rotr(uint32_t x, int n);
};

// ============================================================================
// Transcript for Fiat-Shamir
// ============================================================================

/**
 * Transcript for Fiat-Shamir transformation
 * 
 * Converts interactive proofs to non-interactive using hash-based challenges.
 */
class Transcript {
public:
    /**
     * Create transcript with domain separator
     */
    explicit Transcript(const std::string& label);
    
    /**
     * Append a field element to transcript
     */
    void append_field_element(const std::string& label, const FieldElement256& elem);
    
    /**
     * Append a point to transcript
     */
    void append_point(const std::string& label, const AffinePoint256& point);
    void append_point(const std::string& label, const AffinePoint384& point);
    
    /**
     * Append raw bytes to transcript
     */
    void append_bytes(const std::string& label, const uint8_t* data, size_t len);
    
    /**
     * Get challenge field element
     */
    FieldElement256 challenge_field_element(const std::string& label);
    
    /**
     * Get challenge bytes
     */
    std::array<uint8_t, 32> challenge_bytes(const std::string& label);
    
    /**
     * Fork transcript (for parallel proving)
     */
    Transcript fork(const std::string& label) const;
    
private:
    Blake2sHash::Hasher hasher_;
    std::vector<uint8_t> state_;
    
    void append_message(const std::string& label, const uint8_t* data, size_t len);
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get default Poseidon parameters for BN254
 */
PoseidonParams default_poseidon_params_bn254();

/**
 * Get default Poseidon parameters for BLS12-381
 */
PoseidonParams default_poseidon_params_bls12_381();

} // namespace zk
} // namespace fhe_accelerate
