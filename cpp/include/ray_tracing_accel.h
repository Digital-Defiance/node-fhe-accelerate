/**
 * Ray Tracing Hardware Exploitation for FHE
 * 
 * Exploits Metal's ray tracing hardware for tree-structured operations:
 * - BVH traversal for key switching decomposition trees
 * - Decomposition digit extraction via ray tracing
 * - Merkle tree traversal for ZK proofs
 * 
 * Requirements 14.12, 14.13, 14.14
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fhe_accelerate {
namespace ray_tracing {

/**
 * Check if ray tracing hardware is available
 */
bool ray_tracing_available();

/**
 * BVH-based decomposition tree for key switching
 * 
 * Key switching involves decomposing coefficients into digits.
 * We can encode this as a BVH where:
 * - Rays represent coefficient values
 * - BVH nodes represent decomposition levels
 * - Intersections give decomposition digits
 */
class DecompositionBVH {
public:
    /**
     * Create decomposition BVH
     * 
     * @param base Decomposition base
     * @param levels Number of decomposition levels
     */
    DecompositionBVH(uint32_t base, uint32_t levels);
    ~DecompositionBVH();
    
    /**
     * Build BVH structure
     */
    void build();
    
    /**
     * Decompose coefficients using ray tracing
     * 
     * @param coeffs Input coefficients
     * @param digits Output decomposition digits (levels * count elements)
     * @param count Number of coefficients
     */
    void decompose(const uint64_t* coeffs, uint32_t* digits, size_t count);
    
    /**
     * Check if ray tracing is beneficial for this configuration
     */
    bool is_beneficial() const;
    
private:
    uint32_t base_;
    uint32_t levels_;
    void* bvh_structure_;  // Metal acceleration structure
    void* pipeline_;       // Metal compute pipeline
};

/**
 * Merkle tree traversal using ray tracing
 * 
 * For ZK proofs, we need to traverse Merkle trees.
 * BVH traversal hardware can accelerate this.
 */
class MerkleTreeBVH {
public:
    /**
     * Create Merkle tree BVH
     * 
     * @param depth Tree depth
     */
    explicit MerkleTreeBVH(size_t depth);
    ~MerkleTreeBVH();
    
    /**
     * Build BVH from Merkle tree
     * 
     * @param nodes Tree nodes (2^depth - 1 elements)
     */
    void build(const uint8_t* nodes);
    
    /**
     * Generate proof path using ray tracing
     * 
     * @param leaf_index Index of leaf to prove
     * @param proof_path Output proof path
     * @return Number of nodes in proof path
     */
    size_t generate_proof(size_t leaf_index, uint8_t* proof_path);
    
    /**
     * Verify proof using ray tracing
     * 
     * @param leaf Leaf value
     * @param proof_path Proof path
     * @param path_length Length of proof path
     * @param root Expected root
     * @return true if proof is valid
     */
    bool verify_proof(const uint8_t* leaf, const uint8_t* proof_path,
                      size_t path_length, const uint8_t* root);
    
private:
    size_t depth_;
    void* bvh_structure_;
};

/**
 * Benchmark ray tracing vs CPU for tree operations
 */
struct RayTracingBenchmark {
    double cpu_time_us;
    double rt_time_us;
    double speedup;
    std::string operation;
};

RayTracingBenchmark benchmark_decomposition(uint32_t base, uint32_t levels, size_t count);
RayTracingBenchmark benchmark_merkle_tree(size_t depth, size_t num_proofs);

} // namespace ray_tracing
} // namespace fhe_accelerate
