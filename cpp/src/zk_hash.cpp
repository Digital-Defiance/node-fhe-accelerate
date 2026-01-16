/**
 * Zero-Knowledge Cryptographic Hash Functions Implementation
 * 
 * Implements Poseidon and Blake2s hash functions.
 * 
 * Requirements: 19, 20.4
 */

#include "zk_hash.h"
#include <cstring>
#include <algorithm>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Poseidon Hash Implementation
// ============================================================================

PoseidonHash::PoseidonHash(const PoseidonParams& params)
    : params_(params), field_(bn254_fr()) {
    if (params_.round_constants.empty()) {
        generate_round_constants();
    }
    if (params_.mds_matrix.empty()) {
        generate_mds_matrix();
    }
}

PoseidonHash::PoseidonHash()
    : params_(default_poseidon_params_bn254()), field_(bn254_fr()) {
    generate_round_constants();
    generate_mds_matrix();
}

void PoseidonHash::generate_round_constants() {
    // Generate round constants deterministically
    // Using a simple PRNG seeded with "Poseidon"
    
    size_t total_rounds = params_.full_rounds + params_.partial_rounds;
    size_t num_constants = total_rounds * params_.t;
    params_.round_constants.resize(num_constants);
    
    // Simple deterministic generation
    uint64_t seed = 0x506F736569646F6EULL;  // "Poseidon" in hex
    
    for (size_t i = 0; i < num_constants; ++i) {
        // Mix seed
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        
        // Generate field element
        FieldElement256 elem;
        elem.limbs[0] = seed;
        elem.limbs[1] = seed ^ (seed >> 17);
        elem.limbs[2] = seed ^ (seed << 23);
        elem.limbs[3] = (seed >> 32) ^ seed;
        
        // Reduce modulo field
        while (elem >= field_.modulus()) {
            elem.limbs[3] >>= 1;
        }
        
        params_.round_constants[i] = field_.to_montgomery(elem);
    }
}

void PoseidonHash::generate_mds_matrix() {
    // Generate MDS (Maximum Distance Separable) matrix
    // Using Cauchy matrix construction
    
    size_t t = params_.t;
    params_.mds_matrix.resize(t, std::vector<FieldElement256>(t));
    
    // x_i = i, y_j = t + j
    // M[i][j] = 1 / (x_i + y_j) = 1 / (i + t + j)
    
    for (size_t i = 0; i < t; ++i) {
        for (size_t j = 0; j < t; ++j) {
            uint64_t val = i + t + j + 1;  // +1 to avoid division by zero
            FieldElement256 elem = field_.to_montgomery(FieldElement256(val));
            params_.mds_matrix[i][j] = field_.inv(elem);
        }
    }
}

FieldElement256 PoseidonHash::sbox(const FieldElement256& x) const {
    // S-box: x^alpha (alpha=5 for BN254)
    FieldElement256 x2 = field_.square(x);
    FieldElement256 x4 = field_.square(x2);
    return field_.mul(x4, x);  // x^5
}

void PoseidonHash::mds_multiply(std::vector<FieldElement256>& state) const {
    std::vector<FieldElement256> new_state(params_.t);
    
    for (size_t i = 0; i < params_.t; ++i) {
        new_state[i] = field_.zero();
        for (size_t j = 0; j < params_.t; ++j) {
            FieldElement256 term = field_.mul(params_.mds_matrix[i][j], state[j]);
            new_state[i] = field_.add(new_state[i], term);
        }
    }
    
    state = std::move(new_state);
}

void PoseidonHash::add_round_constants(std::vector<FieldElement256>& state, 
                                        size_t round) const {
    size_t offset = round * params_.t;
    for (size_t i = 0; i < params_.t; ++i) {
        state[i] = field_.add(state[i], params_.round_constants[offset + i]);
    }
}

void PoseidonHash::full_round(std::vector<FieldElement256>& state, size_t round) const {
    add_round_constants(state, round);
    for (size_t i = 0; i < params_.t; ++i) {
        state[i] = sbox(state[i]);
    }
    mds_multiply(state);
}

void PoseidonHash::partial_round(std::vector<FieldElement256>& state, size_t round) const {
    add_round_constants(state, round);
    state[0] = sbox(state[0]);  // S-box only on first element
    mds_multiply(state);
}


void PoseidonHash::permutation(std::vector<FieldElement256>& state) const {
    size_t round = 0;
    
    // First half of full rounds
    for (size_t i = 0; i < params_.full_rounds / 2; ++i) {
        full_round(state, round++);
    }
    
    // Partial rounds
    for (size_t i = 0; i < params_.partial_rounds; ++i) {
        partial_round(state, round++);
    }
    
    // Second half of full rounds
    for (size_t i = 0; i < params_.full_rounds / 2; ++i) {
        full_round(state, round++);
    }
}

FieldElement256 PoseidonHash::hash(const FieldElement256& input) const {
    std::vector<FieldElement256> state(params_.t);
    state[0] = input;
    // Capacity elements initialized to zero
    
    permutation(state);
    
    return state[0];
}

FieldElement256 PoseidonHash::hash2(const FieldElement256& left,
                                     const FieldElement256& right) const {
    std::vector<FieldElement256> state(params_.t);
    state[0] = left;
    state[1] = right;
    
    permutation(state);
    
    return state[0];
}

FieldElement256 PoseidonHash::hash_many(const FieldElement256* inputs, 
                                         size_t count) const {
    if (count == 0) return field_.zero();
    if (count == 1) return hash(inputs[0]);
    if (count == 2) return hash2(inputs[0], inputs[1]);
    
    // Sponge construction for multiple inputs
    std::vector<FieldElement256> state(params_.t);
    
    size_t absorbed = 0;
    while (absorbed < count) {
        // Absorb up to 'rate' elements
        for (size_t i = 0; i < params_.rate && absorbed < count; ++i) {
            state[i] = field_.add(state[i], inputs[absorbed++]);
        }
        permutation(state);
    }
    
    return state[0];
}

FieldElement256 PoseidonHash::hash_bytes(const uint8_t* data, size_t len) const {
    // Convert bytes to field elements (31 bytes per element to stay in field)
    std::vector<FieldElement256> elements;
    
    for (size_t i = 0; i < len; i += 31) {
        std::array<uint8_t, 32> bytes = {0};
        size_t chunk_len = std::min(size_t(31), len - i);
        std::memcpy(bytes.data() + 1, data + i, chunk_len);  // Leave MSB as 0
        
        // Reverse for little-endian
        std::reverse(bytes.begin(), bytes.end());
        
        FieldElement256 elem = FieldElement256::from_bytes(bytes);
        elements.push_back(field_.to_montgomery(elem));
    }
    
    return hash_many(elements.data(), elements.size());
}

FieldElement256 PoseidonHash::build_merkle_tree(
    const std::vector<FieldElement256>& leaves,
    std::vector<std::vector<FieldElement256>>& tree) const {
    
    if (leaves.empty()) return field_.zero();
    
    tree.clear();
    tree.push_back(leaves);
    
    while (tree.back().size() > 1) {
        const auto& prev = tree.back();
        size_t prev_size = prev.size();
        size_t new_size = (prev_size + 1) / 2;
        
        std::vector<FieldElement256> layer(new_size);
        
        for (size_t i = 0; i < new_size; ++i) {
            size_t left_idx = 2 * i;
            size_t right_idx = 2 * i + 1;
            
            if (right_idx < prev_size) {
                layer[i] = hash2(prev[left_idx], prev[right_idx]);
            } else {
                layer[i] = prev[left_idx];
            }
        }
        
        tree.push_back(std::move(layer));
    }
    
    return tree.back()[0];
}

FieldElement256 PoseidonHash::build_merkle_tree_gpu(
    const std::vector<FieldElement256>& leaves,
    std::vector<std::vector<FieldElement256>>& tree) const {
    // TODO: Implement Metal GPU acceleration
    return build_merkle_tree(leaves, tree);
}

std::vector<FieldElement256> PoseidonHash::get_merkle_path(
    const std::vector<std::vector<FieldElement256>>& tree,
    size_t leaf_index) const {
    
    std::vector<FieldElement256> path;
    size_t index = leaf_index;
    
    for (size_t level = 0; level < tree.size() - 1; ++level) {
        size_t sibling_idx = (index % 2 == 0) ? index + 1 : index - 1;
        
        if (sibling_idx < tree[level].size()) {
            path.push_back(tree[level][sibling_idx]);
        } else {
            path.push_back(tree[level][index]);
        }
        
        index /= 2;
    }
    
    return path;
}

bool PoseidonHash::verify_merkle_path(const FieldElement256& root,
                                       const FieldElement256& leaf,
                                       size_t leaf_index,
                                       const std::vector<FieldElement256>& path) const {
    FieldElement256 current = leaf;
    size_t index = leaf_index;
    
    for (const auto& sibling : path) {
        if (index % 2 == 0) {
            current = hash2(current, sibling);
        } else {
            current = hash2(sibling, current);
        }
        index /= 2;
    }
    
    return current == root;
}

void PoseidonHash::batch_hash2(const FieldElement256* lefts,
                                const FieldElement256* rights,
                                FieldElement256* outputs,
                                size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        outputs[i] = hash2(lefts[i], rights[i]);
    }
}

void PoseidonHash::batch_hash2_gpu(const FieldElement256* lefts,
                                    const FieldElement256* rights,
                                    FieldElement256* outputs,
                                    size_t count) const {
    // TODO: Implement Metal GPU acceleration
    batch_hash2(lefts, rights, outputs, count);
}


// ============================================================================
// Blake2s Hash Implementation
// ============================================================================

uint32_t Blake2sHash::rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

void Blake2sHash::G(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                    uint32_t x, uint32_t y) {
    a = a + b + x;
    d = rotr(d ^ a, 16);
    c = c + d;
    b = rotr(b ^ c, 12);
    a = a + b + y;
    d = rotr(d ^ a, 8);
    c = c + d;
    b = rotr(b ^ c, 7);
}

Blake2sHash::Hasher::Hasher() {
    reset();
}

void Blake2sHash::Hasher::reset() {
    h_ = IV;
    h_[0] ^= 0x01010020;  // digest_length=32, fanout=1, depth=1
    buf_len_ = 0;
    total_len_ = 0;
}

void Blake2sHash::Hasher::compress(const uint8_t* block, bool is_last) {
    uint32_t m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = static_cast<uint32_t>(block[i * 4]) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 8) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 3]) << 24);
    }
    
    uint32_t v[16];
    for (int i = 0; i < 8; ++i) {
        v[i] = h_[i];
        v[i + 8] = IV[i];
    }
    
    v[12] ^= static_cast<uint32_t>(total_len_);
    v[13] ^= static_cast<uint32_t>(total_len_ >> 32);
    
    if (is_last) {
        v[14] = ~v[14];
    }
    
    // 10 rounds
    for (int round = 0; round < 10; ++round) {
        const auto& s = SIGMA[round];
        G(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
        G(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]]);
        G(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]]);
        G(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]]);
        G(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]]);
        G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        G(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        G(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }
    
    for (int i = 0; i < 8; ++i) {
        h_[i] ^= v[i] ^ v[i + 8];
    }
}

void Blake2sHash::Hasher::update(const uint8_t* data, size_t len) {
    while (len > 0) {
        size_t to_copy = std::min(len, BLOCK_SIZE - buf_len_);
        std::memcpy(buf_.data() + buf_len_, data, to_copy);
        buf_len_ += to_copy;
        data += to_copy;
        len -= to_copy;
        
        if (buf_len_ == BLOCK_SIZE) {
            total_len_ += BLOCK_SIZE;
            compress(buf_.data(), false);
            buf_len_ = 0;
        }
    }
}

std::array<uint8_t, Blake2sHash::DIGEST_SIZE> Blake2sHash::Hasher::finalize() {
    total_len_ += buf_len_;
    
    // Pad with zeros
    std::memset(buf_.data() + buf_len_, 0, BLOCK_SIZE - buf_len_);
    compress(buf_.data(), true);
    
    std::array<uint8_t, DIGEST_SIZE> digest;
    for (int i = 0; i < 8; ++i) {
        digest[i * 4] = static_cast<uint8_t>(h_[i]);
        digest[i * 4 + 1] = static_cast<uint8_t>(h_[i] >> 8);
        digest[i * 4 + 2] = static_cast<uint8_t>(h_[i] >> 16);
        digest[i * 4 + 3] = static_cast<uint8_t>(h_[i] >> 24);
    }
    
    return digest;
}

std::array<uint8_t, Blake2sHash::DIGEST_SIZE> Blake2sHash::hash(
    const uint8_t* data, size_t len) {
    Hasher hasher;
    hasher.update(data, len);
    return hasher.finalize();
}

std::array<uint8_t, Blake2sHash::DIGEST_SIZE> Blake2sHash::hash_with_personal(
    const uint8_t* data, size_t len,
    const uint8_t* personal, size_t personal_len) {
    
    // Simplified: just prepend personalization
    Hasher hasher;
    hasher.update(personal, personal_len);
    hasher.update(data, len);
    return hasher.finalize();
}

std::array<uint8_t, Blake2sHash::DIGEST_SIZE> Blake2sHash::hash2(
    const std::array<uint8_t, DIGEST_SIZE>& left,
    const std::array<uint8_t, DIGEST_SIZE>& right) {
    
    Hasher hasher;
    hasher.update(left.data(), DIGEST_SIZE);
    hasher.update(right.data(), DIGEST_SIZE);
    return hasher.finalize();
}

FieldElement256 Blake2sHash::to_field_element(
    const std::array<uint8_t, DIGEST_SIZE>& hash) {
    
    std::array<uint8_t, 32> bytes;
    std::copy(hash.begin(), hash.end(), bytes.begin());
    
    FieldElement256 elem = FieldElement256::from_bytes(bytes);
    
    // Reduce modulo field if necessary
    const Field256& field = bn254_fr();
    while (elem >= field.modulus()) {
        elem.limbs[3] >>= 1;
    }
    
    return field.to_montgomery(elem);
}

std::array<uint8_t, Blake2sHash::DIGEST_SIZE> Blake2sHash::build_merkle_tree(
    const std::vector<std::array<uint8_t, DIGEST_SIZE>>& leaves,
    std::vector<std::vector<std::array<uint8_t, DIGEST_SIZE>>>& tree) {
    
    if (leaves.empty()) {
        return std::array<uint8_t, DIGEST_SIZE>{};
    }
    
    tree.clear();
    tree.push_back(leaves);
    
    while (tree.back().size() > 1) {
        const auto& prev = tree.back();
        size_t prev_size = prev.size();
        size_t new_size = (prev_size + 1) / 2;
        
        std::vector<std::array<uint8_t, DIGEST_SIZE>> layer(new_size);
        
        for (size_t i = 0; i < new_size; ++i) {
            size_t left_idx = 2 * i;
            size_t right_idx = 2 * i + 1;
            
            if (right_idx < prev_size) {
                layer[i] = hash2(prev[left_idx], prev[right_idx]);
            } else {
                layer[i] = prev[left_idx];
            }
        }
        
        tree.push_back(std::move(layer));
    }
    
    return tree.back()[0];
}

std::array<uint8_t, Blake2sHash::DIGEST_SIZE> Blake2sHash::build_merkle_tree_gpu(
    const std::vector<std::array<uint8_t, DIGEST_SIZE>>& leaves,
    std::vector<std::vector<std::array<uint8_t, DIGEST_SIZE>>>& tree) {
    // TODO: Implement Metal GPU acceleration
    return build_merkle_tree(leaves, tree);
}

void Blake2sHash::batch_hash(const uint8_t* const* inputs, const size_t* lengths,
                              std::array<uint8_t, DIGEST_SIZE>* outputs, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        outputs[i] = hash(inputs[i], lengths[i]);
    }
}


// ============================================================================
// Transcript Implementation
// ============================================================================

Transcript::Transcript(const std::string& label) {
    // Initialize with domain separator
    append_message("domain-sep", 
                   reinterpret_cast<const uint8_t*>(label.data()), 
                   label.size());
}

void Transcript::append_message(const std::string& label, 
                                 const uint8_t* data, size_t len) {
    // Append label length and label
    uint32_t label_len = static_cast<uint32_t>(label.size());
    hasher_.update(reinterpret_cast<const uint8_t*>(&label_len), 4);
    hasher_.update(reinterpret_cast<const uint8_t*>(label.data()), label.size());
    
    // Append data length and data
    uint32_t data_len = static_cast<uint32_t>(len);
    hasher_.update(reinterpret_cast<const uint8_t*>(&data_len), 4);
    hasher_.update(data, len);
}

void Transcript::append_field_element(const std::string& label, 
                                       const FieldElement256& elem) {
    auto bytes = elem.to_bytes();
    append_message(label, bytes.data(), bytes.size());
}

void Transcript::append_point(const std::string& label, const AffinePoint256& point) {
    if (point.is_infinity) {
        uint8_t infinity_marker = 0xFF;
        append_message(label, &infinity_marker, 1);
    } else {
        auto x_bytes = point.x.to_bytes();
        auto y_bytes = point.y.to_bytes();
        
        std::vector<uint8_t> combined(64);
        std::copy(x_bytes.begin(), x_bytes.end(), combined.begin());
        std::copy(y_bytes.begin(), y_bytes.end(), combined.begin() + 32);
        
        append_message(label, combined.data(), combined.size());
    }
}

void Transcript::append_point(const std::string& label, const AffinePoint384& point) {
    if (point.is_infinity) {
        uint8_t infinity_marker = 0xFF;
        append_message(label, &infinity_marker, 1);
    } else {
        auto x_bytes = point.x.to_bytes();
        auto y_bytes = point.y.to_bytes();
        
        std::vector<uint8_t> combined(96);
        std::copy(x_bytes.begin(), x_bytes.end(), combined.begin());
        std::copy(y_bytes.begin(), y_bytes.end(), combined.begin() + 48);
        
        append_message(label, combined.data(), combined.size());
    }
}

void Transcript::append_bytes(const std::string& label, 
                               const uint8_t* data, size_t len) {
    append_message(label, data, len);
}

FieldElement256 Transcript::challenge_field_element(const std::string& label) {
    auto bytes = challenge_bytes(label);
    return Blake2sHash::to_field_element(bytes);
}

std::array<uint8_t, 32> Transcript::challenge_bytes(const std::string& label) {
    // Append challenge label
    append_message("challenge", 
                   reinterpret_cast<const uint8_t*>(label.data()),
                   label.size());
    
    // Get current hash state
    Blake2sHash::Hasher temp_hasher = hasher_;
    auto challenge = temp_hasher.finalize();
    
    // Update state with challenge (for domain separation)
    hasher_.update(challenge.data(), challenge.size());
    
    return challenge;
}

Transcript Transcript::fork(const std::string& label) const {
    Transcript forked = *this;
    forked.append_message("fork", 
                          reinterpret_cast<const uint8_t*>(label.data()),
                          label.size());
    return forked;
}

// ============================================================================
// Utility Functions
// ============================================================================

PoseidonParams default_poseidon_params_bn254() {
    PoseidonParams params;
    params.t = 3;           // State width
    params.rate = 2;        // Rate
    params.capacity = 1;    // Capacity
    params.full_rounds = 8;
    params.partial_rounds = 57;
    params.alpha = 5;       // S-box exponent
    return params;
}

PoseidonParams default_poseidon_params_bls12_381() {
    PoseidonParams params;
    params.t = 3;
    params.rate = 2;
    params.capacity = 1;
    params.full_rounds = 8;
    params.partial_rounds = 56;
    params.alpha = 5;
    return params;
}

} // namespace zk
} // namespace fhe_accelerate
