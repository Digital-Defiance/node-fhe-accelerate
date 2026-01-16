/**
 * @file key_serializer.cpp
 * @brief FHE Key Serialization Implementation
 * 
 * Implements binary serialization for FHE keys with integrity checksums,
 * streaming support, and versioning.
 * 
 * Requirements: 4.5, 4.6, 16
 */

#include "key_serializer.h"
#include <cstring>
#include <sstream>

namespace fhe_accelerate {

// ============================================================================
// CRC32 Implementation (IEEE 802.3 polynomial)
// ============================================================================

static const uint32_t crc32_table[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924
};

uint32_t KeySerializer::compute_crc32(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

uint32_t KeySerializer::compute_sha256_truncated(const uint8_t* data, size_t length) {
    // Simplified: use CRC32 as placeholder
    // In production, use CommonCrypto or OpenSSL for SHA256
    return compute_crc32(data, length);
}

// ============================================================================
// KeySerializer Implementation
// ============================================================================

KeySerializer::KeySerializer()
    : checksum_type_(ChecksumType::CRC32)
    , compression_type_(CompressionType::NONE)
{}

KeySerializer::~KeySerializer() = default;

void KeySerializer::report_progress(size_t bytes_processed, size_t total_bytes) {
    if (progress_callback_) {
        progress_callback_(bytes_processed, total_bytes);
    }
}

uint32_t KeySerializer::compute_checksum(const uint8_t* data, size_t length) {
    switch (checksum_type_) {
        case ChecksumType::CRC32:
            return compute_crc32(data, length);
        case ChecksumType::SHA256:
            return compute_sha256_truncated(data, length);
        case ChecksumType::NONE:
        default:
            return 0;
    }
}

bool KeySerializer::verify_checksum(const uint8_t* data, size_t length, 
                                     uint32_t expected, ChecksumType type) {
    if (type == ChecksumType::NONE) return true;
    
    uint32_t computed;
    switch (type) {
        case ChecksumType::CRC32:
            computed = compute_crc32(data, length);
            break;
        case ChecksumType::SHA256:
            computed = compute_sha256_truncated(data, length);
            break;
        default:
            return true;
    }
    
    return computed == expected;
}

void KeySerializer::write_header(std::ostream& out, const SerializationHeader& header) {
    out.write(reinterpret_cast<const char*>(&header.magic), sizeof(header.magic));
    out.write(reinterpret_cast<const char*>(&header.version), sizeof(header.version));
    out.write(reinterpret_cast<const char*>(&header.key_type), sizeof(header.key_type));
    out.write(reinterpret_cast<const char*>(&header.key_id), sizeof(header.key_id));
    out.write(reinterpret_cast<const char*>(&header.poly_degree), sizeof(header.poly_degree));
    out.write(reinterpret_cast<const char*>(&header.modulus), sizeof(header.modulus));
    out.write(reinterpret_cast<const char*>(&header.data_size), sizeof(header.data_size));
    out.write(reinterpret_cast<const char*>(&header.checksum_type), sizeof(header.checksum_type));
    out.write(reinterpret_cast<const char*>(&header.compression), sizeof(header.compression));
    out.write(reinterpret_cast<const char*>(header.reserved), sizeof(header.reserved));
    out.write(reinterpret_cast<const char*>(&header.checksum), sizeof(header.checksum));
}

bool KeySerializer::read_header_impl(std::istream& in, SerializationHeader& header) {
    in.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
    in.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    in.read(reinterpret_cast<char*>(&header.key_type), sizeof(header.key_type));
    in.read(reinterpret_cast<char*>(&header.key_id), sizeof(header.key_id));
    in.read(reinterpret_cast<char*>(&header.poly_degree), sizeof(header.poly_degree));
    in.read(reinterpret_cast<char*>(&header.modulus), sizeof(header.modulus));
    in.read(reinterpret_cast<char*>(&header.data_size), sizeof(header.data_size));
    in.read(reinterpret_cast<char*>(&header.checksum_type), sizeof(header.checksum_type));
    in.read(reinterpret_cast<char*>(&header.compression), sizeof(header.compression));
    in.read(reinterpret_cast<char*>(header.reserved), sizeof(header.reserved));
    in.read(reinterpret_cast<char*>(&header.checksum), sizeof(header.checksum));
    return in.good();
}

void KeySerializer::write_polynomial(std::ostream& out, const Polynomial& poly) {
    // Write coefficients as raw bytes
    uint32_t degree = poly.degree();
    out.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
    out.write(reinterpret_cast<const char*>(poly.data()), degree * sizeof(uint64_t));
}

Polynomial KeySerializer::read_polynomial(std::istream& in, uint32_t degree, uint64_t modulus) {
    uint32_t read_degree;
    in.read(reinterpret_cast<char*>(&read_degree), sizeof(read_degree));
    
    if (read_degree != degree) {
        throw std::runtime_error("Polynomial degree mismatch");
    }
    
    std::vector<uint64_t> coeffs(degree);
    in.read(reinterpret_cast<char*>(coeffs.data()), degree * sizeof(uint64_t));
    
    return Polynomial(std::move(coeffs), modulus, false);
}

// ============================================================================
// Secret Key Serialization
// ============================================================================

SerializationResult KeySerializer::serialize_secret_key(
    const SecretKey& sk,
    std::vector<uint8_t>& output
) {
    std::ostringstream oss(std::ios::binary);
    auto result = serialize_secret_key(sk, oss);
    if (result.success) {
        std::string str = oss.str();
        output.assign(str.begin(), str.end());
    }
    return result;
}

SerializationResult KeySerializer::serialize_secret_key(
    const SecretKey& sk,
    std::ostream& output
) {
    // Serialize polynomial data first to compute checksum
    std::ostringstream data_stream(std::ios::binary);
    
    // Write distribution type
    uint8_t dist = static_cast<uint8_t>(sk.distribution);
    data_stream.write(reinterpret_cast<const char*>(&dist), sizeof(dist));
    
    // Write polynomial
    write_polynomial(data_stream, sk.poly);
    
    std::string data_str = data_stream.str();
    
    // Prepare header
    SerializationHeader header;
    header.magic = MAGIC_SECRET_KEY;
    header.version = SERIALIZATION_VERSION;
    header.key_type = 0;  // Secret key
    header.key_id = sk.key_id;
    header.poly_degree = sk.poly.degree();
    header.modulus = sk.poly.modulus();
    header.data_size = static_cast<uint32_t>(data_str.size());
    header.checksum_type = checksum_type_;
    header.compression = compression_type_;
    header.checksum = compute_checksum(
        reinterpret_cast<const uint8_t*>(data_str.data()),
        data_str.size()
    );
    
    // Write header and data
    write_header(output, header);
    output.write(data_str.data(), data_str.size());
    
    size_t total_size = sizeof(SerializationHeader) + data_str.size();
    report_progress(total_size, total_size);
    
    return SerializationResult::ok(total_size);
}

DeserializationResult<SecretKey> KeySerializer::deserialize_secret_key(
    const std::vector<uint8_t>& input,
    uint64_t modulus
) {
    std::istringstream iss(std::string(input.begin(), input.end()), std::ios::binary);
    return deserialize_secret_key(iss, modulus);
}

DeserializationResult<SecretKey> KeySerializer::deserialize_secret_key(
    std::istream& input,
    uint64_t modulus
) {
    SerializationHeader header;
    if (!read_header_impl(input, header)) {
        return DeserializationResult<SecretKey>::error("Failed to read header");
    }
    
    if (header.magic != MAGIC_SECRET_KEY) {
        return DeserializationResult<SecretKey>::error("Invalid magic bytes for secret key");
    }
    
    if (header.version > SERIALIZATION_VERSION) {
        return DeserializationResult<SecretKey>::error("Unsupported serialization version");
    }
    
    // Read data
    std::vector<uint8_t> data(header.data_size);
    input.read(reinterpret_cast<char*>(data.data()), header.data_size);
    
    // Verify checksum
    if (!verify_checksum(data.data(), data.size(), header.checksum, header.checksum_type)) {
        return DeserializationResult<SecretKey>::error("Checksum verification failed");
    }
    
    // Parse data
    std::istringstream data_stream(std::string(data.begin(), data.end()), std::ios::binary);
    
    uint8_t dist;
    data_stream.read(reinterpret_cast<char*>(&dist), sizeof(dist));
    SecretKeyDistribution distribution = static_cast<SecretKeyDistribution>(dist);
    
    Polynomial poly = read_polynomial(data_stream, header.poly_degree, 
                                       modulus > 0 ? modulus : header.modulus);
    
    auto sk = std::make_unique<SecretKey>(std::move(poly), distribution, header.key_id);
    
    return DeserializationResult<SecretKey>::ok(
        std::move(sk),
        sizeof(SerializationHeader) + header.data_size
    );
}

// ============================================================================
// Public Key Serialization
// ============================================================================

SerializationResult KeySerializer::serialize_public_key(
    const PublicKey& pk,
    std::vector<uint8_t>& output
) {
    std::ostringstream oss(std::ios::binary);
    auto result = serialize_public_key(pk, oss);
    if (result.success) {
        std::string str = oss.str();
        output.assign(str.begin(), str.end());
    }
    return result;
}

SerializationResult KeySerializer::serialize_public_key(
    const PublicKey& pk,
    std::ostream& output
) {
    std::ostringstream data_stream(std::ios::binary);
    
    // Write both polynomials
    write_polynomial(data_stream, pk.a);
    write_polynomial(data_stream, pk.b);
    
    std::string data_str = data_stream.str();
    
    SerializationHeader header;
    header.magic = MAGIC_PUBLIC_KEY;
    header.version = SERIALIZATION_VERSION;
    header.key_type = 1;  // Public key
    header.key_id = pk.key_id;
    header.poly_degree = pk.a.degree();
    header.modulus = pk.a.modulus();
    header.data_size = static_cast<uint32_t>(data_str.size());
    header.checksum_type = checksum_type_;
    header.compression = compression_type_;
    header.checksum = compute_checksum(
        reinterpret_cast<const uint8_t*>(data_str.data()),
        data_str.size()
    );
    
    write_header(output, header);
    output.write(data_str.data(), data_str.size());
    
    size_t total_size = sizeof(SerializationHeader) + data_str.size();
    report_progress(total_size, total_size);
    
    return SerializationResult::ok(total_size);
}

DeserializationResult<PublicKey> KeySerializer::deserialize_public_key(
    const std::vector<uint8_t>& input
) {
    std::istringstream iss(std::string(input.begin(), input.end()), std::ios::binary);
    return deserialize_public_key(iss);
}

DeserializationResult<PublicKey> KeySerializer::deserialize_public_key(
    std::istream& input
) {
    SerializationHeader header;
    if (!read_header_impl(input, header)) {
        return DeserializationResult<PublicKey>::error("Failed to read header");
    }
    
    if (header.magic != MAGIC_PUBLIC_KEY) {
        return DeserializationResult<PublicKey>::error("Invalid magic bytes for public key");
    }
    
    std::vector<uint8_t> data(header.data_size);
    input.read(reinterpret_cast<char*>(data.data()), header.data_size);
    
    if (!verify_checksum(data.data(), data.size(), header.checksum, header.checksum_type)) {
        return DeserializationResult<PublicKey>::error("Checksum verification failed");
    }
    
    std::istringstream data_stream(std::string(data.begin(), data.end()), std::ios::binary);
    
    Polynomial a = read_polynomial(data_stream, header.poly_degree, header.modulus);
    Polynomial b = read_polynomial(data_stream, header.poly_degree, header.modulus);
    
    auto pk = std::make_unique<PublicKey>(std::move(a), std::move(b), header.key_id);
    
    return DeserializationResult<PublicKey>::ok(
        std::move(pk),
        sizeof(SerializationHeader) + header.data_size
    );
}

// ============================================================================
// Evaluation Key Serialization
// ============================================================================

SerializationResult KeySerializer::serialize_eval_key(
    const EvaluationKey& ek,
    std::vector<uint8_t>& output
) {
    std::ostringstream oss(std::ios::binary);
    auto result = serialize_eval_key(ek, oss);
    if (result.success) {
        std::string str = oss.str();
        output.assign(str.begin(), str.end());
    }
    return result;
}

SerializationResult KeySerializer::serialize_eval_key(
    const EvaluationKey& ek,
    std::ostream& output
) {
    std::ostringstream data_stream(std::ios::binary);
    
    // Write decomposition parameters
    data_stream.write(reinterpret_cast<const char*>(&ek.relin_key.decomp_base_log), 
                      sizeof(ek.relin_key.decomp_base_log));
    data_stream.write(reinterpret_cast<const char*>(&ek.relin_key.decomp_level), 
                      sizeof(ek.relin_key.decomp_level));
    
    // Write number of key pairs
    uint32_t num_keys = static_cast<uint32_t>(ek.relin_key.keys.size());
    data_stream.write(reinterpret_cast<const char*>(&num_keys), sizeof(num_keys));
    
    // Write each key pair
    size_t total_bytes = 0;
    for (const auto& key_pair : ek.relin_key.keys) {
        write_polynomial(data_stream, key_pair.first);
        write_polynomial(data_stream, key_pair.second);
        total_bytes += key_pair.first.degree() * sizeof(uint64_t) * 2;
        report_progress(total_bytes, num_keys * key_pair.first.degree() * sizeof(uint64_t) * 2);
    }
    
    std::string data_str = data_stream.str();
    
    SerializationHeader header;
    header.magic = MAGIC_EVAL_KEY;
    header.version = SERIALIZATION_VERSION;
    header.key_type = 2;  // Evaluation key
    header.key_id = ek.key_id;
    header.poly_degree = ek.relin_key.keys.empty() ? 0 : ek.relin_key.keys[0].first.degree();
    header.modulus = ek.relin_key.keys.empty() ? 0 : ek.relin_key.keys[0].first.modulus();
    header.data_size = static_cast<uint32_t>(data_str.size());
    header.checksum_type = checksum_type_;
    header.compression = compression_type_;
    header.checksum = compute_checksum(
        reinterpret_cast<const uint8_t*>(data_str.data()),
        data_str.size()
    );
    
    write_header(output, header);
    output.write(data_str.data(), data_str.size());
    
    return SerializationResult::ok(sizeof(SerializationHeader) + data_str.size());
}

DeserializationResult<EvaluationKey> KeySerializer::deserialize_eval_key(
    const std::vector<uint8_t>& input
) {
    std::istringstream iss(std::string(input.begin(), input.end()), std::ios::binary);
    return deserialize_eval_key(iss);
}

DeserializationResult<EvaluationKey> KeySerializer::deserialize_eval_key(
    std::istream& input
) {
    SerializationHeader header;
    if (!read_header_impl(input, header)) {
        return DeserializationResult<EvaluationKey>::error("Failed to read header");
    }
    
    if (header.magic != MAGIC_EVAL_KEY) {
        return DeserializationResult<EvaluationKey>::error("Invalid magic bytes");
    }
    
    std::vector<uint8_t> data(header.data_size);
    input.read(reinterpret_cast<char*>(data.data()), header.data_size);
    
    if (!verify_checksum(data.data(), data.size(), header.checksum, header.checksum_type)) {
        return DeserializationResult<EvaluationKey>::error("Checksum verification failed");
    }
    
    std::istringstream data_stream(std::string(data.begin(), data.end()), std::ios::binary);
    
    auto ek = std::make_unique<EvaluationKey>();
    ek->key_id = header.key_id;
    
    data_stream.read(reinterpret_cast<char*>(&ek->relin_key.decomp_base_log), 
                     sizeof(ek->relin_key.decomp_base_log));
    data_stream.read(reinterpret_cast<char*>(&ek->relin_key.decomp_level), 
                     sizeof(ek->relin_key.decomp_level));
    
    uint32_t num_keys;
    data_stream.read(reinterpret_cast<char*>(&num_keys), sizeof(num_keys));
    
    ek->relin_key.keys.reserve(num_keys);
    for (uint32_t i = 0; i < num_keys; ++i) {
        Polynomial a = read_polynomial(data_stream, header.poly_degree, header.modulus);
        Polynomial b = read_polynomial(data_stream, header.poly_degree, header.modulus);
        ek->relin_key.keys.emplace_back(std::move(a), std::move(b));
    }
    
    ek->relin_key.key_id = header.key_id;
    
    return DeserializationResult<EvaluationKey>::ok(
        std::move(ek),
        sizeof(SerializationHeader) + header.data_size
    );
}

// ============================================================================
// Bootstrapping Key Serialization
// ============================================================================

SerializationResult KeySerializer::serialize_bootstrap_key(
    const BootstrapKey& bk,
    std::vector<uint8_t>& output
) {
    std::ostringstream oss(std::ios::binary);
    auto result = serialize_bootstrap_key(bk, oss);
    if (result.success) {
        std::string str = oss.str();
        output.assign(str.begin(), str.end());
    }
    return result;
}

SerializationResult KeySerializer::serialize_bootstrap_key(
    const BootstrapKey& bk,
    std::ostream& output
) {
    std::ostringstream data_stream(std::ios::binary);
    
    // Write LWE dimension
    data_stream.write(reinterpret_cast<const char*>(&bk.lwe_dimension), sizeof(bk.lwe_dimension));
    
    // Write BSK size
    uint32_t bsk_size = static_cast<uint32_t>(bk.bsk.size());
    data_stream.write(reinterpret_cast<const char*>(&bsk_size), sizeof(bsk_size));
    
    // Write BSK entries
    for (const auto& row : bk.bsk) {
        uint32_t row_size = static_cast<uint32_t>(row.size());
        data_stream.write(reinterpret_cast<const char*>(&row_size), sizeof(row_size));
        for (const auto& pair : row) {
            write_polynomial(data_stream, pair.first);
            write_polynomial(data_stream, pair.second);
        }
    }
    
    // Write KSK
    data_stream.write(reinterpret_cast<const char*>(&bk.ksk.decomp_base_log), 
                      sizeof(bk.ksk.decomp_base_log));
    data_stream.write(reinterpret_cast<const char*>(&bk.ksk.decomp_level), 
                      sizeof(bk.ksk.decomp_level));
    
    uint32_t ksk_size = static_cast<uint32_t>(bk.ksk.keys.size());
    data_stream.write(reinterpret_cast<const char*>(&ksk_size), sizeof(ksk_size));
    
    for (const auto& pair : bk.ksk.keys) {
        write_polynomial(data_stream, pair.first);
        write_polynomial(data_stream, pair.second);
    }
    
    std::string data_str = data_stream.str();
    
    SerializationHeader header;
    header.magic = MAGIC_BOOTSTRAP_KEY;
    header.version = SERIALIZATION_VERSION;
    header.key_type = 3;  // Bootstrap key
    header.key_id = bk.key_id;
    header.poly_degree = bk.bsk.empty() || bk.bsk[0].empty() ? 0 : bk.bsk[0][0].first.degree();
    header.modulus = bk.bsk.empty() || bk.bsk[0].empty() ? 0 : bk.bsk[0][0].first.modulus();
    header.data_size = static_cast<uint32_t>(data_str.size());
    header.checksum_type = checksum_type_;
    header.compression = compression_type_;
    header.checksum = compute_checksum(
        reinterpret_cast<const uint8_t*>(data_str.data()),
        data_str.size()
    );
    
    write_header(output, header);
    output.write(data_str.data(), data_str.size());
    
    return SerializationResult::ok(sizeof(SerializationHeader) + data_str.size());
}

DeserializationResult<BootstrapKey> KeySerializer::deserialize_bootstrap_key(
    const std::vector<uint8_t>& input
) {
    std::istringstream iss(std::string(input.begin(), input.end()), std::ios::binary);
    return deserialize_bootstrap_key(iss);
}

DeserializationResult<BootstrapKey> KeySerializer::deserialize_bootstrap_key(
    std::istream& input
) {
    SerializationHeader header;
    if (!read_header_impl(input, header)) {
        return DeserializationResult<BootstrapKey>::error("Failed to read header");
    }
    
    if (header.magic != MAGIC_BOOTSTRAP_KEY) {
        return DeserializationResult<BootstrapKey>::error("Invalid magic bytes");
    }
    
    std::vector<uint8_t> data(header.data_size);
    input.read(reinterpret_cast<char*>(data.data()), header.data_size);
    
    if (!verify_checksum(data.data(), data.size(), header.checksum, header.checksum_type)) {
        return DeserializationResult<BootstrapKey>::error("Checksum verification failed");
    }
    
    std::istringstream data_stream(std::string(data.begin(), data.end()), std::ios::binary);
    
    auto bk = std::make_unique<BootstrapKey>();
    bk->key_id = header.key_id;
    
    data_stream.read(reinterpret_cast<char*>(&bk->lwe_dimension), sizeof(bk->lwe_dimension));
    
    uint32_t bsk_size;
    data_stream.read(reinterpret_cast<char*>(&bsk_size), sizeof(bsk_size));
    
    bk->bsk.reserve(bsk_size);
    for (uint32_t i = 0; i < bsk_size; ++i) {
        uint32_t row_size;
        data_stream.read(reinterpret_cast<char*>(&row_size), sizeof(row_size));
        
        std::vector<std::pair<Polynomial, Polynomial>> row;
        row.reserve(row_size);
        for (uint32_t j = 0; j < row_size; ++j) {
            Polynomial a = read_polynomial(data_stream, header.poly_degree, header.modulus);
            Polynomial b = read_polynomial(data_stream, header.poly_degree, header.modulus);
            row.emplace_back(std::move(a), std::move(b));
        }
        bk->bsk.push_back(std::move(row));
    }
    
    data_stream.read(reinterpret_cast<char*>(&bk->ksk.decomp_base_log), 
                     sizeof(bk->ksk.decomp_base_log));
    data_stream.read(reinterpret_cast<char*>(&bk->ksk.decomp_level), 
                     sizeof(bk->ksk.decomp_level));
    
    uint32_t ksk_size;
    data_stream.read(reinterpret_cast<char*>(&ksk_size), sizeof(ksk_size));
    
    bk->ksk.keys.reserve(ksk_size);
    for (uint32_t i = 0; i < ksk_size; ++i) {
        Polynomial a = read_polynomial(data_stream, header.poly_degree, header.modulus);
        Polynomial b = read_polynomial(data_stream, header.poly_degree, header.modulus);
        bk->ksk.keys.emplace_back(std::move(a), std::move(b));
    }
    
    bk->ksk.key_id = header.key_id;
    
    return DeserializationResult<BootstrapKey>::ok(
        std::move(bk),
        sizeof(SerializationHeader) + header.data_size
    );
}

// ============================================================================
// Partial Deserialization and Verification
// ============================================================================

DeserializationResult<SerializationHeader> KeySerializer::read_header(
    const std::vector<uint8_t>& input
) {
    std::istringstream iss(std::string(input.begin(), input.end()), std::ios::binary);
    return read_header(iss);
}

DeserializationResult<SerializationHeader> KeySerializer::read_header(
    std::istream& input
) {
    auto header = std::make_unique<SerializationHeader>();
    if (!read_header_impl(input, *header)) {
        return DeserializationResult<SerializationHeader>::error("Failed to read header");
    }
    return DeserializationResult<SerializationHeader>::ok(
        std::move(header),
        sizeof(SerializationHeader)
    );
}

bool KeySerializer::verify_integrity(const std::vector<uint8_t>& input) {
    std::istringstream iss(std::string(input.begin(), input.end()), std::ios::binary);
    return verify_integrity(iss);
}

bool KeySerializer::verify_integrity(std::istream& input) {
    SerializationHeader header;
    if (!read_header_impl(input, header)) {
        return false;
    }
    
    std::vector<uint8_t> data(header.data_size);
    input.read(reinterpret_cast<char*>(data.data()), header.data_size);
    
    return verify_checksum(data.data(), data.size(), header.checksum, header.checksum_type);
}

// ============================================================================
// Size Estimation
// ============================================================================

size_t KeySerializer::estimate_size(const SecretKey& sk) {
    return sizeof(SerializationHeader) + 
           sizeof(uint8_t) +  // distribution
           sizeof(uint32_t) + // degree
           sk.poly.degree() * sizeof(uint64_t);
}

size_t KeySerializer::estimate_size(const PublicKey& pk) {
    return sizeof(SerializationHeader) + 
           2 * (sizeof(uint32_t) + pk.a.degree() * sizeof(uint64_t));
}

size_t KeySerializer::estimate_size(const EvaluationKey& ek) {
    size_t data_size = sizeof(uint32_t) * 3;  // decomp params + num_keys
    for (const auto& pair : ek.relin_key.keys) {
        data_size += 2 * (sizeof(uint32_t) + pair.first.degree() * sizeof(uint64_t));
    }
    return sizeof(SerializationHeader) + data_size;
}

size_t KeySerializer::estimate_size(const BootstrapKey& bk) {
    size_t data_size = sizeof(uint32_t) * 2;  // lwe_dimension + bsk_size
    for (const auto& row : bk.bsk) {
        data_size += sizeof(uint32_t);  // row_size
        for (const auto& pair : row) {
            data_size += 2 * (sizeof(uint32_t) + pair.first.degree() * sizeof(uint64_t));
        }
    }
    data_size += sizeof(uint32_t) * 3;  // ksk params
    for (const auto& pair : bk.ksk.keys) {
        data_size += 2 * (sizeof(uint32_t) + pair.first.degree() * sizeof(uint64_t));
    }
    return sizeof(SerializationHeader) + data_size;
}

// ============================================================================
// BallotSerializer Implementation
// ============================================================================

BallotSerializer::BallotSerializer()
    : checksum_type_(ChecksumType::CRC32)
{}

BallotSerializer::~BallotSerializer() = default;

SerializationResult BallotSerializer::serialize_ballot(
    const std::vector<std::pair<Polynomial, Polynomial>>& encrypted_choices,
    uint64_t timestamp,
    std::vector<uint8_t>& output
) {
    std::ostringstream data_stream(std::ios::binary);
    
    // Write timestamp
    data_stream.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    
    // Write number of choices
    uint32_t num_choices = static_cast<uint32_t>(encrypted_choices.size());
    data_stream.write(reinterpret_cast<const char*>(&num_choices), sizeof(num_choices));
    
    // Write each encrypted choice (ciphertext pair)
    for (const auto& choice : encrypted_choices) {
        uint32_t degree = choice.first.degree();
        uint64_t modulus = choice.first.modulus();
        
        data_stream.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
        data_stream.write(reinterpret_cast<const char*>(&modulus), sizeof(modulus));
        data_stream.write(reinterpret_cast<const char*>(choice.first.data()), 
                          degree * sizeof(uint64_t));
        data_stream.write(reinterpret_cast<const char*>(choice.second.data()), 
                          degree * sizeof(uint64_t));
    }
    
    std::string data_str = data_stream.str();
    
    // Prepare header
    SerializationHeader header;
    header.magic = MAGIC_BALLOT;
    header.version = SERIALIZATION_VERSION;
    header.key_type = 4;  // Ballot
    header.key_id = timestamp;  // Use timestamp as ID
    header.poly_degree = encrypted_choices.empty() ? 0 : encrypted_choices[0].first.degree();
    header.modulus = encrypted_choices.empty() ? 0 : encrypted_choices[0].first.modulus();
    header.data_size = static_cast<uint32_t>(data_str.size());
    header.checksum_type = checksum_type_;
    header.compression = CompressionType::NONE;
    header.checksum = KeySerializer::compute_crc32(
        reinterpret_cast<const uint8_t*>(data_str.data()),
        data_str.size()
    );
    
    // Write to output
    std::ostringstream oss(std::ios::binary);
    
    oss.write(reinterpret_cast<const char*>(&header.magic), sizeof(header.magic));
    oss.write(reinterpret_cast<const char*>(&header.version), sizeof(header.version));
    oss.write(reinterpret_cast<const char*>(&header.key_type), sizeof(header.key_type));
    oss.write(reinterpret_cast<const char*>(&header.key_id), sizeof(header.key_id));
    oss.write(reinterpret_cast<const char*>(&header.poly_degree), sizeof(header.poly_degree));
    oss.write(reinterpret_cast<const char*>(&header.modulus), sizeof(header.modulus));
    oss.write(reinterpret_cast<const char*>(&header.data_size), sizeof(header.data_size));
    oss.write(reinterpret_cast<const char*>(&header.checksum_type), sizeof(header.checksum_type));
    oss.write(reinterpret_cast<const char*>(&header.compression), sizeof(header.compression));
    oss.write(reinterpret_cast<const char*>(header.reserved), sizeof(header.reserved));
    oss.write(reinterpret_cast<const char*>(&header.checksum), sizeof(header.checksum));
    oss.write(data_str.data(), data_str.size());
    
    std::string result = oss.str();
    output.assign(result.begin(), result.end());
    
    return SerializationResult::ok(output.size());
}

DeserializationResult<BallotSerializer::BallotData> BallotSerializer::deserialize_ballot(
    const std::vector<uint8_t>& input,
    uint32_t poly_degree,
    uint64_t modulus
) {
    if (input.size() < sizeof(SerializationHeader)) {
        return DeserializationResult<BallotData>::error("Input too small");
    }
    
    std::istringstream iss(std::string(input.begin(), input.end()), std::ios::binary);
    
    SerializationHeader header;
    iss.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
    iss.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    iss.read(reinterpret_cast<char*>(&header.key_type), sizeof(header.key_type));
    iss.read(reinterpret_cast<char*>(&header.key_id), sizeof(header.key_id));
    iss.read(reinterpret_cast<char*>(&header.poly_degree), sizeof(header.poly_degree));
    iss.read(reinterpret_cast<char*>(&header.modulus), sizeof(header.modulus));
    iss.read(reinterpret_cast<char*>(&header.data_size), sizeof(header.data_size));
    iss.read(reinterpret_cast<char*>(&header.checksum_type), sizeof(header.checksum_type));
    iss.read(reinterpret_cast<char*>(&header.compression), sizeof(header.compression));
    iss.read(reinterpret_cast<char*>(header.reserved), sizeof(header.reserved));
    iss.read(reinterpret_cast<char*>(&header.checksum), sizeof(header.checksum));
    
    if (header.magic != MAGIC_BALLOT) {
        return DeserializationResult<BallotData>::error("Invalid magic bytes for ballot");
    }
    
    // Read and verify data
    std::vector<uint8_t> data(header.data_size);
    iss.read(reinterpret_cast<char*>(data.data()), header.data_size);
    
    uint32_t computed_checksum = KeySerializer::compute_crc32(data.data(), data.size());
    if (computed_checksum != header.checksum) {
        return DeserializationResult<BallotData>::error("Checksum verification failed");
    }
    
    // Parse data
    std::istringstream data_stream(std::string(data.begin(), data.end()), std::ios::binary);
    
    auto ballot = std::make_unique<BallotData>();
    ballot->version = header.version;
    
    data_stream.read(reinterpret_cast<char*>(&ballot->timestamp), sizeof(ballot->timestamp));
    
    uint32_t num_choices;
    data_stream.read(reinterpret_cast<char*>(&num_choices), sizeof(num_choices));
    
    ballot->encrypted_choices.reserve(num_choices);
    for (uint32_t i = 0; i < num_choices; ++i) {
        uint32_t degree;
        uint64_t mod;
        data_stream.read(reinterpret_cast<char*>(&degree), sizeof(degree));
        data_stream.read(reinterpret_cast<char*>(&mod), sizeof(mod));
        
        std::vector<uint64_t> a_coeffs(degree);
        std::vector<uint64_t> b_coeffs(degree);
        data_stream.read(reinterpret_cast<char*>(a_coeffs.data()), degree * sizeof(uint64_t));
        data_stream.read(reinterpret_cast<char*>(b_coeffs.data()), degree * sizeof(uint64_t));
        
        Polynomial a(std::move(a_coeffs), mod, false);
        Polynomial b(std::move(b_coeffs), mod, false);
        ballot->encrypted_choices.emplace_back(std::move(a), std::move(b));
    }
    
    return DeserializationResult<BallotData>::ok(
        std::move(ballot),
        input.size()
    );
}

size_t BallotSerializer::estimate_ballot_size(uint32_t num_choices, uint32_t poly_degree) {
    // Header + timestamp + num_choices + (degree + modulus + 2*coeffs) per choice
    return sizeof(SerializationHeader) + 
           sizeof(uint64_t) +  // timestamp
           sizeof(uint32_t) +  // num_choices
           num_choices * (sizeof(uint32_t) + sizeof(uint64_t) + 2 * poly_degree * sizeof(uint64_t));
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<KeySerializer> create_key_serializer() {
    return std::make_unique<KeySerializer>();
}

std::unique_ptr<BallotSerializer> create_ballot_serializer() {
    return std::make_unique<BallotSerializer>();
}

} // namespace fhe_accelerate
