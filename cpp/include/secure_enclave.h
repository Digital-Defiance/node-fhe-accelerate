/**
 * Secure Enclave Integration for FHE Key Protection
 * 
 * Uses Apple's Secure Enclave for:
 * - Secret key generation (key never leaves enclave)
 * - Final decryption (key never exposed)
 * - Ciphertext signing for authenticity
 * 
 * Requirements 14.33, 14.34, 14.35
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace fhe_accelerate {
namespace secure_enclave {

/**
 * Check if Secure Enclave is available
 */
bool secure_enclave_available();

/**
 * Opaque handle to a key stored in Secure Enclave
 */
using SecureKeyHandle = uint64_t;

/**
 * Invalid key handle constant
 */
constexpr SecureKeyHandle INVALID_KEY_HANDLE = 0;

/**
 * Secure Enclave key manager
 * 
 * Manages FHE keys stored in the Secure Enclave.
 * Keys never leave the enclave - all operations happen inside.
 */
class SecureEnclaveKeyManager {
public:
    SecureEnclaveKeyManager();
    ~SecureEnclaveKeyManager();
    
    /**
     * Check if Secure Enclave is available
     */
    bool is_available() const { return available_; }
    
    /**
     * Generate a secret key inside Secure Enclave
     * 
     * The key is generated and stored in the Secure Enclave.
     * It never leaves the enclave.
     * 
     * @param key_size Size of the key in bytes
     * @return Handle to the key (or INVALID_KEY_HANDLE on failure)
     */
    SecureKeyHandle generate_secret_key(size_t key_size);
    
    /**
     * Delete a key from Secure Enclave
     * 
     * @param handle Key handle
     * @return true if successful
     */
    bool delete_key(SecureKeyHandle handle);
    
    /**
     * Perform decryption inside Secure Enclave
     * 
     * The ciphertext is sent to the enclave, decrypted using the
     * stored key, and the plaintext is returned. The key never
     * leaves the enclave.
     * 
     * @param handle Key handle
     * @param ciphertext Encrypted data
     * @param ct_size Ciphertext size
     * @param plaintext Output buffer for decrypted data
     * @param pt_size Plaintext buffer size
     * @return Actual plaintext size (or 0 on failure)
     */
    size_t decrypt_in_enclave(SecureKeyHandle handle,
                              const uint8_t* ciphertext, size_t ct_size,
                              uint8_t* plaintext, size_t pt_size);
    
    /**
     * Sign data using a key in Secure Enclave
     * 
     * @param handle Key handle
     * @param data Data to sign
     * @param data_size Data size
     * @param signature Output signature buffer
     * @param sig_size Signature buffer size
     * @return Actual signature size (or 0 on failure)
     */
    size_t sign_data(SecureKeyHandle handle,
                     const uint8_t* data, size_t data_size,
                     uint8_t* signature, size_t sig_size);
    
    /**
     * Verify a signature
     * 
     * @param data Original data
     * @param data_size Data size
     * @param signature Signature to verify
     * @param sig_size Signature size
     * @param public_key Public key for verification
     * @param pk_size Public key size
     * @return true if signature is valid
     */
    bool verify_signature(const uint8_t* data, size_t data_size,
                          const uint8_t* signature, size_t sig_size,
                          const uint8_t* public_key, size_t pk_size);
    
    /**
     * Get the public key for a Secure Enclave key
     * 
     * @param handle Key handle
     * @param public_key Output buffer for public key
     * @param pk_size Buffer size
     * @return Actual public key size (or 0 on failure)
     */
    size_t get_public_key(SecureKeyHandle handle,
                          uint8_t* public_key, size_t pk_size);
    
private:
    bool available_;
    void* keychain_ref_;  // SecKeyRef for Secure Enclave access
};

/**
 * Sign a ciphertext for authenticity
 * 
 * Creates a signature over the ciphertext that can be verified
 * by anyone with the public key.
 * 
 * @param key_manager Key manager with signing key
 * @param key_handle Handle to signing key
 * @param ciphertext Ciphertext to sign
 * @param ct_size Ciphertext size
 * @param signature Output signature
 * @param sig_size Signature buffer size
 * @return Actual signature size
 */
size_t sign_ciphertext(SecureEnclaveKeyManager& key_manager,
                       SecureKeyHandle key_handle,
                       const uint64_t* ciphertext, size_t ct_size,
                       uint8_t* signature, size_t sig_size);

/**
 * Verify a ciphertext signature
 * 
 * @param ciphertext Ciphertext that was signed
 * @param ct_size Ciphertext size
 * @param signature Signature to verify
 * @param sig_size Signature size
 * @param public_key Signer's public key
 * @param pk_size Public key size
 * @return true if signature is valid
 */
bool verify_ciphertext_signature(const uint64_t* ciphertext, size_t ct_size,
                                  const uint8_t* signature, size_t sig_size,
                                  const uint8_t* public_key, size_t pk_size);

} // namespace secure_enclave
} // namespace fhe_accelerate
