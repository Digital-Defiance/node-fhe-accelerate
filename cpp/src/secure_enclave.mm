/**
 * Secure Enclave Integration Implementation
 * 
 * Uses Apple's Security framework to access Secure Enclave.
 * 
 * Requirements 14.33, 14.34, 14.35
 */

#include "secure_enclave.h"
#include <iostream>
#include <cstring>

#ifdef __APPLE__
#include <Security/Security.h>
#include <LocalAuthentication/LocalAuthentication.h>
#endif

namespace fhe_accelerate {
namespace secure_enclave {

bool secure_enclave_available() {
#ifdef __APPLE__
    // Check if Secure Enclave is available
    // All Apple Silicon Macs have Secure Enclave
    SecAccessControlRef access = SecAccessControlCreateWithFlags(
        kCFAllocatorDefault,
        kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        kSecAccessControlPrivateKeyUsage,
        nullptr
    );
    
    if (access) {
        CFRelease(access);
        return true;
    }
#endif
    return false;
}

// ============================================================================
// SecureEnclaveKeyManager Implementation
// ============================================================================

SecureEnclaveKeyManager::SecureEnclaveKeyManager()
    : available_(false)
    , keychain_ref_(nullptr)
{
    available_ = secure_enclave_available();
    
    if (available_) {
        std::cout << "Secure Enclave: Available\n";
    }
}

SecureEnclaveKeyManager::~SecureEnclaveKeyManager() {
    // Cleanup keychain references
}

SecureKeyHandle SecureEnclaveKeyManager::generate_secret_key(size_t key_size) {
    if (!available_) {
        std::cerr << "Secure Enclave not available\n";
        return INVALID_KEY_HANDLE;
    }
    
#ifdef __APPLE__
    // Create access control for Secure Enclave
    CFErrorRef error = nullptr;
    SecAccessControlRef access = SecAccessControlCreateWithFlags(
        kCFAllocatorDefault,
        kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        kSecAccessControlPrivateKeyUsage,
        &error
    );
    
    if (error || !access) {
        std::cerr << "Failed to create access control\n";
        return INVALID_KEY_HANDLE;
    }
    
    // Key attributes for Secure Enclave
    NSDictionary* attributes = @{
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
        (id)kSecAttrKeySizeInBits: @256,
        (id)kSecAttrTokenID: (id)kSecAttrTokenIDSecureEnclave,
        (id)kSecPrivateKeyAttrs: @{
            (id)kSecAttrIsPermanent: @YES,
            (id)kSecAttrAccessControl: (__bridge id)access,
        },
    };
    
    // Generate key in Secure Enclave
    SecKeyRef privateKey = SecKeyCreateRandomKey(
        (__bridge CFDictionaryRef)attributes,
        &error
    );
    
    CFRelease(access);
    
    if (error || !privateKey) {
        std::cerr << "Failed to generate key in Secure Enclave\n";
        return INVALID_KEY_HANDLE;
    }
    
    // Return handle (pointer cast to uint64_t)
    return reinterpret_cast<SecureKeyHandle>(privateKey);
#else
    return INVALID_KEY_HANDLE;
#endif
}

bool SecureEnclaveKeyManager::delete_key(SecureKeyHandle handle) {
    if (handle == INVALID_KEY_HANDLE) return false;
    
#ifdef __APPLE__
    SecKeyRef key = reinterpret_cast<SecKeyRef>(handle);
    CFRelease(key);
    return true;
#else
    return false;
#endif
}

size_t SecureEnclaveKeyManager::decrypt_in_enclave(SecureKeyHandle handle,
                                                    const uint8_t* ciphertext, size_t ct_size,
                                                    uint8_t* plaintext, size_t pt_size) {
    if (!available_ || handle == INVALID_KEY_HANDLE) {
        return 0;
    }
    
#ifdef __APPLE__
    SecKeyRef privateKey = reinterpret_cast<SecKeyRef>(handle);
    
    // Create CFData from ciphertext
    CFDataRef cipherData = CFDataCreate(kCFAllocatorDefault, ciphertext, ct_size);
    if (!cipherData) return 0;
    
    CFErrorRef error = nullptr;
    CFDataRef plainData = SecKeyCreateDecryptedData(
        privateKey,
        kSecKeyAlgorithmECIESEncryptionStandardX963SHA256AESGCM,
        cipherData,
        &error
    );
    
    CFRelease(cipherData);
    
    if (error || !plainData) {
        return 0;
    }
    
    CFIndex plainLen = CFDataGetLength(plainData);
    if (static_cast<size_t>(plainLen) > pt_size) {
        CFRelease(plainData);
        return 0;
    }
    
    CFDataGetBytes(plainData, CFRangeMake(0, plainLen), plaintext);
    CFRelease(plainData);
    
    return static_cast<size_t>(plainLen);
#else
    return 0;
#endif
}

size_t SecureEnclaveKeyManager::sign_data(SecureKeyHandle handle,
                                           const uint8_t* data, size_t data_size,
                                           uint8_t* signature, size_t sig_size) {
    if (!available_ || handle == INVALID_KEY_HANDLE) {
        return 0;
    }
    
#ifdef __APPLE__
    SecKeyRef privateKey = reinterpret_cast<SecKeyRef>(handle);
    
    CFDataRef dataRef = CFDataCreate(kCFAllocatorDefault, data, data_size);
    if (!dataRef) return 0;
    
    CFErrorRef error = nullptr;
    CFDataRef signatureRef = SecKeyCreateSignature(
        privateKey,
        kSecKeyAlgorithmECDSASignatureMessageX962SHA256,
        dataRef,
        &error
    );
    
    CFRelease(dataRef);
    
    if (error || !signatureRef) {
        return 0;
    }
    
    CFIndex sigLen = CFDataGetLength(signatureRef);
    if (static_cast<size_t>(sigLen) > sig_size) {
        CFRelease(signatureRef);
        return 0;
    }
    
    CFDataGetBytes(signatureRef, CFRangeMake(0, sigLen), signature);
    CFRelease(signatureRef);
    
    return static_cast<size_t>(sigLen);
#else
    return 0;
#endif
}

bool SecureEnclaveKeyManager::verify_signature(const uint8_t* data, size_t data_size,
                                                const uint8_t* signature, size_t sig_size,
                                                const uint8_t* public_key, size_t pk_size) {
#ifdef __APPLE__
    // Create public key from data
    CFDataRef pkData = CFDataCreate(kCFAllocatorDefault, public_key, pk_size);
    if (!pkData) return false;
    
    NSDictionary* attributes = @{
        (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
        (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPublic,
    };
    
    CFErrorRef error = nullptr;
    SecKeyRef publicKey = SecKeyCreateWithData(
        pkData,
        (__bridge CFDictionaryRef)attributes,
        &error
    );
    
    CFRelease(pkData);
    
    if (error || !publicKey) {
        return false;
    }
    
    CFDataRef dataRef = CFDataCreate(kCFAllocatorDefault, data, data_size);
    CFDataRef sigRef = CFDataCreate(kCFAllocatorDefault, signature, sig_size);
    
    bool valid = SecKeyVerifySignature(
        publicKey,
        kSecKeyAlgorithmECDSASignatureMessageX962SHA256,
        dataRef,
        sigRef,
        &error
    );
    
    CFRelease(dataRef);
    CFRelease(sigRef);
    CFRelease(publicKey);
    
    return valid && !error;
#else
    return false;
#endif
}

size_t SecureEnclaveKeyManager::get_public_key(SecureKeyHandle handle,
                                                uint8_t* public_key, size_t pk_size) {
    if (!available_ || handle == INVALID_KEY_HANDLE) {
        return 0;
    }
    
#ifdef __APPLE__
    SecKeyRef privateKey = reinterpret_cast<SecKeyRef>(handle);
    SecKeyRef publicKey = SecKeyCopyPublicKey(privateKey);
    
    if (!publicKey) return 0;
    
    CFErrorRef error = nullptr;
    CFDataRef pkData = SecKeyCopyExternalRepresentation(publicKey, &error);
    
    CFRelease(publicKey);
    
    if (error || !pkData) {
        return 0;
    }
    
    CFIndex pkLen = CFDataGetLength(pkData);
    if (static_cast<size_t>(pkLen) > pk_size) {
        CFRelease(pkData);
        return 0;
    }
    
    CFDataGetBytes(pkData, CFRangeMake(0, pkLen), public_key);
    CFRelease(pkData);
    
    return static_cast<size_t>(pkLen);
#else
    return 0;
#endif
}

// ============================================================================
// Ciphertext Signing Functions
// ============================================================================

size_t sign_ciphertext(SecureEnclaveKeyManager& key_manager,
                       SecureKeyHandle key_handle,
                       const uint64_t* ciphertext, size_t ct_size,
                       uint8_t* signature, size_t sig_size) {
    // Convert ciphertext to bytes
    const uint8_t* ct_bytes = reinterpret_cast<const uint8_t*>(ciphertext);
    size_t ct_bytes_size = ct_size * sizeof(uint64_t);
    
    return key_manager.sign_data(key_handle, ct_bytes, ct_bytes_size, signature, sig_size);
}

bool verify_ciphertext_signature(const uint64_t* ciphertext, size_t ct_size,
                                  const uint8_t* signature, size_t sig_size,
                                  const uint8_t* public_key, size_t pk_size) {
    SecureEnclaveKeyManager manager;
    
    const uint8_t* ct_bytes = reinterpret_cast<const uint8_t*>(ciphertext);
    size_t ct_bytes_size = ct_size * sizeof(uint64_t);
    
    return manager.verify_signature(ct_bytes, ct_bytes_size, signature, sig_size, public_key, pk_size);
}

} // namespace secure_enclave
} // namespace fhe_accelerate
