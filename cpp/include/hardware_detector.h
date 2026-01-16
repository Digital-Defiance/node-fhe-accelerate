#pragma once

#include "fhe_types.h"

namespace fhe_accelerate {

/// Hardware capability detection
class HardwareDetector {
public:
    /// Detect all available hardware capabilities
    static HardwareCapabilities detect();
    
    /// Check if SME (Scalable Matrix Extension) is available
    static bool has_sme();
    
    /// Check if Metal GPU is available
    static bool has_metal();
    
    /// Check if NEON SIMD is available (always true on ARM64)
    static bool has_neon();
    
    /// Check if AMX (Apple Matrix Coprocessor) is available
    static bool has_amx();
    
    /// Get number of GPU cores (Metal)
    static uint32_t get_metal_gpu_cores();
    
    /// Get unified memory size in bytes
    static uint64_t get_unified_memory_size();
};

// Free functions for cxx bridge
HardwareCapabilities detect();
bool has_sme();
bool has_metal();
bool has_neon();
bool has_amx();
uint32_t get_metal_gpu_cores();
uint64_t get_unified_memory_size();

} // namespace fhe_accelerate
