#include "hardware_detector.h"
#include <sys/sysctl.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_MAC
#include <Metal/Metal.h>
#endif
#endif

namespace fhe_accelerate {

// C++ implementation of HardwareDetector methods
HardwareCapabilities HardwareDetector::detect() {
    HardwareCapabilities caps;
    caps.has_sme = has_sme();
    caps.has_metal = has_metal();
    caps.has_neon = has_neon();
    caps.has_amx = has_amx();
    caps.metal_gpu_cores = get_metal_gpu_cores();
    caps.unified_memory_size = get_unified_memory_size();
    return caps;
}

bool HardwareDetector::has_sme() {
    // Check for SME support via sysctl
    // SME is available on M4 and later
    #ifdef __aarch64__
    int has_feat_sme = 0;
    size_t size = sizeof(has_feat_sme);
    
    // Try to query hw.optional.arm.FEAT_SME
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &has_feat_sme, &size, nullptr, 0) == 0) {
        return has_feat_sme != 0;
    }
    #endif
    return false;
}

bool HardwareDetector::has_metal() {
    #ifdef __APPLE__
    #if TARGET_OS_MAC
    // Check if Metal is available
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
    #endif
    #endif
    return false;
}

bool HardwareDetector::has_neon() {
    // NEON is always available on ARM64
    #ifdef __aarch64__
    return true;
    #else
    return false;
    #endif
}

bool HardwareDetector::has_amx() {
    // AMX is available on M1 and later
    #ifdef __aarch64__
    int has_amx = 0;
    size_t size = sizeof(has_amx);
    
    // Try to query hw.optional.arm.FEAT_AMX
    if (sysctlbyname("hw.optional.amx_version", &has_amx, &size, nullptr, 0) == 0) {
        return has_amx != 0;
    }
    #endif
    return false;
}

uint32_t HardwareDetector::get_metal_gpu_cores() {
    #ifdef __APPLE__
    #if TARGET_OS_MAC
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device != nil) {
            // M4 Max has 40 GPU cores
            // This is a simplified detection - in practice, we'd query device properties
            return 40; // Placeholder - actual detection would be more sophisticated
        }
    }
    #endif
    #endif
    return 0;
}

uint64_t HardwareDetector::get_unified_memory_size() {
    #ifdef __APPLE__
    uint64_t memsize = 0;
    size_t size = sizeof(memsize);
    
    if (sysctlbyname("hw.memsize", &memsize, &size, nullptr, 0) == 0) {
        return memsize;
    }
    #endif
    return 0;
}

// Wrapper functions for cxx bridge (must be free functions in the namespace)
HardwareCapabilities detect() {
    return HardwareDetector::detect();
}

bool has_sme() {
    return HardwareDetector::has_sme();
}

bool has_metal() {
    return HardwareDetector::has_metal();
}

bool has_neon() {
    return HardwareDetector::has_neon();
}

bool has_amx() {
    return HardwareDetector::has_amx();
}

uint32_t get_metal_gpu_cores() {
    return HardwareDetector::get_metal_gpu_cores();
}

uint64_t get_unified_memory_size() {
    return HardwareDetector::get_unified_memory_size();
}

} // namespace fhe_accelerate
