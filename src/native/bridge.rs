//! Bridge between Rust and C++ code using cxx

#[cxx::bridge(namespace = "fhe_accelerate")]
pub mod ffi {
    /// Hardware capabilities from C++
    struct HardwareCapabilities {
        has_sme: bool,
        has_metal: bool,
        has_neon: bool,
        has_amx: bool,
        metal_gpu_cores: u32,
        unified_memory_size: u64,
    }

    unsafe extern "C++" {
        include!("hardware_detector.h");
        include!("modular_arithmetic.h");

        type HardwareCapabilities;
        type ModularArithmetic;

        /// Detect hardware capabilities
        fn detect() -> HardwareCapabilities;
        
        /// Check for specific hardware features
        fn has_sme() -> bool;
        fn has_metal() -> bool;
        fn has_neon() -> bool;
        fn has_amx() -> bool;
        fn get_metal_gpu_cores() -> u32;
        fn get_unified_memory_size() -> u64;
        
        /// Modular arithmetic operations
        fn create_modular_arithmetic(modulus: u64) -> UniquePtr<ModularArithmetic>;
        fn montgomery_mul(self: &ModularArithmetic, a: u64, b: u64) -> u64;
        fn mod_add(self: &ModularArithmetic, a: u64, b: u64) -> u64;
        fn mod_sub(self: &ModularArithmetic, a: u64, b: u64) -> u64;
        fn to_montgomery(self: &ModularArithmetic, a: u64) -> u64;
        fn from_montgomery(self: &ModularArithmetic, a: u64) -> u64;
        fn get_modulus(self: &ModularArithmetic) -> u64;
    }
}

/// Convert C++ HardwareCapabilities to Rust struct
pub fn detect_hardware() -> crate::HardwareCapabilities {
    let cpp_caps = ffi::detect();
    
    crate::HardwareCapabilities {
        has_sme: cpp_caps.has_sme,
        has_metal: cpp_caps.has_metal,
        has_neon: cpp_caps.has_neon,
        has_amx: cpp_caps.has_amx,
        metal_gpu_cores: cpp_caps.metal_gpu_cores,
        unified_memory_size: cpp_caps.unified_memory_size as i64,  // Convert to i64 for napi
    }
}
