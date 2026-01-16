#![deny(clippy::all)]

//! # node-fhe-accelerate
//!
//! High-performance Fully Homomorphic Encryption (FHE) acceleration library
//! optimized for Apple M4 Max hardware.
//!
//! This library provides native bindings to C++ FHE implementations with
//! hardware acceleration via SME, Metal GPU, NEON, and AMX.

use napi::bindgen_prelude::*;
use napi_derive::napi;

mod bridge;

/// Initialize the FHE accelerate library
/// 
/// This function performs one-time initialization including:
/// - Hardware capability detection (SME, Metal, NEON, AMX)
/// - Memory pool setup for unified memory
/// - Shader compilation for Metal backend
#[napi]
pub fn initialize() -> Result<()> {
    // Perform hardware detection to validate the environment
    let _caps = bridge::detect_hardware();
    
    // TODO: Initialize memory pools and compile shaders
    Ok(())
}

/// Get hardware capabilities available on this system
#[napi(object)]
pub struct HardwareCapabilities {
    pub has_sme: bool,
    pub has_metal: bool,
    pub has_neon: bool,
    pub has_amx: bool,
    pub metal_gpu_cores: u32,
    pub unified_memory_size: i64,  // Using i64 for napi compatibility
}

/// Modular arithmetic engine
#[napi]
pub struct ModularArithmetic {
    inner: cxx::UniquePtr<bridge::ffi::ModularArithmetic>,
}

#[napi]
impl ModularArithmetic {
    /// Create a new modular arithmetic engine with the given modulus
    #[napi(constructor)]
    pub fn new(modulus: i64) -> Result<Self> {
        if modulus <= 0 {
            return Err(Error::from_reason("Modulus must be positive"));
        }
        
        let inner = bridge::ffi::create_modular_arithmetic(modulus as u64);
        Ok(Self { inner })
    }
    
    /// Montgomery multiplication: (a * b * R^-1) mod q
    #[napi]
    pub fn montgomery_mul(&self, a: i64, b: i64) -> Result<i64> {
        if a < 0 || b < 0 {
            return Err(Error::from_reason("Inputs must be non-negative"));
        }
        
        let result = self.inner.montgomery_mul(a as u64, b as u64);
        Ok(result as i64)
    }
    
    /// Modular addition: (a + b) mod q
    #[napi]
    pub fn mod_add(&self, a: i64, b: i64) -> Result<i64> {
        if a < 0 || b < 0 {
            return Err(Error::from_reason("Inputs must be non-negative"));
        }
        
        let result = self.inner.mod_add(a as u64, b as u64);
        Ok(result as i64)
    }
    
    /// Modular subtraction: (a - b) mod q
    #[napi]
    pub fn mod_sub(&self, a: i64, b: i64) -> Result<i64> {
        if a < 0 || b < 0 {
            return Err(Error::from_reason("Inputs must be non-negative"));
        }
        
        let result = self.inner.mod_sub(a as u64, b as u64);
        Ok(result as i64)
    }
    
    /// Convert to Montgomery form: a * R mod q
    #[napi]
    pub fn to_montgomery(&self, a: i64) -> Result<i64> {
        if a < 0 {
            return Err(Error::from_reason("Input must be non-negative"));
        }
        
        let result = self.inner.to_montgomery(a as u64);
        Ok(result as i64)
    }
    
    /// Convert from Montgomery form: a * R^-1 mod q
    #[napi]
    pub fn from_montgomery(&self, a: i64) -> Result<i64> {
        if a < 0 {
            return Err(Error::from_reason("Input must be non-negative"));
        }
        
        let result = self.inner.from_montgomery(a as u64);
        Ok(result as i64)
    }
    
    /// Get the modulus
    #[napi]
    pub fn get_modulus(&self) -> i64 {
        self.inner.get_modulus() as i64
    }
}

/// Detect available hardware capabilities
#[napi]
pub fn detect_hardware() -> Result<HardwareCapabilities> {
    Ok(bridge::detect_hardware())
}

/// Version information
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
