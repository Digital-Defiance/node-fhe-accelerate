//
// metal_shader_loader.h
// Metal shader loading and management
//
// This header provides utilities for loading compiled Metal shaders (.metallib)
// and creating compute pipeline states for FHE operations.
//

#ifndef FHE_ACCELERATE_METAL_SHADER_LOADER_H
#define FHE_ACCELERATE_METAL_SHADER_LOADER_H

#ifdef __APPLE__

#include <Metal/Metal.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>

namespace fhe_accelerate {

/// Metal shader loader and pipeline manager
///
/// This class handles loading the compiled metallib, creating compute pipeline states,
/// and managing shader hot-reloading in development mode.
class MetalShaderLoader {
public:
    /// Constructor
    /// @param device Metal device to use
    explicit MetalShaderLoader(id<MTLDevice> device);
    
    /// Destructor
    ~MetalShaderLoader();
    
    // Disable copy and move
    MetalShaderLoader(const MetalShaderLoader&) = delete;
    MetalShaderLoader& operator=(const MetalShaderLoader&) = delete;
    MetalShaderLoader(MetalShaderLoader&&) = delete;
    MetalShaderLoader& operator=(MetalShaderLoader&&) = delete;
    
    /// Load the metallib from the default location
    /// @return true if successful, false otherwise
    bool loadDefaultLibrary();
    
    /// Load a metallib from a specific path
    /// @param path Path to the .metallib file
    /// @return true if successful, false otherwise
    bool loadLibrary(const std::string& path);
    
    /// Get a compute pipeline state for a shader function
    /// @param functionName Name of the kernel function
    /// @return Pipeline state, or nullptr if not found
    id<MTLComputePipelineState> getPipelineState(const std::string& functionName);
    
    /// Check if a shader function exists
    /// @param functionName Name of the kernel function
    /// @return true if the function exists
    bool hasFunction(const std::string& functionName) const;
    
    /// Get the Metal device
    /// @return Metal device
    id<MTLDevice> getDevice() const { return device_; }
    
    /// Get the loaded library
    /// @return Metal library, or nullptr if not loaded
    id<MTLLibrary> getLibrary() const { return library_; }
    
    /// Enable hot-reloading (development only)
    /// @param enable true to enable, false to disable
    void setHotReloadEnabled(bool enable);
    
    /// Check if hot-reloading is enabled
    /// @return true if enabled
    bool isHotReloadEnabled() const { return hot_reload_enabled_; }
    
    /// Reload the shader library (for hot-reloading)
    /// @return true if successful
    bool reload();
    
    /// Get the path to the currently loaded metallib
    /// @return Path to metallib, or empty string if not loaded
    std::string getLibraryPath() const { return library_path_; }
    
    /// List all available shader functions
    /// @return Vector of function names
    std::vector<std::string> listFunctions() const;
    
private:
    /// Create a compute pipeline state for a function
    /// @param functionName Name of the kernel function
    /// @return Pipeline state, or nullptr on error
    id<MTLComputePipelineState> createPipelineState(const std::string& functionName);
    
    /// Find the default metallib path
    /// @return Path to metallib, or empty optional if not found
    std::optional<std::string> findDefaultLibraryPath() const;
    
    /// Clear all cached pipeline states
    void clearPipelineCache();
    
    id<MTLDevice> device_;
    id<MTLLibrary> library_;
    std::string library_path_;
    
    // Cache of compiled pipeline states
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache_;
    
    // Hot-reload support
    bool hot_reload_enabled_;
    time_t last_modified_time_;
};

/// Shader function names (constants for type safety)
namespace ShaderFunctions {
    // NTT shaders
    constexpr const char* NTT_FORWARD_STAGE = "ntt_forward_stage";
    constexpr const char* NTT_FORWARD_BATCH = "ntt_forward_batch";
    constexpr const char* NTT_BIT_REVERSE = "ntt_bit_reverse";
    constexpr const char* NTT_INVERSE_STAGE = "ntt_inverse_stage";
    constexpr const char* NTT_INVERSE_BATCH = "ntt_inverse_batch";
    
    // Modular arithmetic shaders
    constexpr const char* MODMUL_BATCH = "modmul_batch";
    constexpr const char* MODADD_BATCH = "modadd_batch";
    constexpr const char* MODSUB_BATCH = "modsub_batch";
    
    // Bootstrap shaders
    constexpr const char* BOOTSTRAP_ACCUMULATOR = "bootstrap_accumulator";
    constexpr const char* BLIND_ROTATE = "blind_rotate";
    constexpr const char* SAMPLE_EXTRACT = "sample_extract";
}

/// Helper function to get the optimal threadgroup size for a kernel
/// @param pipeline Pipeline state
/// @param total_threads Total number of threads needed
/// @return Optimal threadgroup size
MTLSize getOptimalThreadgroupSize(id<MTLComputePipelineState> pipeline, size_t total_threads);

/// Helper function to compute grid size from total threads and threadgroup size
/// @param total_threads Total number of threads needed
/// @param threadgroup_size Threadgroup size
/// @return Grid size (number of threadgroups)
MTLSize computeGridSize(size_t total_threads, MTLSize threadgroup_size);

} // namespace fhe_accelerate

#endif // __APPLE__

#endif // FHE_ACCELERATE_METAL_SHADER_LOADER_H
