//
// metal_shader_loader.mm
// Metal shader loading and management implementation
//

#include "metal_shader_loader.h"
#include <Foundation/Foundation.h>
#include <sys/stat.h>
#include <iostream>
#include <cmath>

namespace fhe_accelerate {

MetalShaderLoader::MetalShaderLoader(id<MTLDevice> device)
    : device_(device)
    , library_(nil)
    , library_path_()
    , pipeline_cache_()
    , hot_reload_enabled_(false)
    , last_modified_time_(0)
{
    if (device_ == nil) {
        throw std::runtime_error("Metal device is nil");
    }
}

MetalShaderLoader::~MetalShaderLoader() {
    clearPipelineCache();
    // ARC will automatically release library_
    library_ = nil;
}

bool MetalShaderLoader::loadDefaultLibrary() {
    auto path = findDefaultLibraryPath();
    if (!path.has_value()) {
        std::cerr << "Failed to find default metallib" << std::endl;
        return false;
    }
    return loadLibrary(path.value());
}

bool MetalShaderLoader::loadLibrary(const std::string& path) {
    // Clear existing library and cache
    // ARC will automatically release the old library_
    library_ = nil;
    clearPipelineCache();
    
    // Convert path to NSString
    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    
    // Check if file exists
    NSFileManager* fileManager = [NSFileManager defaultManager];
    if (![fileManager fileExistsAtPath:nsPath]) {
        std::cerr << "Metallib not found at path: " << path << std::endl;
        return false;
    }
    
    // Load the library
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:nsPath];
    library_ = [device_ newLibraryWithURL:url error:&error];
    
    if (library_ == nil || error != nil) {
        if (error != nil) {
            NSString* errorDesc = [error localizedDescription];
            std::cerr << "Failed to load metallib: " << [errorDesc UTF8String] << std::endl;
        }
        return false;
    }
    
    library_path_ = path;
    
    // Get file modification time for hot-reload
    struct stat file_stat;
    if (stat(path.c_str(), &file_stat) == 0) {
        last_modified_time_ = file_stat.st_mtime;
    }
    
    std::cout << "Loaded metallib from: " << path << std::endl;
    
    // List available functions
    NSArray<NSString*>* functionNames = [library_ functionNames];
    std::cout << "Available shader functions (" << [functionNames count] << "):" << std::endl;
    for (NSString* name in functionNames) {
        std::cout << "  - " << [name UTF8String] << std::endl;
    }
    
    return true;
}

id<MTLComputePipelineState> MetalShaderLoader::getPipelineState(const std::string& functionName) {
    // Check cache first
    auto it = pipeline_cache_.find(functionName);
    if (it != pipeline_cache_.end()) {
        return it->second;
    }
    
    // Create new pipeline state
    id<MTLComputePipelineState> pipeline = createPipelineState(functionName);
    if (pipeline != nil) {
        pipeline_cache_[functionName] = pipeline;
    }
    
    return pipeline;
}

bool MetalShaderLoader::hasFunction(const std::string& functionName) const {
    if (library_ == nil) {
        return false;
    }
    
    NSString* nsName = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> function = [library_ newFunctionWithName:nsName];
    bool exists = (function != nil);
    
    // ARC will automatically release function
    
    return exists;
}

void MetalShaderLoader::setHotReloadEnabled(bool enable) {
    hot_reload_enabled_ = enable;
    if (enable) {
        std::cout << "Hot-reload enabled for shaders" << std::endl;
    }
}

bool MetalShaderLoader::reload() {
    if (library_path_.empty()) {
        std::cerr << "Cannot reload: no library path set" << std::endl;
        return false;
    }
    
    // Check if file has been modified
    struct stat file_stat;
    if (stat(library_path_.c_str(), &file_stat) != 0) {
        std::cerr << "Cannot stat metallib file" << std::endl;
        return false;
    }
    
    if (file_stat.st_mtime <= last_modified_time_) {
        // File hasn't changed
        return true;
    }
    
    std::cout << "Reloading shaders from: " << library_path_ << std::endl;
    return loadLibrary(library_path_);
}

std::vector<std::string> MetalShaderLoader::listFunctions() const {
    std::vector<std::string> functions;
    
    if (library_ == nil) {
        return functions;
    }
    
    NSArray<NSString*>* functionNames = [library_ functionNames];
    for (NSString* name in functionNames) {
        functions.push_back([name UTF8String]);
    }
    
    return functions;
}

id<MTLComputePipelineState> MetalShaderLoader::createPipelineState(const std::string& functionName) {
    if (library_ == nil) {
        std::cerr << "Cannot create pipeline: library not loaded" << std::endl;
        return nil;
    }
    
    // Get the function
    NSString* nsName = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> function = [library_ newFunctionWithName:nsName];
    
    if (function == nil) {
        std::cerr << "Shader function not found: " << functionName << std::endl;
        return nil;
    }
    
    // Create pipeline state
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function
                                                                                   error:&error];
    
    // ARC will automatically release function
    
    if (pipeline == nil || error != nil) {
        if (error != nil) {
            NSString* errorDesc = [error localizedDescription];
            std::cerr << "Failed to create pipeline for " << functionName << ": "
                      << [errorDesc UTF8String] << std::endl;
        }
        return nil;
    }
    
    std::cout << "Created pipeline for: " << functionName << std::endl;
    std::cout << "  Max threads per threadgroup: " << [pipeline maxTotalThreadsPerThreadgroup] << std::endl;
    std::cout << "  Thread execution width: " << [pipeline threadExecutionWidth] << std::endl;
    
    return pipeline;
}

std::optional<std::string> MetalShaderLoader::findDefaultLibraryPath() const {
    // Search locations in order of preference:
    // 1. Environment variable FHE_METALLIB_PATH
    // 2. dist/shaders/fhe_shaders.metallib (build output)
    // 3. fhe_shaders.metallib (project root)
    // 4. Embedded in bundle (for packaged apps)
    
    // Check environment variable
    const char* env_path = std::getenv("FHE_METALLIB_PATH");
    if (env_path != nullptr) {
        NSString* nsPath = [NSString stringWithUTF8String:env_path];
        if ([[NSFileManager defaultManager] fileExistsAtPath:nsPath]) {
            return std::string(env_path);
        }
    }
    
    // Check dist/shaders
    NSString* distPath = @"dist/shaders/fhe_shaders.metallib";
    if ([[NSFileManager defaultManager] fileExistsAtPath:distPath]) {
        return std::string([distPath UTF8String]);
    }
    
    // Check project root
    NSString* rootPath = @"fhe_shaders.metallib";
    if ([[NSFileManager defaultManager] fileExistsAtPath:rootPath]) {
        return std::string([rootPath UTF8String]);
    }
    
    // Check bundle
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* bundlePath = [bundle pathForResource:@"fhe_shaders" ofType:@"metallib"];
    if (bundlePath != nil) {
        return std::string([bundlePath UTF8String]);
    }
    
    return std::nullopt;
}

void MetalShaderLoader::clearPipelineCache() {
    // ARC will automatically release all pipeline states when the map is cleared
    pipeline_cache_.clear();
}

// Helper functions

MTLSize getOptimalThreadgroupSize(id<MTLComputePipelineState> pipeline, size_t total_threads) {
    if (pipeline == nil) {
        return MTLSizeMake(1, 1, 1);
    }
    
    NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger thread_width = [pipeline threadExecutionWidth];
    
    // For 1D workloads, use multiples of thread execution width
    NSUInteger threads_x = std::min(total_threads, max_threads);
    
    // Round down to multiple of thread execution width
    threads_x = (threads_x / thread_width) * thread_width;
    
    // Ensure at least one thread execution width
    if (threads_x == 0) {
        threads_x = thread_width;
    }
    
    return MTLSizeMake(threads_x, 1, 1);
}

MTLSize computeGridSize(size_t total_threads, MTLSize threadgroup_size) {
    size_t threadgroup_threads = threadgroup_size.width * threadgroup_size.height * threadgroup_size.depth;
    
    if (threadgroup_threads == 0) {
        return MTLSizeMake(1, 1, 1);
    }
    
    // Compute number of threadgroups needed
    size_t num_groups = (total_threads + threadgroup_threads - 1) / threadgroup_threads;
    
    return MTLSizeMake(num_groups, 1, 1);
}

} // namespace fhe_accelerate
