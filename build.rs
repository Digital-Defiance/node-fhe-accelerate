use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get the target triple
    let target = env::var("TARGET").unwrap();
    
    println!("cargo:rerun-if-changed=src/native");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=cpp/shaders");
    
    // Only build C++ components on macOS ARM64
    if target == "aarch64-apple-darwin" {
        build_cpp_components();
        compile_metal_shaders();
    } else {
        println!("cargo:warning=This library is optimized for Apple Silicon (aarch64-apple-darwin)");
        println!("cargo:warning=Building on {} may result in reduced performance or missing features", target);
    }
    
    // Run napi-build
    napi_build::setup();
}

fn build_cpp_components() {
    // Get the project root directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = manifest_dir.join("cpp");
    let cpp_src_dir = cpp_dir.join("src");
    let cpp_include_dir = cpp_dir.join("include");
    
    // Ensure directories exist
    std::fs::create_dir_all(&cpp_src_dir).ok();
    std::fs::create_dir_all(&cpp_include_dir).ok();
    
    // Build C++ code with cxx
    let mut build = cxx_build::bridge("src/native/bridge.rs");
    
    // Add C++ source files (Objective-C++ for Metal framework)
    build
        .file(cpp_src_dir.join("hardware_detector.mm"))
        .file(cpp_src_dir.join("metal_shader_loader.mm"))
        .file(cpp_src_dir.join("modular_arithmetic.cpp"))
        .file(cpp_src_dir.join("bridge_modular.cpp"))
        .include(&cpp_include_dir)
        // C++ standard
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-stdlib=libc++")
        // ARM64 architecture flags for M4 Max
        .flag_if_supported("-march=armv8.6-a")
        // Optimization flags
        .flag_if_supported("-O3")
        .flag_if_supported("-ffast-math")
        // Objective-C++ specific
        .flag_if_supported("-fobjc-arc");
    
    // Compile the bridge
    build.compile("fhe_accelerate_cpp");
    
    // Link against Apple frameworks
    println!("cargo:rustc-link-arg=-framework");
    println!("cargo:rustc-link-arg=Foundation");
    println!("cargo:rustc-link-arg=-framework");
    println!("cargo:rustc-link-arg=Metal");
    println!("cargo:rustc-link-arg=-framework");
    println!("cargo:rustc-link-arg=Accelerate");
    println!("cargo:rustc-link-arg=-framework");
    println!("cargo:rustc-link-arg=CoreFoundation");
    
    // Link against Objective-C runtime
    println!("cargo:rustc-link-lib=objc");
    
    // Link against C++ standard library
    println!("cargo:rustc-link-lib=c++");
    
    // Set up library search paths
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=framework=/System/Library/Frameworks");
}

fn compile_metal_shaders() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_script = manifest_dir.join("scripts/compile-shaders.sh");
    
    // Check if shader script exists
    if !shader_script.exists() {
        println!("cargo:warning=Shader compilation script not found at {:?}", shader_script);
        return;
    }
    
    // Determine build mode
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let build_mode = if profile == "release" { "release" } else { "debug" };
    
    println!("cargo:warning=Compiling Metal shaders in {} mode", build_mode);
    
    // Run shader compilation script
    let output = Command::new("bash")
        .arg(&shader_script)
        .arg(build_mode)
        .output();
    
    match output {
        Ok(output) => {
            if output.status.success() {
                println!("cargo:warning=Metal shaders compiled successfully");
                
                // Print stdout for visibility
                if !output.stdout.is_empty() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        println!("cargo:warning=[SHADER] {}", line);
                    }
                }
            } else {
                println!("cargo:warning=Metal shader compilation failed");
                
                // Print stderr
                if !output.stderr.is_empty() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    for line in stderr.lines() {
                        println!("cargo:warning=[SHADER ERROR] {}", line);
                    }
                }
                
                // Don't fail the build, just warn
                println!("cargo:warning=Continuing build without compiled shaders");
            }
        }
        Err(e) => {
            println!("cargo:warning=Failed to execute shader compilation script: {}", e);
            println!("cargo:warning=Continuing build without compiled shaders");
        }
    }
    
    // Tell cargo where to find the compiled metallib
    let metallib_path = manifest_dir.join("dist/shaders/fhe_shaders.metallib");
    if metallib_path.exists() {
        println!("cargo:rustc-env=FHE_METALLIB_PATH={}", metallib_path.display());
    }
}
