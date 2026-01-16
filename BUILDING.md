# Building @digitaldefiance/node-fhe-accelerate

This document provides detailed instructions for building the FHE acceleration library from source.

## Prerequisites

### Required Software

1. **macOS** (Sequoia 15.2 or later recommended)
   - This library is specifically optimized for Apple Silicon (arm64)
   - Intel Macs are not supported

2. **Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

3. **Rust Toolchain** (1.70 or later)
   ```bash
   # Install Rust via rustup
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Add arm64 macOS target
   rustup target add aarch64-apple-darwin
   
   # Verify installation
   cargo --version
   rustc --version
   ```

4. **Node.js** (18.0.0 or later)
   ```bash
   # Using Homebrew
   brew install node
   
   # Verify installation
   node --version  # Should be >= 18.0.0
   ```

5. **Yarn** (4.0.0 or later)
   ```bash
   # Enable corepack (comes with Node.js 16.10+)
   corepack enable
   
   # Set Yarn version
   yarn set version stable
   
   # Verify installation
   yarn --version  # Should be >= 4.0.0
   ```

### Hardware Requirements

- **Apple Silicon Mac** (M1, M2, M3, or M4 series)
- **Minimum 8 GB RAM** (16 GB or more recommended for development)
- **5 GB free disk space** (for dependencies and build artifacts)

## Build Process

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Digital-Defiance/node-fhe-accelerate.git
cd node-fhe-accelerate

# Install Node.js dependencies
yarn install
```

### 2. Build the Project

#### Full Build (TypeScript + Native + Shaders)

```bash
yarn build
```

This command:
1. Compiles Metal shaders to `.metallib` format
2. Compiles TypeScript to JavaScript (CommonJS, ESM, and type definitions)
3. Compiles Rust code with napi-rs bindings
4. Compiles C++ code with hardware acceleration support
5. Links against Apple frameworks (Metal, Accelerate, Foundation)
6. Produces optimized `.node` binary

#### Build Metal Shaders Only

```bash
# Release build (optimized shaders)
yarn build:shaders

# Debug build (with debug symbols and source info)
yarn build:shaders:debug

# Watch mode (auto-recompile on changes)
yarn build:shaders:watch
```

The shader compilation process:
1. Compiles each `.metal` file to `.air` (Apple Intermediate Representation)
2. Links all `.air` files into a single `fhe_shaders.metallib`
3. Generates a manifest file with shader metadata
4. Copies the metallib to the appropriate locations

#### Build Native Addon Only

```bash
# Debug build (faster compilation, includes debug symbols)
yarn build:debug

# Release build (optimized, production-ready)
yarn build:native
```

#### Build TypeScript Only

```bash
yarn build:ts
```

### 3. Test the Build

```bash
# Run all tests
yarn test

# Test native addon specifically
yarn test:native

# Run tests in watch mode
yarn test:watch
```

## Build Artifacts

After a successful build, you'll find:

```
.
├── index.node                    # Native addon binary (debug or release)
├── fhe_shaders.metallib          # Compiled Metal shaders (release builds)
├── dist/
│   ├── cjs/                      # CommonJS output
│   │   └── index.js
│   ├── esm/                      # ES Module output
│   │   └── index.js
│   ├── types/                    # TypeScript type definitions
│   │   └── index.d.ts
│   └── shaders/                  # Shader build artifacts
│       ├── fhe_shaders.metallib  # Compiled shader library
│       └── shader_manifest.json  # Shader metadata
└── target/
    ├── shaders/                  # Intermediate shader build files (.air)
    └── aarch64-apple-darwin/
        ├── debug/                # Debug build artifacts
        └── release/              # Release build artifacts
```

## Build Configuration

### Cargo.toml

Rust package configuration:
- **Dependencies**: napi-rs, cxx, tokio, serde
- **Crate type**: `cdylib` (dynamic library for Node.js)
- **Optimization**: LTO enabled, single codegen unit for release builds

### build.rs

Custom build script that:
- Detects target platform (must be `aarch64-apple-darwin`)
- Compiles C++ code using `cxx-build`
- Links against Apple frameworks
- Sets up compiler flags for ARM64 optimization

### napi.config.json

napi-rs build configuration:
- **Target**: `aarch64-apple-darwin`
- **Binary name**: `node-fhe-accelerate`
- **Package manager**: yarn

### .cargo/config.toml

Rust compiler configuration:
- **Default target**: `aarch64-apple-darwin`
- **Target CPU**: `native` (uses all available CPU features)
- **Target features**: NEON, CRC

## Compiler Flags

### Rust Flags

- `-C target-cpu=native`: Optimize for the build machine's CPU
- `-C target-feature=+neon,+crc`: Enable NEON SIMD and CRC instructions

### C++ Flags

- `-std=c++17`: C++17 standard
- `-stdlib=libc++`: Use libc++ standard library
- `-march=armv8.6-a`: Target ARMv8.6-A architecture (M4 Max)
- `-O3`: Maximum optimization
- `-ffast-math`: Aggressive floating-point optimizations
- `-fobjc-arc`: Automatic Reference Counting for Objective-C++

### Linked Frameworks

- **Foundation**: Core macOS APIs
- **Metal**: GPU compute shaders and shader compilation
- **Accelerate**: BLAS, LAPACK, vDSP (AMX access)
- **CoreFoundation**: Core Foundation APIs

## Metal Shader Development

### Shader Directory Structure

Metal compute shaders are located in `cpp/shaders/`:
- `common/` - Shared utilities and type definitions
- `ntt/` - Number Theoretic Transform kernels
- `modular/` - Modular arithmetic kernels
- `bootstrap/` - Bootstrapping operation kernels

### Shader Compilation Pipeline

The shader compilation process uses Apple's Metal compiler toolchain:

1. **Compilation**: `.metal` → `.air` (Apple Intermediate Representation)
   ```bash
   xcrun -sdk macosx metal -std=metal3.0 -O3 -c shader.metal -o shader.air
   ```

2. **Linking**: Multiple `.air` → Single `.metallib`
   ```bash
   xcrun -sdk macosx metallib shader1.air shader2.air -o library.metallib
   ```

3. **Loading**: Runtime loading via `MTLDevice.newLibrary()`

### Hot-Reloading (Development)

For rapid shader iteration:

```bash
# Enable hot-reload
export FHE_SHADER_HOT_RELOAD=1

# Build in debug mode with shader watching
yarn build:shaders:watch

# In another terminal, run your application
yarn build:debug
node your_test.js
```

Shaders will automatically recompile when you save changes. Requires `fswatch`:
```bash
brew install fswatch
```

### Shader Debugging

1. **Compilation Errors**: Shown during `yarn build:shaders`
2. **Runtime Errors**: Use Xcode's Metal debugger
3. **Performance**: Profile with Instruments GPU template

See `cpp/shaders/DEVELOPMENT.md` for detailed shader development guide.

## Troubleshooting

### Build Fails with "cxx bridge compilation failed"

**Cause**: C++ compiler or Xcode Command Line Tools not installed

**Solution**:
```bash
xcode-select --install
```

### Build Fails with "Rust target not found"

**Cause**: ARM64 macOS target not installed

**Solution**:
```bash
rustup target add aarch64-apple-darwin
```

### Build Fails with "Framework not found"

**Cause**: Building on non-macOS platform or missing frameworks

**Solution**: This library only builds on macOS. Ensure you're on macOS and have Xcode Command Line Tools installed.

### Runtime Error: "Cannot find module '.node'"

**Cause**: Native addon not built

**Solution**:
```bash
yarn build:native
```

### Runtime Error: "Symbol not found"

**Cause**: Stale build artifacts or linking issues

**Solution**:
```bash
yarn clean
yarn build
```

### Slow Compilation

**Cause**: Debug build or first-time compilation

**Solutions**:
- Use `yarn build:debug` for faster debug builds
- Subsequent builds will be faster due to incremental compilation
- Use `cargo build --release` for optimized builds (slower but produces faster binaries)

## Development Workflow

### Incremental Builds

For faster development iteration:

```bash
# Watch TypeScript files and rebuild on changes
yarn build:ts --watch

# For native code changes, rebuild manually
yarn build:debug
```

### Cleaning Build Artifacts

```bash
# Clean all build artifacts
yarn clean

# Clean Rust build artifacts only
cargo clean
```

### Debugging

#### Debug Build

```bash
yarn build:debug
```

Debug builds include:
- Debug symbols for LLDB/GDB
- Assertions enabled
- No optimizations
- Detailed error messages

#### Using LLDB

```bash
# Run with LLDB debugger
lldb -- node test-native.js

# Set breakpoints in C++ code
(lldb) breakpoint set --name fhe_accelerate::detect
(lldb) run
```

## Performance Optimization

### Release Builds

Release builds are highly optimized:
- Link-Time Optimization (LTO)
- Single codegen unit for maximum optimization
- Target CPU native features
- Aggressive math optimizations

### Profiling

Use Xcode Instruments to profile:

```bash
# Build with release optimizations
yarn build:native

# Profile with Instruments
instruments -t "Time Profiler" node your_script.js
```

## Continuous Integration

For CI/CD pipelines:

```bash
# Install dependencies
yarn install --immutable

# Build and test
yarn build
yarn test

# Verify native addon works
yarn test:native
```

## Platform Support

### Supported

- ✅ macOS arm64 (Apple Silicon M1/M2/M3/M4)

### Not Supported

- ❌ macOS x86_64 (Intel Macs)
- ❌ Linux (any architecture)
- ❌ Windows (any architecture)

This library is specifically designed and optimized for Apple Silicon hardware and will not build or run on other platforms.

## Additional Resources

- [napi-rs Documentation](https://napi.rs/)
- [Rust Book](https://doc.rust-lang.org/book/)
- [cxx Documentation](https://cxx.rs/)
- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [ARM Architecture Reference](https://developer.arm.com/documentation/)

## Getting Help

If you encounter build issues:

1. Check this document for troubleshooting steps
2. Ensure all prerequisites are installed and up to date
3. Try cleaning and rebuilding: `yarn clean && yarn build`
4. Open an issue on GitHub with:
   - Your macOS version
   - Your hardware (M1/M2/M3/M4)
   - Complete error output
   - Steps to reproduce
