# Native Addon Structure

This document describes the native addon structure for `@digitaldefiance/node-fhe-accelerate`.

## Architecture

The native addon uses **napi-rs** to provide Node.js bindings to a C++ implementation of FHE primitives.

### Directory Structure

```
.
├── Cargo.toml              # Rust package configuration
├── build.rs                # Build script for C++ compilation
├── .cargo/
│   └── config.toml         # Cargo build configuration
├── src/
│   └── native/
│       ├── lib.rs          # Main Rust entry point (napi-rs bindings)
│       └── bridge.rs       # Rust-C++ bridge using cxx
└── cpp/
    ├── include/            # C++ header files
    │   ├── fhe_types.h     # Core FHE type definitions
    │   └── hardware_detector.h
    └── src/                # C++ implementation files
        └── hardware_detector.cpp
```

## Technology Stack

### Rust Layer (napi-rs)
- **Purpose**: Provides Node.js bindings with minimal overhead
- **Library**: napi-rs v2 with async support
- **Features**: 
  - Zero-copy buffer sharing
  - Async operations via Tokio
  - TypeScript type generation

### C++ Layer
- **Purpose**: Core FHE implementation with hardware acceleration
- **Standard**: C++17
- **Frameworks**:
  - Metal (GPU compute)
  - Accelerate (AMX, BLAS)
  - ARM intrinsics (NEON, SME)

### Bridge (cxx)
- **Purpose**: Safe Rust-C++ interop
- **Library**: cxx v1.0
- **Features**:
  - Type-safe FFI
  - Automatic binding generation
  - Zero-cost abstractions

## Build Process

### Prerequisites
- Rust toolchain (1.70+)
- Xcode Command Line Tools
- Node.js 18+
- Yarn 4.0+

### Build Commands

```bash
# Build TypeScript and native code
yarn build

# Build only native addon (debug)
yarn build:debug

# Build only native addon (release)
yarn build:native

# Clean build artifacts
yarn clean
```

### Build Steps

1. **Cargo Build**: Compiles Rust code
2. **cxx-build**: Generates C++ bindings and compiles C++ code
3. **Linking**: Links against macOS frameworks (Metal, Accelerate, Foundation)
4. **napi-rs**: Generates TypeScript type definitions
5. **Output**: Produces `.node` binary in `native/` directory

## Hardware Detection

The native addon detects available hardware capabilities at runtime:

- **SME (Scalable Matrix Extension)**: M4 and later
- **Metal GPU**: All Apple Silicon Macs
- **NEON SIMD**: All ARM64 processors
- **AMX (Apple Matrix Coprocessor)**: M1 and later

Detection is performed via:
- `sysctl` queries for CPU features
- Metal framework for GPU capabilities
- Runtime feature detection

## Optimization Flags

### Rust
- `target-cpu=native`: Optimize for the build machine's CPU
- `opt-level=3`: Maximum optimization
- `lto=true`: Link-time optimization
- `codegen-units=1`: Single codegen unit for better optimization

### C++
- `-std=c++17`: C++17 standard
- `-march=armv8.6-a`: Target ARMv8.6-A (M4 architecture)
- `-O3`: Maximum optimization
- `-fno-exceptions`: Disable exceptions for performance
- `-fno-rtti`: Disable RTTI for smaller binary size

## Framework Linking

The addon links against macOS frameworks:

- **Foundation**: Core macOS APIs
- **Metal**: GPU compute shaders
- **Accelerate**: BLAS, LAPACK, vDSP (AMX access)

## Future Additions

As implementation progresses, the following will be added:

1. **NTT Processor** (C++)
   - NEON-optimized butterfly operations
   - SME-accelerated matrix operations
   - Metal GPU batch processing

2. **Polynomial Ring** (C++)
   - Cache-optimized memory layouts
   - RNS (Residue Number System) operations

3. **FHE Engine** (C++)
   - Key generation
   - Encryption/decryption
   - Homomorphic operations

4. **Metal Shaders** (`.metal` files)
   - Batch NTT kernels
   - Modular arithmetic kernels
   - Bootstrap accumulator

## Testing

Native code can be tested at multiple levels:

1. **Rust Unit Tests**: `cargo test`
2. **C++ Unit Tests**: (To be added with Google Test)
3. **Integration Tests**: Via Node.js/TypeScript tests
4. **Property Tests**: Via fast-check in TypeScript

## Debugging

### Debug Build
```bash
yarn build:debug
```

Debug builds include:
- Debug symbols
- Assertions enabled
- No optimizations
- Detailed error messages

### Logging
- Rust: Use `println!` or `log` crate
- C++: Use `std::cerr` or `NSLog` (macOS)

### LLDB Debugging
```bash
lldb -- node --inspect-brk test.js
```

## Performance Considerations

- **Zero-copy**: Use napi-rs `Buffer` for large data transfers
- **Async**: Long-running operations use Tokio async runtime
- **Unified Memory**: Leverage Apple Silicon unified memory for CPU/GPU sharing
- **Cache Alignment**: Align data structures to 128-byte cache lines

## Security

- **Memory Safety**: Rust provides memory safety guarantees
- **Secure Enclave**: Future integration for key protection
- **Constant-time**: Critical operations will use constant-time implementations

## References

- [napi-rs Documentation](https://napi.rs/)
- [cxx Documentation](https://cxx.rs/)
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
