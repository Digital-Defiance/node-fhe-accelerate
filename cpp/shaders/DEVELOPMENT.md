# Metal Shader Development Guide

This guide covers developing and debugging Metal compute shaders for the FHE acceleration library.

## Quick Start

### Compiling Shaders

```bash
# Compile shaders in release mode (optimized)
yarn build:shaders

# Compile shaders in debug mode (with debug symbols)
yarn build:shaders:debug

# Watch for changes and auto-recompile (development)
yarn build:shaders:watch
```

### Hot-Reloading

For rapid iteration during development:

1. Set the environment variable:
   ```bash
   export FHE_SHADER_HOT_RELOAD=1
   ```

2. Run your application in debug mode:
   ```bash
   yarn build:debug
   node your_test.js
   ```

3. Edit shader files - they will be automatically recompiled and reloaded

**Note**: Hot-reloading only works in debug builds and requires `fswatch` to be installed:
```bash
brew install fswatch
```

## Shader Organization

### Directory Structure

```
cpp/shaders/
├── common/           # Shared utilities and types
│   └── fhe_common.metal
├── ntt/             # Number Theoretic Transform shaders
│   ├── ntt_forward.metal
│   └── ntt_inverse.metal
├── modular/         # Modular arithmetic shaders
│   └── modmul.metal
└── bootstrap/       # Bootstrapping shaders
    └── accumulator.metal
```

### Naming Conventions

- **Files**: Use lowercase with underscores (e.g., `ntt_forward.metal`)
- **Kernels**: Use lowercase with underscores (e.g., `ntt_forward_stage`)
- **Types**: Use `_t` suffix (e.g., `coeff_t`, `index_t`)
- **Constants**: Use UPPER_CASE (e.g., `MAX_DEGREE`)

## Writing Shaders

### Basic Kernel Template

```metal
#include "../common/fhe_common.metal"

/// Brief description of what this kernel does
///
/// Detailed explanation of the algorithm, thread organization,
/// and performance characteristics.
///
/// @param input Description of input buffer
/// @param output Description of output buffer
/// @param params Description of parameters
kernel void my_kernel(
    device const coeff_t* input [[buffer(0)]],
    device coeff_t* output [[buffer(1)]],
    constant MyParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]]
) {
    // Bounds checking
    if (gid.x >= params.size) return;
    
    // Kernel implementation
    // ...
}
```

### Common Patterns

#### 1. Parallel Element-Wise Operations

```metal
kernel void element_wise_op(
    device const coeff_t* input [[buffer(0)]],
    device coeff_t* output [[buffer(1)]],
    constant uint32_t& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Process one element per thread
    output[gid] = process(input[gid]);
}
```

#### 2. Reduction Operations

```metal
kernel void reduction(
    device const coeff_t* input [[buffer(0)]],
    device coeff_t* output [[buffer(1)]],
    constant uint32_t& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    threadgroup coeff_t shared[256];
    
    // Load and reduce in shared memory
    shared[tid] = input[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        output[0] = shared[0];
    }
}
```

#### 3. Batch Processing

```metal
kernel void batch_operation(
    device const coeff_t* input [[buffer(0)]],
    device coeff_t* output [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    constant uint32_t& element_size [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    uint element_idx = gid.x;
    
    if (batch_idx >= batch_size || element_idx >= element_size) return;
    
    uint offset = batch_idx * element_size;
    output[offset + element_idx] = process(input[offset + element_idx]);
}
```

## Performance Optimization

### Thread Group Sizing

Choose threadgroup sizes based on:
- **Thread execution width**: Typically 32 on Apple GPUs
- **Shared memory usage**: Limited to 32KB per threadgroup
- **Register pressure**: More threads = fewer registers per thread

Recommended sizes:
- Simple operations: 256-512 threads
- Complex operations: 64-128 threads
- Operations with shared memory: Depends on memory usage

### Memory Access Patterns

1. **Coalesced Access**: Adjacent threads should access adjacent memory
   ```metal
   // Good: coalesced
   coeff_t value = input[gid];
   
   // Bad: strided
   coeff_t value = input[gid * stride];
   ```

2. **Shared Memory**: Use for data reuse within threadgroup
   ```metal
   threadgroup coeff_t shared[256];
   shared[tid] = input[gid];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   ```

3. **Avoid Bank Conflicts**: Pad shared memory if needed
   ```metal
   // Add padding to avoid bank conflicts
   threadgroup coeff_t shared[256 + 8];
   ```

### Arithmetic Optimization

1. **Use Montgomery Form**: Keep coefficients in Montgomery form
2. **Minimize Modular Reductions**: Batch operations when possible
3. **Exploit SIMD**: Metal compiler auto-vectorizes when possible

## Debugging

### Shader Compilation Errors

Compilation errors are shown during the build:
```bash
yarn build:shaders:debug
```

Common errors:
- **Undefined function**: Check includes and function names
- **Type mismatch**: Ensure buffer types match kernel parameters
- **Invalid buffer index**: Buffer indices must be compile-time constants

### Runtime Debugging

1. **Metal Debugger** (Xcode):
   - Attach to your Node.js process
   - Capture GPU frame
   - Inspect shader execution

2. **Printf Debugging**:
   Metal doesn't support printf, but you can:
   - Write debug values to a separate buffer
   - Read back and print from C++

3. **Validation Layer**:
   Enable Metal validation in debug builds:
   ```bash
   export METAL_DEVICE_WRAPPER_TYPE=1
   export METAL_DEBUG_ERROR_MODE=0
   ```

### Performance Profiling

Use Xcode Instruments:
```bash
# Build with release optimizations
yarn build:shaders

# Profile with Instruments
instruments -t "GPU" node your_script.js
```

Key metrics:
- **GPU utilization**: Should be >80% for compute-bound workloads
- **Memory bandwidth**: Check for memory bottlenecks
- **Occupancy**: Percentage of GPU cores active

## Testing

### Unit Tests

Test individual kernels with known inputs/outputs:

```cpp
// C++ test
void test_ntt_forward() {
    // Create test data
    std::vector<uint64_t> input = {1, 2, 3, 4};
    std::vector<uint64_t> expected = {10, -2, -2, -2};
    
    // Run kernel
    auto output = run_ntt_forward(input);
    
    // Verify
    assert(output == expected);
}
```

### Property-Based Tests

Test mathematical properties:

```typescript
// TypeScript property test
test.prop([fc.array(fc.bigInt(), {minLength: 1024, maxLength: 1024})])
  ('NTT round-trip preserves values', (coeffs) => {
    const ntt_result = ntt_forward(coeffs);
    const recovered = ntt_inverse(ntt_result);
    expect(recovered).toEqual(coeffs);
  });
```

### Benchmarking

Compare shader performance:

```cpp
auto start = std::chrono::high_resolution_clock::now();
run_kernel(data);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "Kernel took: " << duration.count() << " μs" << std::endl;
```

## Best Practices

1. **Document Everything**: Explain algorithm, thread organization, and performance
2. **Bounds Checking**: Always check thread indices against buffer sizes
3. **Use Common Utilities**: Leverage functions from `fhe_common.metal`
4. **Test Incrementally**: Test each kernel independently before integration
5. **Profile Early**: Identify bottlenecks before optimizing
6. **Version Control**: Commit working shaders before major changes
7. **Reference Design Doc**: Link to relevant design document sections

## Common Issues

### Issue: Shader Not Found

**Symptom**: Runtime error "Shader function not found"

**Solutions**:
- Ensure shader is compiled: `yarn build:shaders`
- Check function name matches exactly (case-sensitive)
- Verify metallib is in the correct location

### Issue: Incorrect Results

**Symptom**: Kernel produces wrong output

**Solutions**:
- Check modular arithmetic (overflow, reduction)
- Verify thread indexing (off-by-one errors)
- Test with small, known inputs
- Compare against CPU reference implementation

### Issue: Poor Performance

**Symptom**: Kernel is slower than expected

**Solutions**:
- Profile with Instruments to identify bottleneck
- Check GPU utilization (should be >80%)
- Optimize memory access patterns
- Adjust threadgroup size
- Reduce register pressure

### Issue: Compilation Warnings

**Symptom**: Warnings during shader compilation

**Solutions**:
- Fix unused variable warnings
- Address type conversion warnings
- Enable `-Werror` to treat warnings as errors

## Resources

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/best_practices)
- [GPU Programming Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)

## Getting Help

If you encounter issues:

1. Check this guide and the design document
2. Review existing shader implementations
3. Test with minimal examples
4. Profile to identify bottlenecks
5. Ask for help with specific error messages and profiling data
