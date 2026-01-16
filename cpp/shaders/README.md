# Metal Shaders

This directory contains Metal compute shaders for GPU-accelerated FHE operations.

## Directory Structure

- `ntt/` - Number Theoretic Transform shaders
- `modular/` - Modular arithmetic shaders
- `bootstrap/` - Bootstrapping operation shaders
- `common/` - Shared utility functions and headers

## Shader Compilation

Shaders are compiled to `.metallib` format during the build process using the `metal` and `metallib` command-line tools.

### Build Process

1. Individual `.metal` files are compiled to `.air` (Apple Intermediate Representation)
2. `.air` files are linked into a single `.metallib` library
3. The `.metallib` is embedded into the native addon or loaded at runtime

### Development

For hot-reloading during development:
- Set `FHE_SHADER_HOT_RELOAD=1` environment variable
- Shaders will be recompiled and reloaded on file changes
- Only available in debug builds

## Shader Organization

Each shader file should include:
- Clear documentation of inputs/outputs
- Thread group size configuration
- Performance characteristics
- References to design document sections
