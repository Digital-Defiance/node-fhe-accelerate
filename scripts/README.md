# Build Scripts

This directory contains build and development scripts for the FHE acceleration library.

## Scripts

### compile-shaders.sh

Compiles Metal compute shaders to `.metallib` format.

**Usage:**
```bash
./scripts/compile-shaders.sh [debug|release]
```

**What it does:**
1. Finds all `.metal` files in `cpp/shaders/`
2. Compiles each to `.air` (Apple Intermediate Representation)
3. Links all `.air` files into `fhe_shaders.metallib`
4. Generates `shader_manifest.json` with metadata
5. Copies artifacts to appropriate locations

**Build Modes:**
- `debug`: Includes debug symbols and source information
- `release`: Optimized for performance (default)

**Environment Variables:**
- `FHE_SHADER_HOT_RELOAD`: Enable hot-reloading (requires `fswatch`)

**Requirements:**
- macOS with Xcode Command Line Tools
- Metal compiler (`xcrun metal`)
- `fswatch` (optional, for hot-reload)

**Examples:**
```bash
# Release build (optimized)
./scripts/compile-shaders.sh release

# Debug build (with symbols)
./scripts/compile-shaders.sh debug

# Hot-reload mode (watches for changes)
FHE_SHADER_HOT_RELOAD=1 ./scripts/compile-shaders.sh debug
```

**Output:**
- `target/shaders/*.air` - Intermediate files
- `dist/shaders/fhe_shaders.metallib` - Compiled library
- `dist/shaders/shader_manifest.json` - Metadata
- `fhe_shaders.metallib` - Copy in project root (release only)

**Integration:**
This script is automatically called by:
- `yarn build` - Full build
- `yarn build:shaders` - Shader-only build
- `yarn build:debug` - Debug build

**Troubleshooting:**

*Error: Metal compiler not found*
```bash
xcode-select --install
```

*Error: Shader compilation failed*
- Check `.metal` file syntax
- Review error messages in output
- Ensure all includes are correct

*Warning: No .metal files found*
- Verify `cpp/shaders/` directory exists
- Check that shader files have `.metal` extension

## Adding New Scripts

When adding new build scripts:

1. Make them executable: `chmod +x scripts/your-script.sh`
2. Add usage documentation in this README
3. Use consistent error handling and output formatting
4. Test on clean checkout
5. Update `package.json` scripts if needed

## Script Conventions

- Use `set -e` to exit on errors
- Print colored status messages (GREEN for success, RED for errors)
- Check for required tools before running
- Provide helpful error messages
- Support both debug and release modes where applicable
