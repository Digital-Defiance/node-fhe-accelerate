#!/bin/bash
#
# compile-shaders.sh
# Compile Metal shaders to metallib format
#
# This script compiles all .metal shader files to Apple Intermediate Representation (.air)
# and then links them into a single .metallib library file.
#
# Usage:
#   ./scripts/compile-shaders.sh [debug|release]
#
# Environment Variables:
#   FHE_SHADER_HOT_RELOAD - Enable hot-reloading (development only)
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHADER_DIR="$PROJECT_ROOT/cpp/shaders"
BUILD_DIR="$PROJECT_ROOT/target/shaders"
OUTPUT_DIR="$PROJECT_ROOT/dist/shaders"

# Build mode (debug or release)
BUILD_MODE="${1:-release}"

# Compiler flags
METAL_FLAGS="-std=metal3.0"
if [ "$BUILD_MODE" = "debug" ]; then
    METAL_FLAGS="$METAL_FLAGS -gline-tables-only -frecord-sources"
    echo "Building shaders in DEBUG mode"
else
    METAL_FLAGS="$METAL_FLAGS -O3"
    echo "Building shaders in RELEASE mode"
fi

# Target architecture
METAL_FLAGS="$METAL_FLAGS -target air64-apple-macos14.0"

# Create build directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[SHADER]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if Metal compiler is available
if ! command -v xcrun &> /dev/null; then
    print_error "xcrun not found. Please install Xcode Command Line Tools."
    exit 1
fi

if ! xcrun -sdk macosx metal --version &> /dev/null; then
    print_warning "Metal compiler not found. Shaders will not be compiled."
    print_warning "To install Metal toolchain: xcodebuild -downloadComponent MetalToolchain"
    print_warning "Skipping shader compilation..."
    exit 0
fi

print_status "Metal compiler version:"
xcrun -sdk macosx metal --version

# Find all .metal files
METAL_FILES=$(find "$SHADER_DIR" -name "*.metal" -type f)
AIR_FILES=()

if [ -z "$METAL_FILES" ]; then
    print_warning "No .metal files found in $SHADER_DIR"
    exit 0
fi

print_status "Found $(echo "$METAL_FILES" | wc -l | tr -d ' ') shader files"

# Compile each .metal file to .air
for METAL_FILE in $METAL_FILES; do
    # Get relative path and create corresponding .air filename
    REL_PATH="${METAL_FILE#$SHADER_DIR/}"
    AIR_FILE="$BUILD_DIR/${REL_PATH%.metal}.air"
    AIR_DIR="$(dirname "$AIR_FILE")"
    
    # Create subdirectory if needed
    mkdir -p "$AIR_DIR"
    
    print_status "Compiling $(basename "$METAL_FILE")..."
    
    # Compile .metal to .air
    if xcrun -sdk macosx metal $METAL_FLAGS \
        -I "$SHADER_DIR" \
        -c "$METAL_FILE" \
        -o "$AIR_FILE" 2>&1; then
        print_status "  ✓ $(basename "$AIR_FILE")"
        AIR_FILES+=("$AIR_FILE")
    else
        print_error "Failed to compile $METAL_FILE"
        exit 1
    fi
done

# Link all .air files into a single .metallib
METALLIB_FILE="$OUTPUT_DIR/fhe_shaders.metallib"
print_status "Linking ${#AIR_FILES[@]} .air files into metallib..."

if xcrun -sdk macosx metallib \
    "${AIR_FILES[@]}" \
    -o "$METALLIB_FILE" 2>&1; then
    print_status "✓ Created $METALLIB_FILE"
else
    print_error "Failed to create metallib"
    exit 1
fi

# Print metallib info
METALLIB_SIZE=$(du -h "$METALLIB_FILE" | cut -f1)
print_status "Metallib size: $METALLIB_SIZE"

# List functions in the metallib
print_status "Shader functions:"
xcrun -sdk macosx metal-objdump -function-list "$METALLIB_FILE" 2>/dev/null || true

# Copy to additional locations if needed
if [ "$BUILD_MODE" = "release" ]; then
    # Copy to root for easy access by native addon
    cp "$METALLIB_FILE" "$PROJECT_ROOT/fhe_shaders.metallib"
    print_status "Copied metallib to project root"
fi

# Generate shader manifest (JSON file with metadata)
MANIFEST_FILE="$OUTPUT_DIR/shader_manifest.json"
print_status "Generating shader manifest..."

cat > "$MANIFEST_FILE" << EOF
{
  "version": "1.0.0",
  "build_mode": "$BUILD_MODE",
  "build_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "metallib": "fhe_shaders.metallib",
  "shaders": [
EOF

# Add shader entries
FIRST=true
for METAL_FILE in $METAL_FILES; do
    REL_PATH="${METAL_FILE#$SHADER_DIR/}"
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$MANIFEST_FILE"
    fi
    
    # Extract kernel names from the file (simple grep for 'kernel void')
    KERNELS=$(grep -o 'kernel void [a-zA-Z_][a-zA-Z0-9_]*' "$METAL_FILE" | sed 's/kernel void //' || echo "")
    
    cat >> "$MANIFEST_FILE" << EOF
    {
      "source": "$REL_PATH",
      "kernels": [$(echo "$KERNELS" | sed 's/^/"/;s/$/"/' | paste -sd ',' -)]
    }
EOF
done

cat >> "$MANIFEST_FILE" << EOF

  ]
}
EOF

print_status "✓ Generated $MANIFEST_FILE"

# Hot-reload setup (development only)
if [ "$BUILD_MODE" = "debug" ] && [ -n "$FHE_SHADER_HOT_RELOAD" ]; then
    print_status "Hot-reload enabled - watching for shader changes..."
    print_warning "This is a development feature only"
    
    # Use fswatch if available
    if command -v fswatch &> /dev/null; then
        fswatch -o "$SHADER_DIR" | while read; do
            print_status "Shader change detected, recompiling..."
            "$0" debug
        done
    else
        print_warning "fswatch not installed. Install with: brew install fswatch"
        print_warning "Hot-reload will not work without fswatch"
    fi
fi

print_status "Shader compilation complete!"
