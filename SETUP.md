# Setup Guide

Quick start guide for setting up the development environment for `@digitaldefiance/node-fhe-accelerate`.

## Prerequisites

This library requires:
- **macOS** (Apple Silicon only - M1/M2/M3/M4)
- **Xcode Command Line Tools**
- **Rust** (1.70+)
- **Node.js** (18+)
- **Yarn** (4.0+)

## Quick Setup

### 1. Install Xcode Command Line Tools

```bash
xcode-select --install
```

### 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add aarch64-apple-darwin
```

### 3. Install Node.js and Yarn

```bash
# Install Node.js via Homebrew
brew install node

# Enable Yarn
corepack enable
yarn set version stable
```

### 4. Install Dependencies

```bash
yarn install
```

### 5. Build the Project

```bash
# Full build (TypeScript + Native)
yarn build

# Or just the native addon
yarn build:native
```

### 6. Run Tests

```bash
yarn test
```

## Verification

To verify your setup is working:

```bash
# Check versions
node --version    # Should be >= 18.0.0
cargo --version   # Should be >= 1.70.0
yarn --version    # Should be >= 4.0.0

# Build and test
yarn build
yarn test
```

## Next Steps

- See [BUILDING.md](./BUILDING.md) for detailed build instructions
- See [NATIVE_ADDON.md](./NATIVE_ADDON.md) for native addon architecture
- See [README.md](./README.md) for usage examples

## Troubleshooting

### "xcode-select: error: command line tools are already installed"

This is fine - the tools are already installed.

### "cargo: command not found"

Restart your terminal after installing Rust, or run:
```bash
source "$HOME/.cargo/env"
```

### Build fails with framework errors

Ensure Xcode Command Line Tools are installed:
```bash
xcode-select --install
xcode-select -p  # Should show /Library/Developer/CommandLineTools
```

### "This library is optimized for Apple Silicon"

This library only works on Apple Silicon Macs (M1/M2/M3/M4). Intel Macs are not supported.

## Getting Help

If you encounter issues:
1. Check [BUILDING.md](./BUILDING.md) for detailed troubleshooting
2. Ensure all prerequisites are installed
3. Try: `yarn clean && yarn build`
4. Open an issue on GitHub with error details
