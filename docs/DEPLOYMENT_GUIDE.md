# Deployment Guide for Mac Studio/M4 Max

This guide provides comprehensive instructions for deploying the `@digitaldefiance/node-fhe-accelerate` voting system on Mac Studio with M4 Max hardware.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [System Prerequisites](#system-prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Production Deployment](#production-deployment)
6. [Performance Tuning](#performance-tuning)
7. [Verification](#verification)

## Hardware Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **Machine** | Mac Studio (M4 Max) |
| **CPU** | Apple M4 Max (12 P-cores + 4 E-cores) |
| **RAM** | 64 GB Unified Memory |
| **Storage** | 1 TB SSD (NVMe) |
| **Network** | 10 Gbps Ethernet |

### Recommended for Large Elections (>100K voters)

| Component | Specification |
|-----------|---------------|
| **Machine** | Mac Studio (M4 Max) |
| **CPU** | Apple M4 Max |
| **RAM** | 128 GB Unified Memory |
| **Storage** | 2 TB SSD (NVMe) |
| **Network** | 10 Gbps Ethernet (redundant) |
| **Backup** | External NVMe enclosure |

### Hardware Acceleration Features

The M4 Max provides the following acceleration capabilities:

- **SME (Scalable Matrix Extension)**: Matrix operations for NTT
- **Metal GPU**: 40 cores for parallel ballot processing
- **Neural Engine**: 38 TOPS for hash computations
- **NEON SIMD**: 128-bit vector operations
- **Unified Memory**: 400 GB/s bandwidth for zero-copy operations

## System Prerequisites

### 1. macOS Configuration

```bash
# Verify macOS version (Sequoia 15.2+ recommended)
sw_vers

# Disable sleep and screen saver for production
sudo pmset -a sleep 0
sudo pmset -a displaysleep 0
sudo pmset -a disksleep 0

# Enable performance mode
sudo nvram boot-args="serverperfmode=1"
```

### 2. Install Xcode Command Line Tools

```bash
xcode-select --install
```

### 3. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 4. Install Node.js (v20 LTS)

```bash
brew install node@20
echo 'export PATH="/opt/homebrew/opt/node@20/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 5. Install Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add aarch64-apple-darwin
source "$HOME/.cargo/env"
```

### 6. Enable Yarn

```bash
corepack enable
yarn set version stable
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Digital-Defiance/node-fhe-accelerate.git
cd node-fhe-accelerate
```

### 2. Install Dependencies

```bash
yarn install
```

### 3. Build for Production

```bash
# Full production build
yarn build

# Verify native addon
node -e "require('./index.node')"
```

### 4. Run Tests

```bash
# Run all tests
yarn test

# Run with coverage
yarn test:coverage
```

## Configuration

### Environment Variables

Create a `.env` file for production configuration:

```bash
# Server Configuration
NODE_ENV=production
PORT=3000
HOST=0.0.0.0

# FHE Configuration
FHE_SECURITY_LEVEL=128
FHE_POLYNOMIAL_DEGREE=16384
FHE_USE_GPU=true
FHE_USE_SME=true

# Election Configuration
ELECTION_MAX_VOTERS=1000000
ELECTION_MAX_CANDIDATES=100
ELECTION_THRESHOLD_MIN=3
ELECTION_THRESHOLD_MAX=10

# Logging
LOG_LEVEL=info
LOG_FORMAT=json
LOG_FILE=/var/log/fhe-voting/app.log

# Security
TLS_CERT_PATH=/etc/ssl/certs/voting.crt
TLS_KEY_PATH=/etc/ssl/private/voting.key
SECURE_ENCLAVE_ENABLED=true

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30000
```

### TLS Certificate Setup

```bash
# Generate self-signed certificate for testing
openssl req -x509 -newkey rsa:4096 -keyout voting.key -out voting.crt -days 365 -nodes

# For production, use Let's Encrypt or your CA
sudo certbot certonly --standalone -d voting.example.com
```

### Firewall Configuration

```bash
# Allow HTTPS traffic
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/node
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/node

# Configure pf firewall (optional)
echo "pass in on en0 proto tcp from any to any port 443" | sudo pfctl -ef -
```

## Production Deployment

### 1. Create Service User

```bash
# Create dedicated user for the voting service
sudo dscl . -create /Users/fhevoting
sudo dscl . -create /Users/fhevoting UserShell /bin/bash
sudo dscl . -create /Users/fhevoting RealName "FHE Voting Service"
sudo dscl . -create /Users/fhevoting UniqueID 550
sudo dscl . -create /Users/fhevoting PrimaryGroupID 20
sudo dscl . -create /Users/fhevoting NFSHomeDirectory /var/fhevoting
sudo mkdir -p /var/fhevoting
sudo chown fhevoting:staff /var/fhevoting
```

### 2. Create LaunchDaemon

Create `/Library/LaunchDaemons/com.digitaldefiance.fhevoting.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.digitaldefiance.fhevoting</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/node</string>
        <string>/opt/fhevoting/server.js</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/opt/fhevoting</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>NODE_ENV</key>
        <string>production</string>
    </dict>
    <key>UserName</key>
    <string>fhevoting</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/fhevoting/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/fhevoting/stderr.log</string>
</dict>
</plist>
```

### 3. Start Service

```bash
# Load the service
sudo launchctl load /Library/LaunchDaemons/com.digitaldefiance.fhevoting.plist

# Check status
sudo launchctl list | grep fhevoting

# View logs
tail -f /var/log/fhevoting/stdout.log
```

### 4. Nginx Reverse Proxy (Optional)

Install and configure Nginx for load balancing:

```bash
brew install nginx
```

Create `/opt/homebrew/etc/nginx/servers/fhevoting.conf`:

```nginx
upstream fhevoting {
    server 127.0.0.1:3000;
    keepalive 64;
}

server {
    listen 443 ssl http2;
    server_name voting.example.com;

    ssl_certificate /etc/ssl/certs/voting.crt;
    ssl_certificate_key /etc/ssl/private/voting.key;
    ssl_protocols TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES256-GCM-SHA384;

    location / {
        proxy_pass http://fhevoting;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    location /health {
        proxy_pass http://fhevoting/health;
        proxy_read_timeout 5s;
    }
}
```

## Performance Tuning

### 1. System Limits

```bash
# Increase file descriptor limits
sudo launchctl limit maxfiles 65536 200000

# Add to /etc/sysctl.conf
sudo sysctl -w kern.maxfiles=65536
sudo sysctl -w kern.maxfilesperproc=65536
```

### 2. Node.js Optimization

```bash
# Set Node.js memory limit (adjust based on available RAM)
export NODE_OPTIONS="--max-old-space-size=32768"

# Enable V8 optimizations
export NODE_OPTIONS="$NODE_OPTIONS --optimize-for-size"
```

### 3. Hardware Acceleration Verification

```bash
# Run hardware benchmark
node -e "
const { detectHardware } = require('./index.node');
const caps = detectHardware();
console.log('Hardware Capabilities:', JSON.stringify(caps, null, 2));
"
```

Expected output:
```json
{
  "sme": true,
  "sme2": true,
  "amx": true,
  "metalGpu": true,
  "metalGpuCores": 40,
  "neuralEngine": true,
  "neuralEngineTops": 38,
  "neon": true,
  "unifiedMemoryGB": 128
}
```

### 4. Benchmark Performance

```bash
# Run comprehensive benchmark
./run_comprehensive_benchmark

# Expected results on M4 Max:
# - Encryption: < 10ms
# - Homomorphic Addition: < 1ms
# - Homomorphic Multiplication: < 50ms
# - Bootstrapping: < 20ms
# - Ballot Processing: > 10,000/second
```

## Verification

### 1. Health Check

```bash
curl -k https://localhost:443/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "hardware": {
    "sme": true,
    "gpu": true,
    "neuralEngine": true
  }
}
```

### 2. Smoke Test

```bash
# Run integration tests
yarn test:integration

# Run voting simulation
node scripts/smoke-test.js
```

### 3. Load Test

```bash
# Install k6 for load testing
brew install k6

# Run load test
k6 run scripts/load-test.js
```

## Troubleshooting

### Common Issues

1. **Native addon fails to load**
   ```bash
   # Rebuild native addon
   yarn clean && yarn build:native
   ```

2. **GPU not detected**
   ```bash
   # Check Metal availability
   system_profiler SPDisplaysDataType
   ```

3. **High memory usage**
   ```bash
   # Monitor memory
   vm_stat 1
   
   # Check for leaks
   leaks -atExit -- node server.js
   ```

4. **Slow performance**
   ```bash
   # Profile with Instruments
   xcrun xctrace record --template 'Time Profiler' --launch -- node server.js
   ```

## Next Steps

- [Election Official Setup](./ELECTION_OFFICIAL_SETUP.md)
- [Security Best Practices](./SECURITY_BEST_PRACTICES.md)
- [Monitoring and Alerting](./MONITORING_ALERTING.md)
- [Disaster Recovery](./DISASTER_RECOVERY.md)
