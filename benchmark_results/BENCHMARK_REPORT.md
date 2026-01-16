# Comprehensive FHE Hardware Benchmark Report

Testing ALL hardware features on Apple Silicon for FHE operations.

## Summary of Winners

| Operation | Winner | Latency | Speedup |
|-----------|--------|---------|--------|
| Modular Multiplication (n=1024) | Barrett Unrolled (4x) | 1.52 µs | - |
| Modular Multiplication (n=4096) | Barrett Unrolled (4x) | 6.30 µs | - |
| Modular Multiplication (n=16384) | Barrett Unrolled (4x) | 27.44 µs | - |
| Modular Multiplication (n=65536) | Neural Engine | 124.65 µs | - |
| NTT (degree=1024) | SME Tile NTT | 8.69 µs | - |
| NTT (degree=4096) | SME Tile NTT | 41.12 µs | - |
| NTT (degree=16384) | Matrix-form NTT (AMX) | 186.72 µs | - |
| Polynomial Multiplication (degree=1024) | NTT-based | 21.08 µs | - |
| Polynomial Multiplication (degree=4096) | NTT-based | 95.69 µs | - |
| Neural Engine Operations (batch=1024) | CPU Direct | 520.44 ns | - |
| Neural Engine Operations (batch=16384) | CPU Direct | 7.32 µs | - |
| Neural Engine Operations (batch=65536) | CPU Direct | 33.89 µs | - |
| Ray Tracing Operations (base=256, levels=4) | CPU Direct | 36.13 µs | - |
| Texture Sampling (degree=1024, points=10000) | Texture Sampling | 5.10 µs | - |
| Memory System (size=64KB) | memcpy | 551.21 ns | - |
| Memory System (size=1024KB) | Cache-Aligned | 12.34 µs | - |
| Memory System (size=16384KB) | memcpy | 209.90 µs | - |
| Pipelined Operations (degree=4096) | Parallel NTTs | 85.59 µs | - |
| Pipelined Operations (degree=16384) | Parallel NTTs | 274.40 µs | - |

## Detailed Results

### Modular Multiplication (n=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 4.07 µs | 3.33 µs | 245.57 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 3.24 µs | 46.74 ns | 308.85 Kops/s | 1.26x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 3.18 µs | 47.43 ns | 314.01 Kops/s | 1.28x | ✓ |
| Montgomery | CPU Scalar | 2.07 µs | 35.33 ns | 483.58 Kops/s | 1.97x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 1.52 µs | 28.71 ns | 657.71 Kops/s | 2.68x | ✓ |
| Neural Engine | Neural Engine | 1.77 µs | 29.20 ns | 563.91 Kops/s | 2.30x | ✓ |

**Winner:** Barrett Unrolled (4x)

### Modular Multiplication (n=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 13.02 µs | 61.32 ns | 76.82 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 12.50 µs | 49.18 ns | 80.02 Kops/s | 1.04x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 12.44 µs | 374.58 ns | 80.36 Kops/s | 1.05x | ✓ |
| Montgomery | CPU Scalar | 8.18 µs | 277.63 ns | 122.30 Kops/s | 1.59x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 6.30 µs | 64.39 ns | 158.79 Kops/s | 2.07x | ✓ |
| Barrett Parallel | Hybrid | 157.42 µs | 19.04 µs | 6.35 Kops/s | 0.08x | ✓ |
| Neural Engine | Neural Engine | 7.13 µs | 371.82 ns | 140.26 Kops/s | 1.83x | ✓ |

**Winner:** Barrett Unrolled (4x)

### Modular Multiplication (n=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 53.99 µs | 1.78 µs | 18.52 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 52.97 µs | 3.30 µs | 18.88 Kops/s | 1.02x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 45.96 µs | 8.53 µs | 21.76 Kops/s | 1.17x | ✓ |
| Montgomery | CPU Scalar | 34.45 µs | 2.78 µs | 29.02 Kops/s | 1.57x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 27.44 µs | 3.04 µs | 36.44 Kops/s | 1.97x | ✓ |
| Barrett Parallel | Hybrid | 183.78 µs | 7.53 µs | 5.44 Kops/s | 0.29x | ✓ |
| Neural Engine | Neural Engine | 27.85 µs | 1.42 µs | 35.91 Kops/s | 1.94x | ✓ |

**Winner:** Barrett Unrolled (4x)

### Modular Multiplication (n=65536)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 201.87 µs | 13.79 µs | 4.95 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 192.81 µs | 10.56 µs | 5.19 Kops/s | 1.05x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 188.44 µs | 10.00 µs | 5.31 Kops/s | 1.07x | ✓ |
| Montgomery | CPU Scalar | 139.47 µs | 12.61 µs | 7.17 Kops/s | 1.45x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 137.11 µs | 39.65 µs | 7.29 Kops/s | 1.47x | ✓ |
| Barrett Parallel | Hybrid | 178.01 µs | 7.08 µs | 5.62 Kops/s | 1.13x | ✓ |
| Neural Engine | Neural Engine | 124.65 µs | 30.70 µs | 8.02 Kops/s | 1.62x | ✓ |

**Winner:** Neural Engine

### NTT (degree=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Scalar NTT | CPU Scalar | 16.94 µs | 457.41 ns | 59.04 Kops/s | 1.00x | ✓ |
| NEON NTT | NEON | 17.24 µs | 1.26 µs | 58.01 Kops/s | 0.98x | ✓ |
| Barrett NTT | CPU Scalar | 13.97 µs | 2.53 µs | 71.56 Kops/s | 1.21x | ✓ |
| Montgomery NTT | CPU Scalar | 8.92 µs | 1.27 µs | 112.15 Kops/s | 1.90x | ✓ |
| Matrix-form NTT (AMX) | AMX | 8.87 µs | 1.69 µs | 112.68 Kops/s | 1.91x | ✓ |
| SME Tile NTT | SME | 8.69 µs | 967.15 ns | 115.13 Kops/s | 1.95x | ✓ |

**Winner:** SME Tile NTT

### NTT (degree=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Scalar NTT | CPU Scalar | 94.17 µs | 40.22 µs | 10.62 Kops/s | 1.00x | ✓ |
| NEON NTT | NEON | 86.90 µs | 7.02 µs | 11.51 Kops/s | 1.08x | ✓ |
| Barrett NTT | CPU Scalar | 67.15 µs | 14.28 µs | 14.89 Kops/s | 1.40x | ✓ |
| Montgomery NTT | CPU Scalar | 55.27 µs | 72.03 µs | 18.09 Kops/s | 1.70x | ✓ |
| Matrix-form NTT (AMX) | AMX | 49.05 µs | 31.28 µs | 20.39 Kops/s | 1.92x | ✓ |
| SME Tile NTT | SME | 41.12 µs | 3.16 µs | 24.32 Kops/s | 2.29x | ✓ |

**Winner:** SME Tile NTT

### NTT (degree=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Scalar NTT | CPU Scalar | 396.95 µs | 27.26 µs | 2.52 Kops/s | 1.00x | ✓ |
| NEON NTT | NEON | 397.19 µs | 16.27 µs | 2.52 Kops/s | 1.00x | ✓ |
| Barrett NTT | CPU Scalar | 386.23 µs | 61.00 µs | 2.59 Kops/s | 1.03x | ✓ |
| Montgomery NTT | CPU Scalar | 187.11 µs | 7.68 µs | 5.34 Kops/s | 2.12x | ✓ |
| Matrix-form NTT (AMX) | AMX | 186.72 µs | 9.00 µs | 5.36 Kops/s | 2.13x | ✓ |
| SME Tile NTT | SME | 187.02 µs | 6.22 µs | 5.35 Kops/s | 2.12x | ✓ |

**Winner:** Matrix-form NTT (AMX)

### Polynomial Multiplication (degree=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| NTT-based | CPU Scalar | 21.08 µs | 1.75 µs | 47.44 Kops/s | 1.00x | ✓ |

**Winner:** NTT-based

### Polynomial Multiplication (degree=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| NTT-based | CPU Scalar | 95.69 µs | 3.98 µs | 10.45 Kops/s | 1.00x | ✓ |

**Winner:** NTT-based

### Neural Engine Operations (batch=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 520.44 ns | 36.31 ns | 1.92 Mops/s | 1.00x | ✓ |
| Neural Engine | Neural Engine | 808.36 ns | 22.16 ns | 1.24 Mops/s | 0.64x | ✓ |
| Neural Engine Poseidon | Neural Engine | 1.95 ms | 45.97 µs | 524.37 Kops/s | 1.00x | ✓ |

**Winner:** CPU Direct

### Neural Engine Operations (batch=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 7.32 µs | 39.67 ns | 136.67 Kops/s | 1.00x | ✓ |
| Neural Engine | Neural Engine | 14.13 µs | 603.21 ns | 70.77 Kops/s | 0.52x | ✓ |
| Neural Engine Poseidon | Neural Engine | 31.43 ms | 403.82 µs | 521.36 Kops/s | 1.00x | ✓ |

**Winner:** CPU Direct

### Neural Engine Operations (batch=65536)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 33.89 µs | 2.22 µs | 29.51 Kops/s | 1.00x | ✓ |
| Neural Engine | Neural Engine | 53.58 µs | 2.17 µs | 18.66 Kops/s | 0.63x | ✓ |
| Neural Engine Poseidon | Neural Engine | 125.56 ms | 630.35 µs | 521.94 Kops/s | 1.00x | ✓ |

**Winner:** CPU Direct

### Ray Tracing Operations (base=256, levels=4)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 36.13 µs | 1.09 µs | 27.68 Kops/s | 1.00x | ✓ |
| Ray Tracing BVH | Ray Tracing | 42.85 µs | 3.78 µs | 23.34 Kops/s | 0.84x | ✓ |

**Winner:** CPU Direct

### Texture Sampling (degree=1024, points=10000)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Horner's Method | CPU Scalar | 8.93 ms | 115.31 µs | 111.94 ops/s | 1.00x | ✓ |
| Texture Sampling | Texture Sampling | 5.10 µs | 272.17 ns | 195.90 Kops/s | 1750.06x | ✓ |

**Winner:** Texture Sampling

### Memory System (size=64KB)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| memcpy | CPU Scalar | 551.21 ns | 17.66 ns | 0.00 ops/s | 1.00x | ✓ |
| Unified Memory | Hybrid | 949.98 ns | 31.32 ns | 0.00 ops/s | 0.58x | ✓ |
| Cache-Aligned | CPU Scalar | 553.34 ns | 19.74 ns | 0.00 ops/s | 1.00x | ✓ |

**Winner:** memcpy

### Memory System (size=1024KB)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| memcpy | CPU Scalar | 12.44 µs | 353.47 ns | 0.00 ops/s | 1.00x | ✓ |
| Unified Memory | Hybrid | 12.96 µs | 64.35 ns | 0.00 ops/s | 0.96x | ✓ |
| Cache-Aligned | CPU Scalar | 12.34 µs | 54.77 ns | 0.00 ops/s | 1.01x | ✓ |

**Winner:** Cache-Aligned

### Memory System (size=16384KB)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| memcpy | CPU Scalar | 209.90 µs | 8.46 µs | 0.00 ops/s | 1.00x | ✓ |
| Unified Memory | Hybrid | 211.53 µs | 6.33 µs | 0.00 ops/s | 0.99x | ✓ |
| Cache-Aligned | CPU Scalar | 210.88 µs | 5.83 µs | 0.00 ops/s | 1.00x | ✓ |

**Winner:** memcpy

### Pipelined Operations (degree=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Sequential | CPU Scalar | 95.91 µs | 3.25 µs | 10.43 Kops/s | 1.00x | ✓ |
| Parallel NTTs | Hybrid | 85.59 µs | 10.71 µs | 11.68 Kops/s | 1.12x | ✓ |

**Winner:** Parallel NTTs

### Pipelined Operations (degree=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Sequential | CPU Scalar | 423.19 µs | 17.30 µs | 2.36 Kops/s | 1.00x | ✓ |
| Parallel NTTs | Hybrid | 274.40 µs | 15.16 µs | 3.64 Kops/s | 1.54x | ✓ |

**Winner:** Parallel NTTs

